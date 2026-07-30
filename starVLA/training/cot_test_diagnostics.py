"""Low-overhead diagnostics for QwenGR00TCoT test runs."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

def _masked_abs_mean(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    values = (pred.float() - target.float()).abs()
    mask = valid.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)


def _masked_rmse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    values = (pred.float() - target.float()).pow(2)
    mask = valid.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mean_square = (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)
    return torch.sqrt(mean_square.clamp_min(0.0))


def _masked_smooth_l1_mean(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    values = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    mask = valid.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)


def compute_gradient_clipping_metrics(
    *,
    pre_clip_norm: float | torch.Tensor,
    threshold: float,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    """Report the already-computed pre-clip norm and implied clipping scale."""

    norm = float(pre_clip_norm.item() if hasattr(pre_clip_norm, "item") else pre_clip_norm)
    threshold = float(threshold)
    if not np.isfinite(norm) or norm < 0.0:
        raise ValueError(f"pre_clip_norm must be finite and non-negative, got {norm}")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"gradient clipping threshold must be positive and finite, got {threshold}")
    scale = min(1.0, threshold / max(norm, float(eps)))
    return {
        "train/grad_norm_pre_clip": norm,
        "train/grad_clip_threshold": threshold,
        "train/grad_clip_triggered": float(norm > threshold),
        "train/grad_clip_scale": scale,
    }


def resolve_post_step_gradient_norm(
    accelerator_norm: float | torch.Tensor | None,
    model: torch.nn.Module,
) -> float | None:
    """Use Accelerate's norm or DeepSpeed's post-step cached global norm."""

    value = accelerator_norm
    if value is None:
        getter = getattr(model, "get_global_grad_norm", None)
        value = getter() if callable(getter) else None
    if value is None:
        return None
    return float(value.item() if hasattr(value, "item") else value)


def compute_token_utilization_metrics(
    tokens: torch.Tensor,
    *,
    prefix: str,
    attention_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize token diversity and optional attention-pooling utilization."""

    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [B,Q,H], got {tuple(tokens.shape)}")
    token_values = tokens.detach().float()
    batch_size, token_count, _ = token_values.shape
    if token_count < 1:
        raise ValueError("token_count must be positive")

    normalized = F.normalize(token_values, dim=-1, eps=1.0e-12)
    cosine = normalized @ normalized.transpose(1, 2)
    if token_count == 1:
        mean_offdiag_cosine = token_values.new_zeros(())
    else:
        diagonal = torch.diagonal(cosine, dim1=1, dim2=2).sum(dim=1)
        mean_offdiag_cosine = (
            (cosine.sum(dim=(1, 2)) - diagonal) / float(token_count * (token_count - 1))
        ).mean()

    effective_ranks = []
    for sample in token_values:
        centered = sample - sample.mean(dim=0, keepdim=True)
        gram = centered @ centered.transpose(0, 1)
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        total = eigenvalues.sum()
        if float(total.item()) <= 1.0e-12:
            effective_ranks.append(eigenvalues.new_zeros(()))
            continue
        probabilities = eigenvalues / total
        positive = probabilities > 0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        effective_ranks.append(entropy.exp())

    metrics = {
        f"{prefix}/token_mean_offdiag_cosine": float(mean_offdiag_cosine.item()),
        f"{prefix}/token_cov_effective_rank": float(torch.stack(effective_ranks).mean().item()),
    }
    if attention_weights is not None:
        weights = attention_weights.detach().float()
        if weights.shape != (batch_size, token_count):
            raise ValueError(
                f"attention_weights must have shape {(batch_size, token_count)}, got {tuple(weights.shape)}"
            )
        if bool((weights < 0).any()):
            raise ValueError("attention_weights must be non-negative")
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        entropy = -(weights * weights.clamp_min(1.0e-12).log()).sum(dim=-1)
        normalized_entropy = (
            entropy / math.log(token_count)
            if token_count > 1
            else torch.zeros_like(entropy)
        )
        metrics.update(
            {
                f"{prefix}/attention_entropy_nats": float(entropy.mean().item()),
                f"{prefix}/attention_entropy_normalized": float(normalized_entropy.mean().item()),
                f"{prefix}/attention_max_weight": float(weights.max(dim=-1).values.mean().item()),
                f"{prefix}/attention_effective_token_count": float(entropy.exp().mean().item()),
            }
        )
    return metrics


def compute_decoder_reliance_metrics(
    predictions: dict[str, Any],
    examples: list[dict],
    *,
    depth_scale: float,
) -> dict[str, float]:
    """Compare decoder-only summary interventions with normal outputs and targets."""

    interventions = predictions.get("decoder_interventions", {})
    if not interventions:
        return {}
    device = predictions["depth_current"].device
    targets = {
        "depth_current": torch.as_tensor(
            np.stack([example["depth_current"] for example in examples]),
            device=device,
        ),
        "depth_future": torch.as_tensor(
            np.stack([example["depth_future"] for example in examples]),
            device=device,
        ),
    }
    valid = {
        "depth_current": torch.as_tensor(
            np.stack([example["depth_current_valid"] for example in examples]),
            device=device,
        ),
        "depth_future": torch.as_tensor(
            np.stack([example["depth_future_valid"] for example in examples]),
            device=device,
        ),
    }
    metrics = {}
    for variant_name, variant_predictions in interventions.items():
        for depth_name in ("depth_current", "depth_future"):
            variant = variant_predictions[depth_name]
            normal = predictions[depth_name]
            prefix = f"decoder_reliance/{variant_name}/{depth_name}"
            metrics[f"{prefix}_delta_mae_m"] = float(
                (_masked_abs_mean(variant, normal, valid[depth_name]) * depth_scale).item()
            )
            metrics[f"{prefix}_target_mae_m"] = float(
                (_masked_abs_mean(variant, targets[depth_name], valid[depth_name]) * depth_scale).item()
            )
    return metrics


def collect_batch_valid_ratios(examples: list[dict]) -> dict[str, float]:
    """Summarize label validity without retaining tensors or image data."""
    output: dict[str, float] = {}
    for key, name in (
        ("depth_current_valid", "data/depth_current_valid_ratio"),
        ("depth_future_valid", "data/depth_future_valid_ratio"),
        ("uvd_valid_mask", "data/uvd_valid_ratio"),
        ("uvd_out_of_frame_mask", "data/uvd_out_of_frame_ratio"),
        ("uvd_boundary_clamp_mask", "data/uvd_boundary_clamp_ratio"),
    ):
        values = [np.asarray(example[key], dtype=np.float32).mean() for example in examples if key in example]
        if values:
            output[name] = float(np.mean(values))
    return output


def _diagnostic_module_parameter_ids(model: torch.nn.Module) -> dict[str, set[int]]:
    """Resolve attribution groups by object identity across model wrappers."""
    unwrapped = model
    while isinstance(getattr(unwrapped, "module", None), torch.nn.Module):
        unwrapped = unwrapped.module
    module_paths = {
        "query": "geometry_query",
        "depth_decoder": "depth_decoder",
        "uvd_head": "uvd_head",
        "action_model": "action_model",
    }
    groups = {}
    for group, module_path in module_paths.items():
        module = getattr(unwrapped, module_path, None)
        groups[group] = {id(parameter) for parameter in module.parameters()} if module is not None else set()
    return groups


def install_module_grad_norm_hooks(model: torch.nn.Module) -> dict[str, Any]:
    """Capture parameter gradient energy before ZeRO can partition/clear it."""
    parameter_ids = _diagnostic_module_parameter_ids(model)
    state: dict[str, Any] = {"sums": {group: None for group in parameter_ids}, "handles": []}
    parameter_to_group = {
        parameter_id: group for group, ids in parameter_ids.items() for parameter_id in ids
    }
    unwrapped = model
    while isinstance(getattr(unwrapped, "module", None), torch.nn.Module):
        unwrapped = unwrapped.module
    for parameter in unwrapped.parameters():
        group = parameter_to_group.get(id(parameter))
        if group is None or not parameter.requires_grad:
            continue

        def record_gradient(gradient, *, group=group):
            value = gradient.detach().float().pow(2).sum()
            previous = state["sums"][group]
            state["sums"][group] = value if previous is None else previous + value
            return gradient

        state["handles"].append(parameter.register_hook(record_gradient))
    return state


def set_module_grad_norm_collection(hook_state: dict[str, Any], enabled: bool) -> None:
    """Enable expensive gradient accumulation only on selected optimizer steps."""
    hook_state["enabled"] = bool(enabled)
    for group in hook_state["sums"]:
        hook_state["sums"][group] = None


def collect_module_grad_norms(
    model: torch.nn.Module,
    *,
    hook_state: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Collect global norms for the attribution-relevant module groups."""
    parameter_ids = _diagnostic_module_parameter_ids(model)
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        return {f"grad/{name}_norm": 0.0 for name in parameter_ids}
    if hook_state is not None:
        sums = {
            name: (
                hook_state["sums"][name]
                if hook_state["sums"][name] is not None
                else torch.zeros((), device=first_parameter.device, dtype=torch.float32)
            )
            for name in parameter_ids
        }
    else:
        # Fallback for non-ZeRO callers where parameter.grad remains available.
        sums = {name: torch.zeros((), device=first_parameter.device, dtype=torch.float32) for name in parameter_ids}
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().pow(2).sum()
            for group, ids in parameter_ids.items():
                if id(parameter) in ids:
                    sums[group] += value
    if dist.is_available() and dist.is_initialized():
        for value in sums.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if hook_state is not None:
        for handle in hook_state["handles"]:
            handle.remove()
        hook_state["handles"].clear()
        for group in hook_state["sums"]:
            hook_state["sums"][group] = None
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    parameter_counts = {name: 0 for name in parameter_ids}
    for parameter in model.parameters():
        for name, ids in parameter_ids.items():
            if id(parameter) in ids:
                parameter_counts[name] += parameter.numel()
    metrics = {}
    for name, value in sums.items():
        norm = torch.sqrt(value.clamp_min(0.0))
        denominator = max(parameter_counts[name] * world_size, 1)
        metrics[f"grad/{name}_norm"] = float(norm.cpu().item())
        metrics[f"grad/{name}_local_rms"] = float((norm / denominator**0.5).cpu().item())
    return metrics


def _pad_uvd_examples(
    examples: list[dict],
    point_count: int,
    *,
    hand_count: int = 1,
    order: str = "hand_major",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hand_count = int(hand_count)
    point_count = int(point_count)
    if hand_count < 1 or point_count % hand_count != 0:
        raise ValueError(f"point_count={point_count} must be divisible by hand_count={hand_count}")
    if order not in {"hand_major", "time_major"}:
        raise ValueError(f"uvd order must be 'hand_major' or 'time_major', got {order!r}")
    points_per_hand = point_count // hand_count
    target = np.zeros((len(examples), point_count, 3), dtype=np.float32)
    valid = np.zeros((len(examples), point_count), dtype=np.bool_)
    times = np.zeros((len(examples), point_count), dtype=np.float32)
    endpoints = np.zeros((len(examples), hand_count, 2), dtype=np.int64)
    for batch_index, example in enumerate(examples):
        uvd = np.asarray(example["uvd"], dtype=np.float32)
        uvd_valid = np.asarray(example["uvd_valid_mask"], dtype=np.bool_)
        if uvd.ndim == 2:
            uvd = uvd[:, None, :]
        if uvd_valid.ndim == 1:
            uvd_valid = uvd_valid[:, None]
        if uvd.ndim != 3 or uvd.shape[-1] != 3 or uvd_valid.shape != uvd.shape[:2]:
            raise ValueError(
                f"uvd/valid must have shapes [T,H,3]/[T,H], got {uvd.shape}/{uvd_valid.shape}"
            )
        if uvd.shape[1] != hand_count:
            raise ValueError(f"expected {hand_count} UVD hands, got {uvd.shape[1]}")
        uvd_time = np.asarray(
            example.get("uvd_time", np.linspace(0.0, 1.0, len(uvd), dtype=np.float32)),
            dtype=np.float32,
        )
        count = min(len(uvd), points_per_hand)
        for time_index in range(count):
            for hand_index in range(hand_count):
                if order == "time_major":
                    token_index = time_index * hand_count + hand_index
                else:
                    token_index = hand_index * points_per_hand + time_index
                target[batch_index, token_index] = uvd[time_index, hand_index]
                valid[batch_index, token_index] = uvd_valid[time_index, hand_index]
                times[batch_index, token_index] = uvd_time[time_index]
        for hand_index in range(hand_count):
            if order == "time_major":
                endpoints[batch_index, hand_index] = [hand_index, (count - 1) * hand_count + hand_index]
            else:
                start = hand_index * points_per_hand
                endpoints[batch_index, hand_index] = [start, start + count - 1]
    return target, valid, times, endpoints


def compute_geometry_metrics(
    predictions: dict[str, torch.Tensor],
    examples: list[dict],
    *,
    depth_scale: float,
    image_size: int,
    uvd_hand_count: int = 1,
    uvd_order: str = "hand_major",
    include_uvd_time_metrics: bool = False,
) -> dict[str, float]:
    """Compute fixed-sample depth/UVD metrics for a diagnostics checkpoint."""
    device = predictions["depth_current"].device
    depth_current_target = torch.as_tensor(np.stack([x["depth_current"] for x in examples]), device=device)
    depth_future_target = torch.as_tensor(np.stack([x["depth_future"] for x in examples]), device=device)
    depth_current_valid = torch.as_tensor(np.stack([x["depth_current_valid"] for x in examples]), device=device)
    depth_future_valid = torch.as_tensor(np.stack([x["depth_future_valid"] for x in examples]), device=device)
    uvd_target_np, uvd_valid_np, _, uvd_endpoints_np = _pad_uvd_examples(
        examples,
        int(predictions["uvd"].shape[1]),
        hand_count=uvd_hand_count,
        order=uvd_order,
    )
    uvd_target = torch.as_tensor(uvd_target_np, device=device)
    uvd_valid = torch.as_tensor(uvd_valid_np, device=device)
    uvd_endpoints = torch.as_tensor(uvd_endpoints_np, device=device)

    depth_current = predictions["depth_current"].float()
    depth_future = predictions["depth_future"].float()
    uvd = predictions["uvd"].float()
    pixel_scale = float(max(image_size - 1, 1))
    batch_indices = torch.arange(uvd.shape[0], device=device)
    endpoint_batch = batch_indices[:, None, None].expand_as(uvd_endpoints)
    uvd_pred_endpoints = uvd[endpoint_batch, uvd_endpoints]
    uvd_target_endpoints = uvd_target[endpoint_batch, uvd_endpoints]
    uvd_endpoint_valid = uvd_valid[endpoint_batch, uvd_endpoints]
    points_per_hand = uvd.shape[1] // int(uvd_hand_count)
    if uvd_order == "time_major":
        uvd_tracks = uvd.view(uvd.shape[0], points_per_hand, int(uvd_hand_count), 3)
        target_tracks = uvd_target.view(uvd.shape[0], points_per_hand, int(uvd_hand_count), 3)
        valid_tracks = uvd_valid.view(uvd.shape[0], points_per_hand, int(uvd_hand_count))
    else:
        uvd_tracks = uvd.view(uvd.shape[0], int(uvd_hand_count), points_per_hand, 3).transpose(1, 2)
        target_tracks = uvd_target.view(uvd.shape[0], int(uvd_hand_count), points_per_hand, 3).transpose(1, 2)
        valid_tracks = uvd_valid.view(uvd.shape[0], int(uvd_hand_count), points_per_hand).transpose(1, 2)
    segment_valid = valid_tracks[:, 1:] & valid_tracks[:, :-1]
    pred_delta = uvd_tracks[:, 1:] - uvd_tracks[:, :-1]
    target_delta = target_tracks[:, 1:] - target_tracks[:, :-1]
    pred_segments = torch.linalg.vector_norm(pred_delta[..., :2] * pixel_scale, dim=-1)
    target_segments = torch.linalg.vector_norm(target_delta[..., :2] * pixel_scale, dim=-1)
    trace_valid = segment_valid.any(dim=1)
    trace_count = trace_valid.sum().clamp_min(1)
    pred_path_mean = ((pred_segments * segment_valid).sum(dim=1) * trace_valid).sum() / trace_count
    target_path_mean = ((target_segments * segment_valid).sum(dim=1) * trace_valid).sum() / trace_count
    metrics = {
        "depth_current_mae_m": float((_masked_abs_mean(depth_current, depth_current_target, depth_current_valid) * depth_scale).item()),
        "depth_future_mae_m": float((_masked_abs_mean(depth_future, depth_future_target, depth_future_valid) * depth_scale).item()),
        "depth_current_rmse_m": float((_masked_rmse(depth_current, depth_current_target, depth_current_valid) * depth_scale).item()),
        "depth_future_rmse_m": float((_masked_rmse(depth_future, depth_future_target, depth_future_valid) * depth_scale).item()),
        "uvd_u_mae_pixel": float((_masked_abs_mean(uvd[..., 0], uvd_target[..., 0], uvd_valid) * pixel_scale).item()),
        "uvd_v_mae_pixel": float((_masked_abs_mean(uvd[..., 1], uvd_target[..., 1], uvd_valid) * pixel_scale).item()),
        "uvd_xy_mae_norm": float(_masked_abs_mean(uvd[..., :2], uvd_target[..., :2], uvd_valid).item()),
        "uvd_xy_mae_pixel": float((_masked_abs_mean(uvd[..., :2], uvd_target[..., :2], uvd_valid) * max(image_size - 1, 1)).item()),
        "uvd_depth_mae_m": float((_masked_abs_mean(uvd[..., 2:3], uvd_target[..., 2:3], uvd_valid) * depth_scale).item()),
        "uvd_u_rmse_pixel": float((_masked_rmse(uvd[..., 0], uvd_target[..., 0], uvd_valid) * pixel_scale).item()),
        "uvd_v_rmse_pixel": float((_masked_rmse(uvd[..., 1], uvd_target[..., 1], uvd_valid) * pixel_scale).item()),
        "uvd_xy_rmse_pixel": float((_masked_rmse(uvd[..., :2], uvd_target[..., :2], uvd_valid) * pixel_scale).item()),
        "uvd_depth_rmse_m": float((_masked_rmse(uvd[..., 2:3], uvd_target[..., 2:3], uvd_valid) * depth_scale).item()),
        "uvd_u_smooth_l1": float(_masked_smooth_l1_mean(uvd[..., 0], uvd_target[..., 0], uvd_valid).item()),
        "uvd_v_smooth_l1": float(_masked_smooth_l1_mean(uvd[..., 1], uvd_target[..., 1], uvd_valid).item()),
        "uvd_depth_smooth_l1": float(_masked_smooth_l1_mean(uvd[..., 2], uvd_target[..., 2], uvd_valid).item()),
        "uvd_adjacent_u_mae_pixel": float((_masked_abs_mean(pred_delta[..., 0], target_delta[..., 0], segment_valid) * pixel_scale).item()),
        "uvd_adjacent_v_mae_pixel": float((_masked_abs_mean(pred_delta[..., 1], target_delta[..., 1], segment_valid) * pixel_scale).item()),
        "uvd_adjacent_depth_mae_m": float((_masked_abs_mean(pred_delta[..., 2], target_delta[..., 2], segment_valid) * depth_scale).item()),
        "uvd_adjacent_relative_smooth_l1": float(_masked_smooth_l1_mean(pred_delta, target_delta, segment_valid).item()),
        "uvd_start_xy_mae_pixel": float((_masked_abs_mean(uvd_pred_endpoints[:, :, 0, :2], uvd_target_endpoints[:, :, 0, :2], uvd_endpoint_valid[:, :, 0]) * pixel_scale).item()),
        "uvd_end_xy_mae_pixel": float((_masked_abs_mean(uvd_pred_endpoints[:, :, 1, :2], uvd_target_endpoints[:, :, 1, :2], uvd_endpoint_valid[:, :, 1]) * pixel_scale).item()),
        "uvd_start_depth_mae_m": float((_masked_abs_mean(uvd_pred_endpoints[:, :, 0, 2:3], uvd_target_endpoints[:, :, 0, 2:3], uvd_endpoint_valid[:, :, 0]) * depth_scale).item()),
        "uvd_end_depth_mae_m": float((_masked_abs_mean(uvd_pred_endpoints[:, :, 1, 2:3], uvd_target_endpoints[:, :, 1, 2:3], uvd_endpoint_valid[:, :, 1]) * depth_scale).item()),
        "pred/uvd_path_length_pixel": float(pred_path_mean.item()),
        "target/uvd_path_length_pixel": float(target_path_mean.item()),
        "uvd_path_length_mae_pixel": float(_masked_abs_mean((pred_segments * segment_valid).sum(dim=1), (target_segments * segment_valid).sum(dim=1), trace_valid).item()),
        "pred/depth_current_mean": float(depth_current.mean().item()),
        "pred/depth_current_min": float(depth_current.min().item()),
        "pred/depth_current_max": float(depth_current.max().item()),
        "pred/depth_future_mean": float(depth_future.mean().item()),
        "pred/uvd_u_min": float(uvd[..., 0].min().item()),
        "pred/uvd_u_max": float(uvd[..., 0].max().item()),
        "pred/uvd_v_min": float(uvd[..., 1].min().item()),
        "pred/uvd_v_max": float(uvd[..., 1].max().item()),
        "pred/uvd_depth_mean": float(uvd[..., 2].mean().item()),
    }
    if include_uvd_time_metrics:
        for time_index in range(points_per_hand):
            time_valid = valid_tracks[:, time_index]
            pred_time = uvd_tracks[:, time_index]
            target_time = target_tracks[:, time_index]
            valid_count = int(time_valid.sum().item())
            metrics[f"uvd/time_{time_index}/valid_count"] = float(valid_count)
            metrics[f"uvd/time_{time_index}/valid_ratio"] = float(
                time_valid.float().mean().item()
            )
            if valid_count == 0:
                continue
            metrics.update(
                {
                    f"uvd/time_{time_index}/u_mae_pixel": float(
                        (_masked_abs_mean(pred_time[..., 0], target_time[..., 0], time_valid) * pixel_scale).item()
                    ),
                    f"uvd/time_{time_index}/v_mae_pixel": float(
                        (_masked_abs_mean(pred_time[..., 1], target_time[..., 1], time_valid) * pixel_scale).item()
                    ),
                    f"uvd/time_{time_index}/depth_mae_m": float(
                        (_masked_abs_mean(pred_time[..., 2], target_time[..., 2], time_valid) * depth_scale).item()
                    ),
                }
            )
    metrics.update(collect_batch_valid_ratios(examples))
    return metrics


def save_prediction_bundle(
    output_dir: str | Path,
    step: int,
    predictions: dict[str, torch.Tensor],
    examples: list[dict],
    *,
    uvd_hand_count: int = 1,
    uvd_order: str = "hand_major",
) -> Path:
    """Save small fixed-sample prediction/target bundles, not hidden states or images."""
    directory = Path(output_dir) / "test_diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"predictions_step_{int(step):08d}.npz"
    uvd_target, uvd_valid, uvd_time, uvd_endpoints = _pad_uvd_examples(
        examples,
        int(predictions["uvd"].shape[1]),
        hand_count=uvd_hand_count,
        order=uvd_order,
    )
    frame_indices = np.full(uvd_valid.shape, -1, dtype=np.int64)
    points_per_hand = frame_indices.shape[1] // int(uvd_hand_count)
    for batch_index, example in enumerate(examples):
        actual_indices = np.asarray(example["uvd_frame_indices"], dtype=np.int64)
        actual_indices = actual_indices[:points_per_hand]
        if uvd_order == "time_major":
            expanded_indices = np.repeat(actual_indices, int(uvd_hand_count))
        else:
            expanded_indices = np.tile(actual_indices, int(uvd_hand_count))
        frame_indices[batch_index, : len(expanded_indices)] = expanded_indices
    np.savez_compressed(
        path,
        # NumPy cannot represent torch.bfloat16. Geometry inference follows the
        # model's mixed-precision dtype, so persist predictions as float32.
        depth_current=predictions["depth_current"].detach().cpu().float().numpy(),
        depth_future=predictions["depth_future"].detach().cpu().float().numpy(),
        uvd=predictions["uvd"].detach().cpu().float().numpy(),
        depth_current_target=np.stack([x["depth_current"] for x in examples]),
        depth_future_target=np.stack([x["depth_future"] for x in examples]),
        uvd_target=uvd_target,
        uvd_valid_mask=uvd_valid,
        uvd_time=uvd_time,
        uvd_endpoint_indices=uvd_endpoints,
        uvd_frame_indices=frame_indices,
    )
    return path


def _git_value(repo_root: Path, args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=False, capture_output=True, text=True
        ).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def write_run_manifest(output_dir: str | Path, config: Any, repo_root: str | Path) -> Path:
    """Record environment and experiment identity before a test run starts."""
    output_dir = Path(output_dir)
    repo_root = Path(repo_root)
    try:
        config_container = config.unwrap() if hasattr(config, "unwrap") else config
        from omegaconf import OmegaConf
        config_container = OmegaConf.to_container(config_container, resolve=True)
    except Exception:
        config_container = {}
    package_versions = {}
    for package in ("torch", "torchvision", "transformers", "accelerate", "deepspeed", "decord"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"
    manifest = {
        "python": sys.executable,
        "python_version": sys.version,
        "packages": package_versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "git_head": _git_value(repo_root, ["rev-parse", "HEAD"]),
        "git_status": _git_value(repo_root, ["status", "--short"]),
        "git_diff_stat": _git_value(repo_root, ["diff", "--stat"]),
        "config": config_container,
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path
