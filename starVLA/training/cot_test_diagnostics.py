"""Low-overhead diagnostics for QwenGR00TCoT test runs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from starVLA.model.modules.cot_losses import endpoint_depth_uvd_consistency


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

def collect_batch_valid_ratios(examples: list[dict]) -> dict[str, float]:
    """Summarize label validity without retaining tensors or image data."""
    output: dict[str, float] = {}
    for key, name in (
        ("depth_current_valid", "data/depth_current_valid_ratio"),
        ("depth_future_valid", "data/depth_future_valid_ratio"),
        ("uvd_valid_mask", "data/uvd_valid_ratio"),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = np.zeros((len(examples), point_count, 3), dtype=np.float32)
    valid = np.zeros((len(examples), point_count), dtype=np.bool_)
    times = np.zeros((len(examples), point_count), dtype=np.float32)
    endpoints = np.zeros((len(examples), 2), dtype=np.int64)
    for batch_index, example in enumerate(examples):
        uvd = np.asarray(example["uvd"], dtype=np.float32)
        uvd_valid = np.asarray(example["uvd_valid_mask"], dtype=np.bool_)
        uvd_time = np.asarray(
            example.get("uvd_time", np.linspace(0.0, 1.0, len(uvd), dtype=np.float32)),
            dtype=np.float32,
        )
        count = min(len(uvd), point_count)
        target[batch_index, :count] = uvd[:count]
        valid[batch_index, :count] = uvd_valid[:count]
        times[batch_index, :count] = uvd_time[:count]
        endpoints[batch_index, 1] = max(count - 1, 0)
    return target, valid, times, endpoints


def compute_geometry_metrics(
    predictions: dict[str, torch.Tensor],
    examples: list[dict],
    *,
    depth_scale: float,
    image_size: int,
) -> dict[str, float]:
    """Compute fixed-sample depth/UVD metrics for a diagnostics checkpoint."""
    device = predictions["depth_current"].device
    depth_current_target = torch.as_tensor(np.stack([x["depth_current"] for x in examples]), device=device)
    depth_future_target = torch.as_tensor(np.stack([x["depth_future"] for x in examples]), device=device)
    depth_current_valid = torch.as_tensor(np.stack([x["depth_current_valid"] for x in examples]), device=device)
    depth_future_valid = torch.as_tensor(np.stack([x["depth_future_valid"] for x in examples]), device=device)
    uvd_target_np, uvd_valid_np, _, uvd_endpoints_np = _pad_uvd_examples(
        examples, int(predictions["uvd"].shape[1])
    )
    uvd_target = torch.as_tensor(uvd_target_np, device=device)
    uvd_valid = torch.as_tensor(uvd_valid_np, device=device)
    uvd_endpoints = torch.as_tensor(uvd_endpoints_np, device=device)

    depth_current = predictions["depth_current"].float()
    depth_future = predictions["depth_future"].float()
    uvd = predictions["uvd"].float()
    pixel_scale = float(max(image_size - 1, 1))
    batch_indices = torch.arange(uvd.shape[0], device=device)
    endpoint_batch = batch_indices[:, None].expand_as(uvd_endpoints)
    uvd_pred_endpoints = uvd[endpoint_batch, uvd_endpoints]
    uvd_target_endpoints = uvd_target[endpoint_batch, uvd_endpoints]
    uvd_endpoint_valid = uvd_valid[endpoint_batch, uvd_endpoints]
    segment_valid = uvd_valid[:, 1:] & uvd_valid[:, :-1]
    pred_segments = torch.linalg.vector_norm(
        (uvd[:, 1:, :2] - uvd[:, :-1, :2]) * pixel_scale, dim=-1
    )
    target_segments = torch.linalg.vector_norm(
        (uvd_target[:, 1:, :2] - uvd_target[:, :-1, :2]) * pixel_scale, dim=-1
    )
    trace_valid = segment_valid.any(dim=1)
    trace_count = trace_valid.sum().clamp_min(1)
    pred_path_mean = ((pred_segments * segment_valid).sum(dim=1) * trace_valid).sum() / trace_count
    target_path_mean = ((target_segments * segment_valid).sum(dim=1) * trace_valid).sum() / trace_count
    metrics = {
        "depth_current_mae_m": float((_masked_abs_mean(depth_current, depth_current_target, depth_current_valid) * depth_scale).item()),
        "depth_future_mae_m": float((_masked_abs_mean(depth_future, depth_future_target, depth_future_valid) * depth_scale).item()),
        "depth_current_rmse_m": float((_masked_rmse(depth_current, depth_current_target, depth_current_valid) * depth_scale).item()),
        "depth_future_rmse_m": float((_masked_rmse(depth_future, depth_future_target, depth_future_valid) * depth_scale).item()),
        "uvd_xy_mae_norm": float(_masked_abs_mean(uvd[..., :2], uvd_target[..., :2], uvd_valid).item()),
        "uvd_xy_mae_pixel": float((_masked_abs_mean(uvd[..., :2], uvd_target[..., :2], uvd_valid) * max(image_size - 1, 1)).item()),
        "uvd_depth_mae_m": float((_masked_abs_mean(uvd[..., 2:3], uvd_target[..., 2:3], uvd_valid) * depth_scale).item()),
        "uvd_xy_rmse_pixel": float((_masked_rmse(uvd[..., :2], uvd_target[..., :2], uvd_valid) * pixel_scale).item()),
        "uvd_depth_rmse_m": float((_masked_rmse(uvd[..., 2:3], uvd_target[..., 2:3], uvd_valid) * depth_scale).item()),
        "uvd_start_xy_mae_pixel": float((_masked_abs_mean(uvd_pred_endpoints[:, 0, :2], uvd_target_endpoints[:, 0, :2], uvd_endpoint_valid[:, 0]) * pixel_scale).item()),
        "uvd_end_xy_mae_pixel": float((_masked_abs_mean(uvd_pred_endpoints[:, 1, :2], uvd_target_endpoints[:, 1, :2], uvd_endpoint_valid[:, 1]) * pixel_scale).item()),
        "uvd_start_depth_mae_m": float((_masked_abs_mean(uvd_pred_endpoints[:, 0, 2:3], uvd_target_endpoints[:, 0, 2:3], uvd_endpoint_valid[:, 0]) * depth_scale).item()),
        "uvd_end_depth_mae_m": float((_masked_abs_mean(uvd_pred_endpoints[:, 1, 2:3], uvd_target_endpoints[:, 1, 2:3], uvd_endpoint_valid[:, 1]) * depth_scale).item()),
        "pred/uvd_path_length_pixel": float(pred_path_mean.item()),
        "target/uvd_path_length_pixel": float(target_path_mean.item()),
        "uvd_path_length_mae_pixel": float(_masked_abs_mean((pred_segments * segment_valid).sum(dim=1), (target_segments * segment_valid).sum(dim=1), trace_valid).item()),
        "endpoint_geometry_mae_m": float(
            endpoint_depth_uvd_consistency(
                depth_current,
                depth_future,
                uvd,
                uvd_valid,
                depth_scale=depth_scale,
                endpoint_indices=uvd_endpoints,
            ).item()
        ),
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
    metrics.update(collect_batch_valid_ratios(examples))
    return metrics


def save_prediction_bundle(
    output_dir: str | Path,
    step: int,
    predictions: dict[str, torch.Tensor],
    examples: list[dict],
) -> Path:
    """Save small fixed-sample prediction/target bundles, not hidden states or images."""
    directory = Path(output_dir) / "test_diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"predictions_step_{int(step):08d}.npz"
    uvd_target, uvd_valid, uvd_time, uvd_endpoints = _pad_uvd_examples(
        examples, int(predictions["uvd"].shape[1])
    )
    frame_indices = np.full(uvd_valid.shape, -1, dtype=np.int64)
    for batch_index, example in enumerate(examples):
        actual_indices = np.asarray(example["uvd_frame_indices"], dtype=np.int64)
        frame_indices[batch_index, : min(len(actual_indices), frame_indices.shape[1])] = actual_indices[: frame_indices.shape[1]]
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
