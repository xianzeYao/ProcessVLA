"""Diagnostics specific to forward coarse-to-local V4 geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _masked_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    mask = valid.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (
        (values * mask).sum()
        / mask.expand_as(values).sum().clamp_min(1.0)
    )


def compute_named_uvd_metrics(
    prediction: torch.Tensor,
    examples: list[dict],
    *,
    value_key: str,
    valid_key: str,
    prefix: str,
    depth_scale: float,
    image_size: int,
    hand_count: int,
    order: str,
) -> dict[str, float]:
    """Compute coordinate and trajectory metrics for one named UVD group."""

    if order not in {"time_major", "hand_major"}:
        raise ValueError(f"unsupported UVD order: {order!r}")
    batch_size, token_count, coordinate_count = prediction.shape
    if coordinate_count != 3 or token_count % int(hand_count):
        raise ValueError(
            f"prediction must be [B,N,3] with N divisible by {hand_count}"
        )
    points_per_hand = token_count // int(hand_count)
    targets = []
    masks = []
    for example in examples:
        values = np.asarray(example[value_key], dtype=np.float32)
        valid = np.asarray(example[valid_key], dtype=np.bool_)
        if int(hand_count) == 1:
            expected_values = (points_per_hand, 3)
            expected_valid = (points_per_hand,)
        else:
            expected_values = (points_per_hand, int(hand_count), 3)
            expected_valid = (points_per_hand, int(hand_count))
        if values.shape != expected_values or valid.shape != expected_valid:
            raise ValueError(
                f"{value_key}/{valid_key} must have shapes "
                f"{expected_values}/{expected_valid}, got "
                f"{values.shape}/{valid.shape}"
            )
        if int(hand_count) == 1:
            values = values[:, None]
            valid = valid[:, None]
        if order == "hand_major":
            values = values.transpose(1, 0, 2)
            valid = valid.transpose(1, 0)
        targets.append(values.reshape(token_count, 3))
        masks.append(valid.reshape(token_count))
    if len(targets) != batch_size:
        raise ValueError(
            f"prediction batch has {batch_size} samples, got {len(targets)} targets"
        )

    pred = prediction.float()
    target = torch.as_tensor(
        np.stack(targets),
        device=pred.device,
    )
    valid = torch.as_tensor(
        np.stack(masks),
        device=pred.device,
        dtype=torch.bool,
    )
    pixel_scale = float(max(image_size - 1, 1))
    absolute = (pred - target).abs()
    square = (pred - target).square()

    if order == "time_major":
        pred_tracks = pred.reshape(
            batch_size,
            points_per_hand,
            int(hand_count),
            3,
        )
        target_tracks = target.reshape(
            batch_size,
            points_per_hand,
            int(hand_count),
            3,
        )
        valid_tracks = valid.reshape(
            batch_size,
            points_per_hand,
            int(hand_count),
        )
    else:
        pred_tracks = pred.reshape(
            batch_size,
            int(hand_count),
            points_per_hand,
            3,
        ).transpose(1, 2)
        target_tracks = target.reshape(
            batch_size,
            int(hand_count),
            points_per_hand,
            3,
        ).transpose(1, 2)
        valid_tracks = valid.reshape(
            batch_size,
            int(hand_count),
            points_per_hand,
        ).transpose(1, 2)

    pred_delta = pred_tracks[:, 1:] - pred_tracks[:, :-1]
    target_delta = target_tracks[:, 1:] - target_tracks[:, :-1]
    segment_valid = valid_tracks[:, 1:] & valid_tracks[:, :-1]
    pred_segments = torch.linalg.vector_norm(
        pred_delta[..., :2] * pixel_scale,
        dim=-1,
    )
    target_segments = torch.linalg.vector_norm(
        target_delta[..., :2] * pixel_scale,
        dim=-1,
    )
    trace_valid = segment_valid.any(dim=1)
    pred_path = (pred_segments * segment_valid).sum(dim=1)
    target_path = (target_segments * segment_valid).sum(dim=1)

    return {
        f"{prefix}_u_mae_pixel": float(
            (_masked_mean(absolute[..., 0], valid) * pixel_scale).item()
        ),
        f"{prefix}_v_mae_pixel": float(
            (_masked_mean(absolute[..., 1], valid) * pixel_scale).item()
        ),
        f"{prefix}_xy_mae_norm": float(
            _masked_mean(absolute[..., :2], valid).item()
        ),
        f"{prefix}_xy_mae_pixel": float(
            (_masked_mean(absolute[..., :2], valid) * pixel_scale).item()
        ),
        f"{prefix}_depth_mae_m": float(
            (
                _masked_mean(absolute[..., 2], valid)
                * float(depth_scale)
            ).item()
        ),
        f"{prefix}_xy_rmse_pixel": float(
            (
                torch.sqrt(
                    _masked_mean(square[..., :2], valid).clamp_min(0.0)
                )
                * pixel_scale
            ).item()
        ),
        f"{prefix}_depth_rmse_m": float(
            (
                torch.sqrt(
                    _masked_mean(square[..., 2], valid).clamp_min(0.0)
                )
                * float(depth_scale)
            ).item()
        ),
        f"{prefix}_adjacent_xy_mae_pixel": float(
            (
                _masked_mean(
                    (pred_delta[..., :2] - target_delta[..., :2]).abs(),
                    segment_valid,
                )
                * pixel_scale
            ).item()
        ),
        f"{prefix}_adjacent_depth_mae_m": float(
            (
                _masked_mean(
                    (pred_delta[..., 2] - target_delta[..., 2]).abs(),
                    segment_valid,
                )
                * float(depth_scale)
            ).item()
        ),
        f"{prefix}_start_xy_mae_pixel": float(
            (
                _masked_mean(
                    (pred_tracks[:, 0, :, :2] - target_tracks[:, 0, :, :2]).abs(),
                    valid_tracks[:, 0],
                )
                * pixel_scale
            ).item()
        ),
        f"{prefix}_end_xy_mae_pixel": float(
            (
                _masked_mean(
                    (pred_tracks[:, -1, :, :2] - target_tracks[:, -1, :, :2]).abs(),
                    valid_tracks[:, -1],
                )
                * pixel_scale
            ).item()
        ),
        f"pred/{prefix}_path_length_pixel": float(
            _masked_mean(pred_path, trace_valid).item()
        ),
        f"target/{prefix}_path_length_pixel": float(
            _masked_mean(target_path, trace_valid).item()
        ),
        f"{prefix}_path_length_mae_pixel": float(
            _masked_mean((pred_path - target_path).abs(), trace_valid).item()
        ),
    }


def compute_cross_scale_overlap_metrics(
    coarse_prediction: torch.Tensor,
    local_prediction: torch.Tensor,
    examples: list[dict],
    *,
    image_size: int,
    hand_count: int,
) -> dict[str, float]:
    batch_size = coarse_prediction.shape[0]
    coarse = coarse_prediction.reshape(
        batch_size,
        -1,
        hand_count,
        3,
    )
    local = local_prediction.reshape(
        batch_size,
        -1,
        hand_count,
        3,
    )
    overlap_count = min(
        coarse.shape[1],
        local.shape[1] // 2,
    )
    if overlap_count == 0:
        return {"cross_scale/overlap_valid_count": 0.0}
    local_indices = (
        torch.arange(overlap_count, device=local.device) * 2 + 1
    )
    coarse = coarse[:, :overlap_count]
    local = local.index_select(1, local_indices)
    coarse_valid = torch.as_tensor(
        np.stack(
            [
                example["uvd_coarse_valid_mask"]
                for example in examples
            ]
        ),
        device=coarse.device,
        dtype=torch.bool,
    ).reshape(batch_size, -1, hand_count)[:, :overlap_count]
    local_valid = torch.as_tensor(
        np.stack(
            [example["uvd_valid_mask"] for example in examples]
        ),
        device=local.device,
        dtype=torch.bool,
    ).reshape(batch_size, -1, hand_count).index_select(
        1,
        local_indices,
    )
    valid = coarse_valid & local_valid
    if not valid.any():
        return {"cross_scale/overlap_valid_count": 0.0}
    xy_gap = torch.linalg.vector_norm(
        coarse[..., :2] - local[..., :2],
        dim=-1,
    )
    return {
        "cross_scale/overlap_valid_count": float(
            valid.sum().item()
        ),
        "cross_scale/overlap_xy_gap_pixel": float(
            (
                xy_gap[valid].mean()
                * float(max(image_size - 1, 1))
            ).item()
        ),
    }


def terminal_repeat_sample_ratio(examples: list[dict]) -> float:
    repeated = []
    for example in examples:
        local = np.asarray(
            example["uvd_frame_indices"],
            dtype=np.int64,
        )
        coarse = np.asarray(
            example["uvd_coarse_frame_indices"],
            dtype=np.int64,
        )
        repeated.append(
            bool(
                np.any(local[1:] == local[:-1])
                or np.any(coarse[1:] == coarse[:-1])
            )
        )
    return float(np.mean(repeated)) if repeated else 0.0


def compute_v4_geometry_metrics(
    predictions: dict[str, Any],
    examples: list[dict],
    *,
    depth_scale: float,
    image_size: int,
    hand_count: int,
) -> dict[str, float]:
    metrics = compute_named_uvd_metrics(
        predictions["uvd_coarse"],
        examples,
        value_key="uvd_coarse",
        valid_key="uvd_coarse_valid_mask",
        prefix="uvd_coarse",
        depth_scale=depth_scale,
        image_size=image_size,
        hand_count=hand_count,
        order="time_major",
    )
    metrics.update(
        compute_cross_scale_overlap_metrics(
            predictions["uvd_coarse"],
            predictions["uvd"],
            examples,
            image_size=image_size,
            hand_count=hand_count,
        )
    )
    metrics["data/terminal_repeat_sample_ratio"] = (
        terminal_repeat_sample_ratio(examples)
    )
    return metrics
