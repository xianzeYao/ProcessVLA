
"""Masked auxiliary losses for geometric CoT supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate_cot_total_loss(
    action_loss: torch.Tensor,
    depth_current_loss: torch.Tensor,
    depth_future_loss: torch.Tensor,
    uvd_loss: torch.Tensor,
    *,
    lambda_action: float,
    lambda_depth_current: float,
    lambda_depth_future: float,
    lambda_uvd: float,
) -> torch.Tensor:
    """Aggregate Action, current/future Depth, and UVD objectives."""
    return (
        float(lambda_action) * action_loss
        + float(lambda_depth_current) * depth_current_loss
        + float(lambda_depth_future) * depth_future_loss
        + float(lambda_uvd) * uvd_loss
    )


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    numerator = (values * mask).sum()
    denominator = mask.expand_as(values).sum().clamp_min(1.0)
    return numerator / denominator


def masked_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 averaged over valid scalar elements/pixels only."""
    values = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    return _masked_mean(values, valid)


def uvd_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    coordinate_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Masked UVD regression loss with optional per-coordinate weights."""
    values = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    weights = torch.as_tensor(coordinate_weights, device=values.device, dtype=values.dtype).view(1, 1, 3)
    return _masked_mean(values * weights, valid)


def uvd_adjacent_relative_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    hand_count: int,
    coordinate_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Smooth-L1 on adjacent same-hand UVD deltas in time-major order."""
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(
            f"pred/target must share shape [B,K,3], got {tuple(pred.shape)}/{tuple(target.shape)}"
        )
    if valid.shape != pred.shape[:2]:
        raise ValueError(f"valid must have shape {tuple(pred.shape[:2])}, got {tuple(valid.shape)}")
    hand_count = int(hand_count)
    if hand_count < 1 or pred.shape[1] % hand_count != 0:
        raise ValueError(
            f"token count {pred.shape[1]} must be divisible by positive hand_count={hand_count}"
        )
    point_count = pred.shape[1] // hand_count
    if point_count < 2:
        raise ValueError(f"relative UVD loss requires at least two points per hand, got {point_count}")

    pred_tracks = pred.float().reshape(pred.shape[0], point_count, hand_count, 3)
    target_tracks = target.float().reshape(target.shape[0], point_count, hand_count, 3)
    valid_tracks = valid.to(dtype=torch.bool).reshape(valid.shape[0], point_count, hand_count)
    pred_delta = pred_tracks[:, 1:] - pred_tracks[:, :-1]
    target_delta = target_tracks[:, 1:] - target_tracks[:, :-1]
    segment_valid = valid_tracks[:, 1:] & valid_tracks[:, :-1]
    values = F.smooth_l1_loss(pred_delta, target_delta, reduction="none")
    weights = torch.as_tensor(
        coordinate_weights,
        device=values.device,
        dtype=values.dtype,
    ).view(1, 1, 1, 3)
    return _masked_mean(values * weights, segment_valid)
