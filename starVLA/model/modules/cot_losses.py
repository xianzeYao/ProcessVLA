
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


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    coordinate_weights: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Apply the mask before computing Smooth-L1 so invalid NaNs stay inert."""
    mask = valid.to(dtype=torch.bool)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(pred)
    pred_values = torch.where(mask, pred.float(), 0.0)
    target_values = torch.where(mask, target.float(), 0.0)
    values = F.smooth_l1_loss(pred_values, target_values, reduction="none")
    if coordinate_weights is not None:
        weights = torch.as_tensor(
            coordinate_weights,
            device=values.device,
            dtype=values.dtype,
        ).flatten()
        if weights.numel() < values.shape[-1]:
            raise ValueError(
                f"coordinate_weights has {weights.numel()} values for "
                f"{values.shape[-1]} coordinates"
            )
        weights = weights[: values.shape[-1]].view(
            *([1] * (values.ndim - 1)), values.shape[-1]
        )
        values = values * weights
    denominator = mask.to(dtype=values.dtype).sum().clamp_min(1.0)
    return values.sum() / denominator


def masked_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 averaged over valid scalar elements/pixels only."""
    return _masked_smooth_l1(pred, target, valid)


def uvd_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    coordinate_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Masked UVD regression loss with optional per-coordinate weights."""
    return _masked_smooth_l1(
        pred,
        target,
        valid,
        coordinate_weights=coordinate_weights,
    )


def uvd_adjacent_relative_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    hand_count: int,
    coordinate_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Smooth-L1 on adjacent same-hand UVD deltas in time-major order."""
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] not in (2, 3):
        raise ValueError(
            "pred/target must share shape [B,K,C] with C in {2,3}, "
            f"got {tuple(pred.shape)}/{tuple(target.shape)}"
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

    coordinate_dim = pred.shape[-1]
    pred_tracks = pred.float().reshape(
        pred.shape[0], point_count, hand_count, coordinate_dim
    )
    target_tracks = target.float().reshape(
        target.shape[0], point_count, hand_count, coordinate_dim
    )
    valid_tracks = valid.to(dtype=torch.bool).reshape(valid.shape[0], point_count, hand_count)
    pred_delta = pred_tracks[:, 1:] - pred_tracks[:, :-1]
    target_delta = target_tracks[:, 1:] - target_tracks[:, :-1]
    segment_valid = valid_tracks[:, 1:] & valid_tracks[:, :-1]
    return _masked_smooth_l1(
        pred_delta,
        target_delta,
        segment_valid,
        coordinate_weights=coordinate_weights,
    )


def uvd_triangle_shape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    coordinate_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Match translation-invariant grasp and midpoint-to-wrist vectors."""
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(
            f"pred/target must share shape [B,K,3], got {tuple(pred.shape)}/{tuple(target.shape)}"
        )
    if valid.shape != pred.shape[:2]:
        raise ValueError(
            f"valid must have shape {tuple(pred.shape[:2])}, got {tuple(valid.shape)}"
        )
    if pred.shape[1] % 3 != 0:
        raise ValueError(
            f"triangle shape loss requires a token count divisible by 3, got {pred.shape[1]}"
        )
    time_points = pred.shape[1] // 3
    pred_triangle = pred.float().reshape(pred.shape[0], time_points, 3, 3)
    target_triangle = target.float().reshape(target.shape[0], time_points, 3, 3)
    valid_triangle = valid.to(dtype=torch.bool).reshape(valid.shape[0], time_points, 3)
    complete = torch.all(valid_triangle, dim=-1)
    if not torch.any(complete):
        return pred.sum() * 0.0

    def vectors(triangle: torch.Tensor) -> torch.Tensor:
        left, right, wrist = triangle.unbind(dim=2)
        grasp_axis = right - left
        wrist_axis = wrist - 0.5 * (left + right)
        return torch.stack([grasp_axis, wrist_axis], dim=2)

    return _masked_smooth_l1(
        vectors(pred_triangle),
        vectors(target_triangle),
        complete,
        coordinate_weights=coordinate_weights,
    )
