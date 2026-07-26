
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
    """Aggregate the active V1 objective without endpoint geometry consistency.

    UVD depth is the camera-space EEF reference-point depth, while a rendered
    depth map stores the visible surface depth. Without a visibility target,
    coupling them with a sampled-depth loss can impose a wrong constraint.
    The endpoint helper remains available for future visibility-aware
    experiments, but is not part of the active V1 objective.
    """
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


def endpoint_depth_uvd_consistency(
    depth_current: torch.Tensor,
    depth_future: torch.Tensor,
    uvd: torch.Tensor,
    valid: torch.Tensor,
    *,
    depth_scale: float,
    endpoint_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Require predicted UVD endpoint depth to agree with predicted depth maps."""
    if depth_scale <= 0:
        raise ValueError(f"depth_scale must be positive, got {depth_scale}")
    batch_size = uvd.shape[0]
    if endpoint_indices is None:
        current_indices = torch.zeros(batch_size, 1, dtype=torch.long, device=uvd.device)
        future_indices = torch.full(
            (batch_size, 1), uvd.shape[1] - 1, dtype=torch.long, device=uvd.device
        )
    else:
        endpoint_indices = endpoint_indices.to(device=uvd.device, dtype=torch.long)
        if endpoint_indices.shape == (batch_size, 2):
            endpoint_indices = endpoint_indices.unsqueeze(1)
        if endpoint_indices.ndim != 3 or endpoint_indices.shape[0] != batch_size or endpoint_indices.shape[2] != 2:
            raise ValueError(
                f"endpoint_indices must have shape [B, 2] or [B, H, 2] with B={batch_size}, got {tuple(endpoint_indices.shape)}"
            )
        current_indices, future_indices = endpoint_indices[..., 0], endpoint_indices[..., 1]
    hand_count = current_indices.shape[1]
    batch = torch.arange(batch_size, device=uvd.device)[:, None]
    current_uvd = uvd[batch, current_indices]
    future_uvd = uvd[batch, future_indices]
    current_grid = current_uvd[..., :2].mul(2.0).sub(1.0).reshape(batch_size * hand_count, 1, 1, 2)
    future_grid = future_uvd[..., :2].mul(2.0).sub(1.0).reshape(batch_size * hand_count, 1, 1, 2)
    depth_current = depth_current.float().repeat_interleave(hand_count, dim=0)
    depth_future = depth_future.float().repeat_interleave(hand_count, dim=0)
    sampled_current = F.grid_sample(depth_current, current_grid, align_corners=True).reshape(batch_size, hand_count)
    sampled_future = F.grid_sample(depth_future, future_grid, align_corners=True).reshape(batch_size, hand_count)
    target_current = current_uvd[..., 2].float() * depth_scale
    target_future = future_uvd[..., 2].float() * depth_scale
    values = torch.stack([sampled_current - target_current, sampled_future - target_future], dim=-1).abs()
    endpoint_valid = torch.stack([valid[batch, current_indices], valid[batch, future_indices]], dim=-1)
    return _masked_mean(values, endpoint_valid)
