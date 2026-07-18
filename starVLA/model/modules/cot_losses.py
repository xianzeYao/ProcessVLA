
"""Masked auxiliary losses for geometric CoT supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
    batch = torch.arange(uvd.shape[0], device=uvd.device)
    if endpoint_indices is None:
        current_indices = torch.zeros(uvd.shape[0], dtype=torch.long, device=uvd.device)
        future_indices = torch.full(
            (uvd.shape[0],), uvd.shape[1] - 1, dtype=torch.long, device=uvd.device
        )
    else:
        endpoint_indices = endpoint_indices.to(device=uvd.device, dtype=torch.long)
        if endpoint_indices.shape != (uvd.shape[0], 2):
            raise ValueError(
                f"endpoint_indices must have shape {(uvd.shape[0], 2)}, got {tuple(endpoint_indices.shape)}"
            )
        current_indices, future_indices = endpoint_indices[:, 0], endpoint_indices[:, 1]
    current_uvd = uvd[batch, current_indices]
    future_uvd = uvd[batch, future_indices]
    current_grid = current_uvd[:, None, :2].mul(2.0).sub(1.0).unsqueeze(2)
    future_grid = future_uvd[:, None, :2].mul(2.0).sub(1.0).unsqueeze(2)
    sampled_current = F.grid_sample(depth_current.float(), current_grid, align_corners=True).flatten(1)
    sampled_future = F.grid_sample(depth_future.float(), future_grid, align_corners=True).flatten(1)
    target_current = current_uvd[:, 2:3].float() * depth_scale
    target_future = future_uvd[:, 2:3].float() * depth_scale
    values = torch.cat([sampled_current - target_current, sampled_future - target_future], dim=1).abs()
    endpoint_valid = torch.stack([valid[batch, current_indices], valid[batch, future_indices]], dim=1)
    return _masked_mean(values, endpoint_valid)
