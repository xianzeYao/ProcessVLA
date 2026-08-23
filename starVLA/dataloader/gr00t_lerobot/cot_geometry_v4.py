"""Forward coarse-to-local geometry targets for QwenGR00TCoTV4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from starVLA.dataloader.gr00t_lerobot.cot_geometry import (
    CoTLeRobotSingleDataset,
    _resize_depth,
    transform_uvd_to_model_space,
)


@dataclass(frozen=True)
class V4HorizonSpec:
    action_horizon: int
    local_uvd_num_points: int
    coarse_uvd_num_points: int
    coarse_uvd_stride: int
    terminal_repeat: bool

    def __post_init__(self) -> None:
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if self.local_uvd_num_points != self.action_horizon:
            raise ValueError(
                "local_uvd_num_points must equal action_horizon, "
                f"got {self.local_uvd_num_points}/{self.action_horizon}"
            )
        if self.coarse_uvd_num_points != self.action_horizon:
            raise ValueError(
                "coarse_uvd_num_points must equal action_horizon, "
                f"got {self.coarse_uvd_num_points}/{self.action_horizon}"
            )
        if self.coarse_uvd_stride != 2:
            raise ValueError(
                f"coarse_uvd_stride must be 2, got {self.coarse_uvd_stride}"
            )
        if not self.terminal_repeat:
            raise ValueError("terminal_repeat must be true for V4")

    @classmethod
    def from_data_cfg(cls, data_cfg: Any) -> "V4HorizonSpec":
        geometry = data_cfg.get("cot_geometry", {})
        horizon = int(geometry.get("action_horizon", 8))
        return cls(
            action_horizon=horizon,
            local_uvd_num_points=int(
                geometry.get("local_uvd_num_points", horizon)
            ),
            coarse_uvd_num_points=int(
                geometry.get("coarse_uvd_num_points", horizon)
            ),
            coarse_uvd_stride=int(geometry.get("coarse_uvd_stride", 2)),
            terminal_repeat=bool(geometry.get("terminal_repeat", False)),
        )

    def local_indices(self, start: int, terminal: int) -> np.ndarray:
        return sample_forward_uvd_indices(
            start,
            terminal,
            num_points=self.local_uvd_num_points,
            stride=1,
        )

    def coarse_indices(self, start: int, terminal: int) -> np.ndarray:
        return sample_forward_uvd_indices(
            start,
            terminal,
            num_points=self.coarse_uvd_num_points,
            stride=self.coarse_uvd_stride,
        )


def sample_forward_uvd_indices(
    start: int,
    terminal: int,
    *,
    num_points: int,
    stride: int,
) -> np.ndarray:
    start = int(start)
    terminal = int(terminal)
    num_points = int(num_points)
    stride = int(stride)
    if terminal < start:
        raise ValueError(
            f"terminal must be >= start, got start={start}, terminal={terminal}"
        )
    if num_points < 1 or stride < 1:
        raise ValueError("num_points and stride must be positive")
    offsets = stride * np.arange(1, num_points + 1, dtype=np.int64)
    return np.minimum(start + offsets, terminal)


class CoTV4LeRobotSingleDataset(CoTLeRobotSingleDataset):
    """LeRobot sample adapter with dense-local and stride-2 coarse UVD targets."""

    def _uvd_group_targets(
        self,
        *,
        key: str,
        indices: np.ndarray,
        base_index: int,
        horizon: int,
        depth: np.ndarray,
        eef_uvd: np.ndarray,
        eef_valid: np.ndarray,
        target_size: int,
        depth_scale: float,
    ) -> dict[str, np.ndarray]:
        sampled_pixels = eef_uvd[indices]
        uvd, boundary_clamp = transform_uvd_to_model_space(
            sampled_pixels,
            source_width=int(depth.shape[-1]),
            source_height=int(depth.shape[-2]),
            target_width=target_size,
            target_height=target_size,
            depth_scale=depth_scale,
            return_boundary_clamp_mask=True,
        )
        valid = np.asarray(eef_valid[indices], dtype=np.bool_)
        finite_positive = np.isfinite(sampled_pixels).all(axis=-1) & (
            sampled_pixels[..., 2] > 0.0
        )
        in_frame = (
            (sampled_pixels[..., 0] >= 0.0)
            & (sampled_pixels[..., 0] <= float(depth.shape[-1] - 1))
            & (sampled_pixels[..., 1] >= 0.0)
            & (sampled_pixels[..., 1] <= float(depth.shape[-2] - 1))
        )
        out_of_frame = finite_positive & ~in_frame
        boundary_clamp = np.asarray(boundary_clamp, dtype=np.bool_) & valid
        nominal_time = (
            np.arange(1, horizon + 1, dtype=np.float32) / float(horizon)
        )
        prefix = "" if key == "uvd" else "_coarse"
        return {
            key: np.asarray(uvd, dtype=np.float32),
            f"uvd{prefix}_valid_mask": valid,
            f"uvd{prefix}_out_of_frame_mask": np.asarray(
                out_of_frame, dtype=np.bool_
            ),
            f"uvd{prefix}_boundary_clamp_mask": boundary_clamp,
            f"uvd{prefix}_frame_indices": np.asarray(indices, dtype=np.int64),
            f"uvd{prefix}_time": nominal_time,
            f"uvd{prefix}_endpoint_indices": np.asarray(
                [0, horizon - 1], dtype=np.int64
            ),
        }

    def _geometry_targets(self) -> dict[str, np.ndarray]:
        if (
            self._cot_current_trajectory_id is None
            or self._cot_current_base_index is None
        ):
            raise RuntimeError("CoT dataset sample position is not initialized")
        trajectory_id = self._cot_current_trajectory_id
        depth, eef_uvd, eef_valid, _ = self._load_episode_geometry(trajectory_id)
        if len(depth) == 0:
            raise RuntimeError(f"trajectory {trajectory_id} has no depth frames")
        if len(eef_uvd) != len(depth) or len(eef_valid) != len(depth):
            raise RuntimeError(
                "depth, UVD, and UVD validity must have equal episode lengths"
            )

        terminal = len(depth) - 1
        base_index = min(max(int(self._cot_current_base_index), 0), terminal)
        spec = V4HorizonSpec.from_data_cfg(self._cot_data_cfg)
        local_indices = spec.local_indices(base_index, terminal)
        coarse_indices = spec.coarse_indices(base_index, terminal)
        future_index = min(base_index + spec.action_horizon, terminal)

        target_size = int(self._cot_option("image_size", 224))
        depth_scale = float(self._cot_option("uvd_depth_scale", 1.0))
        target_hw = (target_size, target_size)
        current_depth = depth[base_index]
        future_depth = depth[future_index]
        current_valid = np.isfinite(current_depth) & (current_depth > 0.0)
        future_valid = np.isfinite(future_depth) & (future_depth > 0.0)
        current_depth, current_valid = _resize_depth(
            current_depth, current_valid, target_hw
        )
        future_depth, future_valid = _resize_depth(
            future_depth, future_valid, target_hw
        )

        targets = {
            "depth_current": current_depth[None].astype(np.float32),
            "depth_future": future_depth[None].astype(np.float32),
            "depth_current_valid": current_valid[None].astype(np.bool_),
            "depth_future_valid": future_valid[None].astype(np.bool_),
        }
        targets.update(
            self._uvd_group_targets(
                key="uvd",
                indices=local_indices,
                base_index=base_index,
                horizon=spec.action_horizon,
                depth=depth,
                eef_uvd=eef_uvd,
                eef_valid=eef_valid,
                target_size=target_size,
                depth_scale=depth_scale,
            )
        )
        targets.update(
            self._uvd_group_targets(
                key="uvd_coarse",
                indices=coarse_indices,
                base_index=base_index,
                horizon=spec.action_horizon,
                depth=depth,
                eef_uvd=eef_uvd,
                eef_valid=eef_valid,
                target_size=target_size,
                depth_scale=depth_scale,
            )
        )
        return targets
