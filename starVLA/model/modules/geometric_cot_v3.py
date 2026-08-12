"""Time-major landmark geometry tokens for QwenGR00TCoTV3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from starVLA.model.modules.geometric_cot_v2 import (
    GeometrySequenceSlices,
    SharedDepthAttentionPool,
    append_geometry_slots,
    build_depth_summary_interventions,
)


@dataclass(frozen=True)
class LandmarkGeometryTokenLayout:
    depth_query_count: int
    uvd_time_points: int
    landmark_count: int = 3

    def __post_init__(self) -> None:
        if int(self.depth_query_count) < 1:
            raise ValueError(
                f"depth_query_count must be positive, got {self.depth_query_count}"
            )
        if int(self.uvd_time_points) < 2:
            raise ValueError(
                f"uvd_time_points must be at least 2, got {self.uvd_time_points}"
            )
        if int(self.landmark_count) != 3:
            raise ValueError(
                f"landmark_count must be exactly 3 for [L,R,W], got {self.landmark_count}"
            )

    @property
    def uvd_token_count(self) -> int:
        return int(self.uvd_time_points) * int(self.landmark_count)

    @property
    def geometry_token_count(self) -> int:
        return 2 * int(self.depth_query_count) + self.uvd_token_count

    @property
    def geometry_current_slice(self) -> slice:
        return slice(0, int(self.depth_query_count))

    @property
    def geometry_future_slice(self) -> slice:
        start = int(self.depth_query_count)
        return slice(start, start + int(self.depth_query_count))

    @property
    def geometry_uvd_slice(self) -> slice:
        start = 2 * int(self.depth_query_count)
        return slice(start, start + self.uvd_token_count)

    def sequence_slices(self, native_token_count: int) -> GeometrySequenceSlices:
        native_token_count = int(native_token_count)
        if native_token_count < 1:
            raise ValueError(
                f"native_token_count must be positive, got {native_token_count}"
            )
        current_start = native_token_count
        future_start = current_start + int(self.depth_query_count)
        uvd_start = future_start + int(self.depth_query_count)
        return GeometrySequenceSlices(
            native=slice(0, native_token_count),
            depth_current=slice(current_start, future_start),
            depth_future=slice(future_start, uvd_start),
            uvd=slice(uvd_start, uvd_start + self.uvd_token_count),
        )


@dataclass(frozen=True)
class PackedLandmarkUVDTargets:
    target: torch.Tensor
    valid: torch.Tensor
    times: torch.Tensor
    landmark_ids: torch.Tensor


def build_time_major_landmark_ids(
    layout: LandmarkGeometryTokenLayout,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.arange(
        int(layout.landmark_count), device=device, dtype=torch.long
    ).repeat(int(layout.uvd_time_points))


def build_time_major_default_times(
    layout: LandmarkGeometryTokenLayout,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    times = torch.linspace(
        0.0, 1.0, int(layout.uvd_time_points), device=device, dtype=dtype
    )
    return times.repeat_interleave(int(layout.landmark_count))


class LandmarkGeometryTokenEmbedding(nn.Module):
    def __init__(
        self, hidden_dim: int, layout: LandmarkGeometryTokenLayout
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layout = layout
        self.current_depth_queries = nn.Parameter(
            torch.randn(1, int(layout.depth_query_count), self.hidden_dim) * 0.02
        )
        self.future_depth_queries = nn.Parameter(
            torch.randn(1, int(layout.depth_query_count), self.hidden_dim) * 0.02
        )
        self.trajectory_seed = nn.Parameter(
            torch.randn(1, 1, self.hidden_dim) * 0.02
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.landmark_embedding = nn.Embedding(
            int(layout.landmark_count), self.hidden_dim
        )

    def forward(
        self,
        *,
        batch_size: int,
        uvd_times: torch.Tensor | None = None,
        uvd_landmark_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        device = self.current_depth_queries.device
        dtype = self.current_depth_queries.dtype
        expected_shape = (batch_size, self.layout.uvd_token_count)
        if uvd_times is None:
            uvd_times = build_time_major_default_times(
                self.layout, device=device, dtype=dtype
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            uvd_times = uvd_times.to(device=device, dtype=dtype)
            if tuple(uvd_times.shape) != expected_shape:
                raise ValueError(
                    f"uvd_times must have shape {expected_shape}, got {tuple(uvd_times.shape)}"
                )
        if uvd_landmark_ids is None:
            uvd_landmark_ids = build_time_major_landmark_ids(
                self.layout, device=device
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            uvd_landmark_ids = uvd_landmark_ids.to(
                device=device, dtype=torch.long
            )
            if tuple(uvd_landmark_ids.shape) != expected_shape:
                raise ValueError(
                    "uvd_landmark_ids must have shape "
                    f"{expected_shape}, got {tuple(uvd_landmark_ids.shape)}"
                )
        if torch.any(uvd_landmark_ids < 0) or torch.any(
            uvd_landmark_ids >= int(self.layout.landmark_count)
        ):
            raise ValueError(
                f"uvd_landmark_ids must be in [0,{self.layout.landmark_count})"
            )
        current = self.current_depth_queries.expand(batch_size, -1, -1)
        future = self.future_depth_queries.expand(batch_size, -1, -1)
        trajectory = self.trajectory_seed.expand(
            batch_size, self.layout.uvd_token_count, -1
        )
        trajectory = trajectory + self.time_embedding(uvd_times.unsqueeze(-1))
        trajectory = trajectory + self.landmark_embedding(uvd_landmark_ids).to(
            dtype=trajectory.dtype
        )
        return torch.cat([current, future, trajectory], dim=1)


def build_landmark_geometry_full_attention_mask(
    appended_attention_mask: torch.Tensor,
    layout: LandmarkGeometryTokenLayout,
) -> torch.Tensor:
    if appended_attention_mask.ndim != 2:
        raise ValueError(
            "appended_attention_mask must have shape [B,S], got "
            f"{tuple(appended_attention_mask.shape)}"
        )
    sequence_length = int(appended_attention_mask.shape[1])
    slices = layout.sequence_slices(sequence_length - layout.geometry_token_count)
    device = appended_attention_mask.device
    positions = torch.arange(sequence_length, device=device)
    query_positions = positions[:, None]
    key_positions = positions[None, :]
    allowed = key_positions <= query_positions
    current_full = (
        (query_positions >= slices.depth_current.start)
        & (query_positions < slices.depth_current.stop)
        & (key_positions >= slices.depth_current.start)
        & (key_positions < slices.depth_current.stop)
    )
    future_full = (
        (query_positions >= slices.depth_future.start)
        & (query_positions < slices.depth_future.stop)
        & (key_positions >= slices.depth_future.start)
        & (key_positions < slices.depth_future.stop)
    )
    query_is_uvd = (query_positions >= slices.uvd.start) & (
        query_positions < slices.uvd.stop
    )
    key_is_uvd = (key_positions >= slices.uvd.start) & (
        key_positions < slices.uvd.stop
    )
    query_time = torch.div(
        query_positions - slices.uvd.start,
        int(layout.landmark_count),
        rounding_mode="floor",
    )
    key_time = torch.div(
        key_positions - slices.uvd.start,
        int(layout.landmark_count),
        rounding_mode="floor",
    )
    allowed = allowed | current_full | future_full | (
        query_is_uvd & key_is_uvd & (query_time == key_time)
    )
    key_valid = appended_attention_mask.to(dtype=torch.bool)[:, None, None, :]
    return allowed[None, None, :, :] & key_valid


def pack_landmark_uvd_targets_time_major(
    examples: list[dict[str, Any]],
    layout: LandmarkGeometryTokenLayout,
    *,
    device: torch.device,
) -> PackedLandmarkUVDTargets:
    batch_size = len(examples)
    target = torch.zeros(
        batch_size, layout.uvd_token_count, 3, device=device, dtype=torch.float32
    )
    valid = torch.zeros(
        batch_size, layout.uvd_token_count, device=device, dtype=torch.bool
    )
    times = torch.zeros(
        batch_size, layout.uvd_token_count, device=device, dtype=torch.float32
    )
    landmark_ids = build_time_major_landmark_ids(
        layout, device=device
    ).unsqueeze(0).expand(batch_size, -1)
    canonical_ids = np.arange(3, dtype=np.int64)

    for batch_index, example in enumerate(examples):
        uvd = np.asarray(example["uvd"], dtype=np.float32)
        uvd_valid = np.asarray(example["uvd_valid_mask"], dtype=np.bool_)
        if uvd.ndim != 3 or uvd.shape[-1] != 3:
            raise ValueError(f"uvd must have shape [T,3,3], got {uvd.shape}")
        if uvd.shape[1] != int(layout.landmark_count):
            raise ValueError(
                f"uvd must contain exactly 3 landmarks, got {uvd.shape[1]}"
            )
        if uvd_valid.shape != uvd.shape[:2]:
            raise ValueError(
                f"uvd_valid_mask must have shape {uvd.shape[:2]}, got {uvd_valid.shape}"
            )
        count = int(uvd.shape[0])
        if count > int(layout.uvd_time_points):
            raise ValueError(
                f"uvd has {count} time points but layout allows {layout.uvd_time_points}"
            )
        example_times = np.asarray(
            example.get(
                "uvd_time",
                np.linspace(0.0, 1.0, count, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if example_times.shape != (count,):
            raise ValueError(
                f"uvd_time must have shape {(count,)}, got {example_times.shape}"
            )
        example_ids = np.asarray(
            example.get(
                "uvd_landmark_ids",
                np.broadcast_to(canonical_ids, (count, 3)),
            ),
            dtype=np.int64,
        )
        if example_ids.shape != (count, 3) or not np.array_equal(
            example_ids, np.broadcast_to(canonical_ids, (count, 3))
        ):
            raise ValueError(
                "uvd_landmark_ids must have shape [T,3] with every row [0,1,2]"
            )
        flat_count = count * int(layout.landmark_count)
        target[batch_index, :flat_count] = torch.as_tensor(
            uvd.reshape(flat_count, 3), device=device
        )
        valid[batch_index, :flat_count] = torch.as_tensor(
            uvd_valid.reshape(flat_count), device=device
        )
        times[batch_index, :flat_count] = torch.as_tensor(
            np.repeat(example_times, 3), device=device
        )
    return PackedLandmarkUVDTargets(
        target=target,
        valid=valid,
        times=times,
        landmark_ids=landmark_ids,
    )


__all__ = [
    "LandmarkGeometryTokenLayout",
    "LandmarkGeometryTokenEmbedding",
    "PackedLandmarkUVDTargets",
    "SharedDepthAttentionPool",
    "append_geometry_slots",
    "build_depth_summary_interventions",
    "build_time_major_landmark_ids",
    "pack_landmark_uvd_targets_time_major",
    "build_landmark_geometry_full_attention_mask",
]
