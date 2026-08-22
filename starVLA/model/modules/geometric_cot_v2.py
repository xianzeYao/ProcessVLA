"""Geometry-token construction and masks for QwenGR00TCoTV2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


class SharedDepthAttentionPool(nn.Module):
    """Pool a depth-token group with one shared learned scoring function."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.score = nn.Linear(int(hidden_dim), 1, bias=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"depth tokens must have shape [B,Q,H], got {tuple(tokens.shape)}")
        scores = self.score(self.norm(tokens)).squeeze(-1)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=tokens.dtype)
        summary = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return summary, weights


def build_depth_summary_interventions(
    current: torch.Tensor,
    future: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Construct deterministic decoder-only interventions on pooled summaries."""

    if current.shape != future.shape or current.ndim != 2:
        raise ValueError(
            f"current/future summaries must share [B,H], got {tuple(current.shape)}/{tuple(future.shape)}"
        )
    variants = {
        "normal": (current, future),
        "zero": (torch.zeros_like(current), torch.zeros_like(future)),
        "swap": (future, current),
    }
    if current.shape[0] > 1:
        variants["shuffle"] = (
            torch.roll(current, shifts=1, dims=0),
            torch.roll(future, shifts=1, dims=0),
        )
    return variants


@dataclass(frozen=True)
class GeometrySequenceSlices:
    """Absolute slices after geometry tokens are appended to native Qwen tokens."""

    native: slice
    depth_current: slice
    depth_future: slice
    uvd_full: slice
    uvd: slice


@dataclass(frozen=True)
class GeometryTokenLayout:
    """Fixed geometry-token counts and ordering for one model configuration."""

    depth_query_count: int
    uvd_points_per_hand: int
    hand_count: int = 1
    full_uvd_points_per_hand: int = 0

    def __post_init__(self) -> None:
        if int(self.depth_query_count) < 1:
            raise ValueError(f"depth_query_count must be positive, got {self.depth_query_count}")
        if int(self.uvd_points_per_hand) < 2:
            raise ValueError(f"uvd_points_per_hand must be at least 2, got {self.uvd_points_per_hand}")
        if int(self.hand_count) < 1:
            raise ValueError(f"hand_count must be positive, got {self.hand_count}")
        if int(self.full_uvd_points_per_hand) < 0:
            raise ValueError(
                "full_uvd_points_per_hand must be non-negative, "
                f"got {self.full_uvd_points_per_hand}"
            )

    @property
    def uvd_token_count(self) -> int:
        return int(self.uvd_points_per_hand) * int(self.hand_count)

    @property
    def full_uvd_token_count(self) -> int:
        return int(self.full_uvd_points_per_hand) * int(self.hand_count)

    @property
    def geometry_token_count(self) -> int:
        return (
            2 * int(self.depth_query_count)
            + self.full_uvd_token_count
            + self.uvd_token_count
        )

    @property
    def geometry_current_slice(self) -> slice:
        return slice(0, int(self.depth_query_count))

    @property
    def geometry_future_slice(self) -> slice:
        start = int(self.depth_query_count)
        return slice(start, start + int(self.depth_query_count))

    @property
    def geometry_uvd_slice(self) -> slice:
        start = 2 * int(self.depth_query_count) + self.full_uvd_token_count
        return slice(start, start + self.uvd_token_count)

    @property
    def geometry_full_uvd_slice(self) -> slice:
        start = 2 * int(self.depth_query_count)
        return slice(start, start + self.full_uvd_token_count)

    def sequence_slices(self, native_token_count: int) -> GeometrySequenceSlices:
        native_token_count = int(native_token_count)
        if native_token_count < 1:
            raise ValueError(f"native_token_count must be positive, got {native_token_count}")
        current_start = native_token_count
        future_start = current_start + int(self.depth_query_count)
        full_uvd_start = future_start + int(self.depth_query_count)
        uvd_start = full_uvd_start + self.full_uvd_token_count
        return GeometrySequenceSlices(
            native=slice(0, native_token_count),
            depth_current=slice(current_start, future_start),
            depth_future=slice(future_start, full_uvd_start),
            uvd_full=slice(full_uvd_start, uvd_start),
            uvd=slice(uvd_start, uvd_start + self.uvd_token_count),
        )


@dataclass(frozen=True)
class PackedUVDTargets:
    """Fixed-size time-major UVD supervision for one batch."""

    target: torch.Tensor
    valid: torch.Tensor
    times: torch.Tensor
    hand_ids: torch.Tensor


def build_time_major_hand_ids(layout: GeometryTokenLayout, *, device: torch.device | None = None) -> torch.Tensor:
    """Return ``[0..H-1, 0..H-1, ...]`` for each temporal UVD slot."""

    return torch.arange(int(layout.hand_count), device=device, dtype=torch.long).repeat(
        int(layout.uvd_points_per_hand)
    )


def build_time_major_full_hand_ids(
    layout: GeometryTokenLayout,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return time-major hand IDs for reverse full-UVD slots."""

    return torch.arange(int(layout.hand_count), device=device, dtype=torch.long).repeat(
        int(layout.full_uvd_points_per_hand)
    )


def build_time_major_default_times(
    layout: GeometryTokenLayout,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return normalized UVD times repeated for every hand at each time."""

    times = torch.linspace(0.0, 1.0, int(layout.uvd_points_per_hand), device=device, dtype=dtype)
    return times.repeat_interleave(int(layout.hand_count))


def build_time_major_default_full_times(
    layout: GeometryTokenLayout,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return descending physical times for goal-to-current full-UVD slots."""

    times = torch.linspace(
        1.0,
        0.0,
        int(layout.full_uvd_points_per_hand),
        device=device,
        dtype=dtype,
    )
    return times.repeat_interleave(int(layout.hand_count))


class GeometryTokenEmbedding(nn.Module):
    """Create depth and time/hand-aware UVD embeddings before Qwen processing."""

    def __init__(self, hidden_dim: int, layout: GeometryTokenLayout) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layout = layout
        self.current_depth_queries = nn.Parameter(
            torch.randn(1, int(layout.depth_query_count), self.hidden_dim) * 0.02
        )
        self.future_depth_queries = nn.Parameter(
            torch.randn(1, int(layout.depth_query_count), self.hidden_dim) * 0.02
        )
        self.full_trajectory_seed = (
            nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
            if layout.full_uvd_token_count
            else None
        )
        self.trajectory_seed = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.hand_embedding = (
            nn.Embedding(int(layout.hand_count), self.hidden_dim) if int(layout.hand_count) > 1 else None
        )

    def forward(
        self,
        *,
        batch_size: int,
        uvd_times: torch.Tensor | None = None,
        uvd_hand_ids: torch.Tensor | None = None,
        full_uvd_times: torch.Tensor | None = None,
        full_uvd_hand_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        device = self.current_depth_queries.device
        dtype = self.current_depth_queries.dtype
        expected_shape = (batch_size, self.layout.uvd_token_count)

        if uvd_times is None:
            uvd_times = build_time_major_default_times(self.layout, device=device, dtype=dtype)
            uvd_times = uvd_times.unsqueeze(0).expand(batch_size, -1)
        else:
            uvd_times = uvd_times.to(device=device, dtype=dtype)
            if tuple(uvd_times.shape) != expected_shape:
                raise ValueError(f"uvd_times must have shape {expected_shape}, got {tuple(uvd_times.shape)}")

        if uvd_hand_ids is None:
            uvd_hand_ids = build_time_major_hand_ids(self.layout, device=device)
            uvd_hand_ids = uvd_hand_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            uvd_hand_ids = uvd_hand_ids.to(device=device, dtype=torch.long)
            if tuple(uvd_hand_ids.shape) != expected_shape:
                raise ValueError(
                    f"uvd_hand_ids must have shape {expected_shape}, got {tuple(uvd_hand_ids.shape)}"
                )
        if torch.any(uvd_hand_ids < 0) or torch.any(uvd_hand_ids >= int(self.layout.hand_count)):
            raise ValueError(f"uvd_hand_ids must be in [0, {self.layout.hand_count})")

        current = self.current_depth_queries.expand(batch_size, -1, -1)
        future = self.future_depth_queries.expand(batch_size, -1, -1)
        full_trajectory = current[:, :0]
        if self.full_trajectory_seed is not None:
            full_expected_shape = (batch_size, self.layout.full_uvd_token_count)
            if full_uvd_times is None:
                full_uvd_times = build_time_major_default_full_times(
                    self.layout,
                    device=device,
                    dtype=dtype,
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                full_uvd_times = full_uvd_times.to(device=device, dtype=dtype)
                if tuple(full_uvd_times.shape) != full_expected_shape:
                    raise ValueError(
                        f"full_uvd_times must have shape {full_expected_shape}, "
                        f"got {tuple(full_uvd_times.shape)}"
                    )
            if full_uvd_hand_ids is None:
                full_uvd_hand_ids = build_time_major_full_hand_ids(
                    self.layout,
                    device=device,
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                full_uvd_hand_ids = full_uvd_hand_ids.to(device=device, dtype=torch.long)
                if tuple(full_uvd_hand_ids.shape) != full_expected_shape:
                    raise ValueError(
                        f"full_uvd_hand_ids must have shape {full_expected_shape}, "
                        f"got {tuple(full_uvd_hand_ids.shape)}"
                    )
            if torch.any(full_uvd_hand_ids < 0) or torch.any(
                full_uvd_hand_ids >= int(self.layout.hand_count)
            ):
                raise ValueError(
                    f"full_uvd_hand_ids must be in [0, {self.layout.hand_count})"
                )
            full_trajectory = self.full_trajectory_seed.expand(
                batch_size,
                self.layout.full_uvd_token_count,
                -1,
            )
            full_trajectory = full_trajectory + self.time_embedding(
                full_uvd_times.unsqueeze(-1)
            )
            if self.hand_embedding is not None:
                full_trajectory = full_trajectory + self.hand_embedding(
                    full_uvd_hand_ids
                ).to(dtype=full_trajectory.dtype)
        trajectory = self.trajectory_seed.expand(batch_size, self.layout.uvd_token_count, -1)
        trajectory = trajectory + self.time_embedding(uvd_times.unsqueeze(-1))
        if self.hand_embedding is not None:
            trajectory = trajectory + self.hand_embedding(uvd_hand_ids).to(dtype=trajectory.dtype)
        return torch.cat([current, future, full_trajectory, trajectory], dim=1)


def append_geometry_slots(
    qwen_inputs: Mapping[str, Any],
    layout: GeometryTokenLayout,
    *,
    placeholder_token_id: int,
) -> dict[str, Any]:
    """Append fixed geometry placeholders that are active in train and inference."""

    if "input_ids" not in qwen_inputs or "attention_mask" not in qwen_inputs:
        raise KeyError("qwen_inputs must contain input_ids and attention_mask")
    input_ids = qwen_inputs["input_ids"]
    attention_mask = qwen_inputs["attention_mask"]
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"input_ids and attention_mask must share [B,S], got {input_ids.shape}, {attention_mask.shape}"
        )
    batch_size = input_ids.shape[0]
    device = input_ids.device

    placeholders = torch.full(
        (batch_size, layout.geometry_token_count),
        int(placeholder_token_id),
        device=device,
        dtype=input_ids.dtype,
    )
    geometry_valid = torch.ones(
        batch_size,
        layout.geometry_token_count,
        device=device,
        dtype=attention_mask.dtype,
    )

    output = dict(qwen_inputs)
    output["input_ids"] = torch.cat([input_ids, placeholders], dim=1)
    output["attention_mask"] = torch.cat([attention_mask, geometry_valid], dim=1)
    if "mm_token_type_ids" in qwen_inputs:
        mm_token_type_ids = qwen_inputs["mm_token_type_ids"]
        if mm_token_type_ids.shape != input_ids.shape:
            raise ValueError(
                "mm_token_type_ids must match input_ids before geometry slots are appended, "
                f"got {mm_token_type_ids.shape} and {input_ids.shape}"
            )
        geometry_types = torch.zeros_like(placeholders, dtype=mm_token_type_ids.dtype)
        output["mm_token_type_ids"] = torch.cat([mm_token_type_ids, geometry_types], dim=1)
    return output


def build_geometry_full_attention_mask(
    appended_attention_mask: torch.Tensor,
    layout: GeometryTokenLayout,
) -> torch.Tensor:
    """Build the full-layer boolean mask where ``True`` denotes an allowed read."""

    if appended_attention_mask.ndim != 2:
        raise ValueError(
            f"appended_attention_mask must have shape [B,S], got {tuple(appended_attention_mask.shape)}"
        )
    sequence_length = int(appended_attention_mask.shape[1])
    native_token_count = sequence_length - layout.geometry_token_count
    slices = layout.sequence_slices(native_token_count)
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
    def same_group_time(group: slice) -> torch.Tensor:
        query_in_group = (query_positions >= group.start) & (query_positions < group.stop)
        key_in_group = (key_positions >= group.start) & (key_positions < group.stop)
        query_time = torch.div(
            query_positions - group.start,
            int(layout.hand_count),
            rounding_mode="floor",
        )
        key_time = torch.div(
            key_positions - group.start,
            int(layout.hand_count),
            rounding_mode="floor",
        )
        return query_in_group & key_in_group & (query_time == key_time)

    allowed = (
        allowed
        | current_full
        | future_full
        | same_group_time(slices.uvd_full)
        | same_group_time(slices.uvd)
    )
    key_valid = appended_attention_mask.to(dtype=torch.bool)[:, None, None, :]
    return allowed[None, None, :, :] & key_valid


def pack_uvd_targets_time_major(
    examples: list[dict[str, Any]],
    layout: GeometryTokenLayout,
    *,
    device: torch.device,
) -> PackedUVDTargets:
    """Pack `[time, hand, 3]` labels into fixed `time-major` UVD slots."""

    return _pack_uvd_targets_time_major(
        examples,
        points_per_hand=int(layout.uvd_points_per_hand),
        hand_count=int(layout.hand_count),
        value_key="uvd",
        valid_key="uvd_valid_mask",
        time_key="uvd_time",
        default_reverse_time=False,
        device=device,
    )


def pack_full_uvd_targets_time_major(
    examples: list[dict[str, Any]],
    layout: GeometryTokenLayout,
    *,
    device: torch.device,
) -> PackedUVDTargets:
    """Pack reverse full-UVD labels into their independent fixed slots."""

    if int(layout.full_uvd_points_per_hand) < 1:
        raise ValueError("full_uvd_points_per_hand must be positive when packing full UVD")
    return _pack_uvd_targets_time_major(
        examples,
        points_per_hand=int(layout.full_uvd_points_per_hand),
        hand_count=int(layout.hand_count),
        value_key="uvd_full",
        valid_key="uvd_full_valid_mask",
        time_key="uvd_full_time",
        default_reverse_time=True,
        device=device,
    )


def _pack_uvd_targets_time_major(
    examples: list[dict[str, Any]],
    *,
    points_per_hand: int,
    hand_count: int,
    value_key: str,
    valid_key: str,
    time_key: str,
    default_reverse_time: bool,
    device: torch.device,
) -> PackedUVDTargets:
    """Pack one named UVD group into fixed time-major slots."""

    batch_size = len(examples)
    token_count = int(points_per_hand) * int(hand_count)
    target = torch.zeros(batch_size, token_count, 3, device=device, dtype=torch.float32)
    valid = torch.zeros(batch_size, token_count, device=device, dtype=torch.bool)
    times = torch.zeros(batch_size, token_count, device=device, dtype=torch.float32)
    hand_ids = torch.arange(hand_count, device=device, dtype=torch.long).repeat(
        points_per_hand
    ).unsqueeze(0).expand(batch_size, -1)

    for batch_index, example in enumerate(examples):
        uvd = np.asarray(example[value_key], dtype=np.float32)
        uvd_valid = np.asarray(example[valid_key], dtype=np.bool_)
        if uvd.ndim == 2:
            uvd = uvd[:, None, :]
        if uvd.ndim != 3 or uvd.shape[-1] != 3:
            raise ValueError(
                f"{value_key} must have shape [T,3] or [T,H,3], got {uvd.shape}"
            )
        if uvd_valid.ndim == 1:
            uvd_valid = uvd_valid[:, None]
        if uvd_valid.shape != uvd.shape[:2]:
            raise ValueError(
                f"{valid_key} must have shape {uvd.shape[:2]}, got {uvd_valid.shape}"
            )
        if uvd.shape[0] > points_per_hand:
            raise ValueError(
                f"{value_key} has {uvd.shape[0]} time points but the fixed layout allows "
                f"{points_per_hand}"
            )
        if uvd.shape[1] != hand_count:
            raise ValueError(
                f"{value_key} must contain exactly {hand_count} hands for this layout, "
                f"got {uvd.shape[1]}"
            )

        count = int(uvd.shape[0])
        default_times = (
            np.linspace(1.0, 0.0, count, dtype=np.float32)
            if default_reverse_time and count > 1
            else np.linspace(0.0, 1.0, count, dtype=np.float32)
            if count > 1
            else np.zeros(count, dtype=np.float32)
        )
        example_times = np.asarray(
            example.get(time_key, default_times),
            dtype=np.float32,
        )
        if example_times.shape != (count,):
            raise ValueError(
                f"{time_key} must have shape {(count,)}, got {example_times.shape}"
            )

        for time_index in range(count):
            for hand_index in range(int(uvd.shape[1])):
                token_index = time_index * hand_count + hand_index
                target[batch_index, token_index] = torch.as_tensor(uvd[time_index, hand_index], device=device)
                valid[batch_index, token_index] = bool(uvd_valid[time_index, hand_index])
                times[batch_index, token_index] = float(example_times[time_index])

    return PackedUVDTargets(target=target, valid=valid, times=times, hand_ids=hand_ids)
