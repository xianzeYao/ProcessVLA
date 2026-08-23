"""Geometry tokens and masks for forward coarse-to-local QwenGR00TCoTV4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from starVLA.model.modules.geometric_cot_v2 import (
    SharedDepthAttentionPool,
    build_depth_summary_interventions,
)


@dataclass(frozen=True)
class GeometrySequenceSlices:
    native: slice
    depth_current: slice
    depth_future: slice
    uvd_coarse: slice
    uvd_local: slice


@dataclass(frozen=True)
class PackedUVDTargets:
    target: torch.Tensor
    valid: torch.Tensor
    times: torch.Tensor
    hand_ids: torch.Tensor


@dataclass(frozen=True)
class GeometryTokenLayout:
    depth_query_count: int
    local_uvd_points_per_hand: int
    coarse_uvd_points_per_hand: int
    hand_count: int = 1

    def __post_init__(self) -> None:
        if int(self.depth_query_count) < 1:
            raise ValueError(
                f"depth_query_count must be positive, got {self.depth_query_count}"
            )
        if int(self.local_uvd_points_per_hand) < 2:
            raise ValueError(
                "local_uvd_points_per_hand must be at least 2, "
                f"got {self.local_uvd_points_per_hand}"
            )
        if int(self.coarse_uvd_points_per_hand) < 2:
            raise ValueError(
                "coarse_uvd_points_per_hand must be at least 2, "
                f"got {self.coarse_uvd_points_per_hand}"
            )
        if int(self.hand_count) < 1:
            raise ValueError(f"hand_count must be positive, got {self.hand_count}")

    @property
    def local_uvd_token_count(self) -> int:
        return int(self.local_uvd_points_per_hand) * int(self.hand_count)

    @property
    def coarse_uvd_token_count(self) -> int:
        return int(self.coarse_uvd_points_per_hand) * int(self.hand_count)

    @property
    def geometry_token_count(self) -> int:
        return (
            2 * int(self.depth_query_count)
            + self.coarse_uvd_token_count
            + self.local_uvd_token_count
        )

    def sequence_slices(self, native_token_count: int) -> GeometrySequenceSlices:
        native_token_count = int(native_token_count)
        if native_token_count < 1:
            raise ValueError(
                f"native_token_count must be positive, got {native_token_count}"
            )
        current = slice(
            native_token_count,
            native_token_count + int(self.depth_query_count),
        )
        future = slice(
            current.stop,
            current.stop + int(self.depth_query_count),
        )
        coarse = slice(
            future.stop,
            future.stop + self.coarse_uvd_token_count,
        )
        local = slice(
            coarse.stop,
            coarse.stop + self.local_uvd_token_count,
        )
        return GeometrySequenceSlices(
            native=slice(0, native_token_count),
            depth_current=current,
            depth_future=future,
            uvd_coarse=coarse,
            uvd_local=local,
        )


def build_time_major_hand_ids(
    *,
    points_per_hand: int,
    hand_count: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.arange(
        int(hand_count),
        device=device,
        dtype=torch.long,
    ).repeat(int(points_per_hand))


def build_time_major_default_times(
    *,
    points_per_hand: int,
    hand_count: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    times = torch.arange(
        1,
        int(points_per_hand) + 1,
        device=device,
        dtype=dtype,
    ) / float(points_per_hand)
    return times.repeat_interleave(int(hand_count))


class GeometryTokenEmbedding(nn.Module):
    """Create independent depth, coarse, and local learnable query groups."""

    def __init__(self, hidden_dim: int, layout: GeometryTokenLayout) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layout = layout
        self.current_depth_queries = nn.Parameter(
            torch.randn(
                1,
                int(layout.depth_query_count),
                self.hidden_dim,
            )
            * 0.02
        )
        self.future_depth_queries = nn.Parameter(
            torch.randn(
                1,
                int(layout.depth_query_count),
                self.hidden_dim,
            )
            * 0.02
        )
        self.coarse_queries = nn.Parameter(
            torch.randn(
                1,
                layout.coarse_uvd_token_count,
                self.hidden_dim,
            )
            * 0.02
        )
        self.local_queries = nn.Parameter(
            torch.randn(
                1,
                layout.local_uvd_token_count,
                self.hidden_dim,
            )
            * 0.02
        )
        self.coarse_time_embedding = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.local_time_embedding = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        if int(layout.hand_count) > 1:
            self.coarse_hand_embedding = nn.Embedding(
                int(layout.hand_count),
                self.hidden_dim,
            )
            self.local_hand_embedding = nn.Embedding(
                int(layout.hand_count),
                self.hidden_dim,
            )
        else:
            self.coarse_hand_embedding = None
            self.local_hand_embedding = None

    def _trajectory_group(
        self,
        *,
        batch_size: int,
        queries: torch.Tensor,
        time_embedding: nn.Module,
        hand_embedding: nn.Embedding | None,
        points_per_hand: int,
        times: torch.Tensor | None,
        hand_ids: torch.Tensor | None,
        name: str,
    ) -> torch.Tensor:
        token_count = int(points_per_hand) * int(self.layout.hand_count)
        expected_shape = (batch_size, token_count)
        device = queries.device
        dtype = queries.dtype
        if times is None:
            times = build_time_major_default_times(
                points_per_hand=points_per_hand,
                hand_count=self.layout.hand_count,
                device=device,
                dtype=dtype,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            times = times.to(device=device, dtype=dtype)
            if tuple(times.shape) != expected_shape:
                raise ValueError(
                    f"{name}_uvd_times must have shape {expected_shape}, "
                    f"got {tuple(times.shape)}"
                )
        if hand_ids is None:
            hand_ids = build_time_major_hand_ids(
                points_per_hand=points_per_hand,
                hand_count=self.layout.hand_count,
                device=device,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            hand_ids = hand_ids.to(device=device, dtype=torch.long)
            if tuple(hand_ids.shape) != expected_shape:
                raise ValueError(
                    f"{name}_uvd_hand_ids must have shape {expected_shape}, "
                    f"got {tuple(hand_ids.shape)}"
                )
        if torch.any(hand_ids < 0) or torch.any(
            hand_ids >= int(self.layout.hand_count)
        ):
            raise ValueError(
                f"{name}_uvd_hand_ids must be in [0, {self.layout.hand_count})"
            )

        group = queries.expand(batch_size, -1, -1)
        group = group + time_embedding(times.unsqueeze(-1))
        if hand_embedding is not None:
            group = group + hand_embedding(hand_ids).to(dtype=group.dtype)
        return group

    def forward(
        self,
        *,
        batch_size: int,
        coarse_uvd_times: torch.Tensor | None = None,
        coarse_uvd_hand_ids: torch.Tensor | None = None,
        local_uvd_times: torch.Tensor | None = None,
        local_uvd_hand_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        current = self.current_depth_queries.expand(batch_size, -1, -1)
        future = self.future_depth_queries.expand(batch_size, -1, -1)
        coarse = self._trajectory_group(
            batch_size=batch_size,
            queries=self.coarse_queries,
            time_embedding=self.coarse_time_embedding,
            hand_embedding=self.coarse_hand_embedding,
            points_per_hand=self.layout.coarse_uvd_points_per_hand,
            times=coarse_uvd_times,
            hand_ids=coarse_uvd_hand_ids,
            name="coarse",
        )
        local = self._trajectory_group(
            batch_size=batch_size,
            queries=self.local_queries,
            time_embedding=self.local_time_embedding,
            hand_embedding=self.local_hand_embedding,
            points_per_hand=self.layout.local_uvd_points_per_hand,
            times=local_uvd_times,
            hand_ids=local_uvd_hand_ids,
            name="local",
        )
        return torch.cat([current, future, coarse, local], dim=1)


def append_geometry_slots(
    qwen_inputs: Mapping[str, Any],
    layout: GeometryTokenLayout,
    *,
    placeholder_token_id: int,
) -> dict[str, Any]:
    if "input_ids" not in qwen_inputs or "attention_mask" not in qwen_inputs:
        raise KeyError("qwen_inputs must contain input_ids and attention_mask")
    input_ids = qwen_inputs["input_ids"]
    attention_mask = qwen_inputs["attention_mask"]
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "input_ids and attention_mask must share [B,S], "
            f"got {input_ids.shape}/{attention_mask.shape}"
        )
    placeholders = torch.full(
        (input_ids.shape[0], layout.geometry_token_count),
        int(placeholder_token_id),
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    geometry_valid = torch.ones(
        input_ids.shape[0],
        layout.geometry_token_count,
        device=input_ids.device,
        dtype=attention_mask.dtype,
    )
    output = dict(qwen_inputs)
    output["input_ids"] = torch.cat([input_ids, placeholders], dim=1)
    output["attention_mask"] = torch.cat(
        [attention_mask, geometry_valid],
        dim=1,
    )
    if "mm_token_type_ids" in qwen_inputs:
        mm_token_type_ids = qwen_inputs["mm_token_type_ids"]
        if mm_token_type_ids.shape != input_ids.shape:
            raise ValueError(
                "mm_token_type_ids must match input_ids before geometry append"
            )
        geometry_types = torch.zeros_like(
            placeholders,
            dtype=mm_token_type_ids.dtype,
        )
        output["mm_token_type_ids"] = torch.cat(
            [mm_token_type_ids, geometry_types],
            dim=1,
        )
    return output


def build_geometry_full_attention_mask(
    appended_attention_mask: torch.Tensor,
    layout: GeometryTokenLayout,
) -> torch.Tensor:
    if appended_attention_mask.ndim != 2:
        raise ValueError(
            "appended_attention_mask must have shape [B,S], "
            f"got {tuple(appended_attention_mask.shape)}"
        )
    sequence_length = int(appended_attention_mask.shape[1])
    native_token_count = sequence_length - layout.geometry_token_count
    slices = layout.sequence_slices(native_token_count)
    device = appended_attention_mask.device
    positions = torch.arange(sequence_length, device=device)
    query_positions = positions[:, None]
    key_positions = positions[None, :]
    allowed = key_positions <= query_positions

    def fully_connected(group: slice) -> torch.Tensor:
        return (
            (query_positions >= group.start)
            & (query_positions < group.stop)
            & (key_positions >= group.start)
            & (key_positions < group.stop)
        )

    def same_time(group: slice) -> torch.Tensor:
        query_in_group = (
            (query_positions >= group.start)
            & (query_positions < group.stop)
        )
        key_in_group = (
            (key_positions >= group.start)
            & (key_positions < group.stop)
        )
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
        | fully_connected(slices.depth_current)
        | fully_connected(slices.depth_future)
        | same_time(slices.uvd_coarse)
        | same_time(slices.uvd_local)
    )
    key_valid = appended_attention_mask.to(dtype=torch.bool)[:, None, None, :]
    return allowed[None, None, :, :] & key_valid


def _pack_named_uvd_targets_time_major(
    examples: list[dict[str, Any]],
    *,
    points_per_hand: int,
    hand_count: int,
    value_key: str,
    valid_key: str,
    time_key: str,
    device: torch.device,
) -> PackedUVDTargets:
    batch_size = len(examples)
    token_count = int(points_per_hand) * int(hand_count)
    target = torch.empty(
        batch_size,
        token_count,
        3,
        device=device,
        dtype=torch.float32,
    )
    valid = torch.empty(
        batch_size,
        token_count,
        device=device,
        dtype=torch.bool,
    )
    times = torch.empty(
        batch_size,
        token_count,
        device=device,
        dtype=torch.float32,
    )
    hand_ids = build_time_major_hand_ids(
        points_per_hand=points_per_hand,
        hand_count=hand_count,
        device=device,
    )
    for batch_index, example in enumerate(examples):
        values = np.asarray(example[value_key], dtype=np.float32)
        masks = np.asarray(example[valid_key], dtype=np.bool_)
        nominal_times = np.asarray(example[time_key], dtype=np.float32)
        expected_values = (
            (points_per_hand, 3)
            if int(hand_count) == 1
            else (points_per_hand, hand_count, 3)
        )
        expected_masks = (
            (points_per_hand,)
            if int(hand_count) == 1
            else (points_per_hand, hand_count)
        )
        if values.shape != expected_values or masks.shape != expected_masks:
            raise ValueError(
                f"{value_key} must contain exactly {points_per_hand} "
                f"time points and {hand_count} hands"
            )
        if nominal_times.shape != (points_per_hand,):
            raise ValueError(
                f"{time_key} must have shape ({points_per_hand},)"
            )
        target[batch_index] = torch.as_tensor(
            values.reshape(token_count, 3),
            device=device,
        )
        valid[batch_index] = torch.as_tensor(
            masks.reshape(token_count),
            device=device,
        )
        times[batch_index] = torch.as_tensor(
            np.repeat(nominal_times, hand_count),
            device=device,
        )
    return PackedUVDTargets(
        target=target,
        valid=valid,
        times=times,
        hand_ids=hand_ids.unsqueeze(0).expand(batch_size, -1),
    )


def pack_local_uvd_targets_time_major(
    examples: list[dict[str, Any]],
    layout: GeometryTokenLayout,
    *,
    device: torch.device,
) -> PackedUVDTargets:
    return _pack_named_uvd_targets_time_major(
        examples,
        points_per_hand=layout.local_uvd_points_per_hand,
        hand_count=layout.hand_count,
        value_key="uvd",
        valid_key="uvd_valid_mask",
        time_key="uvd_time",
        device=device,
    )


def pack_coarse_uvd_targets_time_major(
    examples: list[dict[str, Any]],
    layout: GeometryTokenLayout,
    *,
    device: torch.device,
) -> PackedUVDTargets:
    return _pack_named_uvd_targets_time_major(
        examples,
        points_per_hand=layout.coarse_uvd_points_per_hand,
        hand_count=layout.hand_count,
        value_key="uvd_coarse",
        valid_key="uvd_coarse_valid_mask",
        time_key="uvd_coarse_time",
        device=device,
    )
