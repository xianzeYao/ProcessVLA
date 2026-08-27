"""Decoder-only bilateral LRW expansion for RoboCasa CoT V5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from starVLA.model.modules.geometric_cot_v2 import (
    GeometryTokenEmbedding,
    GeometryTokenLayout,
    SharedDepthAttentionPool,
    append_geometry_slots,
    build_depth_summary_interventions,
    build_geometry_full_attention_mask,
)


@dataclass(frozen=True)
class HandConfigurationTokenLayout(GeometryTokenLayout):
    """Keep one Qwen token per `(time, hand)` and decode three landmarks."""

    landmark_count: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.hand_count) != 2:
            raise ValueError(
                f"hand_count must be exactly 2 for bilateral LRW, got {self.hand_count}"
            )
        if int(self.landmark_count) != 3:
            raise ValueError(
                "landmark_count must be exactly 3 for [thumb,index,wrist], "
                f"got {self.landmark_count}"
            )

    @property
    def output_point_count(self) -> int:
        return self.uvd_token_count * int(self.landmark_count)


@dataclass(frozen=True)
class PackedHandLRWTargets:
    """Fixed `[batch,time,hand,landmark]` supervision flattened for decoding."""

    target: torch.Tensor
    valid: torch.Tensor
    times: torch.Tensor
    hand_ids: torch.Tensor
    landmark_ids: torch.Tensor


def build_time_major_hand_landmark_ids(
    layout: HandConfigurationTokenLayout,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return IDs in `[L_left,R_left,W_left,L_right,R_right,W_right]` order."""

    per_time_hand = torch.arange(
        int(layout.hand_count), device=device, dtype=torch.long
    ).repeat_interleave(int(layout.landmark_count))
    per_time_landmark = torch.arange(
        int(layout.landmark_count), device=device, dtype=torch.long
    ).repeat(int(layout.hand_count))
    return (
        per_time_hand.repeat(int(layout.uvd_points_per_hand)),
        per_time_landmark.repeat(int(layout.uvd_points_per_hand)),
    )


def _canonical_example_ids(
    count: int,
    layout: HandConfigurationTokenLayout,
) -> tuple[np.ndarray, np.ndarray]:
    hand_ids = np.broadcast_to(
        np.arange(int(layout.hand_count), dtype=np.int64)[None, :, None],
        (count, int(layout.hand_count), int(layout.landmark_count)),
    )
    landmark_ids = np.broadcast_to(
        np.arange(int(layout.landmark_count), dtype=np.int64)[None, None, :],
        (count, int(layout.hand_count), int(layout.landmark_count)),
    )
    return hand_ids, landmark_ids


def pack_hand_lrw_targets_time_major(
    examples: list[dict[str, Any]],
    layout: HandConfigurationTokenLayout,
    *,
    device: torch.device,
) -> PackedHandLRWTargets:
    """Strictly pack `[T,2,3,3]` labels into fixed 36-point supervision."""

    batch_size = len(examples)
    target = torch.zeros(
        batch_size,
        layout.output_point_count,
        3,
        device=device,
        dtype=torch.float32,
    )
    valid = torch.zeros(
        batch_size,
        layout.output_point_count,
        device=device,
        dtype=torch.bool,
    )
    times = torch.zeros(
        batch_size,
        layout.output_point_count,
        device=device,
        dtype=torch.float32,
    )
    flat_hand_ids, flat_landmark_ids = build_time_major_hand_landmark_ids(
        layout, device=device
    )
    hand_ids = flat_hand_ids.unsqueeze(0).expand(batch_size, -1)
    landmark_ids = flat_landmark_ids.unsqueeze(0).expand(batch_size, -1)

    for batch_index, example in enumerate(examples):
        uvd = np.asarray(example["uvd"], dtype=np.float32)
        expected_tail = (
            int(layout.hand_count),
            int(layout.landmark_count),
            3,
        )
        if uvd.ndim != 4 or uvd.shape[1:] != expected_tail:
            raise ValueError(
                "uvd must have shape [T,2,3,3] in [time,hand,landmark,coord] "
                f"order, got {uvd.shape}"
            )
        count = int(uvd.shape[0])
        if count > int(layout.uvd_points_per_hand):
            raise ValueError(
                f"uvd has {count} time points but layout allows "
                f"{layout.uvd_points_per_hand}"
            )

        uvd_valid = np.asarray(example["uvd_valid_mask"], dtype=np.bool_)
        if uvd_valid.shape != uvd.shape[:-1]:
            raise ValueError(
                f"uvd_valid_mask must have shape {uvd.shape[:-1]}, got "
                f"{uvd_valid.shape}"
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

        canonical_hand_ids, canonical_landmark_ids = _canonical_example_ids(
            count, layout
        )
        example_hand_ids = np.asarray(
            example.get("uvd_hand_ids"), dtype=np.int64
        )
        if example_hand_ids.shape != canonical_hand_ids.shape or not np.array_equal(
            example_hand_ids, canonical_hand_ids
        ):
            raise ValueError(
                "uvd_hand_ids must have shape [T,2,3] with each time equal to "
                "[[0,0,0],[1,1,1]]"
            )
        example_landmark_ids = np.asarray(
            example.get("uvd_landmark_ids"), dtype=np.int64
        )
        if (
            example_landmark_ids.shape != canonical_landmark_ids.shape
            or not np.array_equal(example_landmark_ids, canonical_landmark_ids)
        ):
            raise ValueError(
                "uvd_landmark_ids must have shape [T,2,3] with each hand equal "
                "to [0,1,2]"
            )

        points_per_time = int(layout.hand_count) * int(layout.landmark_count)
        flat_count = count * points_per_time
        target[batch_index, :flat_count] = torch.as_tensor(
            uvd.reshape(flat_count, 3), device=device
        )
        valid[batch_index, :flat_count] = torch.as_tensor(
            uvd_valid.reshape(flat_count), device=device
        )
        times[batch_index, :flat_count] = torch.as_tensor(
            np.repeat(example_times, points_per_time), device=device
        )

    return PackedHandLRWTargets(
        target=target,
        valid=valid,
        times=times,
        hand_ids=hand_ids,
        landmark_ids=landmark_ids,
    )


class HandLRWDecoder(nn.Module):
    """Expand each Qwen `(time,hand)` state into thumb/index/wrist UVD."""

    def __init__(self, hidden_dim: int, landmark_count: int = 3) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        landmark_count = int(landmark_count)
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if landmark_count != 3:
            raise ValueError(
                f"landmark_count must be exactly 3 for LRW, got {landmark_count}"
            )
        self.hidden_dim = hidden_dim
        self.landmark_count = landmark_count
        self.landmark_embedding = nn.Embedding(landmark_count, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, hand_tokens: torch.Tensor) -> torch.Tensor:
        if hand_tokens.ndim != 3 or int(hand_tokens.shape[-1]) != self.hidden_dim:
            raise ValueError(
                "hand_tokens must have shape [B,N,H] with H="
                f"{self.hidden_dim}, got {tuple(hand_tokens.shape)}"
            )
        expanded = (
            hand_tokens[:, :, None, :]
            + self.landmark_embedding.weight[None, None, :, :]
        )
        decoded = self.mlp(expanded)
        return decoded.reshape(hand_tokens.shape[0], -1, 3)


__all__ = [
    "GeometryTokenEmbedding",
    "HandConfigurationTokenLayout",
    "HandLRWDecoder",
    "PackedHandLRWTargets",
    "SharedDepthAttentionPool",
    "append_geometry_slots",
    "build_depth_summary_interventions",
    "build_geometry_full_attention_mask",
    "build_time_major_hand_landmark_ids",
    "pack_hand_lrw_targets_time_major",
]
