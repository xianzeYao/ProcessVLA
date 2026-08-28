from __future__ import annotations

import numpy as np
import pytest
import torch

from starVLA.model.modules.geometric_cot_v2 import (
    GeometryTokenEmbedding,
    build_geometry_full_attention_mask,
)
from starVLA.model.modules.geometric_cot_v5 import (
    HandConfigurationTokenLayout,
    HandLRWDecoder,
    build_time_major_hand_landmark_ids,
    pack_hand_lrw_targets_time_major,
)


def _example(time_count: int = 2) -> dict:
    uvd = np.zeros((time_count, 2, 3, 3), dtype=np.float32)
    for time in range(time_count):
        for hand in range(2):
            for landmark in range(3):
                base = 100 * time + 10 * hand + landmark
                uvd[time, hand, landmark] = [base, base + 1, base + 2]
    return {
        "uvd": uvd,
        "uvd_valid_mask": np.asarray(
            [
                [[True, True, True], [True, False, True]],
                [[True, True, False], [True, True, True]],
            ][:time_count],
            dtype=np.bool_,
        ),
        "uvd_time": np.asarray([0.25, 0.75][:time_count], dtype=np.float32),
        "uvd_hand_ids": np.broadcast_to(
            np.asarray([[[0, 0, 0], [1, 1, 1]]], dtype=np.int64),
            (time_count, 2, 3),
        ).copy(),
        "uvd_landmark_ids": np.broadcast_to(
            np.asarray([[[0, 1, 2], [0, 1, 2]]], dtype=np.int64),
            (time_count, 2, 3),
        ).copy(),
    }


def _single_hand_example(time_count: int = 2) -> dict:
    bilateral = _example(time_count)
    return {
        "uvd": bilateral["uvd"][:, :1].copy(),
        "uvd_valid_mask": bilateral["uvd_valid_mask"][:, :1].copy(),
        "uvd_time": bilateral["uvd_time"].copy(),
        "uvd_hand_ids": np.zeros((time_count, 1, 3), dtype=np.int64),
        "uvd_landmark_ids": np.broadcast_to(
            np.asarray([[[0, 1, 2]]], dtype=np.int64),
            (time_count, 1, 3),
        ).copy(),
    }


def test_layout_keeps_12_qwen_tokens_but_declares_36_outputs():
    layout = HandConfigurationTokenLayout(
        depth_query_count=8,
        uvd_points_per_hand=6,
        hand_count=2,
        landmark_count=3,
    )

    assert layout.uvd_token_count == 12
    assert layout.output_point_count == 36
    assert layout.geometry_token_count == 28
    assert layout.geometry_uvd_slice == slice(16, 28)


def test_single_hand_layout_keeps_one_qwen_token_per_time_and_three_outputs():
    layout = HandConfigurationTokenLayout(
        depth_query_count=8,
        uvd_points_per_hand=4,
        hand_count=1,
        landmark_count=3,
    )

    assert layout.uvd_token_count == 4
    assert layout.output_point_count == 12
    assert layout.geometry_token_count == 20
    assert layout.geometry_uvd_slice == slice(16, 20)


@pytest.mark.parametrize("hand_count", [0, 3])
def test_layout_rejects_hand_counts_outside_single_or_bilateral(hand_count):
    with pytest.raises(ValueError, match="one or two"):
        HandConfigurationTokenLayout(
            depth_query_count=1,
            uvd_points_per_hand=2,
            hand_count=hand_count,
        )


def test_time_major_ids_match_lrw_order_within_each_hand_token():
    layout = HandConfigurationTokenLayout(
        depth_query_count=1, uvd_points_per_hand=2, hand_count=2
    )

    hand_ids, landmark_ids = build_time_major_hand_landmark_ids(layout)

    assert hand_ids.tolist() == [0, 0, 0, 1, 1, 1] * 2
    assert landmark_ids.tolist() == [0, 1, 2, 0, 1, 2] * 2


def test_packer_flattens_time_hand_landmark_and_pads_only_as_invalid():
    layout = HandConfigurationTokenLayout(
        depth_query_count=1, uvd_points_per_hand=3, hand_count=2
    )

    packed = pack_hand_lrw_targets_time_major(
        [_example()], layout, device=torch.device("cpu")
    )

    assert packed.target.shape == (1, 18, 3)
    assert packed.target[0, :12].tolist() == [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [10, 11, 12],
        [11, 12, 13],
        [12, 13, 14],
        [100, 101, 102],
        [101, 102, 103],
        [102, 103, 104],
        [110, 111, 112],
        [111, 112, 113],
        [112, 113, 114],
    ]
    assert packed.valid[0, :12].tolist() == [
        True, True, True, True, False, True,
        True, True, False, True, True, True,
    ]
    assert not packed.valid[0, 12:].any()
    assert torch.equal(packed.target[0, 12:], torch.zeros(6, 3))
    assert packed.times[0, :12].tolist() == [0.25] * 6 + [0.75] * 6
    assert packed.hand_ids[0].tolist() == [0, 0, 0, 1, 1, 1] * 3
    assert packed.landmark_ids[0].tolist() == [0, 1, 2, 0, 1, 2] * 3


def test_packer_accepts_single_hand_lrw_targets_in_time_major_order():
    layout = HandConfigurationTokenLayout(
        depth_query_count=1,
        uvd_points_per_hand=2,
        hand_count=1,
    )

    packed = pack_hand_lrw_targets_time_major(
        [_single_hand_example()], layout, device=torch.device("cpu")
    )

    assert packed.target.shape == (1, 6, 3)
    assert packed.target[0].tolist() == [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [100, 101, 102],
        [101, 102, 103],
        [102, 103, 104],
    ]
    assert packed.valid[0].tolist() == [True, True, True, True, True, False]
    assert packed.times[0].tolist() == [0.25] * 3 + [0.75] * 3
    assert packed.hand_ids[0].tolist() == [0, 0, 0] * 2
    assert packed.landmark_ids[0].tolist() == [0, 1, 2] * 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("uvd", value["uvd"][:, 0]), r"\[T,2,3,3\]"),
        (
            lambda value: value.__setitem__(
                "uvd_valid_mask", np.ones((2, 2), dtype=np.bool_)
            ),
            "uvd_valid_mask",
        ),
        (
            lambda value: value["uvd_hand_ids"].__setitem__((0, 0, 0), 1),
            "uvd_hand_ids",
        ),
        (
            lambda value: value["uvd_landmark_ids"].__setitem__((0, 0, 0), 2),
            "uvd_landmark_ids",
        ),
    ],
)
def test_packer_rejects_wrong_axes_or_noncanonical_ids(mutation, message):
    layout = HandConfigurationTokenLayout(
        depth_query_count=1, uvd_points_per_hand=3, hand_count=2
    )
    example = _example()
    mutation(example)

    with pytest.raises(ValueError, match=message):
        pack_hand_lrw_targets_time_major(
            [example], layout, device=torch.device("cpu")
        )


def test_decoder_expands_one_hand_token_to_lrw_without_qwen_tokens():
    decoder = HandLRWDecoder(hidden_dim=4, landmark_count=3)
    hidden = torch.zeros(2, 12, 4, requires_grad=True)

    raw = decoder(hidden)

    assert raw.shape == (2, 36, 3)
    assert decoder.landmark_embedding.weight.shape == (3, 4)
    raw.square().mean().backward()
    assert hidden.grad is not None
    assert decoder.landmark_embedding.weight.grad is not None


def test_v5_reuses_v2_qwen_geometry_length_embedding_and_attention():
    layout = HandConfigurationTokenLayout(
        depth_query_count=8, uvd_points_per_hand=6, hand_count=2
    )
    embedding = GeometryTokenEmbedding(hidden_dim=4, layout=layout)
    geometry = embedding(batch_size=1)
    appended_mask = torch.ones(1, 5 + layout.geometry_token_count, dtype=torch.bool)

    allowed = build_geometry_full_attention_mask(appended_mask, layout)

    assert geometry.shape == (1, 28, 4)
    assert allowed.shape == (1, 1, 33, 33)
    slices = layout.sequence_slices(native_token_count=5)
    assert slices.uvd == slice(21, 33)
    assert allowed[0, 0, 21, 22]
    assert not allowed[0, 0, 21, 23]
