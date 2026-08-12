from __future__ import annotations

import numpy as np
import pytest
import torch

from starVLA.model.modules.geometric_cot_v3 import (
    LandmarkGeometryTokenEmbedding,
    LandmarkGeometryTokenLayout,
    build_landmark_geometry_full_attention_mask,
    pack_landmark_uvd_targets_time_major,
)


def test_layout_and_packer_use_exact_time_major_lrw_order() -> None:
    layout = LandmarkGeometryTokenLayout(
        depth_query_count=1, uvd_time_points=2, landmark_count=3
    )
    packed = pack_landmark_uvd_targets_time_major(
        [
            {
                "uvd": np.asarray(
                    [
                        [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
                        [[40, 41, 42], [50, 51, 52], [60, 61, 62]],
                    ],
                    dtype=np.float32,
                ),
                "uvd_valid_mask": np.asarray(
                    [[True, True, True], [True, False, True]], dtype=np.bool_
                ),
                "uvd_time": np.asarray([0.25, 0.75], dtype=np.float32),
                "uvd_landmark_ids": np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
            }
        ],
        layout,
        device=torch.device("cpu"),
    )

    assert layout.uvd_token_count == 6
    assert packed.target[0].tolist() == [
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
        [40, 41, 42],
        [50, 51, 52],
        [60, 61, 62],
    ]
    assert packed.valid[0].tolist() == [True, True, True, True, False, True]
    assert packed.times[0].tolist() == [0.25, 0.25, 0.25, 0.75, 0.75, 0.75]
    assert packed.landmark_ids[0].tolist() == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize(
    ("uvd_shape", "valid_shape", "message"),
    [
        ((2, 3), (2,), r"\[T,3,3\]"),
        ((2, 2, 3), (2, 2), "exactly 3 landmarks"),
    ],
)
def test_packer_rejects_non_triangle_targets(uvd_shape, valid_shape, message) -> None:
    layout = LandmarkGeometryTokenLayout(
        depth_query_count=1, uvd_time_points=2, landmark_count=3
    )
    with pytest.raises(ValueError, match=message):
        pack_landmark_uvd_targets_time_major(
            [
                {
                    "uvd": np.zeros(uvd_shape, dtype=np.float32),
                    "uvd_valid_mask": np.ones(valid_shape, dtype=np.bool_),
                }
            ],
            layout,
            device=torch.device("cpu"),
        )


def test_landmark_embedding_distinguishes_lrw_at_equal_time() -> None:
    layout = LandmarkGeometryTokenLayout(
        depth_query_count=1, uvd_time_points=2, landmark_count=3
    )
    module = LandmarkGeometryTokenEmbedding(hidden_dim=3, layout=layout)
    with torch.no_grad():
        module.trajectory_seed.zero_()
        for parameter in module.time_embedding.parameters():
            parameter.zero_()
        module.landmark_embedding.weight.copy_(torch.eye(3))

    output = module(
        batch_size=1,
        uvd_times=torch.zeros(1, 6),
        uvd_landmark_ids=torch.tensor([[0, 1, 2, 0, 1, 2]]),
    )

    assert torch.equal(output[0, layout.geometry_uvd_slice], torch.eye(3).repeat(2, 1))


def test_attention_is_same_time_full_and_cross_time_causal() -> None:
    layout = LandmarkGeometryTokenLayout(
        depth_query_count=1, uvd_time_points=2, landmark_count=3
    )
    appended = torch.ones(1, 2 + layout.geometry_token_count, dtype=torch.bool)

    allowed = build_landmark_geometry_full_attention_mask(appended, layout)[0, 0]
    uvd = layout.sequence_slices(native_token_count=2).uvd

    assert uvd == slice(4, 10)
    assert allowed[4, 5] and allowed[4, 6]
    assert allowed[5, 4] and allowed[6, 4]
    assert not allowed[4, 7] and not allowed[6, 9]
    assert allowed[7, 4] and allowed[7, 5] and allowed[7, 6]
    assert allowed[7, 8] and allowed[7, 9]

