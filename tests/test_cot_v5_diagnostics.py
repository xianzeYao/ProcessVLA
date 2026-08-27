from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from starVLA.training.cot_test_diagnostics import (
    _pad_uvd_examples,
    compute_geometry_metrics,
    save_prediction_bundle,
)
from starVLA.training.train_starvla_cot_v1 import CotV1Trainer


def _structured_example(time_count: int = 2) -> dict:
    uvd = np.zeros((time_count, 2, 3, 3), dtype=np.float32)
    for time in range(time_count):
        for hand in range(2):
            for landmark in range(3):
                uvd[time, hand, landmark] = [
                    0.1 * time + 0.2 * hand,
                    0.05 * landmark,
                    1.0 + 0.1 * hand,
                ]
    return {
        "depth_current": np.ones((1, 2, 2), dtype=np.float32),
        "depth_future": np.ones((1, 2, 2), dtype=np.float32),
        "depth_current_valid": np.ones((1, 2, 2), dtype=np.bool_),
        "depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
        "uvd": uvd,
        "uvd_valid_mask": np.ones((time_count, 2, 3), dtype=np.bool_),
        "uvd_time": np.linspace(0.0, 1.0, time_count, dtype=np.float32),
        "uvd_frame_indices": np.arange(9, 9 + time_count, dtype=np.int64),
    }


def test_metrics_flatten_hand_and_landmark_as_six_temporal_tracks():
    example = _structured_example()
    predictions = {
        "depth_current": torch.ones(1, 1, 2, 2),
        "depth_future": torch.ones(1, 1, 2, 2),
        "uvd": torch.as_tensor(example["uvd"].reshape(1, 12, 3)),
    }

    metrics = compute_geometry_metrics(
        predictions,
        [example],
        depth_scale=1.0,
        image_size=224,
        uvd_hand_count=6,
        uvd_order="time_major",
        include_uvd_time_metrics=True,
    )

    assert metrics["uvd/time_0/valid_count"] == 6.0
    assert metrics["uvd/time_1/valid_count"] == 6.0
    assert metrics["uvd_adjacent_relative_smooth_l1"] == 0.0
    assert metrics["uvd_xy_mae_norm"] == 0.0


def test_trainer_prefers_explicit_uvd_track_count():
    model = SimpleNamespace(
        uvd_track_count=6,
        landmark_count=3,
        uvd_hand_count=2,
    )

    assert CotV1Trainer._uvd_track_count(model) == 6


def test_prediction_bundle_expands_each_frame_index_for_six_tracks(tmp_path):
    example = _structured_example()
    predictions = {
        "depth_current": torch.ones(1, 1, 2, 2),
        "depth_future": torch.ones(1, 1, 2, 2),
        "uvd": torch.as_tensor(example["uvd"].reshape(1, 12, 3)),
    }

    path = save_prediction_bundle(
        tmp_path,
        3,
        predictions,
        [example],
        uvd_hand_count=6,
        uvd_order="time_major",
    )

    with np.load(path, allow_pickle=False) as payload:
        assert payload["uvd_target"].shape == (1, 12, 3)
        assert payload["uvd_frame_indices"].tolist() == [
            [9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10]
        ]


def test_rank_four_targets_require_explicit_combined_track_count():
    example = _structured_example()

    with pytest.raises(ValueError, match="combined track count"):
        _pad_uvd_examples(
            [example],
            point_count=12,
            hand_count=2,
            order="time_major",
        )


def test_existing_rank_three_v2_packing_is_unchanged():
    example = {
        "uvd": np.asarray(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.float32,
        ),
        "uvd_valid_mask": np.asarray(
            [[True, False], [True, True]], dtype=np.bool_
        ),
        "uvd_time": np.asarray([0.25, 0.75], dtype=np.float32),
    }

    target, valid, times, endpoints = _pad_uvd_examples(
        [example], point_count=6, hand_count=2, order="time_major"
    )

    np.testing.assert_array_equal(
        target[0],
        np.asarray(
            [
                [1, 2, 3], [4, 5, 6],
                [7, 8, 9], [10, 11, 12],
                [0, 0, 0], [0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(valid[0], [True, False, True, True, False, False])
    np.testing.assert_array_equal(times[0], [0.25, 0.25, 0.75, 0.75, 0.0, 0.0])
    np.testing.assert_array_equal(endpoints[0], [[0, 2], [1, 3]])
