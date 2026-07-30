import numpy as np
import torch

from starVLA.training.cot_test_diagnostics import save_prediction_bundle


def test_step_1000_bundle_accepts_bfloat16_predictions(tmp_path):
    predictions = {
        "depth_current": torch.full((1, 2, 2), 0.25, dtype=torch.bfloat16),
        "depth_future": torch.full((1, 2, 2), 0.50, dtype=torch.bfloat16),
        "uvd": torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=torch.bfloat16),
    }
    examples = [
        {
            "depth_current": np.full((2, 2), 0.25, dtype=np.float32),
            "depth_future": np.full((2, 2), 0.50, dtype=np.float32),
            "uvd": np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
            "uvd_valid_mask": np.asarray([True, True]),
            "uvd_time": np.asarray([0.0, 1.0], dtype=np.float32),
            "uvd_frame_indices": np.asarray([0, 7], dtype=np.int64),
        }
    ]

    path = save_prediction_bundle(
        tmp_path,
        1000,
        predictions,
        examples,
    )

    assert path.name == "predictions_step_00001000.npz"
    with np.load(path) as bundle:
        assert bundle["depth_current"].dtype == np.float32
        assert bundle["depth_future"].dtype == np.float32
        assert bundle["uvd"].dtype == np.float32
        np.testing.assert_allclose(bundle["depth_current"], 0.25)
        np.testing.assert_allclose(bundle["depth_future"], 0.50)
        np.testing.assert_allclose(
            bundle["uvd"], predictions["uvd"].float().numpy(), rtol=0.0, atol=0.0
        )


def test_dual_hand_time_major_bundle_preserves_target_and_frame_order(tmp_path):
    uvd = np.asarray(
        [
            [[0.1, 0.0, 1.0], [0.9, 0.0, 1.0]],
            [[0.2, 0.0, 1.0], [0.8, 0.0, 1.0]],
            [[0.3, 0.0, 1.0], [0.7, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    predictions = {
        "depth_current": torch.ones(1, 1, 2, 2),
        "depth_future": torch.ones(1, 1, 2, 2),
        "uvd": torch.from_numpy(uvd.reshape(1, 6, 3)),
    }
    examples = [
        {
            "depth_current": np.ones((1, 2, 2), dtype=np.float32),
            "depth_future": np.ones((1, 2, 2), dtype=np.float32),
            "uvd": uvd,
            "uvd_valid_mask": np.ones((3, 2), dtype=np.bool_),
            "uvd_time": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            "uvd_frame_indices": np.asarray([10, 20, 30], dtype=np.int64),
        }
    ]

    path = save_prediction_bundle(
        tmp_path,
        1000,
        predictions,
        examples,
        uvd_hand_count=2,
        uvd_order="time_major",
    )

    with np.load(path) as bundle:
        np.testing.assert_allclose(bundle["uvd_target"], uvd.reshape(1, 6, 3))
        np.testing.assert_array_equal(
            bundle["uvd_frame_indices"],
            np.asarray([[10, 10, 20, 20, 30, 30]], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            bundle["uvd_endpoint_indices"],
            np.asarray([[[0, 4], [1, 5]]], dtype=np.int64),
        )
