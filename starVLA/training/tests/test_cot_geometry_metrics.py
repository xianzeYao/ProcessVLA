import numpy as np
import torch

from starVLA.training.cot_test_diagnostics import compute_geometry_metrics


def test_geometry_metrics_cover_depth_endpoints_and_path_length():
    predictions = {
        "depth_current": torch.full((1, 1, 2, 2), 0.50),
        "depth_future": torch.full((1, 1, 2, 2), 0.75),
        "uvd": torch.tensor([[[0.1, 0.2, 0.5], [0.7, 0.8, 0.75]]]),
    }
    examples = [
        {
            "depth_current": np.full((1, 2, 2), 0.50, dtype=np.float32),
            "depth_future": np.full((1, 2, 2), 0.75, dtype=np.float32),
            "depth_current_valid": np.ones((1, 2, 2), dtype=np.bool_),
            "depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
            "uvd": np.asarray([[0.1, 0.2, 0.5], [0.7, 0.8, 0.75]], dtype=np.float32),
            "uvd_valid_mask": np.asarray([True, True]),
            "uvd_time": np.asarray([0.0, 1.0], dtype=np.float32),
            "uvd_frame_indices": np.asarray([0, 7], dtype=np.int64),
        }
    ]

    metrics = compute_geometry_metrics(
        predictions,
        examples,
        depth_scale=1.0,
        image_size=224,
    )

    assert metrics["depth_current_mae_m"] == 0.0
    assert metrics["depth_future_mae_m"] == 0.0
    assert metrics["uvd_start_xy_mae_pixel"] == 0.0
    assert metrics["uvd_end_xy_mae_pixel"] == 0.0
    assert metrics["uvd_path_length_mae_pixel"] == 0.0
    assert metrics["pred/uvd_path_length_pixel"] > 0.0
    assert all(np.isfinite(value) for value in metrics.values())
