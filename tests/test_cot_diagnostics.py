import unittest

import numpy as np
import torch
from torch import nn

from starVLA.training.cot_test_diagnostics import (
    collect_batch_valid_ratios,
    collect_module_grad_norms,
    compute_geometry_metrics,
    install_module_grad_norm_hooks,
)


class _ToyCotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry_query = nn.Linear(2, 2, bias=False)
        self.depth_decoder = nn.Linear(2, 2, bias=False)
        self.uvd_head = nn.Linear(2, 2, bias=False)
        self.action_model = nn.Linear(2, 2, bias=False)


class _DeepSpeedLikeWrapper(nn.Module):
    """Minimal wrapper reproducing the module.* parameter-name prefix."""

    def __init__(self, module):
        super().__init__()
        self.module = module


class CotDiagnosticsTest(unittest.TestCase):
    def test_module_grad_norms_survive_model_wrapper(self):
        model = _ToyCotModel()
        wrapper = _DeepSpeedLikeWrapper(model)
        hook_state = install_module_grad_norm_hooks(wrapper)
        loss = sum(parameter.sum() for parameter in wrapper.parameters())
        loss.backward()

        metrics = collect_module_grad_norms(wrapper, hook_state=hook_state)

        self.assertGreater(metrics["grad/query_norm"], 0.0)
        self.assertGreater(metrics["grad/depth_decoder_norm"], 0.0)
        self.assertGreater(metrics["grad/uvd_head_norm"], 0.0)
        self.assertGreater(metrics["grad/action_model_norm"], 0.0)

    def test_dual_hand_time_major_geometry_metrics_preserve_temporal_tracks(self):
        uvd = np.asarray(
            [
                [[0.1, 0.2, 0.8], [0.8, 0.2, 0.9]],
                [[0.2, 0.3, 0.85], [0.7, 0.3, 0.95]],
                [[0.3, 0.4, 0.9], [0.6, 0.4, 1.0]],
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
                "depth_current_valid": np.ones((1, 2, 2), dtype=np.bool_),
                "depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
                "uvd": uvd,
                "uvd_valid_mask": np.ones((3, 2), dtype=np.bool_),
                "uvd_time": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            }
        ]

        metrics = compute_geometry_metrics(
            predictions,
            examples,
            depth_scale=1.0,
            image_size=2,
            uvd_hand_count=2,
            uvd_order="time_major",
        )

        self.assertEqual(metrics["uvd_xy_mae_norm"], 0.0)
        self.assertEqual(metrics["uvd_depth_mae_m"], 0.0)
        self.assertEqual(metrics["uvd_path_length_mae_pixel"], 0.0)
        self.assertEqual(metrics["uvd_start_xy_mae_pixel"], 0.0)
        self.assertEqual(metrics["uvd_end_xy_mae_pixel"], 0.0)
        self.assertNotIn("endpoint_geometry_mae_m", metrics)

    def test_uvd_diagnostics_split_coordinates_and_adjacent_motion(self):
        target = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.2, 1.0],
                [0.2, 0.4, 1.0],
            ],
            dtype=np.float32,
        )
        pred = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.2, 0.2, 1.1],
                [0.4, 0.4, 1.2],
            ],
            dtype=np.float32,
        )
        predictions = {
            "depth_current": torch.ones(1, 1, 2, 2),
            "depth_future": torch.ones(1, 1, 2, 2),
            "uvd": torch.from_numpy(pred[None]),
        }
        examples = [
            {
                "depth_current": np.ones((1, 2, 2), dtype=np.float32),
                "depth_future": np.ones((1, 2, 2), dtype=np.float32),
                "depth_current_valid": np.ones((1, 2, 2), dtype=np.bool_),
                "depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
                "uvd": target,
                "uvd_valid_mask": np.ones(3, dtype=np.bool_),
                "uvd_time": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            }
        ]

        metrics = compute_geometry_metrics(
            predictions,
            examples,
            depth_scale=1.0,
            image_size=101,
            uvd_hand_count=1,
            uvd_order="time_major",
        )

        self.assertAlmostEqual(metrics["uvd_u_mae_pixel"], 10.0, places=5)
        self.assertAlmostEqual(metrics["uvd_v_mae_pixel"], 0.0, places=5)
        self.assertAlmostEqual(metrics["uvd_depth_mae_m"], 0.1, places=5)
        self.assertAlmostEqual(metrics["uvd_adjacent_u_mae_pixel"], 10.0, places=5)
        self.assertAlmostEqual(metrics["uvd_adjacent_v_mae_pixel"], 0.0, places=5)
        self.assertAlmostEqual(metrics["uvd_adjacent_depth_mae_m"], 0.1, places=5)
        self.assertAlmostEqual(metrics["uvd_u_smooth_l1"], 1.0 / 120.0, places=6)
        self.assertAlmostEqual(metrics["uvd_v_smooth_l1"], 0.0, places=6)
        self.assertAlmostEqual(metrics["uvd_depth_smooth_l1"], 1.0 / 120.0, places=6)
        self.assertAlmostEqual(metrics["uvd_adjacent_relative_smooth_l1"], 1.0 / 300.0, places=6)

    def test_batch_diagnostics_report_uvd_boundary_reasons(self):
        examples = [
            {
                "uvd_valid_mask": np.asarray([True, False, True]),
                "uvd_out_of_frame_mask": np.asarray([False, True, False]),
                "uvd_boundary_clamp_mask": np.asarray([False, False, True]),
            }
        ]

        metrics = collect_batch_valid_ratios(examples)

        self.assertAlmostEqual(metrics["data/uvd_valid_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["data/uvd_out_of_frame_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["data/uvd_boundary_clamp_ratio"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
