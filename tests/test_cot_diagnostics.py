import unittest

import numpy as np
import torch
from torch import nn

from starVLA.training.cot_test_diagnostics import (
    collect_batch_valid_ratios,
    collect_module_grad_norms,
    compute_gradient_clipping_metrics,
    compute_decoder_reliance_metrics,
    compute_geometry_metrics,
    compute_token_utilization_metrics,
    install_module_grad_norm_hooks,
    resolve_post_step_gradient_norm,
)
from starVLA.training.train_starvla_cot_v1 import CotV1Trainer


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
    def test_v3_landmark_count_takes_precedence_for_diagnostic_tracks(self):
        model = type(
            "V3Model",
            (),
            {"landmark_count": 3, "uvd_hand_count": 1},
        )()

        self.assertEqual(CotV1Trainer._uvd_track_count(model), 3)

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
            include_uvd_time_metrics=True,
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
        self.assertAlmostEqual(metrics["uvd/time_0/u_mae_pixel"], 0.0, places=6)
        self.assertAlmostEqual(metrics["uvd/time_1/u_mae_pixel"], 10.0, places=6)
        self.assertAlmostEqual(metrics["uvd/time_2/u_mae_pixel"], 20.0, places=6)
        self.assertAlmostEqual(metrics["uvd/time_1/depth_mae_m"], 0.1, places=6)
        self.assertEqual(metrics["uvd/time_2/valid_ratio"], 1.0)

    def test_all_invalid_uvd_time_slot_omits_error_instead_of_reporting_zero(self):
        target = np.zeros((3, 3), dtype=np.float32)
        pred = np.ones((3, 3), dtype=np.float32)
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
                "uvd_valid_mask": np.asarray([True, False, True]),
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
            include_uvd_time_metrics=True,
        )

        self.assertEqual(metrics["uvd/time_1/valid_count"], 0.0)
        self.assertEqual(metrics["uvd/time_1/valid_ratio"], 0.0)
        self.assertNotIn("uvd/time_1/u_mae_pixel", metrics)
        self.assertNotIn("uvd/time_1/v_mae_pixel", metrics)
        self.assertNotIn("uvd/time_1/depth_mae_m", metrics)

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

    def test_token_utilization_metrics_are_finite_for_uniform_and_degenerate_tokens(self):
        tokens = torch.ones(2, 4, 3)
        weights = torch.full((2, 4), 0.25)

        metrics = compute_token_utilization_metrics(
            tokens,
            prefix="depth_current",
            attention_weights=weights,
        )

        self.assertAlmostEqual(metrics["depth_current/attention_entropy_nats"], np.log(4.0), places=6)
        self.assertAlmostEqual(metrics["depth_current/attention_entropy_normalized"], 1.0, places=6)
        self.assertAlmostEqual(metrics["depth_current/attention_max_weight"], 0.25, places=6)
        self.assertAlmostEqual(metrics["depth_current/attention_effective_token_count"], 4.0, places=6)
        self.assertAlmostEqual(metrics["depth_current/token_mean_offdiag_cosine"], 1.0, places=6)
        self.assertEqual(metrics["depth_current/token_cov_effective_rank"], 0.0)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_gradient_clipping_metrics_match_clipped_and_unclipped_cases(self):
        clipped = compute_gradient_clipping_metrics(pre_clip_norm=2.5, threshold=1.0)
        unclipped = compute_gradient_clipping_metrics(pre_clip_norm=0.5, threshold=1.0)

        self.assertEqual(clipped["train/grad_clip_triggered"], 1.0)
        self.assertAlmostEqual(clipped["train/grad_clip_scale"], 0.4)
        self.assertEqual(unclipped["train/grad_clip_triggered"], 0.0)
        self.assertEqual(unclipped["train/grad_clip_scale"], 1.0)

    def test_post_step_gradient_norm_falls_back_to_deepspeed_cached_value(self):
        class _DeepSpeedLike:
            @staticmethod
            def get_global_grad_norm():
                return 2.5

        self.assertEqual(resolve_post_step_gradient_norm(None, _DeepSpeedLike()), 2.5)
        self.assertEqual(
            resolve_post_step_gradient_norm(torch.tensor(1.25), _DeepSpeedLike()),
            1.25,
        )

    def test_decoder_reliance_metrics_compare_interventions_to_normal_and_target(self):
        predictions = {
            "depth_current": torch.ones(2, 1, 2, 2),
            "depth_future": torch.full((2, 1, 2, 2), 2.0),
            "decoder_interventions": {
                "zero": {
                    "depth_current": torch.zeros(2, 1, 2, 2),
                    "depth_future": torch.zeros(2, 1, 2, 2),
                }
            },
        }
        examples = [
            {
                "depth_current": np.ones((1, 2, 2), dtype=np.float32),
                "depth_future": np.full((1, 2, 2), 2.0, dtype=np.float32),
                "depth_current_valid": np.ones((1, 2, 2), dtype=np.bool_),
                "depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
            }
            for _ in range(2)
        ]

        metrics = compute_decoder_reliance_metrics(
            predictions,
            examples,
            depth_scale=2.0,
        )

        self.assertEqual(metrics["decoder_reliance/zero/depth_current_delta_mae_m"], 2.0)
        self.assertEqual(metrics["decoder_reliance/zero/depth_future_delta_mae_m"], 4.0)
        self.assertEqual(metrics["decoder_reliance/zero/depth_current_target_mae_m"], 2.0)
        self.assertEqual(metrics["decoder_reliance/zero/depth_future_target_mae_m"], 4.0)


if __name__ == "__main__":
    unittest.main()
