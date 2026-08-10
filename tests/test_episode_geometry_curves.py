import unittest

import numpy as np

from examples.simBenchmarks.CoT.geometry_probe.episode_curves import (
    build_episode_curve_context,
)


def _depth(value):
    return np.full((2, 2), value, dtype=np.float32)


class EpisodeCurveContextTest(unittest.TestCase):
    def test_stitches_libero_horizon_zero_and_preserves_invalid_gap(self):
        samples = [
            {
                "uvd": np.asarray(
                    [[0.10, 0.20, 0.30], [0.11, 0.21, 0.31]], dtype=np.float32
                ),
                "uvd_valid_mask": np.asarray([True, True]),
                "depth_current": _depth(1.0),
                "depth_future": _depth(1.2),
                "metadata": {"timestamp": 2.0},
            },
            {
                "uvd": np.asarray(
                    [[0.40, 0.50, 0.60], [0.41, 0.51, 0.61]], dtype=np.float32
                ),
                "uvd_valid_mask": np.asarray([False, True]),
                "depth_current": _depth(2.0),
                "depth_future": _depth(2.2),
                "metadata": {"timestamp": 2.05},
            },
        ]
        predictions = {
            "v1_5": [
                {
                    "uvd": np.asarray(
                        [[0.12, 0.22, 0.32], [0.13, 0.23, 0.33]], dtype=np.float32
                    ),
                    "depth_current": _depth(0.9),
                    "depth_future": _depth(1.1),
                },
                {
                    "uvd": np.asarray(
                        [[0.42, 0.52, 0.62], [0.43, 0.53, 0.63]], dtype=np.float32
                    ),
                    "depth_current": _depth(1.9),
                    "depth_future": _depth(2.1),
                },
            ],
            "v2": [
                {
                    "uvd": np.asarray(
                        [[0.14, 0.24, 0.34], [0.15, 0.25, 0.35]], dtype=np.float32
                    ),
                    "depth_current": _depth(1.1),
                    "depth_future": _depth(1.3),
                },
                {
                    "uvd": np.asarray(
                        [[0.44, 0.54, 0.64], [0.45, 0.55, 0.65]], dtype=np.float32
                    ),
                    "depth_current": _depth(2.1),
                    "depth_future": _depth(2.3),
                },
            ],
        }

        context = build_episode_curve_context(
            samples, predictions, labels=("v1_5", "v2")
        )

        np.testing.assert_allclose(context.timestamps, [2.0, 2.05])
        self.assertEqual(context.gt.shape, (2, 1, 3))
        self.assertEqual(context.predictions["v1_5"].shape, (2, 1, 3))
        np.testing.assert_allclose(context.gt[0, 0], [0.10, 0.20, 0.30])
        np.testing.assert_allclose(
            context.predictions["v2"][0, 0], [0.14, 0.24, 0.34]
        )
        self.assertTrue(np.isnan(context.gt[1, 0]).all())
        self.assertTrue(np.isnan(context.predictions["v1_5"][1, 0]).all())
        self.assertTrue(np.isnan(context.predictions["v2"][1, 0]).all())
        self.assertFalse(context.valid[1, 0])

    def test_canonicalizes_robocasa_time_hand_layout(self):
        gt = np.asarray(
            [
                [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]],
                [[0.11, 0.21, 0.31], [0.41, 0.51, 0.61]],
            ],
            dtype=np.float32,
        )
        pred = gt + np.float32(0.01)
        samples = [
            {
                "uvd": gt,
                "uvd_valid_mask": np.asarray([[True, True], [True, True]]),
                "depth_current": _depth(1.0),
                "depth_future": _depth(1.5),
                "metadata": {"timestamp": 0.0},
            }
        ]
        predictions = {
            "v1": [{"uvd": pred, "depth_current": _depth(0.9), "depth_future": _depth(1.4)}],
            "v2": [{"uvd": pred + 0.01, "depth_current": _depth(1.1), "depth_future": _depth(1.6)}],
        }

        context = build_episode_curve_context(
            samples, predictions, labels=("v1", "v2")
        )

        self.assertEqual(context.gt.shape, (1, 2, 3))
        np.testing.assert_allclose(context.gt[0], gt[0])
        np.testing.assert_allclose(context.predictions["v1"][0], pred[0])
        self.assertEqual(len(context.depth_limits), 2)
        self.assertLess(context.depth_limits[0][0], 0.30)
        self.assertGreater(context.depth_limits[1][1], 0.61)
        self.assertLessEqual(context.depth_image_limits[0], 0.9)
        self.assertGreaterEqual(context.depth_image_limits[1], 1.6)
        self.assertGreater(context.depth_error_limit, 0.0)


if __name__ == "__main__":
    unittest.main()
