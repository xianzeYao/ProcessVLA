
import unittest
from types import MethodType

import numpy as np
import starVLA.dataloader.gr00t_lerobot.cot_geometry as cot_geometry_module

from starVLA.dataloader.gr00t_lerobot.cot_geometry import (
    CoTLeRobotSingleDataset,
    _EpisodeGeometryCache,
    project_eef_to_agentview_uvd,
    sample_real_uvd_indices,
    transform_uvd_to_model_space,
)


class CotGeometryTest(unittest.TestCase):
    def test_episode_cache_refreshes_hits_and_evicts_least_recently_used(self):
        cache = _EpisodeGeometryCache(capacity=2)
        episode_1 = ("episode-1",)
        episode_2 = ("episode-2",)
        episode_3 = ("episode-3",)

        cache.put(1, episode_1)
        cache.put(2, episode_2)
        self.assertIs(cache.get(1), episode_1)
        cache.put(3, episode_3)

        self.assertIsNone(cache.get(2))
        self.assertIs(cache.get(1), episode_1)
        self.assertIs(cache.get(3), episode_3)

    def test_zero_capacity_episode_cache_retains_nothing(self):
        cache = _EpisodeGeometryCache(capacity=0)

        cache.put(1, ("episode-1",))

        self.assertIsNone(cache.get(1))

    def test_episode_cache_preserves_stored_depth_dtype(self):
        cache = _EpisodeGeometryCache(capacity=1)
        depth = np.zeros((2, 4, 4), dtype=np.float16)
        payload = (depth, np.zeros((2, 3)), np.ones(2, dtype=np.bool_), np.zeros((2, 7)))

        cache.put(1, payload)

        self.assertEqual(cache.get(1)[0].dtype, np.float16)

    def test_project_world_eef_to_camera_uvd_uses_camera_z_depth(self):
        k = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        t_world_camera = np.eye(4, dtype=np.float32)
        xyz = np.array([[0.0, 0.0, 2.0], [0.2, -0.1, 1.0]], dtype=np.float32)

        uvd, valid = project_eef_to_agentview_uvd(xyz, k, t_world_camera, width=64, height=32)

        np.testing.assert_allclose(uvd, [[32.0, 16.0, 2.0], [52.0, 6.0, 1.0]], atol=1e-5)
        np.testing.assert_array_equal(valid, [True, True])

    def test_project_marks_points_behind_camera_or_outside_image_invalid(self):
        k = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz = np.array([[0.0, 0.0, -1.0], [2.0, 0.0, 1.0]], dtype=np.float32)

        uvd, valid = project_eef_to_agentview_uvd(xyz, k, np.eye(4, dtype=np.float32), width=64, height=32)

        self.assertFalse(valid.any())
        self.assertTrue(np.all(np.isfinite(uvd)))

    def test_projection_validity_matches_representable_pixel_centers(self):
        k = np.eye(3, dtype=np.float32)
        xyz = np.array(
            [
                [63.0, 31.0, 1.0],
                [63.5, 31.0, 1.0],
                [63.0, 31.5, 1.0],
            ],
            dtype=np.float32,
        )

        _, valid = project_eef_to_agentview_uvd(
            xyz,
            k,
            np.eye(4, dtype=np.float32),
            width=64,
            height=32,
        )

        np.testing.assert_array_equal(valid, [True, False, False])

    def test_uniform_trace_sampling_keeps_real_endpoints_and_has_no_interpolation(self):
        indices = sample_real_uvd_indices(start=10, end=18, k=4)

        np.testing.assert_array_equal(indices, [10, 13, 15, 18])
        self.assertTrue(np.issubdtype(indices.dtype, np.integer))
        self.assertEqual(len(np.unique(indices)), 4)

    def test_reverse_trace_sampling_walks_from_episode_end_and_keeps_current_frame(self):
        self.assertTrue(hasattr(cot_geometry_module, "sample_reverse_uvd_indices"))
        indices = cot_geometry_module.sample_reverse_uvd_indices(
            start=2,
            end=11,
            stride=4,
            max_points=4,
        )

        np.testing.assert_array_equal(indices, [11, 7, 3, 2])
        self.assertTrue(np.issubdtype(indices.dtype, np.integer))

    def test_reverse_trace_sampling_rejects_capacity_that_would_truncate_endpoint(self):
        self.assertTrue(hasattr(cot_geometry_module, "sample_reverse_uvd_indices"))
        with self.assertRaisesRegex(ValueError, "capacity"):
            cot_geometry_module.sample_reverse_uvd_indices(
                start=0,
                end=17,
                stride=4,
                max_points=5,
            )

    def test_geometry_targets_add_reverse_full_trace_without_changing_local_trace(self):
        dataset = CoTLeRobotSingleDataset.__new__(CoTLeRobotSingleDataset)
        dataset._cot_current_trajectory_id = 7
        dataset._cot_current_base_index = 2
        dataset._cot_data_cfg = {
            "cot_geometry": {
                "action_horizon": 3,
                "uvd_num_points": 3,
                "full_uvd_stride": 4,
                "full_uvd_num_points": 4,
                "image_size": 4,
                "uvd_depth_scale": 1.0,
            }
        }
        depth = np.ones((12, 4, 4), dtype=np.float32)
        eef_uvd = np.stack(
            [
                np.linspace(0.0, 3.0, 12, dtype=np.float32),
                np.ones(12, dtype=np.float32),
                np.ones(12, dtype=np.float32),
            ],
            axis=-1,
        )
        dataset._load_episode_geometry = MethodType(
            lambda self, trajectory_id: (
                depth,
                eef_uvd,
                np.ones(12, dtype=np.bool_),
                np.zeros((12, 7), dtype=np.float32),
            ),
            dataset,
        )

        targets = dataset._geometry_targets()

        np.testing.assert_array_equal(targets["uvd_frame_indices"], [2, 4, 5])
        np.testing.assert_array_equal(targets["uvd_full_frame_indices"], [11, 7, 3, 2])
        np.testing.assert_allclose(targets["uvd_full_time"], [1.0, 5 / 9, 1 / 9, 0.0])
        np.testing.assert_allclose(targets["uvd_full"][:, 0], [1.0, 7 / 11, 3 / 11, 2 / 11])
        np.testing.assert_array_equal(targets["uvd_full_valid_mask"], [True] * 4)
        np.testing.assert_array_equal(targets["uvd_full_endpoint_indices"], [0, 3])

    def test_geometry_targets_omit_full_trace_when_feature_is_not_configured(self):
        dataset = CoTLeRobotSingleDataset.__new__(CoTLeRobotSingleDataset)
        dataset._cot_current_trajectory_id = 7
        dataset._cot_current_base_index = 0
        dataset._cot_data_cfg = {
            "cot_geometry": {
                "action_horizon": 1,
                "uvd_num_points": 2,
                "image_size": 4,
                "uvd_depth_scale": 1.0,
            }
        }
        dataset._load_episode_geometry = MethodType(
            lambda self, trajectory_id: (
                np.ones((2, 4, 4), dtype=np.float32),
                np.ones((2, 3), dtype=np.float32),
                np.ones(2, dtype=np.bool_),
                np.zeros((2, 7), dtype=np.float32),
            ),
            dataset,
        )

        targets = dataset._geometry_targets()

        self.assertNotIn("uvd_full", targets)

    def test_uvd_coordinates_follow_depth_resize_into_zero_one_model_space(self):
        raw = np.array([[0.0, 0.0, 2.0], [255.0, 255.0, 1.0]], dtype=np.float32)

        model_uvd = transform_uvd_to_model_space(
            raw,
            source_width=256,
            source_height=256,
            target_width=224,
            target_height=224,
            depth_scale=1.0,
        )

        np.testing.assert_allclose(model_uvd, [[0.0, 0.0, 2.0], [1.0, 1.0, 1.0]], atol=1e-6)

    def test_uvd_normalization_clamps_only_the_returned_model_coordinates(self):
        raw = np.array(
            [[-1.0e-5, 255.0 + 1.0e-5, 1.0]],
            dtype=np.float32,
        )

        model_uvd, boundary_clamp = transform_uvd_to_model_space(
            raw,
            source_width=256,
            source_height=256,
            target_width=224,
            target_height=224,
            depth_scale=1.0,
            return_boundary_clamp_mask=True,
        )

        np.testing.assert_allclose(model_uvd, [[0.0, 1.0, 1.0]], atol=1e-7)
        np.testing.assert_array_equal(boundary_clamp, [True])
