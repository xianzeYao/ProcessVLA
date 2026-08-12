from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.cot_geometry import _EpisodeGeometryCache
from starVLA.gripper_triangle import gripper_triangle_path


def write_geometry_fixture(root: Path, *, sidecar_frames: int = 5) -> pd.DataFrame:
    depth_relative = Path("depth/chunk-000/episode_000007.npz")
    depth_path = root / depth_relative
    depth_path.parent.mkdir(parents=True)
    np.savez_compressed(
        depth_path, depth_m=np.ones((5, 8, 8), dtype=np.float32)
    )
    frame = pd.DataFrame(
        {
            "observation.depth.image_m_path": [str(depth_relative)] * 5,
            "observation.state": [
                np.asarray([index, 1, 2, 3, 4, 5, 0.1, 0.2], dtype=np.float32)
                for index in range(5)
            ],
        }
    )
    world = np.zeros((sidecar_frames, 3, 3), dtype=np.float32)
    uvd = np.zeros((sidecar_frames, 3, 3), dtype=np.float32)
    for time in range(sidecar_frames):
        world[time] = np.asarray(
            [[-0.02, 0.0, 1.0], [0.02, 0.0, 1.0], [0.0, 0.05, 1.0]],
            dtype=np.float32,
        )
        uvd[time] = np.asarray(
            [
                [1.0 + time, 2.0, 1.0],
                [3.0 + time, 2.0, 1.0],
                [2.0 + time, 4.0, 1.0],
            ],
            dtype=np.float32,
        )
    projection_valid = np.ones((sidecar_frames, 3), dtype=np.bool_)
    in_frame = projection_valid.copy()
    if sidecar_frames == 5:
        uvd[3, 1, 0] = 10.0
        in_frame[3, 1] = False
    sidecar_path = gripper_triangle_path(root, 7)
    sidecar_path.parent.mkdir(parents=True)
    np.savez_compressed(
        sidecar_path,
        world_xyz=world,
        agentview_uvd_pixels=uvd,
        agentview_projection_valid=projection_valid,
        agentview_in_frame=in_frame,
    )
    return frame


def uninitialized_dataset(root: Path, frame: pd.DataFrame):
    from starVLA.dataloader.cot_v3_lerobot_datasets import (
        LiberoGripperTriangleCoTLeRobotSingleDataset,
    )

    dataset = LiberoGripperTriangleCoTLeRobotSingleDataset.__new__(
        LiberoGripperTriangleCoTLeRobotSingleDataset
    )
    dataset._dataset_path = root
    dataset.curr_traj_data = frame
    dataset._cot_cache = _EpisodeGeometryCache(1)
    dataset._cot_current_trajectory_id = 7
    dataset._cot_current_base_index = 1
    dataset._cot_data_cfg = {
        "cot_geometry": {
            "action_horizon": 3,
            "uvd_num_points": 3,
            "image_size": 8,
            "uvd_depth_scale": 1.0,
        }
    }
    return dataset


class LiberoGripperTriangleDatasetTest(unittest.TestCase):
    def test_loads_sidecar_geometry_and_preserves_episode_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = write_geometry_fixture(root)
            dataset = uninitialized_dataset(root, frame)

            depth, uvd, valid, state = dataset._load_episode_geometry(7)

        self.assertEqual(depth.shape, (5, 8, 8))
        self.assertEqual(uvd.shape, (5, 3, 3))
        self.assertEqual(valid.shape, (5, 3))
        self.assertFalse(valid[3, 1])
        np.testing.assert_array_equal(
            state[:, 0], np.asarray([0, 1, 2, 3, 4], dtype=np.float32)
        )

    def test_missing_or_frame_mismatched_sidecars_fail_loudly(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = write_geometry_fixture(root, sidecar_frames=4)
            dataset = uninitialized_dataset(root, frame)
            with self.assertRaisesRegex(ValueError, "frame count"):
                dataset._load_episode_geometry(7)

            gripper_triangle_path(root, 7).unlink()
            dataset._cot_cache = _EpisodeGeometryCache(1)
            with self.assertRaises(FileNotFoundError):
                dataset._load_episode_geometry(7)

    def test_sampled_targets_keep_three_landmarks_and_in_frame_validity(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = write_geometry_fixture(root)
            dataset = uninitialized_dataset(root, frame)

            targets = dataset._geometry_targets()

        self.assertEqual(targets["uvd"].shape, (3, 3, 3))
        self.assertEqual(targets["uvd_valid_mask"].shape, (3, 3))
        self.assertEqual(targets["uvd_landmark_ids"].shape, (3, 3))
        np.testing.assert_array_equal(
            targets["uvd_landmark_ids"], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        )
        np.testing.assert_array_equal(targets["uvd_frame_indices"], [1, 3, 4])
        np.testing.assert_allclose(targets["uvd_time"], [0.0, 2 / 3, 1.0])
        self.assertFalse(targets["uvd_valid_mask"][1, 1])
        self.assertTrue(targets["uvd_boundary_clamp_mask"][1, 1] == 0)


class CotV3DatasetFactoryTest(unittest.TestCase):
    def test_factory_builds_four_ordered_v3_datasets(self) -> None:
        import starVLA.dataloader.cot_v3_lerobot_datasets as module

        data_cfg = {
            "data_root_dir": "/datasets/libero",
            "video_backend": "torchvision_av",
            "delete_pause_frame": False,
            "seed": 7,
        }
        data_cfg = type("Cfg", (dict,), {"data_root_dir": "/datasets/libero"})(data_cfg)
        instances: list[object] = []

        def build_dataset(**kwargs):
            instance = type("Dataset", (), {"kwargs": kwargs})()
            instances.append(instance)
            return instance

        with mock.patch.object(
            module, "LiberoGripperTriangleCoTLeRobotSingleDataset", side_effect=build_dataset
        ), mock.patch.object(module, "LeRobotMixtureDataset", side_effect=lambda mixture, **kwargs: (mixture, kwargs)):
            mixture, kwargs = module.get_vla_dataset(data_cfg=data_cfg)

        self.assertEqual(len(instances), 4)
        self.assertEqual([weight for _, weight in mixture], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(
            [Path(instance.kwargs["dataset_path"]).name for instance in instances],
            [
                "libero_object_no_noops_1.0.0_lerobot",
                "libero_goal_no_noops_1.0.0_lerobot",
                "libero_spatial_no_noops_1.0.0_lerobot",
                "libero_10_no_noops_1.0.0_lerobot",
            ],
        )
        self.assertEqual(kwargs["seed"], 7)


if __name__ == "__main__":
    unittest.main()
