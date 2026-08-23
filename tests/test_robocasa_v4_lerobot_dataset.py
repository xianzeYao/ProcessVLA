from types import MethodType

import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.cot_geometry import _EpisodeGeometryCache
from starVLA.dataloader.robocasa_v4_lerobot_datasets import (
    RoboCasaV4CoTLeRobotSingleDataset,
)


def test_robocasa_v4_targets_preserve_time_major_two_hand_horizons():
    dataset = RoboCasaV4CoTLeRobotSingleDataset.__new__(
        RoboCasaV4CoTLeRobotSingleDataset
    )
    dataset._cot_current_trajectory_id = 3
    dataset._cot_current_base_index = 5
    dataset._cot_data_cfg = {
        "cot_geometry": {
            "action_horizon": 16,
            "local_uvd_num_points": 16,
            "coarse_uvd_num_points": 16,
            "coarse_uvd_stride": 2,
            "terminal_repeat": True,
            "image_size": 8,
            "uvd_depth_scale": 1.0,
        }
    }
    depth = np.ones((40, 8, 8), dtype=np.float32)
    hand_uvd = np.zeros((40, 2, 3), dtype=np.float32)
    hand_uvd[..., 0] = np.asarray([2.0, 5.0], dtype=np.float32)
    hand_uvd[..., 1:] = 1.0
    dataset._load_episode_geometry = MethodType(
        lambda self, trajectory_id: (
            depth,
            hand_uvd,
            np.ones((40, 2), dtype=np.bool_),
            np.zeros((40, 29), dtype=np.float32),
        ),
        dataset,
    )

    targets = dataset._geometry_targets()

    assert targets["uvd"].shape == (16, 2, 3)
    assert targets["uvd_coarse"].shape == (16, 2, 3)
    assert targets["uvd_valid_mask"].shape == (16, 2)
    np.testing.assert_array_equal(targets["uvd_frame_indices"], np.arange(6, 22))
    np.testing.assert_array_equal(
        targets["uvd_coarse_frame_indices"], np.arange(7, 38, 2)
    )


def test_robocasa_v4_loader_keeps_left_right_order_and_masks_dial_padding(tmp_path):
    np.savez(tmp_path / "depth.npz", depth_m=np.ones((2, 4, 4), dtype=np.float32))
    np.savez(
        tmp_path / "camera.npz",
        agentview_K=np.eye(3, dtype=np.float32),
        agentview_T_world_camera=np.eye(4, dtype=np.float32),
    )
    left = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    right = np.asarray([2.0, 0.0, 1.0], dtype=np.float32)
    dataset = RoboCasaV4CoTLeRobotSingleDataset.__new__(
        RoboCasaV4CoTLeRobotSingleDataset
    )
    dataset._dataset_path = tmp_path
    dataset._cot_cache = _EpisodeGeometryCache(1)
    dataset.curr_traj_data = pd.DataFrame(
        [
            {
                "observation.depth.image_m_path": "depth.npz",
                "observation.camera.params_path": "camera.npz",
                "observation.left_thumb_index_pinch_pos": left,
                "observation.right_thumb_index_pinch_pos": right,
                "observation.state": np.zeros(29, dtype=np.float32),
            },
            {
                "observation.depth.image_m_path": "depth.npz",
                "observation.camera.params_path": "camera.npz",
                "observation.left_thumb_index_pinch_pos": left,
                "observation.right_thumb_index_pinch_pos": right,
                "observation.state": np.zeros(29, dtype=np.float32),
            },
        ]
    )

    _, uvd, valid, _ = dataset._load_episode_geometry(11)

    np.testing.assert_allclose(uvd[:, 0], np.asarray([[1.0, 1.0, 1.0]] * 2))
    np.testing.assert_allclose(uvd[:, 1], np.asarray([[2.0, 0.0, 1.0]] * 2))
    np.testing.assert_array_equal(valid[:, 0], [True, True])
    np.testing.assert_array_equal(valid[:, 1], [False, False])
