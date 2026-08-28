from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starVLA.dataloader.gr00t_lerobot.cot_geometry import _EpisodeGeometryCache
from starVLA.gripper_triangle import gripper_triangle_path


def _write_geometry_fixture(root: Path) -> pd.DataFrame:
    depth_relative = Path("depth/chunk-000/episode_000007.npz")
    depth_path = root / depth_relative
    depth_path.parent.mkdir(parents=True)
    np.savez_compressed(
        depth_path,
        depth_m=np.ones((5, 8, 8), dtype=np.float32),
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
    uvd = np.zeros((5, 3, 3), dtype=np.float32)
    for time in range(5):
        uvd[time] = np.asarray(
            [
                [1.0 + time, 2.0, 1.0],
                [3.0 + time, 2.0, 1.0],
                [2.0 + time, 4.0, 1.0],
            ],
            dtype=np.float32,
        )
    in_frame = np.ones((5, 3), dtype=np.bool_)
    in_frame[3, 1] = False
    sidecar = gripper_triangle_path(root, 7)
    sidecar.parent.mkdir(parents=True)
    np.savez_compressed(
        sidecar,
        world_xyz=np.zeros((5, 3, 3), dtype=np.float32),
        agentview_uvd_pixels=uvd,
        agentview_projection_valid=np.ones((5, 3), dtype=np.bool_),
        agentview_in_frame=in_frame,
    )
    return frame


def test_libero_v5_targets_add_one_hand_axis_without_changing_lrw_values(tmp_path):
    try:
        module = importlib.import_module(
            "starVLA.dataloader.libero_v5_lerobot_datasets"
        )
    except ModuleNotFoundError:
        pytest.fail("LIBERO V5 dataset adapter is missing")

    dataset_type = module.LiberoV5CoTLeRobotSingleDataset
    dataset = dataset_type.__new__(dataset_type)
    dataset._dataset_path = tmp_path
    dataset.curr_traj_data = _write_geometry_fixture(tmp_path)
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

    targets = dataset._geometry_targets()

    assert targets["uvd"].shape == (3, 1, 3, 3)
    assert targets["uvd_valid_mask"].shape == (3, 1, 3)
    assert targets["uvd_out_of_frame_mask"].shape == (3, 1, 3)
    assert targets["uvd_boundary_clamp_mask"].shape == (3, 1, 3)
    assert targets["uvd_hand_ids"].shape == (3, 1, 3)
    assert targets["uvd_landmark_ids"].shape == (3, 1, 3)
    np.testing.assert_array_equal(targets["uvd_frame_indices"], [1, 3, 4])
    np.testing.assert_allclose(
        targets["uvd"][:, 0, :, 0],
        [[2 / 7, 4 / 7, 3 / 7], [4 / 7, 6 / 7, 5 / 7], [5 / 7, 1, 6 / 7]],
    )
    np.testing.assert_array_equal(targets["uvd_hand_ids"], 0)
    np.testing.assert_array_equal(
        targets["uvd_landmark_ids"][:, 0],
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    )
    assert not targets["uvd_valid_mask"][1, 0, 1]
