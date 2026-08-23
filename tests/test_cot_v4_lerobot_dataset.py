from types import MethodType

import numpy as np
import pytest

from starVLA.dataloader.gr00t_lerobot.cot_geometry_v4 import (
    CoTV4LeRobotSingleDataset,
)


def make_dataset(*, base_index: int, terminal_valid: bool = True):
    dataset = CoTV4LeRobotSingleDataset.__new__(CoTV4LeRobotSingleDataset)
    dataset._cot_current_trajectory_id = 7
    dataset._cot_current_base_index = base_index
    dataset._cot_data_cfg = {
        "cot_geometry": {
            "action_horizon": 8,
            "local_uvd_num_points": 8,
            "coarse_uvd_num_points": 8,
            "coarse_uvd_stride": 2,
            "terminal_repeat": True,
            "image_size": 4,
            "uvd_depth_scale": 1.0,
        }
    }
    depth = np.ones((20, 4, 4), dtype=np.float32)
    eef_uvd = np.stack(
        [
            np.linspace(0.0, 3.0, 20, dtype=np.float32),
            np.ones(20, dtype=np.float32),
            np.ones(20, dtype=np.float32),
        ],
        axis=-1,
    )
    valid = np.ones(20, dtype=np.bool_)
    valid[-1] = terminal_valid
    dataset._load_episode_geometry = MethodType(
        lambda self, trajectory_id: (
            depth,
            eef_uvd,
            valid,
            np.zeros((20, 7), dtype=np.float32),
        ),
        dataset,
    )
    return dataset


def test_libero_targets_emit_forward_local_and_stride2_coarse_groups():
    targets = make_dataset(base_index=2)._geometry_targets()

    np.testing.assert_array_equal(targets["uvd_frame_indices"], np.arange(3, 11))
    np.testing.assert_array_equal(
        targets["uvd_coarse_frame_indices"], np.arange(4, 19, 2)
    )
    np.testing.assert_allclose(targets["uvd_time"], np.arange(1, 9) / 8)
    np.testing.assert_allclose(targets["uvd_coarse_time"], np.arange(1, 9) / 8)
    assert targets["uvd"].shape == (8, 3)
    assert targets["uvd_coarse"].shape == (8, 3)
    assert targets["uvd_valid_mask"].all()
    assert targets["uvd_coarse_valid_mask"].all()


@pytest.mark.parametrize("terminal_valid", [True, False])
def test_terminal_repeat_supervises_every_slot_subject_to_geometry_validity(
    terminal_valid,
):
    targets = make_dataset(
        base_index=18,
        terminal_valid=terminal_valid,
    )._geometry_targets()

    np.testing.assert_array_equal(targets["uvd_frame_indices"], np.full(8, 19))
    np.testing.assert_array_equal(
        targets["uvd_coarse_frame_indices"], np.full(8, 19)
    )
    assert bool(targets["uvd_valid_mask"].all()) is terminal_valid
    assert bool(targets["uvd_coarse_valid_mask"].all()) is terminal_valid
    np.testing.assert_array_equal(targets["uvd_endpoint_indices"], [0, 7])
    np.testing.assert_array_equal(targets["uvd_coarse_endpoint_indices"], [0, 7])
