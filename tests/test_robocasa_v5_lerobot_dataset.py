from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starVLA.dataloader.gr00t_lerobot.cot_geometry import _EpisodeGeometryCache
from starVLA.dataloader.robocasa_v5_lerobot_datasets import (
    RoboCasaV5CoTLeRobotSingleDataset,
)
from starVLA.robocasa_hand_lrw import hand_lrw_path


def _write_sidecar(root, episode_id: int, frame_count: int = 5):
    world = np.zeros((frame_count, 2, 3, 3), dtype=np.float32)
    uvd = np.zeros_like(world)
    for frame in range(frame_count):
        for hand in range(2):
            for landmark in range(3):
                value = 10 * frame + 3 * hand + landmark
                world[frame, hand, landmark] = [value, hand, landmark]
                uvd[frame, hand, landmark] = [1 + hand, 2 + landmark, 0.5 + frame]
    projection_valid = np.ones((frame_count, 2, 3), dtype=np.bool_)
    in_frame = projection_valid.copy()
    in_frame[0, 1, 2] = False
    path = hand_lrw_path(root, episode_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        world_xyz=world,
        agentview_uvd_pixels=uvd,
        agentview_projection_valid=projection_valid,
        agentview_in_frame=in_frame,
    )
    return uvd, in_frame


def _uninitialized_dataset(tmp_path, *, frame_count: int = 5):
    np.savez(
        tmp_path / "depth.npz",
        depth_m=np.ones((frame_count, 8, 8), dtype=np.float32),
    )
    dataset = RoboCasaV5CoTLeRobotSingleDataset.__new__(
        RoboCasaV5CoTLeRobotSingleDataset
    )
    dataset._dataset_path = tmp_path
    dataset._cot_cache = _EpisodeGeometryCache(1)
    dataset.curr_traj_data = pd.DataFrame(
        [
            {
                "observation.depth.image_m_path": "depth.npz",
                "observation.state": np.full(29, frame, dtype=np.float32),
            }
            for frame in range(frame_count)
        ]
    )
    return dataset


def test_loader_keeps_time_hand_landmark_axes_and_dial_mask(tmp_path):
    episode_id = 7
    expected_uvd, expected_in_frame = _write_sidecar(tmp_path, episode_id)
    dataset = _uninitialized_dataset(tmp_path)

    depth, uvd, valid, state = dataset._load_episode_geometry(episode_id)

    assert depth.shape == (5, 8, 8)
    assert uvd.shape == (5, 2, 3, 3)
    assert valid.shape == (5, 2, 3)
    assert state.shape == (5, 29)
    np.testing.assert_allclose(uvd, expected_uvd)
    np.testing.assert_array_equal(valid, expected_in_frame)
    assert not valid[0, 1, 2]


def test_loader_rejects_missing_and_frame_mismatched_sidecars(tmp_path):
    dataset = _uninitialized_dataset(tmp_path, frame_count=5)
    with pytest.raises(FileNotFoundError):
        dataset._load_episode_geometry(3)

    _write_sidecar(tmp_path, 3, frame_count=4)
    with pytest.raises(ValueError, match="frame count"):
        dataset._load_episode_geometry(3)


def test_targets_use_v2_six_time_indices_and_explicit_ids(tmp_path):
    frame_count = 21
    dataset = _uninitialized_dataset(tmp_path, frame_count=frame_count)
    dataset._cot_current_trajectory_id = 9
    dataset._cot_current_base_index = 1
    dataset._cot_data_cfg = {
        "cot_geometry": {
            "action_horizon": 16,
            "uvd_num_points": 6,
            "image_size": 8,
            "uvd_depth_scale": 1.0,
        }
    }
    uvd = np.zeros((frame_count, 2, 3, 3), dtype=np.float32)
    uvd[..., 0] = np.asarray([[[1.0], [2.0]]], dtype=np.float32)
    uvd[..., 1] = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    uvd[..., 2] = 0.5
    valid = np.ones((frame_count, 2, 3), dtype=np.bool_)
    valid[4, 1, 2] = False
    dataset._load_episode_geometry = MethodType(
        lambda self, trajectory_id: (
            np.ones((frame_count, 8, 8), dtype=np.float32),
            uvd,
            valid,
            np.zeros((frame_count, 29), dtype=np.float32),
        ),
        dataset,
    )

    targets = dataset._geometry_targets()

    assert targets["uvd"].shape == (6, 2, 3, 3)
    assert targets["uvd_valid_mask"].shape == (6, 2, 3)
    np.testing.assert_array_equal(
        targets["uvd_frame_indices"], np.asarray([1, 4, 7, 11, 14, 17])
    )
    np.testing.assert_array_equal(
        targets["uvd_hand_ids"][0], [[0, 0, 0], [1, 1, 1]]
    )
    np.testing.assert_array_equal(
        targets["uvd_landmark_ids"][0], [[0, 1, 2], [0, 1, 2]]
    )
    assert not targets["uvd_valid_mask"][1, 1, 2]


def test_factory_preserves_resolved_dataset_order_and_weights(tmp_path, monkeypatch):
    from starVLA.dataloader import robocasa_v5_lerobot_datasets as module

    paths = [tmp_path / "task_b", tmp_path / "task_a"]
    for path in paths:
        path.mkdir()
    observed = []

    class FakeDataset:
        def __init__(self, path, **kwargs):
            observed.append((path, kwargs))

    class FakeMixture:
        def __init__(self, mixture, **kwargs):
            self.mixture = mixture
            self.kwargs = kwargs

    monkeypatch.setattr(module, "dataset_specs", lambda cfg, root: (paths, [0.25, 0.75]))
    monkeypatch.setattr(module, "RoboCasaV5CoTLeRobotSingleDataset", FakeDataset)
    monkeypatch.setattr(module, "LeRobotMixtureDataset", FakeMixture)
    cfg = SimpleNamespace(data_root_dir=str(tmp_path))
    cfg.get = lambda key, default=None: default

    mixture = module.get_vla_dataset(cfg)

    assert [path for path, _ in observed] == paths
    assert [weight for _, weight in mixture.mixture] == [0.25, 0.75]
