from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from starVLA.dataloader.cot_v3_lerobot_datasets import (
    LiberoGripperTriangleCoTLeRobotSingleDataset,
)
from starVLA.dataloader.gr00t_lerobot.cot_geometry import _EpisodeGeometryCache
from starVLA.gripper_triangle import gripper_triangle_path
from starVLA.model.framework.VLM4A.QwenGR00TCoTV3 import Qwen_GR00T_CoT_V3
from starVLA.model.modules.geometric_cot_v3 import (
    LandmarkGeometryTokenEmbedding,
    LandmarkGeometryTokenLayout,
)


def _write_two_frame_fixture(root: Path) -> pd.DataFrame:
    depth_relative = Path("depth/chunk-000/episode_000007.npz")
    depth_path = root / depth_relative
    depth_path.parent.mkdir(parents=True)
    np.savez_compressed(
        depth_path,
        depth_m=np.ones((2, 8, 8), dtype=np.float32),
    )
    frame = pd.DataFrame(
        {
            "observation.depth.image_m_path": [str(depth_relative)] * 2,
            "observation.state": [
                np.zeros(8, dtype=np.float32),
                np.ones(8, dtype=np.float32),
            ],
        }
    )
    uvd = np.asarray(
        [
            [[1, 2, 1], [3, 2, 1], [2, 4, 1]],
            [[2, 2, 1], [4, 2, 1], [3, 4, 1]],
        ],
        dtype=np.float32,
    )
    sidecar = gripper_triangle_path(root, 7)
    sidecar.parent.mkdir(parents=True)
    np.savez_compressed(
        sidecar,
        world_xyz=np.zeros((2, 3, 3), dtype=np.float32),
        agentview_uvd_pixels=uvd,
        agentview_projection_valid=np.ones((2, 3), dtype=np.bool_),
        agentview_in_frame=np.ones((2, 3), dtype=np.bool_),
    )
    return frame


def _dataset_fixture(root: Path, frame: pd.DataFrame):
    dataset = LiberoGripperTriangleCoTLeRobotSingleDataset.__new__(
        LiberoGripperTriangleCoTLeRobotSingleDataset
    )
    dataset._dataset_path = root
    dataset.curr_traj_data = frame
    dataset._cot_cache = _EpisodeGeometryCache(1)
    dataset._cot_current_trajectory_id = 7
    dataset._cot_current_base_index = 0
    dataset._cot_data_cfg = {
        "cot_geometry": {
            "action_horizon": 1,
            "uvd_num_points": 2,
            "image_size": 8,
            "uvd_depth_scale": 1.0,
        }
    }
    return dataset


def test_libero_v3_loader_packer_embedding_and_losses_share_one_contract() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = _dataset_fixture(root, _write_two_frame_fixture(root))
        sample = dataset._geometry_targets()

    expected = torch.tensor(
        [
            [1 / 7, 2 / 7, 1],
            [3 / 7, 2 / 7, 1],
            [2 / 7, 4 / 7, 1],
            [2 / 7, 2 / 7, 1],
            [4 / 7, 2 / 7, 1],
            [3 / 7, 4 / 7, 1],
        ],
        dtype=torch.float32,
    )
    layout = LandmarkGeometryTokenLayout(
        depth_query_count=1,
        uvd_time_points=2,
        landmark_count=3,
    )
    model = Qwen_GR00T_CoT_V3.__new__(Qwen_GR00T_CoT_V3)
    model.geometry_layout = layout
    model.lambda_uvd_temporal = 0.1
    model.lambda_uvd_shape = 0.0
    packed = model._prepare_uvd_targets([sample], torch.device("cpu"))

    torch.testing.assert_close(packed.target[0], expected)
    assert packed.valid[0].tolist() == [True] * 6
    assert packed.times[0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert packed.landmark_ids[0].tolist() == [0, 1, 2, 0, 1, 2]

    embedding = LandmarkGeometryTokenEmbedding(hidden_dim=4, layout=layout)
    embedded = embedding(
        batch_size=1,
        uvd_times=packed.times,
        uvd_landmark_ids=packed.landmark_ids,
    )
    assert embedded.shape == (1, layout.geometry_token_count, 4)
    assert layout.uvd_token_count == 6

    pred = packed.target.clone()
    pred[0, 4, 0] += 0.5
    losses = model._compute_uvd_losses(pred, packed)
    assert losses["shape"] > 0
    torch.testing.assert_close(
        losses["total"],
        losses["absolute"] + 0.1 * losses["temporal"],
    )
