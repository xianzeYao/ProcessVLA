"""RoboCasa GR1 forward coarse-to-local V4 dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.modelExtensions.CoT.scripts.robocasa_rerender_geometry import (
    dial_content_region_mask,
)
from starVLA.dataloader.gr00t_lerobot.cot_geometry import (
    _read_npz_array,
    project_eef_to_agentview_uvd,
)
from starVLA.dataloader.gr00t_lerobot.cot_geometry_v4 import (
    CoTV4LeRobotSingleDataset,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.robocasa_eef_fields import (
    select_robocasa_uvd_world_columns,
)
from starVLA.dataloader.robocasa_lerobot_datasets import (
    RoboCasaGR1DataConfig,
    dataset_specs,
)


class RoboCasaV4CoTLeRobotSingleDataset(CoTV4LeRobotSingleDataset):
    """V4 geometry variant using bilateral thumb-index pinch points."""

    def _load_episode_geometry(
        self,
        trajectory_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cot_cache.get(trajectory_id)
        if cached is not None:
            return cached
        if self.curr_traj_data is None:
            raise RuntimeError("trajectory data is not loaded")
        row0 = self.curr_traj_data.iloc[0]
        depth = _read_npz_array(
            self.dataset_path / str(row0["observation.depth.image_m_path"]),
            "depth_m",
        )
        camera = self.dataset_path / str(
            row0["observation.camera.params_path"]
        )
        camera_k = _read_npz_array(camera, "agentview_K").astype(np.float32)
        camera_pose = _read_npz_array(
            camera, "agentview_T_world_camera"
        ).astype(np.float32)
        left_key, right_key = select_robocasa_uvd_world_columns(
            self.curr_traj_data.columns
        )
        left = np.stack(self.curr_traj_data[left_key].to_numpy()).astype(
            np.float32
        )
        right = np.stack(self.curr_traj_data[right_key].to_numpy()).astype(
            np.float32
        )
        world = np.stack([left, right], axis=1)
        uvd, valid = project_eef_to_agentview_uvd(
            world,
            camera_k,
            camera_pose,
            width=int(depth.shape[-1]),
            height=int(depth.shape[-2]),
        )
        valid = dial_content_region_mask(
            uvd,
            valid,
            output_size=int(depth.shape[-1]),
        )
        state = np.stack(
            self.curr_traj_data["observation.state"].to_numpy()
        ).astype(np.float32)
        value = (depth, uvd, valid, state)
        self._cot_cache.put(trajectory_id, value)
        return value


def get_vla_dataset(
    data_cfg: Any,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    **kwargs: Any,
):
    root = Path(str(data_cfg.data_root_dir))
    paths, weights = dataset_specs(data_cfg, root)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"RoboCasa dataset does not exist: {missing}")
    config = RoboCasaGR1DataConfig()
    datasets = [
        RoboCasaV4CoTLeRobotSingleDataset(
            path,
            modality_configs=config.modality_config(),
            transforms=config.transform(),
            embodiment_tag=config.embodiment_tag,
            video_backend=data_cfg.get("video_backend", "torchvision_av"),
            delete_pause_frame=bool(
                data_cfg.get("delete_pause_frame", False)
            ),
            data_cfg=data_cfg,
        )
        for path in paths
    ]
    return LeRobotMixtureDataset(
        list(zip(datasets, weights)),
        mode=mode,
        balance_dataset_weights=bool(balance_dataset_weights),
        balance_trajectory_weights=bool(balance_trajectory_weights),
        seed=int(data_cfg.get("seed", 42)),
        data_cfg=data_cfg,
        **kwargs,
    )


def collate_fn(batch):
    return batch
