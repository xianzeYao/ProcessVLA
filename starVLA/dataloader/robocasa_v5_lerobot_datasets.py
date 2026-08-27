"""RoboCasa GR1 bilateral LRW hand-configuration dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from starVLA.dataloader.gr00t_lerobot.cot_geometry import (
    CoTLeRobotSingleDataset,
    _read_npz_array,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.robocasa_lerobot_datasets import (
    RoboCasaGR1DataConfig,
    dataset_specs,
)
from starVLA.robocasa_hand_lrw import (
    hand_lrw_path,
    load_hand_lrw_sidecar,
    validate_hand_lrw_dataset_metadata,
)


class RoboCasaV5CoTLeRobotSingleDataset(CoTLeRobotSingleDataset):
    """V2 sampling and depth targets backed by bilateral LRW sidecars."""

    def __init__(self, dataset_path: str | Path, *args: Any, **kwargs: Any) -> None:
        validate_hand_lrw_dataset_metadata(dataset_path)
        super().__init__(dataset_path, *args, **kwargs)
        self._hand_lrw_metadata_validated = True

    def _load_episode_geometry(
        self,
        trajectory_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not getattr(self, "_hand_lrw_metadata_validated", False):
            validate_hand_lrw_dataset_metadata(self.dataset_path)
            self._hand_lrw_metadata_validated = True
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
        frame_count = len(self.curr_traj_data)
        sidecar = load_hand_lrw_sidecar(
            hand_lrw_path(self.dataset_path, trajectory_id),
            frame_count=frame_count,
            width=int(depth.shape[-1]),
            height=int(depth.shape[-2]),
        )
        state = np.stack(
            self.curr_traj_data["observation.state"].to_numpy()
        ).astype(np.float32)
        value = (
            depth,
            sidecar.agentview_uvd_pixels,
            sidecar.agentview_in_frame,
            state,
        )
        self._cot_cache.put(trajectory_id, value)
        return value

    def _geometry_targets(self) -> dict[str, np.ndarray]:
        targets = super()._geometry_targets()
        time_count = int(targets["uvd"].shape[0])
        targets["uvd_hand_ids"] = np.broadcast_to(
            np.asarray([[[0, 0, 0], [1, 1, 1]]], dtype=np.int64),
            (time_count, 2, 3),
        ).copy()
        targets["uvd_landmark_ids"] = np.broadcast_to(
            np.asarray([[[0, 1, 2], [0, 1, 2]]], dtype=np.int64),
            (time_count, 2, 3),
        ).copy()
        return targets


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
        RoboCasaV5CoTLeRobotSingleDataset(
            path,
            modality_configs=config.modality_config(),
            transforms=config.transform(),
            embodiment_tag=config.embodiment_tag,
            video_backend=data_cfg.get("video_backend", "torchvision_av"),
            delete_pause_frame=bool(data_cfg.get("delete_pause_frame", False)),
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


__all__ = [
    "RoboCasaV5CoTLeRobotSingleDataset",
    "collate_fn",
    "get_vla_dataset",
]
