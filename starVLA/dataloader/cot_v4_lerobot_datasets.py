"""Dedicated LIBERO forward coarse-to-local V4 dataloader factory."""

from __future__ import annotations

from pathlib import Path

from starVLA.dataloader.cot_lerobot_datasets import (
    LIBERO_COT_DATASETS,
    LiberoCoTDataConfig,
)
from starVLA.dataloader.gr00t_lerobot.cot_geometry_v4 import (
    CoTV4LeRobotSingleDataset,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset


def get_vla_dataset(
    data_cfg,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    **kwargs,
):
    root = Path(str(data_cfg.data_root_dir))
    data_config = LiberoCoTDataConfig()
    video_backend = data_cfg.get("video_backend", "torchvision_av")
    delete_pause_frame = bool(data_cfg.get("delete_pause_frame", False))
    mixture = [
        (
            CoTV4LeRobotSingleDataset(
                dataset_path=root / dataset_name,
                modality_configs=data_config.modality_config(),
                transforms=data_config.transform(),
                embodiment_tag=data_config.embodiment_tag,
                video_backend=video_backend,
                delete_pause_frame=delete_pause_frame,
                data_cfg=data_cfg,
            ),
            weight,
        )
        for dataset_name, weight in LIBERO_COT_DATASETS
    ]
    return LeRobotMixtureDataset(
        mixture,
        mode=mode,
        balance_dataset_weights=bool(balance_dataset_weights),
        balance_trajectory_weights=bool(balance_trajectory_weights),
        seed=int(data_cfg.get("seed", 42)),
        data_cfg=data_cfg,
        **kwargs,
    )


def collate_fn(batch):
    return batch
