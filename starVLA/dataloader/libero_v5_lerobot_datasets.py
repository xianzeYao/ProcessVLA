"""LIBERO single-hand `[L,R,W]` dataset adapter for QwenGR00TCoTV5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from starVLA.dataloader.cot_v3_lerobot_datasets import (
    LIBERO_COT_V3_DATASETS,
    LiberoCoTV3DataConfig,
    LiberoGripperTriangleCoTLeRobotSingleDataset,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset


class LiberoV5CoTLeRobotSingleDataset(
    LiberoGripperTriangleCoTLeRobotSingleDataset
):
    """Expose V3 gripper triangles with V5's explicit singleton hand axis."""

    def _geometry_targets(self) -> dict[str, np.ndarray]:
        targets = super()._geometry_targets()
        uvd = np.asarray(targets["uvd"])
        if uvd.ndim != 3 or uvd.shape[1:] != (3, 3):
            raise ValueError(
                "LIBERO gripper targets must have shape [T,3,3], "
                f"got {uvd.shape}"
            )
        time_count = int(uvd.shape[0])
        for key in (
            "uvd",
            "uvd_valid_mask",
            "uvd_out_of_frame_mask",
            "uvd_boundary_clamp_mask",
            "uvd_landmark_ids",
        ):
            values = np.asarray(targets[key])
            expected_tail = (3, 3) if key == "uvd" else (3,)
            if values.shape != (time_count, *expected_tail):
                raise ValueError(
                    f"{key} must have shape {(time_count, *expected_tail)}, "
                    f"got {values.shape}"
                )
            targets[key] = values[:, None].copy()
        targets["uvd_hand_ids"] = np.zeros(
            (time_count, 1, 3), dtype=np.int64
        )
        return targets


def get_vla_dataset(
    data_cfg: Any,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    **kwargs: Any,
) -> LeRobotMixtureDataset:
    root = Path(str(data_cfg.data_root_dir))
    data_config = LiberoCoTV3DataConfig()
    video_backend = data_cfg.get("video_backend", "torchvision_av")
    delete_pause_frame = bool(data_cfg.get("delete_pause_frame", False))
    mixture = [
        (
            LiberoV5CoTLeRobotSingleDataset(
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
        for dataset_name, weight in LIBERO_COT_V3_DATASETS
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


def collate_fn(batch: list[Any]) -> list[Any]:
    return batch


__all__ = [
    "LiberoV5CoTLeRobotSingleDataset",
    "collate_fn",
    "get_vla_dataset",
]
