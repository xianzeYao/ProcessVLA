
"""Dedicated LIBERO Depth-UVD CoT dataloader factory."""

from __future__ import annotations

from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.cot_geometry import CoTLeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.libero_image_views import select_libero_image_views


LIBERO_COT_DATASETS = (
    ("libero_object_no_noops_1.0.0_lerobot", 1.0),
    ("libero_goal_no_noops_1.0.0_lerobot", 1.0),
    ("libero_spatial_no_noops_1.0.0_lerobot", 1.0),
    ("libero_10_no_noops_1.0.0_lerobot", 1.0),
)


class LiberoCoTDataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = ["video.primary_image", "video.wrist_image"]
    state_keys = [
        "state.x", "state.y", "state.z", "state.roll",
        "state.pitch", "state.yaw", "state.pad", "state.gripper",
    ]
    action_keys = [
        "action.x", "action.y", "action.z", "action.roll",
        "action.pitch", "action.yaw", "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def __init__(self, image_views: str = "all") -> None:
        self.video_keys = select_libero_image_views(self.video_keys, image_views)

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max", "action.y": "min_max", "action.z": "min_max",
                    "action.roll": "min_max", "action.pitch": "min_max", "action.yaw": "min_max",
                },
            ),
        ])


def get_vla_dataset(
    data_cfg,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    **kwargs,
):
    root = Path(str(data_cfg.data_root_dir))
    data_config = LiberoCoTDataConfig(image_views=str(data_cfg.get("image_views", "all")))
    video_backend = data_cfg.get("video_backend", "torchvision_av")
    delete_pause_frame = bool(data_cfg.get("delete_pause_frame", False))
    mixture = []
    for dataset_name, weight in LIBERO_COT_DATASETS:
        mixture.append((
            CoTLeRobotSingleDataset(
                dataset_path=root / dataset_name,
                modality_configs=data_config.modality_config(),
                transforms=data_config.transform(),
                embodiment_tag=data_config.embodiment_tag,
                video_backend=video_backend,
                delete_pause_frame=delete_pause_frame,
                data_cfg=data_cfg,
            ),
            weight,
        ))
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
