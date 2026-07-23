"""Calvin-specific LeRobot factories for baseline and geometric CoT training.

Calvin stores an already-normalized relative EEF command in ``rel_actions``.
The data transform therefore converts arrays to tensors but does not apply a
second action normalization or an absolute-to-relative conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from starVLA.dataloader.gr00t_lerobot.cot_geometry import CoTLeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor


class CalvinDataConfig:
    """The seven-dimensional relative EEF action and eight-dimensional state."""

    embodiment_tag = EmbodimentTag.FRANKA
    action_dim = 7
    state_dim = 8
    video_keys = ["video.primary_image", "video.wrist_image"]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(8))

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        # ``rel_actions`` are already clipped/normalized by the official Calvin
        # environment. A tensor conversion is intentionally the only action
        # transform here.
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionToTensor(apply_to=self.action_keys),
            ]
        )


class CalvinCoTLeRobotSingleDataset(CoTLeRobotSingleDataset):
    """CoT dataset using Calvin's explicit depth/camera metadata.

    The rerender converter writes ``observation.camera.params_path`` with the
    arrays ``agentview_K`` and ``agentview_T_world_camera``. Requiring that
    field keeps UVD supervision physically grounded.
    """

    pass


def dataset_specs(data_cfg: Any, root: Path) -> tuple[list[Path], list[float]]:
    names = data_cfg.get("dataset_names")
    if names is None:
        name = data_cfg.get("dataset_name")
        names = [name] if name else []
    elif isinstance(names, str):
        names = [names]
    else:
        names = list(names)
    if not names:
        raise ValueError("Calvin data config requires dataset_name or dataset_names")
    names = [str(name) for name in names]

    weights = data_cfg.get("dataset_weights")
    if weights is None:
        weights = [1.0] * len(names)
    else:
        weights = [float(weight) for weight in weights]
        if len(weights) != len(names):
            raise ValueError(
                f"dataset_weights has {len(weights)} entries but dataset_names has {len(names)}"
            )
        if any(weight <= 0 for weight in weights):
            raise ValueError(f"dataset_weights must be positive, got {weights}")
    return [root / name for name in names], weights


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
        raise FileNotFoundError(f"Calvin LeRobot dataset does not exist: {missing}")

    config = CalvinDataConfig()
    dataset_cls = (
        CalvinCoTLeRobotSingleDataset
        if bool(data_cfg.get("enable_cot_geometry", False))
        else LeRobotSingleDataset
    )
    datasets = [
        dataset_cls(
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

