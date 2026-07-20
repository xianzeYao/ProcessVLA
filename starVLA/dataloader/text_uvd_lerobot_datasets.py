"""Standalone LIBERO dataloader for semantic-text and fine-UVD ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.text_uvd import (
    EmbodiedCotSidecar,
    EpisodeUVDCache,
    build_padded_uvd_trace,
    read_episode_camera_uvd,
)
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


LIBERO_TEXT_UVD_DATASETS = (
    ("libero_10_no_noops_1.0.0_lerobot", 1.0),
    ("libero_goal_no_noops_1.0.0_lerobot", 1.0),
    ("libero_object_no_noops_1.0.0_lerobot", 1.0),
    ("libero_spatial_no_noops_1.0.0_lerobot", 1.0),
)


class LiberoTextUVDDataConfig:
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


class TextUVDLeRobotSingleDataset(LeRobotSingleDataset):
    """Existing LeRobot sample plus optional decoded CoT and fixed-K UVD trace."""

    def __init__(self, *args: Any, data_cfg: Any = None, **kwargs: Any):
        self._text_uvd_cfg = data_cfg or {}
        self._condition_mode = str(self._text_uvd_cfg.get("condition_mode", "base")).lower()
        if self._condition_mode not in {"base", "text", "uvd", "both"}:
            raise ValueError(f"Unknown condition_mode={self._condition_mode}; expected base/text/uvd/both")
        self._use_text = self._condition_mode in {"text", "both"}
        self._use_uvd = self._condition_mode in {"uvd", "both"}
        self._cot_sidecar = EmbodiedCotSidecar(
            Path(args[0]) if args else Path(kwargs["dataset_path"]),
            sidecar_dir=str(self._text_uvd_cfg.get("cot_sidecar_dir", "meta/embodied_cot")),
        ) if self._use_text else None
        self._uvd_cache = EpisodeUVDCache(int(self._option("episode_cache_size", 1)))
        self._current_trajectory_id: int | None = None
        self._current_base_index: int | None = None
        super().__init__(*args, data_cfg=data_cfg, **kwargs)

    def _option(self, key: str, default: Any) -> Any:
        options = self._text_uvd_cfg.get("text_uvd", {}) if hasattr(self._text_uvd_cfg, "get") else {}
        return options.get(key, default) if hasattr(options, "get") else default

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        self._current_trajectory_id = int(trajectory_id)
        self._current_base_index = int(base_index)
        return super().get_step_data(trajectory_id, base_index)

    def _load_episode_uvd(self, trajectory_id: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._uvd_cache.get(trajectory_id)
        if cached is not None:
            return cached
        if self.curr_traj_data is None:
            raise RuntimeError("trajectory data is not loaded")
        source_width = int(self._option("source_width", 256))
        source_height = int(self._option("source_height", 256))
        payload = read_episode_camera_uvd(
            self.dataset_path,
            self.curr_traj_data,
            source_width=source_width,
            source_height=source_height,
        )
        self._uvd_cache.put(trajectory_id, payload)
        return payload

    def _uvd_sample(self) -> dict[str, np.ndarray]:
        if self._current_trajectory_id is None or self._current_base_index is None:
            raise RuntimeError("sample position is not initialized")
        eef_uvd, eef_valid = self._load_episode_uvd(self._current_trajectory_id)
        trajectory_length = len(eef_uvd)
        if trajectory_length == 0:
            raise RuntimeError(f"trajectory {self._current_trajectory_id} has no camera/EEF frames")
        base_index = min(max(self._current_base_index, 0), trajectory_length - 1)
        horizon = int(self._option("action_horizon", 8))
        future_index = min(base_index + horizon, trajectory_length - 1)
        point_count = int(self._option("uvd_num_points", int(np.floor(0.3 * horizon)) + 2))
        trace = build_padded_uvd_trace(
            eef_uvd,
            eef_valid,
            start=base_index,
            end=future_index,
            k=point_count,
            source_width=int(self._option("source_width", 256)),
            source_height=int(self._option("source_height", 256)),
            target_width=int(self._option("target_width", 224)),
            target_height=int(self._option("target_height", 224)),
            depth_scale=float(self._option("uvd_depth_scale", 1.0)),
        )
        trace["uvd_current"] = trace["uvd"][0].copy()
        trace["uvd_current_valid"] = np.asarray(trace["uvd_valid_mask"][0], dtype=np.bool_)
        return trace

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        if self._use_text:
            if self._cot_sidecar is None or self._current_trajectory_id is None or self._current_base_index is None:
                raise RuntimeError("text condition is enabled but the CoT sidecar is not initialized")
            sample["cot_text"] = self._cot_sidecar.get(self._current_trajectory_id, self._current_base_index)
        if self._use_uvd:
            sample.update(self._uvd_sample())
        return sample


def get_vla_dataset(
    data_cfg,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    **kwargs,
):
    data_config = LiberoTextUVDDataConfig()
    video_backend = data_cfg.get("video_backend", "torchvision_av")
    delete_pause_frame = bool(data_cfg.get("delete_pause_frame", False))
    mixture = []
    for dataset_name, weight in LIBERO_TEXT_UVD_DATASETS:
        mixture.append((
            TextUVDLeRobotSingleDataset(
                Path(str(data_cfg.data_root_dir)) / dataset_name,
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
