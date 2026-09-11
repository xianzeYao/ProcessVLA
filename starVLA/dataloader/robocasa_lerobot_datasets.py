"""RoboCasa GR1 dataset factory, with an optional RGB-D/UVD CoT adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from examples.modelExtensions.CoT.scripts.robocasa_rerender_geometry import dial_content_region_mask

from starVLA.dataloader.gr00t_lerobot.cot_geometry import CoTLeRobotSingleDataset, _read_npz_array, project_eef_to_agentview_uvd
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionSinCosTransform, StateActionToTensor, StateActionTransform
from .robocasa_eef_fields import select_robocasa_uvd_world_columns


class RoboCasaGR1DataConfig:
    embodiment_tag = EmbodimentTag.GR1
    video_keys = ["video.ego_view"]
    # The original 44-D vector additionally includes fixed legs and neck. Keep
    # the active arms/hands/waist 29-D interface used by existing GR1 policies.
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand", "action.waist"]
    language_keys = ["annotation.human.coarse_action"]
    action_indices = list(range(16))

    def __init__(self, data_cfg: Any = None) -> None:
        data_cfg = data_cfg or {}
        cot_geometry = data_cfg.get("cot_geometry", {})
        future_alignment = cot_geometry.get("future_image_alignment", False)
        if not isinstance(future_alignment, bool):
            raise ValueError(
                "cot_geometry.future_image_alignment must be a boolean, "
                f"got {future_alignment!r}"
            )
        horizon = int(cot_geometry.get("action_horizon", 16))
        self.observation_indices = [0]
        self.video_observation_indices = (
            [0, horizon] if future_alignment else [0]
        )

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.video_observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={key: "min_max" for key in self.action_keys}),
        ])


class RoboCasaCoTLeRobotSingleDataset(CoTLeRobotSingleDataset):
    """CoT geometry variant that projects bilateral thumb-index pinch fields."""

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        enabled = self._cot_option("future_image_alignment", False)
        if not isinstance(enabled, bool):
            raise ValueError(
                "future_image_alignment must be a boolean, "
                f"got {enabled!r}"
            )
        if not enabled:
            return sample
        video_key = self.modality_keys["video"][0]
        frames = data[video_key]
        if len(frames) != 2:
            raise ValueError(
                "DA3 feature alignment expects exactly current/future RGB frames, "
                f"got {len(frames)} for {video_key}"
            )
        target_size = int(self._cot_option("image_size", 224))
        sample["future_image"] = Image.fromarray(frames[1]).convert("RGB").resize(
            (target_size, target_size)
        )
        return sample

    def _load_episode_geometry(self, trajectory_id: int):
        cached = self._cot_cache.get(trajectory_id)
        if cached is not None:
            return cached
        if self.curr_traj_data is None:
            raise RuntimeError("trajectory data is not loaded")
        row0 = self.curr_traj_data.iloc[0]
        depth = _read_npz_array(self.dataset_path / str(row0["observation.depth.image_m_path"]), "depth_m")
        camera = self.dataset_path / str(row0["observation.camera.params_path"])
        k = _read_npz_array(camera, "agentview_K").astype(np.float32)
        t = _read_npz_array(camera, "agentview_T_world_camera").astype(np.float32)
        left_key, right_key = select_robocasa_uvd_world_columns(self.curr_traj_data.columns)
        left_hand_world = np.stack(self.curr_traj_data[left_key].to_numpy()).astype(np.float32)
        right_hand_world = np.stack(self.curr_traj_data[right_key].to_numpy()).astype(np.float32)
        hand_world = np.stack([left_hand_world, right_hand_world], axis=1)
        thumb_index_uvd, thumb_index_valid = project_eef_to_agentview_uvd(
            hand_world, k, t, width=int(depth.shape[-1]), height=int(depth.shape[-2])
        )
        thumb_index_valid = dial_content_region_mask(thumb_index_uvd, thumb_index_valid, output_size=int(depth.shape[-1]))
        state = np.stack(self.curr_traj_data["observation.state"].to_numpy()).astype(np.float32)
        value = (depth, thumb_index_uvd, thumb_index_valid, state)
        self._cot_cache.put(trajectory_id, value)
        return value


def dataset_specs(data_cfg: Any, root: Path) -> tuple[list[Path], list[float]]:
    """Resolve legacy single-task and explicit ordered multi-task selections."""
    names = data_cfg.get("dataset_names")
    if names is None:
        data_mix = data_cfg.get("data_mix")
        if data_mix is not None:
            try:
                entries = DATASET_NAMED_MIXTURES[str(data_mix)]
            except KeyError as exc:
                raise ValueError(f"Unknown RoboCasa data_mix: {data_mix}") from exc
            names = [entry[0] for entry in entries]
            if data_cfg.get("dataset_weights") is None:
                data_cfg = dict(data_cfg)
                data_cfg["dataset_weights"] = [entry[1] for entry in entries]
        else:
            names = [str(data_cfg.get("dataset_name", "gr1_unified.PnPWineToCabinetClose_rerender"))]
    elif isinstance(names, str):
        names = [names]
    else:
        names = [str(name) for name in names]
    if not names:
        raise ValueError("dataset_names must not be empty")
    weights = data_cfg.get("dataset_weights")
    if weights is None:
        weights = [1.0] * len(names)
    else:
        weights = [float(weight) for weight in weights]
        if len(weights) != len(names):
            raise ValueError(f"dataset_weights has {len(weights)} entries but dataset_names has {len(names)}")
        if any(weight <= 0 for weight in weights):
            raise ValueError(f"dataset_weights must be positive, got {weights}")
    return [root / name for name in names], weights


def get_vla_dataset(data_cfg: Any, mode: str = "train", balance_dataset_weights: bool = False, balance_trajectory_weights: bool = False, **kwargs):
    root = Path(str(data_cfg.data_root_dir))
    paths, weights = dataset_specs(data_cfg, root)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"RoboCasa dataset does not exist: {missing}")
    config = RoboCasaGR1DataConfig(data_cfg)
    dataset_cls = RoboCasaCoTLeRobotSingleDataset if bool(data_cfg.get("enable_cot_geometry", False)) else LeRobotSingleDataset
    datasets = [
        dataset_cls(path, modality_configs=config.modality_config(), transforms=config.transform(), embodiment_tag=config.embodiment_tag, video_backend=data_cfg.get("video_backend", "torchvision_av"), delete_pause_frame=bool(data_cfg.get("delete_pause_frame", False)), data_cfg=data_cfg)
        for path in paths
    ]
    return LeRobotMixtureDataset(list(zip(datasets, weights)), mode=mode, balance_dataset_weights=bool(balance_dataset_weights), balance_trajectory_weights=bool(balance_trajectory_weights), seed=int(data_cfg.get("seed", 42)), data_cfg=data_cfg, **kwargs)


def collate_fn(batch):
    return batch
