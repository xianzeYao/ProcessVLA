"""CALVIN RGB-D replay probe without the optional tactile camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

from .probe_calvin_replay_rgbd import _first_frame, resolve_config_dir


def probe(dataset_root: Path, frame_path: Path | None = None, config_split: str = "auto") -> dict:
    dataset_root = dataset_root.resolve()
    config_dir = resolve_config_dir(dataset_root, config_split)
    config_path = config_dir / ".hydra" / "merged_config.yaml"
    if frame_path is None:
        frame_path = _first_frame(dataset_root / "training")
    frame = np.load(frame_path)
    config = OmegaConf.load(config_path)
    config.env.use_egl = False
    config.env.show_gui = False
    config.env.cameras.pop("tactile", None)
    env = hydra.utils.instantiate(config.env, show_gui=False, use_vr=False, use_scene_info=True)
    try:
        obs = env.reset(robot_obs=frame["robot_obs"], scene_obs=frame["scene_obs"])
        rgb_obs, depth_obs = env.get_camera_obs()
        result = {
            "source_frame": str(frame_path),
            "reset_robot_obs_shape": list(np.asarray(obs["robot_obs"]).shape),
            "reset_scene_obs_shape": list(np.asarray(obs["scene_obs"]).shape),
            "rgb_shapes": {key: list(np.asarray(value).shape) for key, value in rgb_obs.items()},
            "depth_shapes": {key: list(np.asarray(value).shape) for key, value in depth_obs.items()},
            "native_rgb_shapes": {
                "rgb_static": list(frame["rgb_static"].shape),
                "rgb_gripper": list(frame["rgb_gripper"].shape),
            },
            "native_depth_shapes": {
                "depth_static": list(frame["depth_static"].shape),
                "depth_gripper": list(frame["depth_gripper"].shape),
            },
        }
        result["rgb_shape_match"] = result["rgb_shapes"] == {
            "rgb_static": result["native_rgb_shapes"]["rgb_static"],
            "rgb_gripper": result["native_rgb_shapes"]["rgb_gripper"],
        }
        result["depth_shape_match"] = result["depth_shapes"] == {
            "depth_static": result["native_depth_shapes"]["depth_static"],
            "depth_gripper": result["native_depth_shapes"]["depth_gripper"],
        }
        result["ok"] = bool(result["rgb_shape_match"] and result["depth_shape_match"])
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--frame", type=Path, default=None)
    parser.add_argument("--config-split", choices=("auto", "validation", "training"), default="auto")
    args = parser.parse_args()
    print(json.dumps(probe(args.dataset_root, args.frame), indent=2))


if __name__ == "__main__":
    main()
