"""Probe official CALVIN state reset and native camera rendering.

This intentionally does not write a converted dataset. It answers the narrower
question whether the checked-out simulator can reset from one source frame and
produce the expected RGB-D camera shapes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _first_frame(training_dir: Path) -> Path:
    starts = np.load(training_dir / "ep_start_end_ids.npy")
    frame = int(starts[0, 0])
    candidate = training_dir / f"episode_{frame:07d}.npz"
    if not candidate.exists():
        matches = sorted(training_dir.glob("episode_*.npz"))
        if not matches:
            raise FileNotFoundError(f"no episode_*.npz files under {training_dir}")
        candidate = matches[0]
    return candidate


def resolve_config_dir(dataset_root: Path, config_split: str = "auto") -> Path:
    if config_split not in {"auto", "validation", "training"}:
        raise ValueError(f"config_split must be auto, validation, or training; got {config_split!r}")
    candidates = ("validation", "training") if config_split == "auto" else (config_split,)
    for split in candidates:
        config_dir = dataset_root / split
        if (config_dir / ".hydra" / "merged_config.yaml").exists():
            return config_dir
    expected = ", ".join(str(dataset_root / split / ".hydra" / "merged_config.yaml") for split in candidates)
    raise FileNotFoundError(f"missing CALVIN simulator config; checked: {expected}")


def probe(dataset_root: Path, frame_path: Path | None = None, config_split: str = "auto") -> dict:
    dataset_root = dataset_root.resolve()
    config_dir = resolve_config_dir(dataset_root, config_split)
    if frame_path is None:
        frame_path = _first_frame(dataset_root / "training")
    frame = np.load(frame_path)

    calvin_root = os.environ.get("CALVIN_ROOT")
    if calvin_root:
        sys.path.insert(0, str(Path(calvin_root) / "calvin_env"))
    from calvin_env.envs.play_table_env import get_env

    env = get_env(config_dir, show_gui=False)
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
        result["rgb_shape_match"] = (
            result["rgb_shapes"].get("rgb_static") == result["native_rgb_shapes"]["rgb_static"]
            and result["rgb_shapes"].get("rgb_gripper") == result["native_rgb_shapes"]["rgb_gripper"]
        )
        result["depth_shape_match"] = (
            result["depth_shapes"].get("depth_static") == result["native_depth_shapes"]["depth_static"]
            and result["depth_shapes"].get("depth_gripper") == result["native_depth_shapes"]["depth_gripper"]
        )
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
