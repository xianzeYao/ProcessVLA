#!/usr/bin/env python3
"""Validate bilateral-wrist RoboCasa replay-RGBD LeRobot task outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "observation.left_eef_pos",
    "observation.right_eef_pos",
    "observation.left_robot_eef_pos",
    "observation.right_robot_eef_pos",
    "observation.left_pinch_pos",
    "observation.right_pinch_pos",
    "observation.left_thumb_index_pinch_pos",
    "observation.right_thumb_index_pinch_pos",
    "observation.depth.image_m_path",
    "observation.camera.params_path",
    "source.hdf5_demo_id",
}
LANGUAGE_COLUMN = "annotation.human.coarse_action"
ACTIVE_UVD_WORLD_KEYS = [
    "observation.left_thumb_index_pinch_pos",
    "observation.right_thumb_index_pinch_pos",
]
EEF_ABLATION_WORLD_KEYS = [
    "observation.left_robot_eef_pos",
    "observation.right_robot_eef_pos",
]
VIRTUAL_PINCH_WORLD_KEYS = [
    "observation.left_pinch_pos",
    "observation.right_pinch_pos",
]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_task(task_root: Path, *, check_all_episodes: bool = False) -> dict:
    info = json.loads((task_root / "meta" / "info.json").read_text(encoding="utf-8"))
    replay = info.get("replay_rgbd", {})
    if replay.get("wrist_order") != ["left", "right"]:
        raise RuntimeError(f"{task_root}: expected ordered bilateral wrist metadata")
    if replay.get("uvd_world_keys") != ACTIVE_UVD_WORLD_KEYS:
        raise RuntimeError(f"{task_root}: active UVD metadata must use thumb-index fields")
    if replay.get("eef_ablation_world_keys") != EEF_ABLATION_WORLD_KEYS:
        raise RuntimeError(f"{task_root}: EEF ablation metadata is missing or stale")
    if replay.get("virtual_pinch_world_keys") != VIRTUAL_PINCH_WORLD_KEYS:
        raise RuntimeError(f"{task_root}: virtual pinch metadata is missing or stale")
    expected_definition = "midpoint of DIAL gripper0_{left,right}_{L,R}_thumb_distal_link and index_intermediate_link body positions"
    if replay.get("uvd_point_definition") != expected_definition:
        raise RuntimeError(f"{task_root}: active UVD point definition is missing or stale")
    if replay.get("stored_image_transform") != "dial_vertical_flip_crop_pad_resize":
        raise RuntimeError(f"{task_root}: stale or unknown image transform metadata: {replay.get('stored_image_transform')!r}")
    stored_size = tuple(int(value) for value in replay.get("stored_image_size", []))
    if len(stored_size) != 2 or stored_size[0] != stored_size[1]:
        raise RuntimeError(f"{task_root}: invalid stored_image_size={stored_size}")
    if tuple(info.get("features", {}).get("observation.images.ego_view", {}).get("shape", [])) != (*stored_size, 3):
        raise RuntimeError(f"{task_root}: RGB metadata shape does not match stored_image_size={stored_size}")
    if not REQUIRED_COLUMNS <= set(info.get("features", {})):
        raise RuntimeError(f"{task_root}: info features are missing {sorted(REQUIRED_COLUMNS - set(info.get('features', {})))}")
    tasks = _jsonl(task_root / "meta" / "tasks.jsonl")
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}
    if 0 not in task_by_index or task_by_index[0] != "":
        raise RuntimeError(f"{task_root}: tasks.jsonl must reserve task_index 0 for an empty task")
    if any(index != 0 and not text.startswith("unlocked_waist: ") for index, text in task_by_index.items()):
        raise RuntimeError(f"{task_root}: non-empty task metadata is not canonical Fourier language")
    if int(info.get("total_tasks", -1)) != len(task_by_index):
        raise RuntimeError(f"{task_root}: info total_tasks disagrees with tasks.jsonl")
    episodes = {int(row["episode_index"]): int(row["length"]) for row in _jsonl(task_root / "meta" / "episodes.jsonl")}
    parquet_paths = sorted((task_root / "data").glob("chunk-*/episode_*.parquet"))
    found = {int(path.stem.rsplit("_", 1)[1]): path for path in parquet_paths}
    if set(found) != set(episodes):
        raise RuntimeError(f"{task_root}: parquet episode coverage mismatch, expected={len(episodes)} found={len(found)}")
    indexes = sorted(episodes) if check_all_episodes else sorted({min(episodes), max(episodes)})
    for index in indexes:
        frame = pd.read_parquet(found[index])
        if len(frame) != episodes[index]:
            raise RuntimeError(f"{task_root}: episode {index} has {len(frame)} frames, expected {episodes[index]}")
        if not REQUIRED_COLUMNS <= set(frame.columns):
            raise RuntimeError(f"{task_root}: episode {index} missing parquet fields")
        if LANGUAGE_COLUMN not in frame.columns:
            raise RuntimeError(f"{task_root}: episode {index} is missing {LANGUAGE_COLUMN}")
        frame_task_indices = {int(value) for value in frame[LANGUAGE_COLUMN].tolist()}
        if not frame_task_indices <= set(task_by_index):
            raise RuntimeError(f"{task_root}: episode {index} contains unknown task indices {sorted(frame_task_indices - set(task_by_index))}")
        row = frame.iloc[0]
        point_columns = (
            "observation.left_robot_eef_pos",
            "observation.right_robot_eef_pos",
            "observation.left_eef_pos",
            "observation.right_eef_pos",
            "observation.left_pinch_pos",
            "observation.right_pinch_pos",
            "observation.left_thumb_index_pinch_pos",
            "observation.right_thumb_index_pinch_pos",
        )
        bad_point_columns = [column for column in point_columns if np.asarray(row[column]).shape != (3,)]
        if bad_point_columns:
            raise RuntimeError(f"{task_root}: episode {index} point fields are not 3-vectors: {bad_point_columns}")
        depth = np.load(task_root / str(row["observation.depth.image_m_path"]))["depth_m"]
        camera = np.load(task_root / str(row["observation.camera.params_path"]))
        if depth.shape != (len(frame), *stored_size):
            raise RuntimeError(f"{task_root}: episode {index} depth shape {depth.shape} does not match {(len(frame), *stored_size)}")
        if camera["agentview_K"].shape != (len(frame), 3, 3) or camera["agentview_T_world_camera"].shape != (len(frame), 4, 4):
            raise RuntimeError(f"{task_root}: episode {index} RGB-D/camera frame count mismatch")
        if not np.isfinite(camera["agentview_K"]).all() or not np.isfinite(camera["agentview_T_world_camera"]).all():
            raise RuntimeError(f"{task_root}: episode {index} camera matrices contain non-finite values")
        video = task_root / "videos" / f"chunk-{index // 1000:03d}" / "observation.images.ego_view" / f"episode_{index:06d}.mp4"
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError(f"{task_root}: episode {index} video is missing or empty")
    return {"task": task_root.name, "episodes": len(episodes), "checked_episodes": len(indexes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing *_rerender task roots")
    parser.add_argument("--task", type=Path, action="append", help="Specific task root to validate; may be repeated.")
    parser.add_argument("--check-all-episodes", action="store_true")
    args = parser.parse_args()
    tasks = args.task if args.task else sorted(args.root.glob("gr1_unified.*_rerender"))
    if not tasks:
        raise RuntimeError(f"No rerender tasks found under {args.root}")
    reports = [validate_task(task, check_all_episodes=args.check_all_episodes) for task in tasks]
    print(json.dumps({"tasks": reports, "total_tasks": len(reports), "total_episodes": sum(row["episodes"] for row in reports)}, indent=2))


if __name__ == "__main__":
    main()
