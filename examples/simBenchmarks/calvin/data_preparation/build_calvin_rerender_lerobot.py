#!/usr/bin/env python3
"""Convert CALVIN rerender H5 episodes to the StarVLA LeRobot contract.

The source H5 files are produced by ``rerender_calvin_episodes.py``. This
converter keeps the official relative actions and adds explicit per-frame
camera calibration so the Calvin CoT loader can compute/validate EEF UVD.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd

from .build_calvin_native_rgbd_lerobot import (
    FPS,
    build_segments,
    calvin_state,
    load_annotations,
    load_source_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rerender-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-split", default="training", choices=("training", "validation"))
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", type=float, default=FPS)
    return parser.parse_args()


def find_h5_files(rerender_root: Path) -> list[Path]:
    paths = sorted(set(rerender_root.glob("episode_*/episode_*.h5")) | set(rerender_root.glob("episode_*.h5")))
    if not paths:
        raise FileNotFoundError(f"no rerender H5 files under {rerender_root}")
    return paths


def read_h5_frame_ids(path: Path) -> set[int]:
    with h5py.File(path, "r") as handle:
        return {int(value) for value in np.asarray(handle["frame_id"])}

def manifest_segments(rerender_root: Path, h5_files: list[Path]) -> list[tuple[Path, dict]]:
    manifest_path = rerender_root / "summary.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("segments", [])
    by_name = {path.name: path for path in h5_files}
    selected = []
    for entry in entries:
        h5_path = by_name.get(str(entry.get("h5", "")))
        if h5_path is None:
            raise FileNotFoundError(f"summary references missing rerender H5: {entry.get('h5')}")
        selected.append(
            (
                h5_path,
                {
                    "source_start": int(entry["begin"]),
                    "source_end": int(entry["end"]),
                    "source_segment_index": int(entry.get("source_segment_index", -1)),
                    "instruction": str(entry["instruction"]),
                    "task": str(entry["task"]),
                },
            )
        )
    return selected


def prepare_output(output_root: Path, dataset_root: Path, total_segments: int, total_frames: int, fps: float, tasks: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("meta", "data", "videos", "depth", "camera"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[4]
    modality = repo_root / "examples/simBenchmarks/calvin/train_files/modality_calvin_lerobot_relative.json"
    shutil.copy2(modality, output_root / "meta" / "modality.json")
    unique_tasks = sorted(set(tasks))
    with (output_root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task_index, task in enumerate(unique_tasks):
            handle.write(json.dumps({"task_index": task_index, "task": task}, ensure_ascii=False) + "\n")
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": total_segments,
        "total_frames": total_frames,
        "total_tasks": len(unique_tasks),
        "total_videos": total_segments * 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_segments}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "depth_path": "depth/chunk-{episode_chunk:03d}/{depth_key}/episode_{episode_index:06d}.npz",
        "camera_path": "camera/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz",
        "source_dataset": str(dataset_root),
        "replay_rgbd": {
            "status": "rerendered",
            "rgb_source": "CALVIN PyBullet rerender",
            "depth_source": "CALVIN PyBullet rerender",
            "depth_unit": "meters",
            "uvd_source": "EEF robot_obs[:3] projected by rerender camera matrices",
            "fps": fps,
        },
        "features": {
            "observation.images.image": {"dtype": "video", "shape": [200, 200, 3], "names": ["height", "width", "rgb"], "info": {"video.height": 200, "video.width": 200, "video.fps": fps, "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}},
            "observation.images.wrist_image": {"dtype": "video", "shape": [84, 84, 3], "names": ["height", "width", "rgb"], "info": {"video.height": 84, "video.width": 84, "video.fps": fps, "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}},
            "observation.state": {"dtype": "float32", "shape": [8], "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]}},
            "action": {"dtype": "float32", "shape": [7], "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]}},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.depth.image_m_path": {"dtype": "string", "shape": [1], "names": None},
            "observation.depth.wrist_m_path": {"dtype": "string", "shape": [1], "names": None},
            "observation.camera.params_path": {"dtype": "string", "shape": [1], "names": None},
            "source.frame_id": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (output_root / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def convert_segment(segment: dict, segment_index: int, h5_path: Path, output_root: Path, fps: float, task_index: int) -> dict:
    with h5py.File(h5_path, "r") as handle:
        source_ids = np.asarray(handle["frame_id"], dtype=np.int64)
        begin, end = int(segment["source_start"]), int(segment["source_end"])
        positions = np.flatnonzero((source_ids >= begin) & (source_ids <= end))
        if len(positions) != end - begin + 1 or not np.array_equal(source_ids[positions], np.arange(begin, end + 1)):
            raise ValueError(f"rerender H5 {h5_path} does not fully cover {begin}:{end}")
        rgb_static = np.asarray(handle["rgb_static"][positions], dtype=np.uint8)
        rgb_gripper = np.asarray(handle["rgb_gripper"][positions], dtype=np.uint8)
        depth_static = np.asarray(handle["depth_static_m"][positions], dtype=np.float32)
        depth_gripper = np.asarray(handle["depth_gripper_m"][positions], dtype=np.float32)
        robot_obs = np.asarray(handle["robot_obs"][positions], dtype=np.float32)
        actions = np.asarray(handle["rel_actions"][positions], dtype=np.float32)
        static_k = np.asarray(handle["camera_K_static"][positions], dtype=np.float32)
        static_t = np.asarray(handle["camera_T_world_camera_static"][positions], dtype=np.float32)
        gripper_k = np.asarray(handle["camera_K_gripper"][positions], dtype=np.float32)
        gripper_t = np.asarray(handle["camera_T_world_camera_gripper"][positions], dtype=np.float32)

    chunk = segment_index // 1000
    episode = f"episode_{segment_index:06d}"
    video_dir = output_root / "videos" / f"chunk-{chunk:03d}"
    depth_dir = output_root / "depth" / f"chunk-{chunk:03d}"
    camera_dir = output_root / "camera" / f"chunk-{chunk:03d}"
    image_path = video_dir / "observation.images.image" / f"{episode}.mp4"
    wrist_path = video_dir / "observation.images.wrist_image" / f"{episode}.mp4"
    static_depth_path = depth_dir / "observation.depth.image_m" / f"{episode}.npz"
    gripper_depth_path = depth_dir / "observation.depth.wrist_m" / f"{episode}.npz"
    camera_path = camera_dir / f"{episode}.npz"
    for path in (image_path, wrist_path, static_depth_path, gripper_depth_path, camera_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(image_path, list(rgb_static), fps=fps, codec="libx264", macro_block_size=2)
    imageio.mimsave(wrist_path, list(rgb_gripper), fps=fps, codec="libx264", macro_block_size=2)
    np.savez_compressed(static_depth_path, depth_m=depth_static)
    np.savez_compressed(gripper_depth_path, depth_m=depth_gripper)
    np.savez_compressed(
        camera_path,
        agentview_K=static_k,
        agentview_T_world_camera=static_t,
        wrist_K=gripper_k,
        wrist_T_world_camera=gripper_t,
    )

    rows = []
    for local_index, source_id in enumerate(source_ids[positions]):
        rows.append({
            "observation.state": calvin_state(robot_obs[local_index]),
            "action": actions[local_index],
            "timestamp": float(local_index / fps),
            "frame_index": local_index,
            "episode_index": segment_index,
            "index": segment_index * 100000 + local_index,
            "task_index": task_index,
            "observation.depth.image_m_path": static_depth_path.relative_to(output_root).as_posix(),
            "observation.depth.wrist_m_path": gripper_depth_path.relative_to(output_root).as_posix(),
            "observation.camera.params_path": camera_path.relative_to(output_root).as_posix(),
            "source.frame_id": int(source_id),
        })
    parquet = output_root / "data" / f"chunk-{chunk:03d}" / f"{episode}.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    return {"episode_index": segment_index, "tasks": [segment["instruction"]], "task": segment["task"], "length": len(rows), "source_start": begin, "source_end": end, "rerender_h5": str(h5_path)}


def main() -> None:
    args = parse_args()
    source_dir = args.dataset_root / args.source_split
    boundaries = load_source_episodes(source_dir)
    annotations = load_annotations(source_dir)
    h5_files = find_h5_files(args.rerender_root)
    selected = manifest_segments(args.rerender_root, h5_files)
    if not selected:
        for h5_path in h5_files:
            frame_ids = read_h5_frame_ids(h5_path)
            for segment in build_segments(annotations, frame_ids, boundaries):
                selected.append((h5_path, segment))
    if args.max_segments > 0:
        selected = selected[: args.max_segments]
    if not selected:
        raise RuntimeError("no language segments in rerender H5 files")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_root}; pass --overwrite")
        shutil.rmtree(args.output_root)
    tasks = [segment["instruction"] for _, segment in selected]
    total_frames = sum(int(segment["source_end"]) - int(segment["source_start"]) + 1 for _, segment in selected)
    prepare_output(args.output_root, args.dataset_root, len(selected), total_frames, args.fps, tasks)
    task_index = {task: index for index, task in enumerate(sorted(set(tasks)))}
    mapping = []
    for output_index, (h5_path, segment) in enumerate(selected):
        mapping.append(convert_segment(segment, output_index, h5_path, args.output_root, args.fps, task_index[segment["instruction"]]))
        print(f"[rerender-convert {output_index + 1}/{len(selected)}] {segment['source_start']}:{segment['source_end']}", flush=True)
    with (args.output_root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in mapping:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_root / "meta" / "conversion_summary.json").write_text(json.dumps({"status": "success", "source_split": args.source_split, "num_episodes": len(mapping), "num_frames": total_frames, "action_key": "rel_actions", "rgbd_mode": "rerendered"}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
