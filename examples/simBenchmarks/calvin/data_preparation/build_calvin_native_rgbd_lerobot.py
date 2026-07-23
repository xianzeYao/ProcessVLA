#!/usr/bin/env python3
"""Convert official per-timestep CALVIN data to StarVLA LeRobot v2.

This first converter preserves the official RGB/depth observations.  It does
not call the simulator and therefore does not claim a state-based rerender.
The separate replay probe decides whether replacing these observations with
fresh simulator renders is scientifically justified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from bisect import bisect_right
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from .calvin_schema import calvin_state, parse_calvin_split, validate_frame


FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-split", default="training", choices=("training", "validation"))
    parser.add_argument("--action-key", default="rel_actions", choices=("rel_actions", "actions"))
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", type=float, default=FPS)
    return parser.parse_args()


def load_annotations(source_dir: Path) -> dict[str, Any]:
    path = source_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing language annotations: {path}")
    value = np.load(path, allow_pickle=True).item()
    if not isinstance(value, dict):
        raise ValueError(f"language annotations must contain a dict: {path}")
    return value


def load_frame_ids(source_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(source_dir.glob("episode_*.npz")):
        try:
            frame_id = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid CALVIN frame filename: {path.name}") from exc
        if frame_id in result:
            raise ValueError(f"duplicate CALVIN frame id: {frame_id}")
        result[frame_id] = path
    if not result:
        raise FileNotFoundError(f"no episode_*.npz files under {source_dir}")
    return result


def load_source_episodes(source_dir: Path) -> np.ndarray:
    path = source_dir / "ep_start_end_ids.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing episode boundaries: {path}")
    value = np.asarray(np.load(path), dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"episode boundaries must have shape (N, 2), got {value.shape}")
    if np.any(value[:, 1] < value[:, 0]):
        raise ValueError("episode boundaries contain an inverted range")
    return value


def read_frame(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        frame = {key: np.asarray(raw[key]) for key in raw.files}
    validate_frame(frame)
    return frame


def source_episode_for_frame(frame_id: int, boundaries: np.ndarray) -> int:
    starts = boundaries[:, 0]
    candidate = bisect_right(starts.tolist(), int(frame_id)) - 1
    if candidate < 0 or int(frame_id) > int(boundaries[candidate, 1]):
        raise ValueError(f"frame {frame_id} is not covered by ep_start_end_ids.npy")
    return candidate


def build_segments(annotations: dict[str, Any], frame_ids: set[int], boundaries: np.ndarray) -> list[dict[str, Any]]:
    language = annotations.get("language", {})
    info = annotations.get("info", {})
    ranges = np.asarray(info.get("indx", []), dtype=np.int64)
    instructions = list(language.get("ann", []))
    tasks = list(language.get("task", []))
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(f"language info.indx must have shape (N, 2), got {ranges.shape}")
    if len(ranges) != len(instructions) or len(ranges) != len(tasks):
        raise ValueError("language annotations and info.indx have different lengths")
    result = []
    for segment_index, ((start, end), instruction, task) in enumerate(zip(ranges, instructions, tasks)):
        start, end = int(start), int(end)
        if start > end:
            raise ValueError(f"language segment {segment_index} has start > end")
        if start not in frame_ids or end not in frame_ids:
            continue
        source_episode = source_episode_for_frame(start, boundaries)
        if source_episode_for_frame(end, boundaries) != source_episode:
            raise ValueError(f"language segment {segment_index} crosses source episodes")
        result.append({
            "source_segment_index": segment_index,
            "source_start": start,
            "source_end": end,
            "source_episode_index": source_episode,
            "instruction": str(instruction),
            "task": str(task),
        })
    return result


def _video_info(width: int, height: int, fps: float) -> dict[str, Any]:
    return {
        "video.height": height,
        "video.width": width,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }


def prepare_output(output_root: Path, dataset_root: Path, split: tuple[str, str], tasks: list[str], total_segments: int, total_frames: int, fps: float, modality_path: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "videos").mkdir(parents=True, exist_ok=True)
    (output_root / "depth").mkdir(parents=True, exist_ok=True)
    shutil.copy2(modality_path, output_root / "meta" / "modality.json")
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
        "source_dataset": str(dataset_root),
        "source_split": {"train": split[0], "eval": split[1]},
        "replay_rgbd": {
            "status": "native_observations_pending_simulator_probe",
            "rgb_source": "official CALVIN per-timestep NPZ",
            "depth_source": "official CALVIN per-timestep NPZ",
            "depth_unit": "official CALVIN depth convention; verify before geometry training",
            "fps": fps,
        },
        "features": {
            "observation.images.image": {"dtype": "video", "shape": [200, 200, 3], "names": ["height", "width", "rgb"], "info": _video_info(200, 200, fps)},
            "observation.images.wrist_image": {"dtype": "video", "shape": [84, 84, 3], "names": ["height", "width", "rgb"], "info": _video_info(84, 84, fps)},
            "observation.state": {"dtype": "float32", "shape": [8], "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_width", "gripper_action"]}},
            "action": {"dtype": "float32", "shape": [7], "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_action"]}},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.depth.static_m_path": {"dtype": "string", "shape": [1], "names": None},
            "observation.depth.gripper_m_path": {"dtype": "string", "shape": [1], "names": None},
            "source.frame_id": {"dtype": "int64", "shape": [1], "names": None},
            "source.episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "source.segment_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (output_root / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def convert_segment(segment: dict[str, Any], segment_index: int, frame_paths: dict[int, Path], boundaries: np.ndarray, output_root: Path, action_key: str, fps: float, task_index: int) -> dict[str, Any]:
    frame_ids = list(range(int(segment["source_start"]), int(segment["source_end"]) + 1))
    frames = [read_frame(frame_paths[frame_id]) for frame_id in frame_ids]
    chunk = segment_index // 1000
    episode_name = f"episode_{segment_index:06d}"
    video_dir = output_root / "videos" / f"chunk-{chunk:03d}"
    depth_dir = output_root / "depth" / f"chunk-{chunk:03d}"
    image_path = video_dir / "observation.images.image" / f"{episode_name}.mp4"
    wrist_path = video_dir / "observation.images.wrist_image" / f"{episode_name}.mp4"
    static_depth_path = depth_dir / "observation.depth.static_m" / f"{episode_name}.npz"
    gripper_depth_path = depth_dir / "observation.depth.gripper_m" / f"{episode_name}.npz"
    for path in (image_path, wrist_path, static_depth_path, gripper_depth_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(image_path, [frame["rgb_static"] for frame in frames], fps=fps, codec="libx264", macro_block_size=2)
    imageio.mimsave(wrist_path, [frame["rgb_gripper"] for frame in frames], fps=fps, codec="libx264", macro_block_size=2)
    np.savez_compressed(static_depth_path, depth_static=np.stack([frame["depth_static"] for frame in frames]).astype(np.float32))
    np.savez_compressed(gripper_depth_path, depth_gripper=np.stack([frame["depth_gripper"] for frame in frames]).astype(np.float32))
    rows = []
    for local_index, (source_id, frame) in enumerate(zip(frame_ids, frames)):
        rows.append({
            "observation.state": calvin_state(frame["robot_obs"]),
            "action": np.asarray(frame[action_key], dtype=np.float32),
            "timestamp": float(local_index / fps),
            "frame_index": local_index,
            "episode_index": segment_index,
            "index": segment_index * 100000 + local_index,
            "task_index": task_index,
            "observation.depth.static_m_path": static_depth_path.relative_to(output_root).as_posix(),
            "observation.depth.gripper_m_path": gripper_depth_path.relative_to(output_root).as_posix(),
            "source.frame_id": source_id,
            "source.episode_index": int(segment["source_episode_index"]),
            "source.segment_index": int(segment["source_segment_index"]),
        })
    parquet_path = output_root / "data" / f"chunk-{chunk:03d}" / f"{episode_name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return {
        "episode_index": segment_index,
        "tasks": [segment["instruction"]],
        "task": segment["task"],
        "length": len(rows),
        "source_start": int(segment["source_start"]),
        "source_end": int(segment["source_end"]),
        "source_episode_index": int(segment["source_episode_index"]),
        "source_segment_index": int(segment["source_segment_index"]),
    }


def main() -> None:
    args = parse_args()
    split = parse_calvin_split(args.dataset_root.name)
    source_dir = args.dataset_root / args.source_split
    frame_paths = load_frame_ids(source_dir)
    boundaries = load_source_episodes(source_dir)
    annotations = load_annotations(source_dir)
    segments = build_segments(annotations, set(frame_paths), boundaries)
    if args.max_segments > 0:
        segments = segments[: args.max_segments]
    if not segments:
        raise RuntimeError("no language segments selected")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_root}; pass --overwrite")
        shutil.rmtree(args.output_root)
    tasks = [str(segment["instruction"]) for segment in segments]
    repo_root = Path(__file__).resolve().parents[4]
    modality_path = repo_root / "examples/simBenchmarks/calvin/train_files/modality.json"
    prepare_output(args.output_root, args.dataset_root, split, tasks, len(segments), sum(int(s["source_end"]) - int(s["source_start"]) + 1 for s in segments), args.fps, modality_path)
    task_index_by_text = {task: i for i, task in enumerate(sorted(set(tasks)))}
    mapping = []
    for output_index, segment in enumerate(segments):
        mapping.append(convert_segment(segment, output_index, frame_paths, boundaries, args.output_root, args.action_key, args.fps, task_index_by_text[str(segment["instruction"])]))
        print(f"[convert {output_index + 1}/{len(segments)}] source={segment['source_start']}:{segment['source_end']}", flush=True)
    with (args.output_root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in mapping:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "status": "success",
        "source_split": split,
        "source_dir": str(source_dir),
        "action_key": args.action_key,
        "num_episodes": len(mapping),
        "num_frames": sum(int(row["length"]) for row in mapping),
        "rgbd_mode": "native_observations_pending_simulator_probe",
    }
    (args.output_root / "meta" / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
