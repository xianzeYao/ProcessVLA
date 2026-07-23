"""Convert a small public CALVIN LeRobot mirror into the StarVLA contract.

The public mirrors are useful when the original CALVIN raw archive is too large
for a schema/loader smoke test.  They contain real CALVIN PNG observations,
15-D robot state, relative 7-D actions, and task metadata.  This converter
keeps those values, projects state to StarVLA's 8-D Franka interface, and
writes ordinary LeRobot video/parquet metadata.  It intentionally does not
invent depth or camera calibration; the CoT geometry path must use a
simulator-rerendered dataset instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from .calvin_schema import calvin_state


FPS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", type=float, default=FPS)
    return parser.parse_args()


def read_tasks(source_root: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    with (source_root / "meta" / "tasks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[int(row["task_index"])] = str(row["task"])
    return result


def decode_image(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("bytes", value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.item()
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"expected encoded image bytes, got {type(value).__name__}")
    return np.asarray(imageio.imread(BytesIO(bytes(value))), dtype=np.uint8)


def feature_info(fps: float) -> dict[str, Any]:
    video = lambda h, w: {
        "dtype": "video",
        "shape": [h, w, 3],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.height": h,
            "video.width": w,
            "video.fps": fps,
            "video.channels": 3,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }
    return {
        "observation.images.image": video(200, 200),
        "observation.images.wrist_image": video(84, 84),
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]},
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def main() -> None:
    args = parse_args()
    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive for a probe")
    source_root = args.source_root
    parquet_paths = sorted((source_root / "data").glob("chunk-*/*.parquet"))[: args.max_episodes]
    if len(parquet_paths) < args.max_episodes:
        raise FileNotFoundError(f"requested {args.max_episodes} episodes, found {len(parquet_paths)}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_root}; pass --overwrite")
        shutil.rmtree(args.output_root)

    source_tasks = read_tasks(source_root)
    source_info = json.loads((source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    selected_tasks = sorted({int(task) for path in parquet_paths for task in pd.read_parquet(path)["task_index"].unique()})
    task_map = {old: new for new, old in enumerate(selected_tasks)}
    tasks = {new: source_tasks[old] for old, new in task_map.items()}

    for name in ("meta", "data", "videos"):
        (args.output_root / name).mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[4]
    shutil.copy2(
        repo_root / "examples/simBenchmarks/calvin/train_files/modality_calvin_lerobot_relative.json",
        args.output_root / "meta" / "modality.json",
    )
    with (args.output_root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task_index, task in tasks.items():
            handle.write(json.dumps({"task_index": task_index, "task": task}, ensure_ascii=False) + "\n")

    episodes = []
    total_frames = 0
    for episode_index, parquet_path in enumerate(parquet_paths):
        table = pd.read_parquet(parquet_path)
        top = [decode_image(value) for value in table["observation.images.top"]]
        wrist = [decode_image(value) for value in table["observation.images.wrist"]]
        if top[0].shape[:2] != (200, 200) or wrist[0].shape[:2] != (84, 84):
            raise ValueError(f"unexpected public mirror image shapes: {top[0].shape}, {wrist[0].shape}")
        chunk = episode_index // 1000
        episode = f"episode_{episode_index:06d}"
        top_path = args.output_root / "videos" / f"chunk-{chunk:03d}" / "observation.images.image" / f"{episode}.mp4"
        wrist_path = args.output_root / "videos" / f"chunk-{chunk:03d}" / "observation.images.wrist_image" / f"{episode}.mp4"
        top_path.parent.mkdir(parents=True, exist_ok=True)
        wrist_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(top_path, top, fps=args.fps, codec="libx264", macro_block_size=2)
        imageio.mimsave(wrist_path, wrist, fps=args.fps, codec="libx264", macro_block_size=2)

        rows = []
        for local_index, row in table.iterrows():
            old_task = int(row["task_index"])
            rows.append(
                {
                    "observation.state": calvin_state(np.asarray(row["observation.state"], dtype=np.float32)),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "timestamp": float(local_index / args.fps),
                    "frame_index": int(local_index),
                    "episode_index": episode_index,
                    "index": episode_index * 100000 + int(local_index),
                    "task_index": task_map[old_task],
                }
            )
        out_parquet = args.output_root / "data" / f"chunk-{chunk:03d}" / f"{episode}.parquet"
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_parquet, index=False)
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [tasks[task_map[int(table["task_index"].iloc[0])]]],
                "length": len(rows),
                "source_episode": parquet_path.name,
            }
        )
        total_frames += len(rows)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes) * 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "source_dataset": str(source_root),
        "source_info": source_info,
        "replay_rgbd": {
            "status": "public_lerobot_rgb_only",
            "rgb_source": "public CALVIN LeRobot mirror",
            "depth_source": None,
            "note": "Use the simulator-rerendered CoT dataset for depth/UVD supervision.",
            "fps": args.fps,
        },
        "features": feature_info(args.fps),
    }
    (args.output_root / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    with (args.output_root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    (args.output_root / "meta" / "conversion_summary.json").write_text(
        json.dumps(
            {
                "status": "success",
                "source": str(source_root),
                "num_episodes": len(episodes),
                "num_frames": total_frames,
                "action_source": "public mirror action (official Calvin relative action contract)",
                "rgbd_mode": "rgb_only",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output": str(args.output_root), "episodes": len(episodes), "frames": total_frames}, indent=2))


if __name__ == "__main__":
    main()
