"""Small, dependency-light verifier for converted CALVIN LeRobot datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify_dataset(dataset: Path, expected_action_dim: int = 7, expected_state_dim: int = 8) -> dict:
    info = json.loads((dataset / "meta/info.json").read_text())
    tasks = (dataset / "meta/tasks.jsonl").read_text().splitlines()
    episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line]
    parquet_paths = sorted((dataset / "data").glob("*/data-*.parquet"))
    if not parquet_paths:
        parquet_paths = sorted((dataset / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet files under {dataset / 'data'}")
    frame_count = 0
    task_ids = set()
    episode_tables: dict[int, list[pd.DataFrame]] = {}
    for path in parquet_paths:
        table = pd.read_parquet(path)
        for key in ("observation.state", "action", "timestamp", "episode_index", "task_index"):
            if key not in table:
                raise ValueError(f"{path}: missing {key}")
        for key, dim in (("observation.state", expected_state_dim), ("action", expected_action_dim)):
            values = np.vstack([np.asarray(value) for value in table[key]])
            if values.shape[1] != dim or not np.isfinite(values).all():
                raise ValueError(f"{path}: invalid {key} shape/values {values.shape}")
        frame_count += len(table)
        task_ids.update(int(value) for value in table["task_index"].unique())
        episode_ids = table["episode_index"].unique()
        if len(episode_ids) != 1:
            raise ValueError(f"{path}: expected one episode_index, got {episode_ids.tolist()}")
        episode_id = int(episode_ids[0])
        episode_tables.setdefault(episode_id, []).append(table)
    if len(episodes) != info.get("total_episodes"):
        raise ValueError("episode metadata count does not match info.json")
    if len(tasks) != info.get("total_tasks"):
        raise ValueError("task metadata count does not match info.json")

    expected_frames = info.get("total_frames")
    if expected_frames is not None and frame_count != int(expected_frames):
        raise ValueError(f"frame count does not match info.json: found={frame_count} expected={expected_frames}")

    episode_metadata = {int(row["episode_index"]): row for row in episodes}
    if set(episode_metadata) != set(episode_tables):
        raise ValueError(
            "episode ids differ between metadata and parquet: "
            f"metadata={sorted(episode_metadata)} parquet={sorted(episode_tables)}"
        )

    rerendered = info.get("replay_rgbd", {}).get("status") == "rerendered"
    media_checked = 0
    for episode_id, tables in episode_tables.items():
        table = pd.concat(tables, ignore_index=True).sort_values("frame_index", kind="stable")
        frame_indices = table["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(frame_indices, np.arange(len(table), dtype=np.int64)):
            raise ValueError(f"episode {episode_id}: frame_index is not contiguous from zero")
        if not np.all(table["episode_index"].to_numpy(dtype=np.int64) == episode_id):
            raise ValueError(f"episode {episode_id}: episode_index changed inside parquet data")

        metadata = episode_metadata[episode_id]
        expected_length = metadata.get("length")
        if expected_length is not None and len(table) != int(expected_length):
            raise ValueError(
                f"episode {episode_id}: length mismatch found={len(table)} expected={expected_length}"
            )
        rerender_columns = {
            "source.frame_id",
            "observation.depth.image_m_path",
            "observation.depth.wrist_m_path",
            "observation.camera.params_path",
        }
        if rerendered and rerender_columns.issubset(table.columns):
            source_ids = table["source.frame_id"].to_numpy(dtype=np.int64)
            begin, end = int(metadata["source_start"]), int(metadata["source_end"])
            if not np.array_equal(source_ids, np.arange(begin, end + 1, dtype=np.int64)):
                raise ValueError(f"episode {episode_id}: source.frame_id does not match episode metadata")

            for column in ("observation.depth.image_m_path", "observation.depth.wrist_m_path"):
                path_values = set(str(value) for value in table[column])
                if len(path_values) != 1:
                    raise ValueError(f"episode {episode_id}: {column} points to multiple files")
                media_path = dataset / next(iter(path_values))
                if not media_path.is_file():
                    raise FileNotFoundError(media_path)
                with np.load(media_path, allow_pickle=False) as media:
                    if "depth_m" not in media:
                        raise ValueError(f"{media_path}: missing depth_m")
                    if np.asarray(media["depth_m"]).shape[0] != len(table):
                        raise ValueError(f"{media_path}: media length does not match episode {episode_id}")
                media_checked += 1

            camera_values = set(str(value) for value in table["observation.camera.params_path"])
            if len(camera_values) != 1:
                raise ValueError(f"episode {episode_id}: camera params point to multiple files")
            camera_path = dataset / next(iter(camera_values))
            if not camera_path.is_file():
                raise FileNotFoundError(camera_path)
            with np.load(camera_path, allow_pickle=False) as camera:
                for key in ("agentview_K", "agentview_T_world_camera", "wrist_K", "wrist_T_world_camera"):
                    if key not in camera or np.asarray(camera[key]).shape[0] != len(table):
                        raise ValueError(f"{camera_path}: invalid or missing {key}")
            media_checked += 1

            chunk = episode_id // int(info.get("chunks_size", 1000))
            for video_key in ("observation.images.image", "observation.images.wrist_image"):
                video_path = dataset / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_id:06d}.mp4"
                if not video_path.is_file() or video_path.stat().st_size == 0:
                    raise FileNotFoundError(video_path)
                media_checked += 1
    return {
        "dataset": str(dataset),
        "frames": frame_count,
        "episodes": len(episodes),
        "tasks": len(tasks),
        "task_ids_in_data": sorted(task_ids),
        "fps": info.get("fps"),
        "replay_rgbd": info.get("replay_rgbd"),
        "media_files_checked": media_checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_dataset(args.dataset), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
