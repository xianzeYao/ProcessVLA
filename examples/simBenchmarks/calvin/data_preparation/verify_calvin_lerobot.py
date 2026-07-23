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
    if len(episodes) != info.get("total_episodes"):
        raise ValueError("episode metadata count does not match info.json")
    if len(tasks) != info.get("total_tasks"):
        raise ValueError("task metadata count does not match info.json")
    return {
        "dataset": str(dataset),
        "frames": frame_count,
        "episodes": len(episodes),
        "tasks": len(tasks),
        "task_ids_in_data": sorted(task_ids),
        "fps": info.get("fps"),
        "replay_rgbd": info.get("replay_rgbd"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_dataset(args.dataset), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
