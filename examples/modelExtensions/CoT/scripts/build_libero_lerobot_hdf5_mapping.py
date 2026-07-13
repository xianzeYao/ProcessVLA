#!/usr/bin/env python3
"""
Build a mapping from converted LeRobot LIBERO episodes back to original LIBERO HDF5 demos.

The IPEC / OpenVLA no_noops LeRobot datasets retain state/action/RGB but drop the
original MuJoCo `states`.  Original LIBERO HDF5 demos contain `data/demo_x/states`,
which can be used to render metric depth if we know which demo and frame indices
correspond to each LeRobot episode.

Matching strategy, in order:
  1. exact_full_action: same length, all 7 action dims match after gripper convention conversion.
  2. exact_contiguous_motion: LeRobot first 6 action dims match a contiguous HDF5 slice.
  3. exact_ordered_motion: LeRobot first 6 action dims match an ordered HDF5 subsequence,
     allowing skipped HDF5 frames (typical no-op filtering).
  4. best_motion_fallback: best contiguous first-6-dim match; not accepted by default.

Gripper convention observed locally:
  original HDF5 action[:, 6]: -1=open, +1=close
  LeRobot action[:, 6]:       1=open, 0=close
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

DATASET_TO_SUITE = {
    "libero_spatial_no_noops_1.0.0_lerobot": "libero_spatial",
    "libero_object_no_noops_1.0.0_lerobot": "libero_object",
    "libero_goal_no_noops_1.0.0_lerobot": "libero_goal",
    "libero_10_no_noops_1.0.0_lerobot": "libero_10",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Map LeRobot LIBERO episodes to original HDF5 demos.")
    ap.add_argument("--lerobot-root", required=True)
    ap.add_argument("--hdf5-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-episodes", type=int, default=0, help="<=0 means all episodes")
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_parquet_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    p = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if p.exists():
        return p
    matches = list((root / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet"))
    if matches:
        return matches[0]
    raise FileNotFoundError(p)


def load_lerobot_actions(root: Path, episode_index: int) -> np.ndarray:
    df = pd.read_parquet(episode_parquet_path(root, episode_index))
    return np.stack(df["action"].to_numpy()).astype(np.float64)


def normalize_hdf5_action(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float64).copy()
    out[:, 6] = (out[:, 6] < 0).astype(np.float64)
    return out


def load_hdf5_actions(hdf5_root: Path) -> dict[str, list[dict[str, Any]]]:
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hdf5_path in sorted(hdf5_root.glob("*.hdf5")):
        with h5py.File(hdf5_path, "r") as f:
            info = json.loads(f["data"].attrs["problem_info"])
            lang = info["language_instruction"]
            for demo_id in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1])):
                demo = f[f"data/{demo_id}"]
                raw = demo["actions"][()].astype(np.float64)
                by_lang[lang].append(
                    {
                        "hdf5": str(hdf5_path),
                        "hdf5_name": hdf5_path.name,
                        "demo_id": demo_id,
                        "actions_raw": raw,
                        "actions_norm": normalize_hdf5_action(raw),
                        "num_hdf5_frames": int(raw.shape[0]),
                        "states_shape": list(demo["states"].shape) if "states" in demo else None,
                    }
                )
    return by_lang


def max_abs(a: np.ndarray) -> float:
    return float(np.max(np.abs(a))) if a.size else 0.0


def mse(a: np.ndarray) -> float:
    return float(np.mean(np.square(a))) if a.size else 0.0


def exact_full_action(la: np.ndarray, candidates: list[dict[str, Any]], tol: float) -> dict[str, Any] | None:
    for c in candidates:
        ha = c["actions_norm"]
        if len(ha) != len(la):
            continue
        diff = ha - la
        if max_abs(diff) <= tol:
            return make_match(c, "exact_full_action", 0, list(range(len(la))), diff, accepted=True)
    return None


def exact_contiguous_motion(la: np.ndarray, candidates: list[dict[str, Any]], tol: float) -> dict[str, Any] | None:
    for c in candidates:
        ha = c["actions_norm"]
        if len(ha) < len(la):
            continue
        for start in range(len(ha) - len(la) + 1):
            diff = ha[start : start + len(la), :6] - la[:, :6]
            if max_abs(diff) <= tol:
                return make_match(c, "exact_contiguous_motion", start, list(range(start, start + len(la))), diff, accepted=True)
    return None


def exact_ordered_motion(la: np.ndarray, candidates: list[dict[str, Any]], tol: float) -> dict[str, Any] | None:
    for c in candidates:
        ha = c["actions_norm"]
        if len(ha) < len(la):
            continue
        indices: list[int] = []
        j = 0
        for i in range(len(ha)):
            if j < len(la) and max_abs(ha[i, :6] - la[j, :6]) <= tol:
                indices.append(i)
                j += 1
                if j == len(la):
                    diff = ha[indices, :6] - la[:, :6]
                    return make_match(c, "exact_ordered_motion", indices[0], indices, diff, accepted=True)
    return None


def best_motion_fallback(la: np.ndarray, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: tuple[float, float, dict[str, Any], int, list[int], np.ndarray] | None = None
    for c in candidates:
        ha = c["actions_norm"]
        if len(ha) < len(la):
            continue
        for start in range(len(ha) - len(la) + 1):
            diff = ha[start : start + len(la), :6] - la[:, :6]
            score = mse(diff)
            err = max_abs(diff)
            if best is None or (score, err) < (best[0], best[1]):
                best = (score, err, c, start, list(range(start, start + len(la))), diff)
    if best is None:
        return None
    score, err, c, start, indices, diff = best
    return make_match(c, "best_motion_fallback", start, indices, diff, accepted=False)


def make_match(
    c: dict[str, Any],
    match_type: str,
    hdf5_start: int,
    hdf5_indices: list[int],
    diff: np.ndarray,
    accepted: bool,
) -> dict[str, Any]:
    return {
        "match_type": match_type,
        "accepted": bool(accepted),
        "hdf5": c["hdf5"],
        "hdf5_name": c["hdf5_name"],
        "demo_id": c["demo_id"],
        "hdf5_start": int(hdf5_start),
        "hdf5_end_exclusive": int(hdf5_indices[-1] + 1) if hdf5_indices else int(hdf5_start),
        "hdf5_indices": [int(x) for x in hdf5_indices],
        "uses_contiguous_states": bool(hdf5_indices == list(range(hdf5_indices[0], hdf5_indices[-1] + 1))) if hdf5_indices else False,
        "num_hdf5_frames": int(c["num_hdf5_frames"]),
        "states_shape": c["states_shape"],
        "motion_mse": mse(diff),
        "motion_max_abs_error": max_abs(diff),
    }


def match_episode(la: np.ndarray, candidates: list[dict[str, Any]], tol: float) -> dict[str, Any]:
    for fn in (exact_full_action, exact_contiguous_motion, exact_ordered_motion):
        m = fn(la, candidates, tol)
        if m is not None:
            return m
    m = best_motion_fallback(la, candidates)
    if m is not None:
        return m
    return {"match_type": "unmatched", "accepted": False}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    lerobot_root = Path(args.lerobot_root)
    hdf5_root = Path(args.hdf5_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name or lerobot_root.name
    suite = DATASET_TO_SUITE.get(dataset_name, hdf5_root.name)
    episodes = read_jsonl(lerobot_root / "meta" / "episodes.jsonl")
    if args.max_episodes and args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    hdf5_by_lang = load_hdf5_actions(hdf5_root)

    rows: list[dict[str, Any]] = []
    for ep in episodes:
        episode_index = int(ep["episode_index"])
        task = ep["tasks"][0]
        la = load_lerobot_actions(lerobot_root, episode_index)
        candidates = hdf5_by_lang.get(task, [])
        match = match_episode(la, candidates, args.tol) if candidates else {"match_type": "no_task_candidates", "accepted": False}
        row = {
            "suite": suite,
            "dataset_name": dataset_name,
            "episode_index": episode_index,
            "task": task,
            "lerobot_length": int(len(la)),
            "tol": args.tol,
            **match,
        }
        rows.append(row)

    manifest_jsonl = out_dir / f"{suite}_lerobot_hdf5_mapping.jsonl"
    manifest_json = out_dir / f"{suite}_lerobot_hdf5_mapping.json"
    summary_path = out_dir / f"{suite}_lerobot_hdf5_mapping_summary.json"
    csv_path = out_dir / f"{suite}_lerobot_hdf5_mapping.csv"
    write_jsonl(manifest_jsonl, rows)
    manifest_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "suite", "dataset_name", "episode_index", "task", "lerobot_length", "match_type", "accepted",
        "hdf5_name", "demo_id", "hdf5_start", "hdf5_end_exclusive", "num_hdf5_frames",
        "uses_contiguous_states", "motion_mse", "motion_max_abs_error", "hdf5",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    counts = Counter(row["match_type"] for row in rows)
    accepted = sum(1 for row in rows if row.get("accepted"))
    contiguous = sum(1 for row in rows if row.get("accepted") and row.get("uses_contiguous_states"))
    summary = {
        "suite": suite,
        "dataset_name": dataset_name,
        "lerobot_root": str(lerobot_root),
        "hdf5_root": str(hdf5_root),
        "num_episodes": len(rows),
        "accepted": accepted,
        "rejected": len(rows) - accepted,
        "accepted_contiguous_states": contiguous,
        "accepted_ordered_noncontiguous_states": accepted - contiguous,
        "match_type_counts": dict(counts),
        "outputs": {
            "jsonl": str(manifest_jsonl),
            "json": str(manifest_json),
            "csv": str(csv_path),
        },
        "notes": [
            "exact_full_action compares all 7 dims after mapping HDF5 gripper -1/+1 to LeRobot 1/0.",
            "exact_contiguous_motion compares only first 6 motion dims on a contiguous HDF5 slice.",
            "exact_ordered_motion compares only first 6 motion dims as an ordered HDF5 subsequence and records explicit hdf5_indices.",
            "For exact_ordered_motion, depth export must gather states by hdf5_indices, not a simple contiguous slice.",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
