#!/usr/bin/env python3
"""Strict structural validator for an extracted CALVIN raw split."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    from .calvin_schema import validate_frame
except ImportError:
    from calvin_schema import validate_frame


FRAME_RE = re.compile(r"^episode_(\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument("--sample-frames", type=int, default=32)
    parser.add_argument("--require-config", action="store_true")
    parser.add_argument("--require-scene-info", action="store_true")
    return parser.parse_args()


def load_boundaries(split_dir: Path) -> np.ndarray:
    path = split_dir / "ep_start_end_ids.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    boundaries = np.asarray(np.load(path), dtype=np.int64)
    if boundaries.ndim != 2 or boundaries.shape[1] != 2:
        raise ValueError(f"invalid episode boundaries shape: {boundaries.shape}")
    if np.any(boundaries[:, 1] < boundaries[:, 0]):
        raise ValueError("episode boundaries contain an inverted range")
    if len(boundaries) and np.any(boundaries[1:, 0] <= boundaries[:-1, 0]):
        raise ValueError("episode boundaries are not strictly ordered")
    return boundaries


def load_frame_bitmap(split_dir: Path, max_frame: int) -> tuple[np.ndarray, int, int, list[Path]]:
    present = np.zeros(max_frame + 1, dtype=np.bool_)
    count = 0
    invalid: list[Path] = []
    sampled: list[Path] = []
    min_frame = max_frame + 1
    max_seen = -1
    for entry in __import__("os").scandir(split_dir):
        match = FRAME_RE.match(entry.name)
        if not match or not entry.is_file():
            continue
        frame_id = int(match.group(1))
        if frame_id > max_frame:
            invalid.append(Path(entry.path))
            continue
        if present[frame_id]:
            raise ValueError(f"duplicate frame id: {frame_id}")
        present[frame_id] = True
        count += 1
        min_frame = min(min_frame, frame_id)
        max_seen = max(max_seen, frame_id)
        if len(sampled) < 4:
            sampled.append(Path(entry.path))
    if count:
        sampled.extend([Path(split_dir / f"episode_{int(i):07d}.npz") for i in np.linspace(min_frame, max_seen, 8, dtype=np.int64) if present[int(i)]])
    return present, count, max_seen, list(dict.fromkeys(sampled))


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    split_dir = dataset_root / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)
    boundaries = load_boundaries(split_dir)
    max_end = int(boundaries[:, 1].max()) if len(boundaries) else -1
    expected = np.zeros(max_end + 1, dtype=np.bool_)
    for begin, end in boundaries:
        expected[int(begin) : int(end) + 1] = True
    expected_count = int(expected.sum())
    present, frame_count, max_seen, sampled_paths = load_frame_bitmap(split_dir, max_end)
    missing_ids = np.flatnonzero(expected & ~present)
    extra_ids = np.flatnonzero(present & ~expected)
    if len(missing_ids) or len(extra_ids):
        raise ValueError(
            f"raw frame coverage mismatch: missing={len(missing_ids)} extra={len(extra_ids)} "
            f"first_missing={missing_ids[:8].tolist()} first_extra={extra_ids[:8].tolist()}"
        )
    if frame_count != expected_count:
        raise ValueError(f"frame count mismatch: found={frame_count} expected={expected_count}")

    for name in ("ep_lens.npy", "ep_start_end_ids.npy"):
        if not (split_dir / name).exists():
            raise FileNotFoundError(split_dir / name)
    scene_info_present = (split_dir / "scene_info.npy").exists()
    if args.require_scene_info and not scene_info_present:
        raise FileNotFoundError(split_dir / "scene_info.npy")
    language_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not language_path.exists():
        raise FileNotFoundError(language_path)
    language = np.load(language_path, allow_pickle=True).item()
    if not isinstance(language, dict) or not {"language", "info"}.issubset(language):
        raise ValueError(f"invalid language annotation structure: {language_path}")
    if args.require_config and not (split_dir / ".hydra" / "merged_config.yaml").exists():
        raise FileNotFoundError(split_dir / ".hydra" / "merged_config.yaml")

    ep_lens = np.asarray(np.load(split_dir / "ep_lens.npy"), dtype=np.int64).reshape(-1)
    if len(ep_lens) != len(boundaries):
        raise ValueError(f"ep_lens count {len(ep_lens)} != boundaries count {len(boundaries)}")
    if not np.array_equal(ep_lens, boundaries[:, 1] - boundaries[:, 0] + 1):
        raise ValueError("ep_lens does not match ep_start_end_ids")

    sample_ids = np.linspace(0, max_end, max(1, args.sample_frames), dtype=np.int64)
    sample_ids = sorted(set(int(i) for i in sample_ids if expected[int(i)]))
    sample_ids.extend(int(Path(p).stem.split("_")[1]) for p in sampled_paths)
    checked = []
    for frame_id in sorted(set(sample_ids)):
        path = split_dir / f"episode_{frame_id:07d}.npz"
        with np.load(path, allow_pickle=False) as raw:
            frame = {key: np.asarray(raw[key]) for key in raw.files}
        validate_frame(frame)
        checked.append(frame_id)

    result = {
        "status": "success",
        "dataset_root": str(dataset_root),
        "split": args.split,
        "episodes": int(len(boundaries)),
        "frames": frame_count,
        "min_frame_id": int(np.flatnonzero(present)[0]) if frame_count else None,
        "max_frame_id": max_seen,
        "language_annotations": int(len(language.get("language", {}).get("ann", []))),
        "sampled_frames": checked,
        "require_config": bool(args.require_config),
        "scene_info_present": scene_info_present,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
