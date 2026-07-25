#!/usr/bin/env python3
"""Parallel conversion of rerendered CALVIN H5 segments to LeRobot.

The per-episode writer is intentionally shared with the serial converter so
the action/state/depth/camera contract cannot drift between implementations.
Workers only write distinct global episode indices.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


try:
    from .build_calvin_native_rgbd_lerobot import (
        FPS,
        build_segments,
        load_annotations,
        load_source_episodes,
    )
    from .build_calvin_rerender_lerobot import (
        convert_segment,
        find_h5_files,
        prepare_output,
        read_h5_frame_ids,
    )
except ImportError:  # direct import from the data_preparation test directory
    from build_calvin_native_rgbd_lerobot import FPS, build_segments, load_annotations, load_source_episodes
    from build_calvin_rerender_lerobot import convert_segment, find_h5_files, prepare_output, read_h5_frame_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rerender-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-split", default="training", choices=("training", "validation"))
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fps", type=float, default=FPS)
    return parser.parse_args()


def partition_jobs(jobs: list[int], workers: int) -> list[list[int]]:
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    worker_count = min(workers, max(len(jobs), 1))
    quotient, remainder = divmod(len(jobs), worker_count)
    parts = []
    offset = 0
    for worker_index in range(worker_count):
        size = quotient + (1 if worker_index < remainder else 0)
        parts.append(jobs[offset : offset + size])
        offset += size
    return parts


def _episode_paths(output_root: Path, episode_index: int) -> list[Path]:
    chunk = episode_index // 1000
    episode = f"episode_{episode_index:06d}"
    chunk_name = f"chunk-{chunk:03d}"
    return [
        output_root / "videos" / chunk_name / "observation.images.image" / f"{episode}.mp4",
        output_root / "videos" / chunk_name / "observation.images.wrist_image" / f"{episode}.mp4",
        output_root / "depth" / chunk_name / "observation.depth.image_m" / f"{episode}.npz",
        output_root / "depth" / chunk_name / "observation.depth.wrist_m" / f"{episode}.npz",
        output_root / "camera" / chunk_name / f"{episode}.npz",
        output_root / "data" / chunk_name / f"{episode}.parquet",
    ]


def expected_episode_outputs(output_root: Path, episode_index: int) -> list[Path]:
    return _episode_paths(output_root, episode_index)


def episode_outputs_complete(output_root: Path, episode_index: int) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in _episode_paths(output_root, episode_index))


def _build_h5_indices(rerender_root: Path, h5_files: list[Path]) -> dict:
    by_relative = {}
    by_parent_relative = {}
    by_name = {}
    for path in h5_files:
        if _is_relative_to(path, rerender_root):
            by_relative[path.relative_to(rerender_root).as_posix()] = path
        if _is_relative_to(path, rerender_root.parent):
            by_parent_relative[path.relative_to(rerender_root.parent).as_posix()] = path
        by_name.setdefault(path.name, []).append(path)
    return {
        "by_relative": by_relative,
        "by_parent_relative": by_parent_relative,
        "by_name": by_name,
    }


def _resolve_manifest_h5(rerender_root: Path, reference: str, indices: dict) -> Path:
    ref = Path(reference)
    candidates = []
    if ref.is_absolute():
        candidates.append(ref)
    candidates.extend((rerender_root / ref, rerender_root.parent / ref))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if reference in indices["by_relative"]:
        return indices["by_relative"][reference]
    if reference in indices["by_parent_relative"]:
        return indices["by_parent_relative"][reference]
    matches = indices["by_name"].get(ref.name, [])
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"summary references missing or ambiguous rerender H5: {reference}")


def manifest_segments(rerender_root: Path, h5_files: list[Path]) -> list[tuple[Path, dict]]:
    manifest_path = rerender_root / "summary.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("segments") or payload.get("rendered_segments") or []
    selected = []
    indices = _build_h5_indices(rerender_root, h5_files)
    for entry in entries:
        begin = entry.get("begin", entry.get("source_start"))
        end = entry.get("end", entry.get("source_end"))
        selected.append(
            (
                _resolve_manifest_h5(rerender_root, str(entry["h5"]), indices),
                {
                    "source_start": int(begin),
                    "source_end": int(end),
                    "source_segment_index": int(entry.get("source_segment_index", -1)),
                    "instruction": str(entry["instruction"]),
                    "task": str(entry["task"]),
                },
            )
        )
    return selected


def _fallback_segments(dataset_root: Path, rerender_root: Path, source_split: str) -> list[tuple[Path, dict]]:
    source_dir = dataset_root / source_split
    boundaries = load_source_episodes(source_dir)
    annotations = load_annotations(source_dir)
    selected = []
    for h5_path in find_h5_files(rerender_root):
        frame_ids = read_h5_frame_ids(h5_path)
        selected.extend((h5_path, segment) for segment in build_segments(annotations, frame_ids, boundaries))
    return selected


def _mapping_row(episode_index: int, h5_path: Path, segment: dict) -> dict:
    return {
        "episode_index": episode_index,
        "tasks": [segment["instruction"]],
        "task": segment["task"],
        "length": int(segment["source_end"]) - int(segment["source_start"]) + 1,
        "source_start": int(segment["source_start"]),
        "source_end": int(segment["source_end"]),
        "rerender_h5": str(h5_path),
    }


def _convert_job(job: tuple[int, Path, dict, Path, float, int]) -> dict:
    episode_index, h5_path, segment, output_root, fps, task_index = job
    result = convert_segment(segment, episode_index, h5_path, output_root, fps, task_index)
    return result


def _write_metadata(output_root: Path, mapping: Iterable[dict], total_frames: int, source_split: str) -> None:
    ordered = sorted(mapping, key=lambda row: int(row["episode_index"]))
    with (output_root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_root / "meta" / "conversion_summary.json").write_text(
        json.dumps(
            {
                "status": "success",
                "source_split": source_split,
                "num_episodes": len(ordered),
                "num_frames": total_frames,
                "action_key": "rel_actions",
                "rgbd_mode": "rerendered",
                "converter": "parallel",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    h5_files = find_h5_files(args.rerender_root)
    selected = manifest_segments(args.rerender_root, h5_files)
    if not selected:
        selected = _fallback_segments(args.dataset_root, args.rerender_root, args.source_split)
    if args.max_segments > 0:
        selected = selected[: args.max_segments]
    if not selected:
        raise RuntimeError("no language segments in rerender H5 files")

    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    elif args.output_root.exists() and not args.resume:
        raise FileExistsError(f"output exists: {args.output_root}; pass --overwrite or --resume")

    tasks = [segment["instruction"] for _, segment in selected]
    total_frames = sum(int(segment["source_end"]) - int(segment["source_start"]) + 1 for _, segment in selected)
    prepare_output(args.output_root, args.dataset_root, len(selected), total_frames, args.fps, tasks)
    task_index = {task: index for index, task in enumerate(sorted(set(tasks)))}
    mapping = {}
    jobs = []
    for episode_index, (h5_path, segment) in enumerate(selected):
        row = _mapping_row(episode_index, h5_path, segment)
        if args.resume and episode_outputs_complete(args.output_root, episode_index):
            mapping[episode_index] = row
            continue
        jobs.append((episode_index, h5_path, segment, args.output_root, args.fps, task_index[segment["instruction"]]))

    print(f"[parallel-convert] episodes={len(selected)} pending={len(jobs)} workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_convert_job, job): job[0] for job in jobs}
        for completed_index, future in enumerate(as_completed(futures), start=1):
            episode_index = futures[future]
            mapping[episode_index] = future.result()
            print(f"[parallel-convert {completed_index}/{len(jobs)}] episode={episode_index}", flush=True)

    if len(mapping) != len(selected):
        missing = sorted(set(range(len(selected))) - set(mapping))
        raise RuntimeError(f"conversion did not produce all episodes; missing {missing[:10]}")
    _write_metadata(args.output_root, mapping.values(), total_frames, args.source_split)
    print(json.dumps({"output": str(args.output_root), "episodes": len(mapping), "frames": total_frames}, indent=2), flush=True)


if __name__ == "__main__":
    main()
