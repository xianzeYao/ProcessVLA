#!/usr/bin/env python3
"""CPU-only CALVIN rerender runner.

Samples complete language segments, routes them using ``scene_info.npy``,
and renders isolated worker outputs compatible with the existing H5-to-LeRobot
converter.  The runner never initializes EGL or CUDA.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import hydra
import numpy as np
from omegaconf import OmegaConf

try:
    from examples.simBenchmarks.calvin.data_preparation.build_calvin_native_rgbd_lerobot import (
        build_segments,
        load_annotations,
        load_source_episodes,
    )
    from examples.simBenchmarks.calvin.data_preparation.rerender_calvin_episodes import (
        render_episode,
        resolve_config_dir,
    )
except ImportError:  # pragma: no cover - direct file execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from examples.simBenchmarks.calvin.data_preparation.build_calvin_native_rgbd_lerobot import (
        build_segments,
        load_annotations,
        load_source_episodes,
    )
    from examples.simBenchmarks.calvin.data_preparation.rerender_calvin_episodes import (
        render_episode,
        resolve_config_dir,
    )


DEFAULT_WORKERS = 16
SCENES = frozenset(("A", "B", "C", "D"))


def select_random_segments(segments: Sequence[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Select a reproducible subset; ``count=0`` preserves all segments."""

    if count < 0:
        raise ValueError("count must be non-negative; use 0 for all segments")
    source = list(segments)
    if count == 0:
        return source
    if count > len(source):
        raise ValueError(f"requested {count} segments, but only {len(source)} are available")
    indices = np.random.default_rng(seed).choice(len(source), size=count, replace=False).tolist()
    return [source[index] for index in indices]


def scene_for_frame(scene_intervals: Sequence[tuple[int, int, str]], frame_id: int) -> str:
    matches = [scene for start, end, scene in scene_intervals if start <= frame_id <= end]
    if len(matches) != 1:
        raise ValueError(f"frame {frame_id} maps to {len(matches)} scenes")
    return matches[0]


def balance_segments(segments: Sequence[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    """Greedily balance segment frame counts across CPU workers."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    group_count = min(workers, max(1, len(segments)))
    groups: list[list[dict[str, Any]]] = [[] for _ in range(group_count)]
    loads = [0] * group_count
    weighted = sorted(
        segments,
        key=lambda item: int(item["source_end"]) - int(item["source_start"]) + 1,
        reverse=True,
    )
    for segment in weighted:
        length = int(segment["source_end"]) - int(segment["source_start"]) + 1
        target = min(range(group_count), key=lambda index: (loads[index], index))
        groups[target].append(segment)
        loads[target] += length
    return groups


def load_scene_intervals(split_dir: Path) -> list[tuple[int, int, str]]:
    path = split_dir / "scene_info.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing scene metadata: {path}")
    raw = np.load(path, allow_pickle=True).item()
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"scene metadata must be a non-empty dict: {path}")
    intervals = []
    for name, bounds in raw.items():
        scene = str(name).replace("calvin_scene_", "")
        if scene not in SCENES or len(bounds) != 2:
            raise ValueError(f"invalid scene entry: {name!r}={bounds!r}")
        start, end = int(bounds[0]), int(bounds[1])
        if end < start:
            raise ValueError(f"inverted scene interval: {name!r}={bounds!r}")
        intervals.append((start, end, scene))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] <= previous[1]:
            raise ValueError(f"overlapping scene intervals: {previous} and {current}")
    return intervals


def add_scene_fields(segments: Iterable[dict[str, Any]], scene_intervals: Sequence[tuple[int, int, str]]) -> list[dict[str, Any]]:
    result = []
    for segment in segments:
        start, end = int(segment["source_start"]), int(segment["source_end"])
        scene = scene_for_frame(scene_intervals, start)
        if scene_for_frame(scene_intervals, end) != scene:
            raise ValueError(f"segment {segment['source_segment_index']} crosses scenes: {start}:{end}")
        value = dict(segment)
        value["scene"] = scene
        value["frame_count"] = end - start + 1
        result.append(value)
    return result


def make_cpu_env(config_dir: Path, scene_name: str) -> Any:
    """Build one scene-specific CPU environment without EGL."""

    if scene_name not in SCENES:
        raise ValueError(f"unknown CALVIN scene: {scene_name}")
    config = OmegaConf.load(str(config_dir / ".hydra" / "merged_config.yaml"))
    import calvin_env as calvin_env_package

    env_root = Path(calvin_env_package.__file__).resolve().parents[1]
    scene_config = OmegaConf.load(str(env_root / "conf" / "scene" / f"calvin_scene_{scene_name}.yaml"))
    scene_config.data_path = str(env_root / "data")
    scene_config.euler_obs = config.robot.euler_obs
    config.scene = scene_config
    config.env.scene_cfg = scene_config
    config.cameras.pop("tactile", None)
    config.env.cameras = config.cameras
    config.env.use_egl = False
    config.env.show_gui = False
    return hydra.utils.instantiate(config.env, show_gui=False, use_vr=False, use_scene_info=True)


def worker_main(
    dataset_root: str,
    source_split: str,
    config_split: str,
    segments: list[dict[str, Any]],
    worker_root: str,
    comparison: bool,
    fps: float,
    queue: Any,
) -> None:
    output_root = Path(worker_root)
    output_root.mkdir(parents=True, exist_ok=True)
    environments: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        dataset_path = Path(dataset_root)
        config_dir = resolve_config_dir(dataset_path, config_split)
        for local_index, segment in enumerate(segments):
            scene = str(segment["scene"])
            if scene not in environments:
                environments[scene] = make_cpu_env(config_dir, scene)
            summary = render_episode(
                environments[scene],
                dataset_path / source_split,
                output_root,
                local_index,
                int(segment["source_start"]),
                int(segment["source_end"]),
                fps,
                write_comparison=comparison,
            )
            summary.update(
                {
                    "source_segment_index": int(segment["source_segment_index"]),
                    "source_start": int(segment["source_start"]),
                    "source_end": int(segment["source_end"]),
                    "source_episode_index": int(segment["source_episode_index"]),
                    "instruction": str(segment["instruction"]),
                    "task": str(segment["task"]),
                    "scene": scene,
                    "h5": str((output_root / summary["h5"]).relative_to(output_root.parent)),
                }
            )
            summaries.append(summary)
    except Exception as exc:
        queue.put({"status": "error", "error": repr(exc), "worker_root": str(output_root)})
        raise
    finally:
        for environment in environments.values():
            try:
                environment.close()
            except Exception:
                pass
    payload = {
        "status": "success",
        "worker_root": str(output_root),
        "segments": summaries,
        "frames": sum(int(item["frames"]) for item in summaries),
        "seconds": time.perf_counter() - started,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    queue.put(payload)


def prepare_segments(dataset_root: Path, source_split: str, count: int, seed: int) -> list[dict[str, Any]]:
    source_dir = dataset_root / source_split
    boundaries = load_source_episodes(source_dir)
    annotations = load_annotations(source_dir)
    frame_ids = {int(path.stem.split("_", 1)[1]) for path in source_dir.glob("episode_*.npz")}
    complete = build_segments(annotations, frame_ids, boundaries)
    if not complete:
        raise RuntimeError(f"no complete language segments in {source_dir}")
    return add_scene_fields(select_random_segments(complete, count, seed), load_scene_intervals(source_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-split", choices=("training", "validation"), default="training")
    parser.add_argument("--config-split", choices=("auto", "training", "validation"), default="auto")
    parser.add_argument("--num-random-segments", type=int, default=5, metavar="N", help="random segments; 0 means all")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--comparison", action="store_true", help="write comparison MP4/JPEG files")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    if not 1 <= args.workers <= cpu_count:
        raise ValueError(f"--workers must be between 1 and {cpu_count}")
    if args.num_random_segments < 0:
        raise ValueError("--num-random-segments must be non-negative; use 0 for all")
    dataset_root = args.dataset_root.resolve()
    source_dir = dataset_root / args.source_split
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing source split: {source_dir}")
    selected = prepare_segments(dataset_root, args.source_split, args.num_random_segments, args.seed)
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    groups = balance_segments(selected, min(args.workers, len(selected)))
    print(json.dumps({
        "selected_segments": len(selected),
        "selected_frames": sum(int(item["frame_count"]) for item in selected),
        "workers": len(groups),
        "scene_counts": {scene: sum(item["scene"] == scene for item in selected) for scene in sorted(SCENES)},
    }, indent=2), flush=True)

    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = []
    for worker_index, group in enumerate(groups):
        worker_root = output_root / f"worker-{worker_index:03d}"
        process = context.Process(
            target=worker_main,
            args=(str(dataset_root), args.source_split, args.config_split, group, str(worker_root), args.comparison, args.fps, queue),
        )
        process.start()
        processes.append(process)
    results = [queue.get(timeout=24 * 60 * 60) for _ in processes]
    for process in processes:
        process.join()
    if any(result.get("status") != "success" for result in results) or any(process.exitcode != 0 for process in processes):
        raise RuntimeError(f"CPU rerender worker failure: {results}")
    rendered = [segment for result in results for segment in result["segments"]]
    expected = {(int(item["source_start"]), int(item["source_end"])) for item in selected}
    actual = {(int(item["source_start"]), int(item["source_end"])) for item in rendered}
    if len(rendered) != len(selected) or expected != actual:
        raise RuntimeError("rendered segments do not exactly match selected source ranges")
    summary = {
        "status": "success",
        "dataset_root": str(dataset_root),
        "source_split": args.source_split,
        "num_random_segments": args.num_random_segments,
        "seed": args.seed,
        "workers": len(groups),
        "comparison_visualizations": bool(args.comparison),
        "num_segments": len(rendered),
        "num_frames": sum(int(item["frames"]) for item in rendered),
        # Keep this key separate: the existing converter's legacy summary
        # parser expects a flat h5 basename. It will discover these worker H5s
        # recursively and reconstruct metadata from frame_id when this key is
        # absent.
        "rendered_segments": sorted(rendered, key=lambda item: int(item["source_segment_index"])),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
