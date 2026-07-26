#!/usr/bin/env python3
"""Run/resume the formal 24-task Fourier RoboCasa rerender pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from starVLA.dataloader.robocasa_fourier_tasks import FOURIER_TASKS, fourier_dataset_names, task_for_source_dataset


def canonical_output_name(source_task_name: str) -> str:
    return task_for_source_dataset(source_task_name).official_dataset_name


def legacy_output_name(source_task_name: str) -> str:
    return f"{source_task_name}_rerender"


def migrate_legacy_outputs(root: Path, *, dry_run: bool = False) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for task in FOURIER_TASKS:
        legacy = root / legacy_output_name(task.source_dataset_name)
        canonical = root / task.official_dataset_name
        if not legacy.exists():
            continue
        if canonical.exists():
            raise RuntimeError(f"Both legacy and canonical outputs exist: {legacy} and {canonical}")
        moves.append((legacy, canonical))
        if not dry_run:
            legacy.rename(canonical)
    return moves


def stage_canonical_outputs(root: Path, *, rebuild: bool, dry_run: bool = False) -> None:
    """Stage canonical names under the generic runner's legacy names."""
    for task in FOURIER_TASKS:
        legacy = root / legacy_output_name(task.source_dataset_name)
        canonical = root / task.official_dataset_name
        if rebuild:
            for path in (legacy, canonical):
                if not path.exists():
                    continue
                print(f"[rebuild] removing {path}", flush=True)
                if not dry_run:
                    shutil.rmtree(path)
            continue
        if canonical.exists():
            if legacy.exists():
                raise RuntimeError(f"Both legacy and canonical outputs exist: {legacy} and {canonical}")
            print(f"[stage] {canonical} -> {legacy}", flush=True)
            if not dry_run:
                canonical.rename(legacy)


def load_and_validate_fourier_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = set(fourier_dataset_names())
    tasks = manifest.get("tasks", [])
    names = {task.get("task") for task in tasks}
    if len(tasks) != 24 or names != expected:
        raise RuntimeError(f"Fourier manifest must contain exactly 24 expected tasks, got {len(tasks)}")
    if manifest.get("successful_tasks") != 24 or manifest.get("total_episodes") != 24000:
        raise RuntimeError("Fourier manifest is not complete: expected 24 successful tasks and 24000 episodes")
    return manifest


def _runner_command(args: argparse.Namespace) -> list[str]:
    runner = Path(__file__).with_name("run_robocasa_teleop_rerender.py")
    command = [
        args.python,
        str(runner),
        "--source-root", str(args.source_root),
        "--hdf5-root", str(args.hdf5_root),
        "--output-root", str(args.output_root),
        "--robocasa-repo", str(args.robocasa_repo),
        "--resolution", str(args.resolution),
        "--fps", str(args.fps),
        "--max-episodes-per-process", str(args.max_episodes_per_process),
        "--num-shards", str(args.num_shards),
        "--gpu-ids", str(args.gpu_ids),
        "--processes-per-gpu", str(args.processes_per_gpu),
    ]
    if args.max_tasks > 0:
        command.extend(["--max-tasks", str(args.max_tasks)])
    if args.rebuild:
        command.append("--rebuild")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/root/data/yxz/datasets/robocasa_teleop_lerobot_original"))
    parser.add_argument("--hdf5-root", type=Path, default=Path("/root/data/yxz/datasets/robocasa_gr1_original_hdf5/HDF5"))
    parser.add_argument("--output-root", type=Path, default=Path("/root/data/yxz/datasets/robocasa_fourier_rerender"))
    parser.add_argument("--robocasa-repo", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-episodes-per-process", type=int, default=60)
    parser.add_argument("--num-shards", type=int, default=4,
                        help="Legacy single-device concurrency; ignored when --gpu-ids is set.")
    parser.add_argument("--gpu-ids", default="",
                        help="Comma-separated physical MuJoCo EGL device indices, for example 4,5,6,7.")
    parser.add_argument("--processes-per-gpu", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--migrate-legacy", action="store_true", help="Deprecated compatibility flag; canonical staging is now automatic.")
    parser.add_argument("--rebuild", action="store_true", help="Regenerate existing task outputs with the current renderer.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.processes_per_gpu < 1:
        raise ValueError(f"--processes-per-gpu must be positive, got {args.processes_per_gpu}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    stage_canonical_outputs(args.output_root, rebuild=args.rebuild, dry_run=args.dry_run)

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    subprocess.run(_runner_command(args), check=True, env=env)
    if not args.dry_run:
        for old, new in migrate_legacy_outputs(args.output_root):
            print(f"[migrate] {old} -> {new}", flush=True)


if __name__ == "__main__":
    main()
