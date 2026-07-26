#!/usr/bin/env python3
"""Safely run the bilateral-wrist rerender builder across complete Teleop tasks.

Each task is rendered independently. A task is started only after its downloaded
LeRobot metadata and every declared episode parquet are present, and only if the
matching raw RoboCasa HDF5 file exists. Existing task outputs are never replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    task_name: str
    source_root: Path
    hdf5_path: Path
    episode_count: int


def parse_gpu_ids(value: str | None) -> list[int]:
    """Parse physical EGL device indices from a comma-separated CLI value."""

    if value is None or not value.strip():
        return []
    parts = [part.strip() for part in value.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"--gpu-ids contains an empty entry: {value!r}")
    try:
        gpu_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"--gpu-ids must be comma-separated integers, got {value!r}") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"--gpu-ids must be non-negative, got {gpu_ids}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-ids must not contain duplicates, got {gpu_ids}")
    return gpu_ids


def gpu_slots(gpu_ids: list[int], processes_per_gpu: int) -> list[int]:
    """Return one physical GPU ID for each concurrent renderer slot."""

    if processes_per_gpu < 1:
        raise ValueError(f"processes_per_gpu must be positive, got {processes_per_gpu}")
    return [gpu_id for gpu_id in gpu_ids for _ in range(processes_per_gpu)]


def renderer_environment(gpu_id: int | None, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build a child environment, optionally binding MuJoCo EGL to one GPU."""

    environment = dict(os.environ if base_env is None else base_env)
    if gpu_id is not None:
        environment["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    return environment


def _jsonl_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def ready_tasks(source_root: Path, hdf5_root: Path) -> list[Task]:
    """Return only source tasks whose declared LeRobot episode files are complete."""
    tasks: list[Task] = []
    for task_root in sorted((source_root / "LeRobot").glob("gr1_unified.*")):
        episodes_path = task_root / "meta" / "episodes.jsonl"
        if not (task_root / "meta" / "info.json").is_file() or not episodes_path.is_file():
            continue
        episode_count = _jsonl_count(episodes_path)
        parquet_count = len(list((task_root / "data").glob("chunk-*/episode_*.parquet")))
        if episode_count == 0 or parquet_count != episode_count:
            continue
        hdf5_name = task_root.name.removeprefix("gr1_unified.") + ".hdf5"
        hdf5_path = hdf5_root / hdf5_name
        if not hdf5_path.is_file():
            continue
        tasks.append(Task(task_root.name, task_root, hdf5_path, episode_count))
    return tasks


def is_complete(output_root: Path, episode_count: int) -> bool:
    """Completion is file-level, so independent successful shards compose safely."""
    return len(list((output_root / "data").glob("chunk-*/episode_*.parquet"))) == episode_count


def missing_episode_ranges(
    completed_indices: set[int], *, episode_count: int, max_episodes_per_chunk: int
) -> list[tuple[int, int]]:
    """Return half-open missing episode ranges, splitting at a safe process lifetime."""
    if max_episodes_per_chunk < 1:
        raise ValueError("max_episodes_per_chunk must be positive")
    missing = [index for index in range(episode_count) if index not in completed_indices]
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(missing):
        end = start + 1
        while (
            end < len(missing)
            and missing[end] == missing[end - 1] + 1
            and end - start < max_episodes_per_chunk
        ):
            end += 1
        ranges.append((missing[start], missing[end - 1] + 1))
        start = end
    return ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robocasa-repo", type=Path, required=True)
    parser.add_argument("--builder", type=Path, default=Path(__file__).with_name("build_robocasa_replay_rgbd_lerobot.py"))
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-tasks", type=int, default=0, help="Debug limit; <=0 runs every ready task.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Maximum concurrent short-lived renderer processes per task when --gpu-ids is omitted.")
    parser.add_argument("--gpu-ids", default="",
                        help="Comma-separated physical MuJoCo EGL device indices, for example 4,5,6,7.")
    parser.add_argument("--processes-per-gpu", type=int, default=2,
                        help="Concurrent renderer processes assigned to each GPU in --gpu-ids mode.")
    parser.add_argument("--max-episodes-per-process", type=int, default=60,
                        help="Bound a renderer process lifetime; MuJoCo model reloads are not stable beyond this many episodes.")
    parser.add_argument("--skip-incomplete-output", action="store_true", help="Leave in-progress task outputs untouched and continue to another ready task.")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing task rerenders and regenerate them with the current renderer.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _builder_command(task: Task, output: Path, args: argparse.Namespace, start: int, end: int, *, skip_prepare: bool) -> list[str]:
    command = [
        args.python_executable,
        str(args.builder),
        "--base-lerobot-root", str(task.source_root),
        "--hdf5", str(task.hdf5_path),
        "--output-root", str(output),
        "--robocasa-repo", str(args.robocasa_repo),
        "--resolution", str(args.resolution),
        "--fps", str(args.fps),
        "--episode-start", str(start),
        "--episode-end", str(end),
    ]
    if skip_prepare:
        command.append("--skip-prepare")
    return command


def prepare_command(task: Task, output: Path, args: argparse.Namespace) -> list[str]:
    """Initialize one task's metadata before any parallel render chunk starts."""

    return [
        args.python_executable,
        str(args.builder),
        "--base-lerobot-root", str(task.source_root),
        "--hdf5", str(task.hdf5_path),
        "--output-root", str(output),
        "--robocasa-repo", str(args.robocasa_repo),
        "--resolution", str(args.resolution),
        "--fps", str(args.fps),
        "--episode-start", "0",
        "--episode-end", str(task.episode_count),
        "--prepare-only",
    ]


def fresh_commands(task: Task, output: Path, args: argparse.Namespace) -> list[list[str]]:
    """Build a complete task as safe bounded-lifetime renderer commands."""
    ranges = missing_episode_ranges(
        set(),
        episode_count=task.episode_count,
        max_episodes_per_chunk=args.max_episodes_per_process,
    )
    return [
        _builder_command(task, output, args, start, end, skip_prepare=index > 0)
        for index, (start, end) in enumerate(ranges)
    ]


def resume_commands(task: Task, output: Path, completed_indices: set[int], args: argparse.Namespace) -> list[list[str]]:
    """Build safe short-lived commands for just the episodes not yet rendered."""
    ranges = missing_episode_ranges(
        completed_indices,
        episode_count=task.episode_count,
        max_episodes_per_chunk=args.max_episodes_per_process,
    )
    return [
        _builder_command(task, output, args, start, end, skip_prepare=True)
        for start, end in ranges
    ]


def _run_resume_task(task: Task, output: Path, args: argparse.Namespace) -> None:
    completed = {
        int(path.stem.rsplit("_", 1)[1])
        for path in (output / "data").glob("chunk-*/episode_*.parquet")
    }
    commands = resume_commands(task, output, completed, args)
    slots = gpu_slots(getattr(args, "gpu_ids", []), int(getattr(args, "processes_per_gpu", 2)))
    worker_count = len(slots) if slots else int(args.num_shards)
    mode = f"gpu_slots={slots}" if slots else f"num_shards={worker_count}"
    print(f"[resume {task.task_name}] missing={task.episode_count - len(completed)} chunks={len(commands)} workers={worker_count} {mode}", flush=True)
    for command in commands:
        print(" ".join(command), flush=True)
    if args.dry_run:
        return
    if not slots:
        for command in commands:
            subprocess.run(command, check=True, env=renderer_environment(None))
    else:
        for batch_start in range(0, len(commands), len(slots)):
            active = []
            for command_index, command in enumerate(commands[batch_start:batch_start + len(slots)], start=batch_start):
                gpu_id = slots[command_index % len(slots)]
                active.append(subprocess.Popen(command, env=renderer_environment(gpu_id)))
            exit_codes = [process.wait() for process in active]
            if any(code != 0 for code in exit_codes):
                raise RuntimeError(f"rerender resume chunk exit codes for {task.task_name}: {exit_codes}")
    if not is_complete(output, task.episode_count):
        raise RuntimeError(f"rerender resume finished but episode parquet coverage is incomplete: {output}")


def _run_fresh_task(task: Task, output: Path, args: argparse.Namespace) -> None:
    slots = gpu_slots(getattr(args, "gpu_ids", []), int(getattr(args, "processes_per_gpu", 2)))
    worker_count = len(slots) if slots else int(args.num_shards)
    if worker_count < 1:
        raise ValueError(f"renderer worker count must be positive, got {worker_count}")
    commands = fresh_commands(task, output, args)
    preparation = prepare_command(task, output, args)
    render_commands = [command if "--skip-prepare" in command else command + ["--skip-prepare"] for command in commands]
    mode = f"gpu_slots={slots}" if slots else f"num_shards={worker_count}"
    print(f"[start {task.task_name}] episodes={task.episode_count} chunks={len(render_commands)} workers={worker_count} {mode}", flush=True)
    print("[prepare] " + " ".join(preparation), flush=True)
    for command in render_commands:
        print(" ".join(command), flush=True)
    if args.dry_run:
        return

    preparation_gpu = slots[0] if slots else None
    subprocess.run(preparation, check=True, env=renderer_environment(preparation_gpu))

    def launch(command_index: int) -> subprocess.Popen:
        gpu_id = slots[command_index % len(slots)] if slots else None
        return subprocess.Popen(render_commands[command_index], env=renderer_environment(gpu_id))

    first = launch(0)
    active = [first]
    next_command = 1
    while active:
        while next_command < len(render_commands) and len(active) < worker_count:
            active.append(launch(next_command))
            next_command += 1
        failures = [worker.wait() for worker in active]
        if any(code != 0 for code in failures):
            raise RuntimeError(f"rerender chunk exit codes for {task.task_name}: {failures}")
        active = []
        if next_command < len(render_commands):
            active.append(launch(next_command))
            next_command += 1
    if not is_complete(output, task.episode_count):
        raise RuntimeError(f"rerender finished but episode parquet coverage is incomplete: {output}")


def main() -> None:
    args = parse_args()
    args.gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.processes_per_gpu < 1:
        raise ValueError(f"--processes-per-gpu must be positive, got {args.processes_per_gpu}")
    tasks = ready_tasks(args.source_root, args.hdf5_root)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise RuntimeError("No complete Teleop task is ready: need metadata, every source parquet, and matching HDF5.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        output = args.output_root / f"{task.task_name}_rerender"
        if output.exists():
            if args.rebuild:
                print(f"[rebuild] {task.task_name}: removing {output}", flush=True)
                if not args.dry_run:
                    shutil.rmtree(output)
                else:
                    continue
                _run_fresh_task(task, output, args)
                continue
            if is_complete(output, task.episode_count):
                print(f"[skip complete] {task.task_name}", flush=True)
                continue
            if args.skip_incomplete_output:
                print(f"[skip in progress] {task.task_name}", flush=True)
                continue
            _run_resume_task(task, output, args)
            continue
        _run_fresh_task(task, output, args)


if __name__ == "__main__":
    main()
