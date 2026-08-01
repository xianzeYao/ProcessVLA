"""Pure helpers for reproducible RoboCasa-GR1 evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO


def build_task_ranges(total_tasks: int, num_workers: int) -> list[list[int]]:
    """Assign tasks round-robin so every worker gets a deterministic subset."""
    if total_tasks < 0:
        raise ValueError("total_tasks must be non-negative")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    return [list(range(worker_id, total_tasks, num_workers)) for worker_id in range(num_workers)]


def task_slug(task_index: int, env_name: str) -> str:
    """Return the stable filename stem for one task."""
    basename = env_name.rsplit("/", 1)[-1]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename).strip("_.")
    if not safe_name:
        raise ValueError(f"environment name has no usable basename: {env_name!r}")
    return f"task_{task_index:02d}_{safe_name}"


def build_manifest(
    checkpoint: str,
    gpus: list[str],
    num_episodes: int,
    base_port: int,
    run_dir: Path,
    env_names: list[str],
    save_video: bool,
) -> dict[str, Any]:
    """Describe every task and output path before evaluation starts."""
    if not 1 <= len(gpus) <= 24:
        raise ValueError(
            f"RoboCasa evaluation requires between 1 and 24 GPUs, got {len(gpus)}"
        )
    if len(env_names) != 24:
        raise ValueError(f"RoboCasa evaluation requires exactly 24 tasks, got {len(env_names)}")
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    gpu_values = [int(gpu) for gpu in gpus]
    if len(set(gpu_values)) != len(gpu_values):
        raise ValueError("RoboCasa evaluation requires unique GPU identifiers")
    tasks = []
    for task_index, env_name in enumerate(env_names):
        worker_id = task_index % len(gpu_values)
        slug = task_slug(task_index, env_name)
        tasks.append(
            {
                "task_index": task_index,
                "env_name": env_name,
                "worker_id": worker_id,
                "gpu": gpu_values[worker_id],
                "port": base_port + worker_id,
                "task_slug": slug,
                "task_log": str(run_dir / "logs" / f"{slug}.log"),
                "result_json": str(run_dir / "task_results" / f"{slug}.json"),
                "video_dir": str(run_dir / "videos" / slug) if save_video else None,
            }
        )

    return {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "gpus": gpu_values,
        "num_episodes": num_episodes,
        "num_tasks": len(env_names),
        "run_dir": str(run_dir),
        "save_video": save_video,
        "tasks": tasks,
    }


def emit_episode_progress(
    *,
    task_index: int,
    episode: int,
    total_episodes: int,
    success: bool,
    task_successes: int,
    elapsed_seconds: float,
    stream: TextIO | None = None,
) -> None:
    """Print one immediately visible task-local episode progress line."""
    output = stream if stream is not None else sys.stdout
    success_rate = task_successes / episode if episode else 0.0
    print(
        f"[robocasa] task={task_index:02d} episode={episode}/{total_episodes} "
        f"success={str(bool(success)).lower()} task_successes={task_successes} "
        f"task_success_rate={success_rate:.3f} elapsed={elapsed_seconds:.1f}s",
        file=output,
        flush=True,
    )


def emit_task_complete(
    *,
    task_index: int,
    episodes: int,
    successes: int,
    elapsed_seconds: float,
    stream: TextIO | None = None,
) -> None:
    """Print one immediately visible task completion line."""
    output = stream if stream is not None else sys.stdout
    success_rate = successes / episodes if episodes else 0.0
    print(
        f"[robocasa] task={task_index:02d} complete episodes={episodes} "
        f"successes={successes} success_rate={success_rate:.3f} elapsed={elapsed_seconds:.1f}s",
        file=output,
        flush=True,
    )


def _merge_metadata(payload: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    for key, value in (metadata or {}).items():
        if key in payload and payload[key] != value:
            raise ValueError(f"result metadata cannot override {key!r}")
        payload[key] = value
    return payload


def build_completed_task_payload(
    *,
    task_index: int,
    env_name: str,
    successes: Iterable[bool],
    elapsed_seconds: float,
    gpu: int,
    worker_id: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one successful task result without changing metric definitions."""
    values = [bool(value) for value in successes]
    success_count = sum(values)
    num_episodes = len(values)
    payload = {
        "schema_version": 1,
        "task_index": task_index,
        "env_name": env_name,
        "status": "completed",
        "num_episodes": num_episodes,
        "success_count": success_count,
        "success_rate": success_count / num_episodes if num_episodes else 0.0,
        "successes": values,
        "elapsed_seconds": float(elapsed_seconds),
        "gpu": gpu,
        "worker_id": worker_id,
    }
    return _merge_metadata(payload, metadata)


def build_failed_task_payload(
    *,
    task_index: int,
    env_name: str,
    error: str,
    traceback_text: str,
    elapsed_seconds: float,
    gpu: int,
    worker_id: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one failed task result with its diagnostic context."""
    payload = {
        "schema_version": 1,
        "task_index": task_index,
        "env_name": env_name,
        "status": "failed",
        "error": error,
        "traceback": traceback_text,
        "elapsed_seconds": float(elapsed_seconds),
        "gpu": gpu,
        "worker_id": worker_id,
    }
    return _merge_metadata(payload, metadata)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a formatted UTF-8 JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def aggregate_task_payloads(
    payloads: Iterable[dict[str, Any]],
    expected_task_count: int | None = None,
    expected_num_episodes: int | None = None,
) -> dict[str, Any]:
    """Aggregate task-level results using the official macro task average."""
    payload_list = list(payloads)
    if expected_task_count is not None and len(payload_list) != expected_task_count:
        raise ValueError(f"expected {expected_task_count} task results, got {len(payload_list)}")

    task_indices = []
    for payload in payload_list:
        if payload.get("status", "completed") != "completed":
            raise ValueError(
                f"task result is not completed: task_index={payload.get('task_index')} "
                f"status={payload.get('status')!r}"
            )
        if "task_index" not in payload:
            if expected_task_count is not None:
                raise ValueError("task result is missing task_index")
        else:
            task_indices.append(int(payload["task_index"]))

    if len(task_indices) != len(set(task_indices)):
        raise ValueError("duplicate RoboCasa task index")
    if expected_task_count is not None and task_indices != list(range(expected_task_count)):
        if sorted(task_indices) != list(range(expected_task_count)):
            raise ValueError(
                f"unexpected task indices: expected {list(range(expected_task_count))}, got {sorted(task_indices)}"
            )

    if len(task_indices) == len(payload_list):
        ordered = sorted(payload_list, key=lambda payload: int(payload["task_index"]))
    else:
        ordered = sorted(payload_list, key=lambda payload: str(payload["env_name"]))
    names = [str(payload["env_name"]) for payload in ordered]
    if len(names) != len(set(names)):
        raise ValueError("duplicate RoboCasa task result")

    task_success_rates: dict[str, float] = {}
    total_episodes = 0
    total_successes = 0
    for payload in ordered:
        successes = [bool(value) for value in payload["successes"]]
        if not successes:
            raise ValueError(f"task has no episode results: {payload['env_name']}")
        success_count = sum(successes)
        num_episodes = len(successes)
        success_rate = success_count / num_episodes
        if expected_num_episodes is not None and num_episodes != expected_num_episodes:
            raise ValueError(
                f"task {payload['env_name']} expected {expected_num_episodes} episodes, got {num_episodes}"
            )
        if "num_episodes" in payload and int(payload["num_episodes"]) != num_episodes:
            raise ValueError(
                f"num_episodes mismatch for {payload['env_name']}: "
                f"payload={payload['num_episodes']} successes={num_episodes}"
            )
        if "success_count" in payload and int(payload["success_count"]) != success_count:
            raise ValueError(f"success_count mismatch for {payload['env_name']}")
        if "success_rate" in payload and abs(float(payload["success_rate"]) - success_rate) > 1e-12:
            raise ValueError(f"success_rate mismatch for {payload['env_name']}")
        total_episodes += num_episodes
        total_successes += success_count
        task_success_rates[str(payload["env_name"])] = success_rate

    num_tasks = len(ordered)
    return {
        "num_tasks": num_tasks,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "macro_success_rate": sum(task_success_rates.values()) / max(num_tasks, 1),
        "micro_success_rate": total_successes / max(total_episodes, 1),
        "task_success_rates": task_success_rates,
        "task_results": ordered,
    }


def main(argv: list[str] | None = None) -> None:
    """Write a run manifest or a shell-level fallback failure result."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--checkpoint", required=True)
    manifest_parser.add_argument("--gpus", required=True)
    manifest_parser.add_argument("--num-episodes", required=True, type=int)
    manifest_parser.add_argument("--base-port", required=True, type=int)
    manifest_parser.add_argument("--run-dir", required=True, type=Path)
    manifest_parser.add_argument("--env-name", action="append", dest="env_names", required=True)
    manifest_parser.add_argument("--save-video", action="store_true")
    manifest_parser.add_argument("--output", required=True, type=Path)

    failure_parser = subparsers.add_parser("failure")
    failure_parser.add_argument("--output", required=True, type=Path)
    failure_parser.add_argument("--task-index", required=True, type=int)
    failure_parser.add_argument("--env-name", required=True)
    failure_parser.add_argument("--gpu", required=True, type=int)
    failure_parser.add_argument("--worker-id", required=True, type=int)
    failure_parser.add_argument("--elapsed-seconds", required=True, type=float)
    failure_parser.add_argument("--error", required=True)
    failure_parser.add_argument("--traceback-file", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "manifest":
        manifest = build_manifest(
            checkpoint=args.checkpoint,
            gpus=args.gpus.split(","),
            num_episodes=args.num_episodes,
            base_port=args.base_port,
            run_dir=args.run_dir,
            env_names=args.env_names,
            save_video=args.save_video,
        )
        write_json(args.output, manifest)
        print(f"[robocasa] manifest={args.output} tasks={manifest['num_tasks']}", flush=True)
        return

    traceback_text = args.traceback_file.read_text(encoding="utf-8", errors="replace")
    payload = build_failed_task_payload(
        task_index=args.task_index,
        env_name=args.env_name,
        error=args.error,
        traceback_text=traceback_text,
        elapsed_seconds=args.elapsed_seconds,
        gpu=args.gpu,
        worker_id=args.worker_id,
    )
    write_json(args.output, payload)
    print(f"[robocasa] task={args.task_index:02d} fallback_status=failed result={args.output}", flush=True)


if __name__ == "__main__":
    main()
