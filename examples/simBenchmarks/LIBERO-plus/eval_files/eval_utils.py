"""Pure helpers shared by LIBERO-plus workers and result aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def validate_task_range(start_idx: int, end_idx: int, total_tasks: int) -> tuple[int, int]:
    """Validate and normalize a half-open task range ``[start_idx, end_idx)``."""
    total_tasks = int(total_tasks)
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if total_tasks < 0:
        raise ValueError(f"total_tasks must be non-negative, got {total_tasks}")
    if start_idx < 0 or start_idx > total_tasks:
        raise ValueError(f"start_idx must be in [0, {total_tasks}], got {start_idx}")
    if end_idx < 0 or end_idx > total_tasks:
        raise ValueError(f"end_idx must be in [0, {total_tasks}], got {end_idx}")
    if end_idx < start_idx:
        raise ValueError(f"end_idx must be >= start_idx, got {start_idx}:{end_idx}")
    return start_idx, end_idx


def split_task_range(total_tasks: int, num_splits: int, split_index: int) -> tuple[int, int]:
    """Return a balanced, non-overlapping half-open range for one split."""
    total_tasks = int(total_tasks)
    num_splits = int(num_splits)
    split_index = int(split_index)
    if total_tasks < 0:
        raise ValueError(f"total_tasks must be non-negative, got {total_tasks}")
    if num_splits <= 0:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    if split_index < 0 or split_index >= num_splits:
        raise ValueError(
            f"split_index must be in [0, {num_splits}), got {split_index}"
        )
    base, remainder = divmod(total_tasks, num_splits)
    start_idx = split_index * base + min(split_index, remainder)
    end_idx = start_idx + base + (1 if split_index < remainder else 0)
    return start_idx, end_idx


def plan_suite_jobs(suite_sizes: dict[str, int], worker_count: int) -> list[dict]:
    """Allocate workers across suites and return one range job per worker.

    Every suite receives at least one worker. Remaining workers are assigned to
    the suite with the largest current ``task_count / assigned_workers`` ratio.
    """
    if not suite_sizes:
        raise ValueError("suite_sizes must not be empty")
    worker_count = int(worker_count)
    if worker_count <= 0:
        raise ValueError(f"worker_count must be positive, got {worker_count}")
    if worker_count < len(suite_sizes):
        raise ValueError(
            f"worker_count={worker_count} is smaller than suite_count={len(suite_sizes)}"
        )
    normalized = {str(name): int(size) for name, size in suite_sizes.items()}
    for name, size in normalized.items():
        if size < 0:
            raise ValueError(f"suite {name!r} has negative task count {size}")

    assignments = {name: 1 for name in normalized}
    for _ in range(worker_count - len(normalized)):
        name = max(
            normalized,
            key=lambda candidate: normalized[candidate] / assignments[candidate],
        )
        assignments[name] += 1

    jobs = []
    for suite, size in normalized.items():
        split_count = min(assignments[suite], max(size, 1))
        for split_index in range(split_count):
            start_idx, end_idx = split_task_range(size, split_count, split_index)
            if start_idx == end_idx and size > 0:
                continue
            jobs.append(
                {
                    "suite": suite,
                    "split_index": split_index,
                    "split_count": split_count,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "task_count": end_idx - start_idx,
                }
            )
    return jobs


def _episode_rows(payload: dict) -> list[dict]:
    rows = payload.get("episodes", [])
    if not isinstance(rows, list):
        raise ValueError("slice payload episodes must be a list")
    return rows


def aggregate_slice_results(
    slice_payloads: Iterable[dict],
    expected_total_tasks: int,
) -> dict:
    """Merge slice payloads and require exactly one coverage entry per task."""
    expected_total_tasks = int(expected_total_tasks)
    if expected_total_tasks < 0:
        raise ValueError("expected_total_tasks must be non-negative")

    payloads = list(slice_payloads)
    if not payloads and expected_total_tasks:
        raise ValueError("missing task ids: no slice payloads were provided")

    covered_task_ids: list[int] = []
    episodes: list[dict] = []
    suite_names: set[str] = set()
    slices = []
    for payload in payloads:
        suite = str(payload.get("suite", ""))
        if not suite:
            raise ValueError("slice payload is missing suite")
        suite_names.add(suite)
        task_ids = [int(item) for item in payload.get("task_ids", [])]
        start_idx, end_idx = validate_task_range(
            payload.get("start_idx", 0),
            payload.get("end_idx", 0),
            expected_total_tasks,
        )
        expected_ids = list(range(start_idx, end_idx))
        if task_ids != expected_ids:
            raise ValueError(
                f"slice task_ids do not match range {start_idx}:{end_idx}: {task_ids}"
            )
        covered_task_ids.extend(task_ids)
        rows = _episode_rows(payload)
        episodes.extend(rows)
        slices.append(
            {
                "suite": suite,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "task_count": len(task_ids),
                "episode_count": len(rows),
            }
        )

    duplicates = sorted(
        task_id for task_id in set(covered_task_ids) if covered_task_ids.count(task_id) > 1
    )
    if duplicates:
        raise ValueError(f"duplicate task ids: {duplicates}")
    missing = sorted(set(range(expected_total_tasks)) - set(covered_task_ids))
    if missing:
        raise ValueError(f"missing task ids: {missing}")

    category_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total_count": 0, "success_count": 0}
    )
    success_count = 0
    for episode in episodes:
        success = bool(episode.get("success", False))
        if success:
            success_count += 1
        category = episode.get("category")
        if category:
            category = str(category)
            category_counts[category]["total_count"] += 1
            category_counts[category]["success_count"] += int(success)

    episode_count = len(episodes)
    return {
        "suite": next(iter(suite_names)) if len(suite_names) == 1 else "mixed",
        "suites": sorted(suite_names),
        "slice_count": len(payloads),
        "task_count": len(covered_task_ids),
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": success_count / episode_count if episode_count else 0.0,
        "categories": dict(sorted(category_counts.items())),
        "task_ids": sorted(covered_task_ids),
        "slices": slices,
        "episodes": episodes,
    }

