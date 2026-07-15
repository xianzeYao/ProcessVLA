#!/usr/bin/env python3
"""Aggregate validated LIBERO-plus task-range JSON files."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from eval_utils import aggregate_slice_results

SLICE_PATTERN = re.compile(r"^(?P<suite>.+)_(?P<start>\d+)_(?P<end>\d+)\.json$")
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def load_suite_sizes(libero_home: Path) -> dict[str, int]:
    classification_path = libero_home / "libero/libero/benchmark/task_classification.json"
    with classification_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {suite: len(payload[suite]) for suite in DEFAULT_SUITES if suite in payload}


def discover_slice_files(log_dir: Path, suite: str | None = None) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(log_dir.glob("*.json")):
        match = SLICE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        current_suite = match.group("suite")
        if suite is not None and current_suite != suite:
            continue
        grouped[current_suite].append(path)
    return dict(grouped)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _merge_category_counts(target: dict, source: dict) -> None:
    for category, counts in source.items():
        target.setdefault(category, {"total_count": 0, "success_count": 0})
        target[category]["total_count"] += int(counts.get("total_count", 0))
        target[category]["success_count"] += int(counts.get("success_count", 0))


def _finalize_metrics(payload: dict) -> dict:
    total = int(payload["episode_count"])
    success = int(payload["success_count"])
    payload["success_rate"] = success / total if total else 0.0
    return payload


def aggregate_log_dir(
    log_dir: Path,
    suite_sizes: dict[str, int],
    suite: str | None = None,
) -> dict:
    grouped = discover_slice_files(log_dir, suite=suite)
    requested_suites = [suite] if suite is not None else [name for name in DEFAULT_SUITES if name in grouped]
    if not requested_suites:
        raise ValueError(f"No slice JSON files found under {log_dir}")

    suite_results = {}
    for suite_name in requested_suites:
        if suite_name not in suite_sizes:
            raise ValueError(f"Missing expected task count for suite {suite_name!r}")
        files = grouped.get(suite_name, [])
        if not files:
            raise ValueError(f"No slice JSON files found for suite {suite_name!r}")
        payloads = [read_json(path) for path in files]
        result = aggregate_slice_results(payloads, expected_total_tasks=suite_sizes[suite_name])
        result["source_files"] = [str(path) for path in files]
        suite_results[suite_name] = result

    overall = {
        "suite": "overall",
        "suites": sorted(suite_results),
        "slice_count": sum(item["slice_count"] for item in suite_results.values()),
        "task_count": sum(item["task_count"] for item in suite_results.values()),
        "episode_count": sum(item["episode_count"] for item in suite_results.values()),
        "success_count": sum(item["success_count"] for item in suite_results.values()),
        "categories": {},
    }
    for result in suite_results.values():
        _merge_category_counts(overall["categories"], result["categories"])
    _finalize_metrics(overall)
    return {"overall": overall, "suites": suite_results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--libero-home", default=os.environ.get("LIBERO_HOME"), type=Path)
    parser.add_argument("--suite", default=None, choices=DEFAULT_SUITES)
    parser.add_argument("--expected-total-tasks", default=None, type=int)
    parser.add_argument("--output-path", default=None, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.log_dir.is_dir():
        raise FileNotFoundError(f"Log directory does not exist: {args.log_dir}")
    if args.libero_home is None and args.expected_total_tasks is None:
        raise ValueError("Pass --libero-home or --expected-total-tasks")

    if args.libero_home is not None:
        suite_sizes = load_suite_sizes(args.libero_home)
    else:
        if args.suite is None:
            raise ValueError("--suite is required when using --expected-total-tasks")
        suite_sizes = {args.suite: args.expected_total_tasks}
    if args.expected_total_tasks is not None:
        if args.suite is None:
            raise ValueError("--suite is required with --expected-total-tasks")
        suite_sizes[args.suite] = args.expected_total_tasks

    result = aggregate_log_dir(args.log_dir, suite_sizes, suite=args.suite)
    output_path = args.output_path or (args.log_dir / "overall_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
