#!/usr/bin/env python3
"""Print the deterministic LIBERO-plus task-slice plan as TSV."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-home", required=True, type=Path)
    parser.add_argument("--suites", required=True, help="comma-separated suite names")
    parser.add_argument("--worker-count", required=True, type=int)
    parser.add_argument(
        "--eval-utils", default=Path(__file__).with_name("eval_utils.py"), type=Path
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suites = [item for item in args.suites.split(",") if item]
    classification_path = (
        args.libero_home / "libero/libero/benchmark/task_classification.json"
    )
    with classification_path.open(encoding="utf-8") as handle:
        classification = json.load(handle)
    suite_sizes = {name: len(classification[name]) for name in suites}
    spec = importlib.util.spec_from_file_location("libero_plus_eval_utils", args.eval_utils)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {args.eval_utils}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields = ("suite", "split_index", "split_count", "start_idx", "end_idx", "task_count")
    for job in module.plan_suite_jobs(suite_sizes, args.worker_count):
        print("\t".join(str(job[field]) for field in fields))


if __name__ == "__main__":
    main()
