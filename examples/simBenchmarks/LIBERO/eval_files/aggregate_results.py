#!/usr/bin/env python3
"""Aggregate one structured result JSON per standard LIBERO suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def aggregate_results(payloads: list[dict], expected_suites: tuple[str, ...] = DEFAULT_SUITES) -> dict:
    """Validate one result per suite and return suite plus overall metrics."""
    by_suite = {}
    for payload in payloads:
        suite = str(payload.get("suite", ""))
        if suite not in expected_suites:
            raise ValueError(f"unexpected suite in result: {suite!r}")
        if suite in by_suite:
            raise ValueError(f"duplicate result for suite: {suite}")
        by_suite[suite] = payload

    missing = [suite for suite in expected_suites if suite not in by_suite]
    if missing:
        raise ValueError(f"missing suite results: {missing}")

    episode_count = sum(int(item["episode_count"]) for item in by_suite.values())
    success_count = sum(int(item["success_count"]) for item in by_suite.values())
    overall = {
        "suite": "overall",
        "suites": list(expected_suites),
        "task_count": sum(int(item["task_count"]) for item in by_suite.values()),
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": success_count / episode_count if episode_count else 0.0,
    }
    return {"overall": overall, "suites": by_suite}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.result_dir.glob("*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = aggregate_results(payloads)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(args.output_path), "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
