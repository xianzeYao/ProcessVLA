"""Aggregate RoboCasa task-level result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.simBenchmarks.Robocasa_tabletop.eval_files.robocasa_eval_protocol import (
    aggregate_task_payloads,
    write_json,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-task-count", type=int, default=24)
    parser.add_argument("--expected-num-episodes", type=int, default=50)
    parser.add_argument("payloads", nargs="+", type=Path)
    args = parser.parse_args(argv)

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.payloads]
    aggregate = aggregate_task_payloads(
        payloads,
        expected_task_count=args.expected_task_count,
        expected_num_episodes=args.expected_num_episodes,
    )
    output = {
        "schema_version": 1,
        "num_tasks": aggregate["num_tasks"],
        "total_episodes": aggregate["total_episodes"],
        "total_successes": aggregate["total_successes"],
        "macro_success_rate": aggregate["macro_success_rate"],
        "micro_success_rate": aggregate["micro_success_rate"],
        "task_success_rates": aggregate["task_success_rates"],
        "task_results": aggregate["task_results"],
    }
    write_json(args.output, output)
    print(
        json.dumps(
            {k: output[k] for k in ("num_tasks", "total_episodes", "macro_success_rate", "micro_success_rate")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
