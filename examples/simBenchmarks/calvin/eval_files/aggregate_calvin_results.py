"""Aggregate split-aware CALVIN worker JSON payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.simBenchmarks.calvin.eval_files.calvin_eval_protocol import aggregate_worker_payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("payloads", nargs="+", type=Path)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text()) for path in args.payloads]
    aggregate = aggregate_worker_payloads(payloads)
    metadata = {
        "schema_version": 1,
        "split": payloads[0].get("split", "unknown"),
        "sequence_start": 0,
        "sequence_end": aggregate["num_sequences"],
        "action_chunk_size": payloads[0].get("action_chunk_size"),
        "action_stride": payloads[0].get("action_stride"),
        **aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("split", "num_sequences", "success_rates", "average_tasks_completed")}))


if __name__ == "__main__":
    main()
