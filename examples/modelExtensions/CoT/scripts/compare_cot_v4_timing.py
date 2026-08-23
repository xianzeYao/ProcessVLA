#!/usr/bin/env python3
"""Compare stable V4 model time and memory with a matched baseline."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def summarize_metrics(
    path: Path,
    *,
    warmup_step: int,
) -> dict[str, float | bool]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stable = [
        row
        for row in records
        if int(row.get("step", -1)) >= int(warmup_step)
    ]
    model_times = [
        float(row["timing/model"])
        for row in stable
        if "timing/model" in row
    ]
    losses = [
        float(row["total_loss"])
        for row in stable
        if "total_loss" in row
    ]
    memory = [
        float(row["system/gpu_memory_max_allocated_gb"])
        for row in stable
        if "system/gpu_memory_max_allocated_gb" in row
    ]
    if not model_times or not losses or not memory:
        raise ValueError(
            f"stable metrics are incomplete in {path}"
        )
    median_model = statistics.median(model_times)
    if not math.isfinite(median_model) or median_model <= 0.0:
        raise ValueError(
            f"median model time must be positive in {path}"
        )
    return {
        "median_model_s": median_model,
        "max_allocated_gb": max(memory),
        "losses_finite": all(
            math.isfinite(value) for value in losses
        ),
    }


def compare_runs(
    baseline: Path,
    candidate: Path,
    *,
    warmup_step: int,
    max_model_ratio: float,
) -> dict[str, float | bool]:
    base = summarize_metrics(
        baseline,
        warmup_step=warmup_step,
    )
    cand = summarize_metrics(
        candidate,
        warmup_step=warmup_step,
    )
    ratio = (
        float(cand["median_model_s"])
        / float(base["median_model_s"])
    )
    return {
        "baseline_median_model_s": base["median_model_s"],
        "candidate_median_model_s": cand["median_model_s"],
        "model_time_ratio": ratio,
        "candidate_max_allocated_gb": cand["max_allocated_gb"],
        "passed": bool(
            ratio <= float(max_model_ratio)
            and base["losses_finite"]
            and cand["losses_finite"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--warmup-step", type=int, default=20)
    parser.add_argument("--max-model-ratio", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_runs(
        args.baseline,
        args.candidate,
        warmup_step=args.warmup_step,
        max_model_ratio=args.max_model_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
