#!/usr/bin/env python3
"""Run the Stage A hidden-geometry intervention probe on materialized samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.simBenchmarks.CoT.geometry_probe.trace_intervention_probe import (
    run_trace_intervention_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Training YAML paired with the checkpoint")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--repeat-tolerance",
        type=float,
        default=1e-6,
        help="Maximum allowed correct-condition repeat action error",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "native_only",
            "zero_geometry",
            "within_task_shuffle",
            "cross_task_swap",
            "uvd_only",
            "depth_only",
        ],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.samples_dir.is_dir():
        raise NotADirectoryError(args.samples_dir)
    sample_paths = sorted(args.samples_dir.glob("sample_*.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"no sample_*.npz files in {args.samples_dir}")
    summary = run_trace_intervention_checkpoint(
        args.checkpoint,
        sample_paths=sample_paths,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        variants=args.variants,
        bootstrap_resamples=args.bootstrap_resamples,
        repeat_tolerance=args.repeat_tolerance,
        config_path=args.config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
