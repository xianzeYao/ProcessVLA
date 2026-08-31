"""Compare two RoboCasa action–trace consistency evaluation directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from examples.simBenchmarks.Robocasa_tabletop.eval_files.trace_consistency import (
    load_trace_records,
    summarize_trace_records,
)


_METRIC_KEYS = (
    "u_mae_px",
    "v_mae_px",
    "uv_l2_mean_px",
    "depth_mae_m",
    "depth_rmse_m",
)


def summarize_trace_directory(path: str | Path) -> dict[str, Any]:
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"trace directory does not exist: {directory}")
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise ValueError(f"trace directory contains no JSONL task files: {directory}")
    records = []
    for source in files:
        records.extend(load_trace_records(source))
    summary = summarize_trace_records(records)
    summary["task_file_count"] = len(files)
    return summary


def _metric_delta(
    baseline: Mapping[str, Any], ablation: Mapping[str, Any]
) -> dict[str, Any]:
    result = {
        "sample_count_baseline": int(baseline["sample_count"]),
        "sample_count_ablation": int(ablation["sample_count"]),
    }
    for key in _METRIC_KEYS:
        baseline_value = baseline.get(key)
        ablation_value = ablation.get(key)
        result[key] = (
            None
            if baseline_value is None or ablation_value is None
            else float(ablation_value) - float(baseline_value)
        )
    return result


def _summary_delta(
    baseline: Mapping[str, Any], ablation: Mapping[str, Any]
) -> dict[str, Any]:
    result = {
        "direct": _metric_delta(baseline["direct"], ablation["direct"]),
        "interpolated": _metric_delta(
            baseline["interpolated"], ablation["interpolated"]
        ),
        "per_hand": {},
        "per_offset": {},
    }
    for name in sorted(set(baseline["per_hand"]) & set(ablation["per_hand"])):
        result["per_hand"][name] = _metric_delta(
            baseline["per_hand"][name], ablation["per_hand"][name]
        )
    for offset in sorted(
        set(baseline["per_offset"]) & set(ablation["per_offset"]),
        key=int,
    ):
        result["per_offset"][offset] = _metric_delta(
            baseline["per_offset"][offset], ablation["per_offset"][offset]
        )
    return result


def compare_trace_directories(
    baseline_dir: str | Path,
    ablation_dir: str | Path,
    *,
    baseline_label: str = "baseline",
    ablation_label: str = "ablation",
) -> dict[str, Any]:
    baseline_summary = summarize_trace_directory(baseline_dir)
    ablation_summary = summarize_trace_directory(ablation_dir)
    for key in ("action_horizon", "image_size"):
        if baseline_summary[key] != ablation_summary[key]:
            raise ValueError(
                f"incompatible {key}: baseline={baseline_summary[key]} "
                f"ablation={ablation_summary[key]}"
            )
    return {
        "schema_version": 1,
        "baseline": {
            "label": str(baseline_label),
            "path": str(Path(baseline_dir)),
            "summary": baseline_summary,
        },
        "ablation": {
            "label": str(ablation_label),
            "path": str(Path(ablation_dir)),
            "summary": ablation_summary,
        },
        "ablation_minus_baseline": _summary_delta(
            baseline_summary, ablation_summary
        ),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_trace_dir")
    parser.add_argument("ablation_trace_dir")
    parser.add_argument("--baseline-label", default="v2_q32_nodepthcond")
    parser.add_argument("--ablation-label", default="without_current_depth")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    comparison = compare_trace_directories(
        args.baseline_trace_dir,
        args.ablation_trace_dir,
        baseline_label=args.baseline_label,
        ablation_label=args.ablation_label,
    )
    _write_json_atomic(Path(args.output), comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
