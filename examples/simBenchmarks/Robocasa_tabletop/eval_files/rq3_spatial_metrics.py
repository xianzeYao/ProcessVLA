#!/usr/bin/env python3
"""Offline RQ3 trace-consistency metrics for RoboCasa rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TRACE_METRICS = (
    "uv_consistency_px",
    "depth_consistency_mm",
    "uvd_consistency",
)
DEFAULT_OFFSETS = (0, 3, 6, 10, 13, 16)


def compute_trace_point_errors(
    predicted_uvd: Sequence[float],
    realized_uvd: Sequence[float],
    *,
    image_size: int,
    depth_scale_m: float,
) -> dict[str, float]:
    """Return coordinate-wise trace consistency errors for one valid point."""
    predicted = np.asarray(predicted_uvd, dtype=np.float64)
    realized = np.asarray(realized_uvd, dtype=np.float64)
    if predicted.shape != (3,) or realized.shape != (3,):
        raise ValueError(
            f"predicted/realized UVD must each have shape (3,), got "
            f"{predicted.shape}/{realized.shape}"
        )
    if not np.isfinite(predicted).all() or not np.isfinite(realized).all():
        raise ValueError("predicted/realized UVD must be finite")
    if image_size < 2:
        raise ValueError(f"image_size must be at least 2, got {image_size}")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError(f"depth_scale_m must be positive, got {depth_scale_m}")
    du, dv, dd = np.abs(predicted - realized)
    pixel_scale = float(image_size - 1)
    return {
        "uv_consistency_px": float(pixel_scale * (du + dv) / 2.0),
        "depth_consistency_mm": float(1000.0 * depth_scale_m * dd),
        "uvd_consistency": float((du + dv + dd) / 3.0),
    }


def _episode_metric_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[int, np.ndarray]]:
    grouped: dict[int, dict[int, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row["task_index"])][int(row["episode_index"])].append(
            [float(row[name]) for name in TRACE_METRICS]
        )
    return {
        task: {
            episode: np.asarray(values, dtype=np.float64).mean(axis=0)
            for episode, values in episodes.items()
        }
        for task, episodes in grouped.items()
    }


def _bootstrap_task_stratified_episode_means(
    episode_metrics: Mapping[int, Mapping[int, np.ndarray]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, list[float]]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    replicate_task_means = []
    for task in sorted(episode_metrics):
        values = np.stack(list(episode_metrics[task].values()), axis=0)
        draws = rng.integers(
            0,
            len(values),
            size=(bootstrap_samples, len(values)),
        )
        replicate_task_means.append(values[draws].mean(axis=1))
    replicates = np.stack(replicate_task_means, axis=1).mean(axis=1)
    low, high = np.quantile(replicates, [0.025, 0.975], axis=0)
    return {
        name: [float(low[index]), float(high[index])]
        for index, name in enumerate(TRACE_METRICS)
    }


def aggregate_hierarchical_errors(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 7,
) -> dict[str, Any]:
    """Aggregate point errors using episode-first, task-macro semantics."""
    if not rows:
        raise ValueError("cannot aggregate an empty trace row collection")
    point_values = np.asarray(
        [[float(row[name]) for name in TRACE_METRICS] for row in rows],
        dtype=np.float64,
    )
    episode_metrics = _episode_metric_arrays(rows)
    task_means = {
        task: np.stack(list(episodes.values()), axis=0).mean(axis=0)
        for task, episodes in episode_metrics.items()
    }
    macro = np.stack(list(task_means.values()), axis=0).mean(axis=0)
    micro = point_values.mean(axis=0)
    return {
        "task_macro": {
            name: float(macro[index]) for index, name in enumerate(TRACE_METRICS)
        },
        "micro": {
            name: float(micro[index]) for index, name in enumerate(TRACE_METRICS)
        },
        "bootstrap_95_ci": _bootstrap_task_stratified_episode_means(
            episode_metrics,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "counts": {
            "valid_points": int(len(rows)),
            "valid_episodes": int(sum(len(value) for value in episode_metrics.values())),
            "valid_tasks": int(len(episode_metrics)),
        },
    }


def _manifest_task_names(manifest: Mapping[str, Any]) -> dict[int, str]:
    return {
        int(task["task_index"]): str(task["env_name"])
        for task in manifest.get("tasks", [])
    }


def load_trace_run(
    run_dir: str | Path,
    *,
    variant: str,
    expected_offsets: Sequence[int] = DEFAULT_OFFSETS,
    depth_scale_m: float = 1.0,
) -> dict[str, Any]:
    """Load direct trace records and expand every valid finite hand point."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing rollout manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    task_names = _manifest_task_names(manifest)
    trace_files = sorted((run_dir / "trace_consistency").glob("task_*.jsonl"))
    if not trace_files:
        raise FileNotFoundError(f"no task_*.jsonl under {run_dir / 'trace_consistency'}")

    expected_offsets_tuple = tuple(int(value) for value in expected_offsets)
    point_rows: list[dict[str, Any]] = []
    observed_tasks: set[int] = set()
    observed_episodes: dict[int, set[int]] = defaultdict(set)
    observed_offsets: dict[int, set[int]] = defaultdict(set)
    decision_ids: set[tuple[int, int, int]] = set()
    duplicate_decisions = 0
    offset_patterns: Counter[tuple[int, ...]] = Counter()
    image_sizes: Counter[int] = Counter()
    record_count = 0
    candidate_point_count = 0
    invalid_mask_count = 0
    invalid_nonfinite_count = 0
    abs_error_mismatch_count = 0

    for trace_path in trace_files:
        with trace_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_count += 1
                task = int(record["task_index"])
                episode = int(record["episode_index"])
                decision = int(record["decision_index"])
                decision_id = (task, episode, decision)
                if decision_id in decision_ids:
                    duplicate_decisions += 1
                decision_ids.add(decision_id)
                observed_tasks.add(task)
                observed_episodes[task].add(episode)
                image_size = int(record["image_size"])
                image_sizes[image_size] += 1

                direct = record.get("direct")
                if not isinstance(direct, Mapping):
                    raise ValueError(f"{trace_path}:{line_number} has no direct mapping")
                offsets = tuple(int(value) for value in direct["offsets"])
                offset_patterns[offsets] += 1
                if offsets != expected_offsets_tuple:
                    raise ValueError(
                        f"{trace_path}:{line_number} direct offsets {offsets} do not "
                        f"match expected {expected_offsets_tuple}"
                    )
                observed_offsets[task].update(offsets)
                predicted = np.asarray(direct["predicted_uvd"], dtype=np.float64)
                realized = np.asarray(direct["realized_uvd"], dtype=np.float64)
                valid = np.asarray(direct["valid"], dtype=bool)
                stored_abs = np.asarray(direct["abs_error"], dtype=np.float64)
                expected_shape = (len(offsets), 2, 3)
                if predicted.shape != expected_shape or realized.shape != expected_shape:
                    raise ValueError(
                        f"{trace_path}:{line_number} expected predicted/realized shape "
                        f"{expected_shape}, got {predicted.shape}/{realized.shape}"
                    )
                if valid.shape != expected_shape[:2] or stored_abs.shape != expected_shape:
                    raise ValueError(
                        f"{trace_path}:{line_number} invalid valid/abs_error shapes "
                        f"{valid.shape}/{stored_abs.shape}"
                    )
                finite_pairs = np.isfinite(predicted).all(axis=-1) & np.isfinite(realized).all(axis=-1)
                candidate_point_count += int(valid.size)
                invalid_mask_count += int((~valid).sum())
                invalid_nonfinite_count += int((valid & ~finite_pairs).sum())
                comparable = valid & finite_pairs & np.isfinite(stored_abs).all(axis=-1)
                if comparable.any():
                    mismatch = ~np.isclose(
                        stored_abs[comparable],
                        np.abs(predicted - realized)[comparable],
                        rtol=1e-5,
                        atol=1e-7,
                    ).all(axis=-1)
                    abs_error_mismatch_count += int(mismatch.sum())
                keep = valid & finite_pairs
                for offset_index, hand in np.argwhere(keep):
                    errors = compute_trace_point_errors(
                        predicted[offset_index, hand],
                        realized[offset_index, hand],
                        image_size=image_size,
                        depth_scale_m=depth_scale_m,
                    )
                    point_rows.append(
                        {
                            "variant": variant,
                            "task_index": task,
                            "task_name": task_names.get(task, ""),
                            "episode_index": episode,
                            "decision_index": decision,
                            "offset": int(offsets[offset_index]),
                            "hand": int(hand),
                            **errors,
                        }
                    )

    expected_tasks = set(task_names)
    expected_episode_count = int(manifest.get("num_episodes", 0))
    missing_episodes = {
        str(task): sorted(set(range(expected_episode_count)) - observed_episodes[task])
        for task in sorted(expected_tasks | observed_tasks)
        if set(range(expected_episode_count)) - observed_episodes[task]
    }
    missing_offsets = {
        str(task): sorted(set(expected_offsets_tuple) - observed_offsets[task])
        for task in sorted(expected_tasks | observed_tasks)
        if set(expected_offsets_tuple) - observed_offsets[task]
    }
    return {
        "manifest": manifest,
        "point_rows": point_rows,
        "validation": {
            "used_interpolated": False,
            "trace_file_count": len(trace_files),
            "record_count": record_count,
            "candidate_point_count": candidate_point_count,
            "valid_point_count": len(point_rows),
            "invalid_by_direct_valid_count": invalid_mask_count,
            "invalid_nonfinite_count": invalid_nonfinite_count,
            "abs_error_mismatch_count": abs_error_mismatch_count,
            "duplicate_decision_count": duplicate_decisions,
            "image_size_record_counts": {
                str(key): value for key, value in sorted(image_sizes.items())
            },
            "direct_offset_pattern_counts": {
                ",".join(map(str, key)): value
                for key, value in sorted(offset_patterns.items())
            },
            "missing_task_indices": sorted(expected_tasks - observed_tasks),
            "unexpected_task_indices": sorted(observed_tasks - expected_tasks),
            "missing_record_episode_ids_by_task": missing_episodes,
            "missing_offsets_by_task": missing_offsets,
        },
    }


def _task_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    task_names: Mapping[int, str],
) -> list[dict[str, Any]]:
    output = []
    tasks = sorted({int(row["task_index"]) for row in rows})
    for task in tasks:
        selected = [row for row in rows if int(row["task_index"]) == task]
        aggregate = aggregate_hierarchical_errors(
            selected, bootstrap_samples=10_000, seed=7 + task
        )
        output.append(
            {
                "variant": variant,
                "task_index": task,
                "task_name": task_names.get(task, ""),
                **aggregate["counts"],
                **{f"task_episode_mean_{key}": value for key, value in aggregate["task_macro"].items()},
                **{f"point_micro_{key}": value for key, value in aggregate["micro"].items()},
            }
        )
    return output


def _offset_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    expected_offsets: Sequence[int],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for offset in expected_offsets:
        selected = [row for row in rows if int(row["offset"]) == int(offset)]
        aggregate = aggregate_hierarchical_errors(
            selected,
            bootstrap_samples=bootstrap_samples,
            seed=seed + int(offset),
        )
        flat_ci = {
            f"{metric}_ci95_low": aggregate["bootstrap_95_ci"][metric][0]
            for metric in TRACE_METRICS
        }
        flat_ci.update(
            {
                f"{metric}_ci95_high": aggregate["bootstrap_95_ci"][metric][1]
                for metric in TRACE_METRICS
            }
        )
        output.append(
            {
                "variant": variant,
                "offset": int(offset),
                **aggregate["counts"],
                **{f"task_macro_{key}": value for key, value in aggregate["task_macro"].items()},
                **{f"micro_{key}": value for key, value in aggregate["micro"].items()},
                **flat_ci,
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_trace_cli(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_offsets = tuple(int(value) for value in args.expected_offsets.split(","))
    summary: dict[str, Any] = {
        "schema_version": 1,
        "definition": {
            "source": "direct only; interpolated is never read",
            "validity": "direct.valid and finite predicted/realized UVD",
            "uv_consistency_px": "((W-1)*abs(du) + (H-1)*abs(dv))/2",
            "depth_consistency_mm": "1000*depth_scale_m*abs(dd)",
            "uvd_consistency": "(abs(du)+abs(dv)+abs(dd))/3 in normalized model space",
            "aggregation": "point -> episode mean -> task mean -> equal-weight task macro",
            "bootstrap": "tasks fixed; episodes resampled with replacement within each task",
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(args.seed),
            "expected_offsets": list(expected_offsets),
            "depth_scale_m": float(args.depth_scale_m),
        },
        "variants": {},
    }
    all_task_rows: list[dict[str, Any]] = []
    all_offset_rows: list[dict[str, Any]] = []
    for run_spec in args.run:
        if "=" not in run_spec:
            raise ValueError(f"--run must be VARIANT=PATH, got {run_spec!r}")
        variant, raw_path = run_spec.split("=", 1)
        loaded = load_trace_run(
            raw_path,
            variant=variant,
            expected_offsets=expected_offsets,
            depth_scale_m=float(args.depth_scale_m),
        )
        rows = loaded["point_rows"]
        aggregate = aggregate_hierarchical_errors(
            rows,
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.seed),
        )
        manifest = loaded["manifest"]
        task_names = _manifest_task_names(manifest)
        summary["variants"][variant] = {
            "run_dir": str(Path(raw_path).resolve()),
            "checkpoint": str(manifest.get("checkpoint", "")),
            "checkpoint_step": _checkpoint_step(str(manifest.get("checkpoint", ""))),
            **aggregate,
            "validation": loaded["validation"],
        }
        all_task_rows.extend(
            _task_rows(rows, variant=variant, task_names=task_names)
        )
        all_offset_rows.extend(
            _offset_rows(
                rows,
                variant=variant,
                expected_offsets=expected_offsets,
                bootstrap_samples=int(args.bootstrap_samples),
                seed=int(args.seed),
            )
        )
        del rows, loaded

    summary_path = output_dir / "trace_consistency_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(output_dir / "trace_consistency_by_task.csv", all_task_rows)
    _write_csv(output_dir / "trace_consistency_by_offset.csv", all_offset_rows)
    print(f"wrote {summary_path}")


def _checkpoint_step(path: str) -> int | None:
    stem = Path(path).stem
    prefix = "steps_"
    if not stem.startswith(prefix):
        return None
    remainder = stem[len(prefix) :].split("_", 1)[0]
    return int(remainder) if remainder.isdigit() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    trace = subparsers.add_parser("trace", help="recompute direct trace consistency")
    trace.add_argument("--run", action="append", required=True, metavar="VARIANT=PATH")
    trace.add_argument("--output-dir", required=True)
    trace.add_argument("--expected-offsets", default=",".join(map(str, DEFAULT_OFFSETS)))
    trace.add_argument("--depth-scale-m", type=float, default=1.0)
    trace.add_argument("--bootstrap-samples", type=int, default=10_000)
    trace.add_argument("--seed", type=int, default=7)
    trace.set_defaults(func=run_trace_cli)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
