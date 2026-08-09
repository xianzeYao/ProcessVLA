"""Materialization and local-model helpers for paired geometry probes."""

from __future__ import annotations

import csv
import gc
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from examples.simBenchmarks.CoT.geometry_probe.probe_utils import (
    SampleRef,
    canonicalize_uvd_prediction,
    masked_depth_metrics,
    uvd_trajectory_metrics,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def materialize_samples(
    store: Any,
    plan: Sequence[SampleRef],
    output_dir: str | Path,
) -> list[Path]:
    """Decode each selected dataset sample once and store a stable NPZ bundle."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for sample_index, ref in enumerate(plan):
        path = output_dir / f"sample_{sample_index:04d}.npz"
        if path.exists():
            paths.append(path)
            continue
        sample = store.load_sample(ref)
        images = [np.ascontiguousarray(image) for image in sample["example"]["image"]]
        payload: dict[str, Any] = {
            "image_count": np.asarray(len(images), dtype=np.int64),
            "language": np.asarray(str(sample["example"]["lang"])),
            "depth_current": np.asarray(sample["depth_current"], dtype=np.float32),
            "depth_future": np.asarray(sample["depth_future"], dtype=np.float32),
            "depth_current_valid": np.asarray(sample["depth_current_valid"], dtype=np.bool_),
            "depth_future_valid": np.asarray(sample["depth_future_valid"], dtype=np.bool_),
            "uvd": np.asarray(sample["uvd"], dtype=np.float32),
            "uvd_valid_mask": np.asarray(sample["uvd_valid_mask"], dtype=np.bool_),
            "uvd_out_of_frame_mask": np.asarray(
                sample.get("uvd_out_of_frame_mask", np.zeros_like(sample["uvd_valid_mask"])),
                dtype=np.bool_,
            ),
            "uvd_time": np.asarray(sample["uvd_time"], dtype=np.float32),
            "uvd_frame_indices": np.asarray(sample["uvd_frame_indices"], dtype=np.int64),
            "metadata_json": np.asarray(json.dumps(sample.get("metadata", {}), default=_json_default)),
        }
        for image_index, image in enumerate(images):
            payload[f"image_{image_index}"] = image
        np.savez_compressed(path, **payload)
        paths.append(path)
        if (sample_index + 1) % 10 == 0 or sample_index + 1 == len(plan):
            print(f"materialized {sample_index + 1}/{len(plan)} samples", flush=True)
    return paths


def load_materialized_sample(path: str | Path) -> dict[str, Any]:
    """Load a materialized sample without pickle-backed arrays."""

    with np.load(Path(path), allow_pickle=False) as payload:
        image_count = int(payload["image_count"])
        return {
            "images": [np.asarray(payload[f"image_{index}"]) for index in range(image_count)],
            "language": str(payload["language"].item()),
            "depth_current": np.asarray(payload["depth_current"], dtype=np.float32),
            "depth_future": np.asarray(payload["depth_future"], dtype=np.float32),
            "depth_current_valid": np.asarray(payload["depth_current_valid"], dtype=np.bool_),
            "depth_future_valid": np.asarray(payload["depth_future_valid"], dtype=np.bool_),
            "uvd": np.asarray(payload["uvd"], dtype=np.float32),
            "uvd_valid_mask": np.asarray(payload["uvd_valid_mask"], dtype=np.bool_),
            "uvd_out_of_frame_mask": np.asarray(payload["uvd_out_of_frame_mask"], dtype=np.bool_),
            "uvd_time": np.asarray(payload["uvd_time"], dtype=np.float32),
            "uvd_frame_indices": np.asarray(payload["uvd_frame_indices"], dtype=np.int64),
            "metadata": json.loads(str(payload["metadata_json"].item())),
        }


def build_geometry_example(sample: dict[str, Any]) -> dict[str, Any]:
    """Build an inference example with layout placeholders, never GT geometry coordinates."""

    target_shape = np.asarray(sample["uvd"]).shape
    if len(target_shape) not in (2, 3) or target_shape[-1] != 3:
        raise ValueError(f"materialized UVD target must end in 3, got {target_shape}")
    valid_shape = target_shape[:-1]
    return {
        "image": [np.ascontiguousarray(image) for image in sample["images"]],
        "lang": str(sample["language"]),
        "uvd": np.zeros(target_shape, dtype=np.float32),
        "uvd_valid_mask": np.ones(valid_shape, dtype=np.bool_),
        "uvd_time": np.asarray(sample["uvd_time"], dtype=np.float32),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _squeeze_depth(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value).astype(np.float32, copy=False)
    if array.ndim < 3 or array.shape[0] != 1:
        raise ValueError(f"{name} must have batch size 1, got {array.shape}")
    array = array[0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must resolve to [H,W], got {array.shape}")
    return array


def run_framework_predictions(
    framework: Any,
    *,
    label: str,
    sample_paths: Sequence[str | Path],
    output_dir: str | Path,
    autocast_device: str | None = None,
) -> list[Path]:
    """Run one already-loaded framework and save canonical per-sample predictions."""

    prediction_dir = Path(output_dir) / str(label)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    hand_count = int(getattr(framework, "uvd_hand_count", 1))
    points_per_hand = int(framework._trajectory_point_count())
    token_order = str(getattr(framework, "uvd_token_order", "hand_major"))
    output_paths: list[Path] = []
    for sample_index, sample_path in enumerate(sample_paths):
        output_path = prediction_dir / f"sample_{sample_index:04d}.npz"
        if output_path.exists():
            output_paths.append(output_path)
            continue
        sample = load_materialized_sample(sample_path)
        example = build_geometry_example(sample)
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if autocast_device == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            geometry = framework.predict_geometry([example])
        flat_uvd = _to_numpy(geometry["uvd"]).astype(np.float32, copy=False)
        if flat_uvd.ndim != 3 or flat_uvd.shape[0] != 1:
            raise ValueError(f"uvd output must have shape [1,N,3], got {flat_uvd.shape}")
        canonical_uvd = canonicalize_uvd_prediction(
            flat_uvd[0],
            points_per_hand=points_per_hand,
            hand_count=hand_count,
            token_order=token_order,
        )
        target_uvd = np.asarray(sample["uvd"])
        if target_uvd.ndim == 2 and canonical_uvd.shape[1] == 1:
            canonical_uvd = canonical_uvd[:, 0, :]
        if canonical_uvd.shape != np.asarray(sample["uvd"]).shape:
            raise ValueError(
                f"prediction/GT UVD layout mismatch: {canonical_uvd.shape} vs {np.asarray(sample['uvd']).shape}"
            )
        np.savez_compressed(
            output_path,
            depth_current=_squeeze_depth(geometry["depth_current"], name="depth_current"),
            depth_future=_squeeze_depth(geometry["depth_future"], name="depth_future"),
            uvd=canonical_uvd,
        )
        output_paths.append(output_path)
        if (sample_index + 1) % 10 == 0 or sample_index + 1 == len(sample_paths):
            print(f"{label}: predicted {sample_index + 1}/{len(sample_paths)} samples", flush=True)
    return output_paths


def strip_action_model_for_geometry(framework: Any) -> int:
    """Remove the action-only head and return its parameter storage in bytes."""
    if not callable(getattr(framework, "predict_geometry", None)):
        raise TypeError("framework must implement predict_geometry before geometry-only stripping")
    action_model = getattr(framework, "action_model", None)
    if action_model is None:
        return 0
    parameters = getattr(action_model, "parameters", lambda: ())()
    removed_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in parameters
    )
    delattr(framework, "action_model")
    return int(removed_bytes)


def run_checkpoint(
    checkpoint: str | Path,
    *,
    label: str,
    sample_paths: Sequence[str | Path],
    output_dir: str | Path,
    device: str,
    framework_loader: Any | None = None,
) -> list[Path]:
    """Load one checkpoint, write all predictions, and release its GPU memory."""

    if framework_loader is None:
        from starVLA.model.framework.base_framework import baseframework

        framework_loader = baseframework.from_pretrained
    framework = framework_loader(str(checkpoint))
    removed_bytes = strip_action_model_for_geometry(framework)
    print(
        f"geometry-only: removed action_model ({removed_bytes / 2**20:.1f} MiB)", flush=True
    )
    print(f"loaded {label} checkpoint: {checkpoint}", flush=True)
    framework = framework.to(device).eval()
    try:
        return run_framework_predictions(
            framework,
            label=label,
            sample_paths=sample_paths,
            output_dir=output_dir,
            autocast_device="cuda" if str(device).startswith("cuda") else None,
        )
    finally:
        del framework
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_prediction(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        return {
            "depth_current": np.asarray(payload["depth_current"], dtype=np.float32),
            "depth_future": np.asarray(payload["depth_future"], dtype=np.float32),
            "uvd": np.asarray(payload["uvd"], dtype=np.float32),
        }


def _flatten_sample_metrics(sample: dict[str, Any], prediction: dict[str, np.ndarray], image_size: int) -> dict[str, float | int]:
    current = masked_depth_metrics(
        prediction["depth_current"], sample["depth_current"], sample["depth_current_valid"]
    )
    future = masked_depth_metrics(
        prediction["depth_future"], sample["depth_future"], sample["depth_future_valid"]
    )
    uvd = uvd_trajectory_metrics(
        prediction["uvd"], sample["uvd"], sample["uvd_valid_mask"], image_size=image_size
    )
    flat: dict[str, float | int] = {}
    flat.update({f"depth_current_{key}": value for key, value in current.items()})
    flat.update({f"depth_future_{key}": value for key, value in future.items()})
    flat.update({f"uvd_{key}": value for key, value in uvd.items() if key != "per_hand"})
    for hand, hand_metrics in uvd["per_hand"].items():
        flat.update({f"uvd_{hand}_{key}": value for key, value in hand_metrics.items()})
    return flat


def _mean_numeric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    output: dict[str, float] = {}
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                values.append(float(value))
        if values:
            output[key] = float(np.mean(values))
    return output


def _metric_delta(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    return {
        key: float(second[key] - first[key])
        for key in sorted(set(first) & set(second))
    }


def write_paired_results(
    *,
    sample_paths: Sequence[str | Path],
    prediction_paths: dict[str, Sequence[str | Path]],
    labels: Sequence[str],
    output_dir: str | Path,
    image_size: int,
) -> dict[str, Any]:
    """Compute paired metrics and write recomputable JSONL/JSON/CSV artifacts."""

    labels = [str(label) for label in labels]
    if len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError(f"exactly two distinct labels are required, got {labels}")
    sample_paths = list(sample_paths)
    for label in labels:
        if label not in prediction_paths:
            raise KeyError(f"missing prediction paths for label={label!r}")
        if len(prediction_paths[label]) != len(sample_paths):
            raise ValueError(
                f"prediction count mismatch for {label}: {len(prediction_paths[label])} vs {len(sample_paths)}"
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    metric_rows: dict[str, list[dict[str, float | int]]] = {label: [] for label in labels}
    group_metric_rows: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    csv_rows: list[dict[str, Any]] = []

    for sample_index, sample_path in enumerate(sample_paths):
        sample = load_materialized_sample(sample_path)
        metadata = dict(sample["metadata"])
        group = str(metadata.get("suite", "unknown"))
        metrics_by_label: dict[str, dict[str, float | int]] = {}
        for label in labels:
            prediction = load_prediction(prediction_paths[label][sample_index])
            metrics = _flatten_sample_metrics(sample, prediction, int(image_size))
            metrics_by_label[label] = metrics
            metric_rows[label].append(metrics)
            group_metric_rows.setdefault(group, {name: [] for name in labels})[label].append(metrics)
        delta = _metric_delta(
            {key: float(value) for key, value in metrics_by_label[labels[0]].items()},
            {key: float(value) for key, value in metrics_by_label[labels[1]].items()},
        )
        row = {
            "sample_index": sample_index,
            **metadata,
            "metrics": metrics_by_label,
            f"delta_{labels[1]}_minus_{labels[0]}": delta,
        }
        rows.append(row)
        flat_csv: dict[str, Any] = {"sample_index": sample_index, **metadata}
        for label, metrics in metrics_by_label.items():
            flat_csv.update({f"{label}_{key}": value for key, value in metrics.items()})
        flat_csv.update({f"delta_{key}": value for key, value in delta.items()})
        csv_rows.append(flat_csv)

    overall = {label: _mean_numeric_rows(metric_rows[label]) for label in labels}
    overall[f"delta_{labels[1]}_minus_{labels[0]}"] = _metric_delta(overall[labels[0]], overall[labels[1]])
    by_group: dict[str, Any] = {}
    for group, grouped in group_metric_rows.items():
        group_summary = {label: _mean_numeric_rows(grouped[label]) for label in labels}
        group_summary[f"delta_{labels[1]}_minus_{labels[0]}"] = _metric_delta(
            group_summary[labels[0]], group_summary[labels[1]]
        )
        by_group[group] = group_summary
    summary = {"labels": labels, "sample_count": len(rows), "overall": overall, "by_group": by_group, "samples": rows}

    jsonl_path = output_dir / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    fieldnames = sorted({key for row in csv_rows for key in row})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return summary
