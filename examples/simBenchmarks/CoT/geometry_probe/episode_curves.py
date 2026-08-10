"""Immutable full-episode curve context for geometry comparison videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EpisodeCurveContext:
    """Episode-wide UVD curves and visual limits shared by every video frame."""

    timestamps: np.ndarray
    gt: np.ndarray
    predictions: dict[str, np.ndarray]
    valid: np.ndarray
    depth_limits: tuple[tuple[float, float], ...]
    uv_limits: tuple[tuple[float, float], tuple[float, float]]
    depth_image_limits: tuple[float, float]
    depth_error_limit: float


def _canonical_uvd(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[-1] == 3:
        return array[:, None, :]
    if array.ndim == 3 and array.shape[-1] == 3:
        return array
    raise ValueError(f"UVD must have shape [T,3] or [T,H,3], got {array.shape}")


def _canonical_valid(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.bool_)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    raise ValueError(f"UVD validity must have shape [T] or [T,H], got {array.shape}")


def _padded_limits(values: Sequence[np.ndarray], *, minimum_span: float = 1e-3) -> tuple[float, float]:
    finite = [np.asarray(value)[np.isfinite(value)] for value in values]
    finite = [value for value in finite if value.size]
    if not finite:
        return 0.0, 1.0
    merged = np.concatenate(finite)
    low = float(merged.min())
    high = float(merged.max())
    span = max(high - low, float(minimum_span))
    margin = 0.05 * span
    return low - margin, high + margin


def build_episode_curve_context(
    samples: Sequence[Mapping[str, Any]],
    predictions_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Sequence[str],
) -> EpisodeCurveContext:
    """Stitch horizon-zero UVD points into stable full-episode curves."""

    labels = tuple(str(label) for label in labels)
    if not samples:
        raise ValueError("at least one episode sample is required")
    if len(labels) != 2 or any(label not in predictions_by_label for label in labels):
        raise ValueError(f"two available prediction labels are required, got {labels}")
    count = len(samples)
    if any(len(predictions_by_label[label]) != count for label in labels):
        raise ValueError("sample and prediction counts must match")

    first_gt = _canonical_uvd(samples[0]["uvd"])
    hands = first_gt.shape[1]
    gt_points: list[np.ndarray] = []
    valid_points: list[np.ndarray] = []
    timestamps: list[float] = []
    prediction_points: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    depth_arrays: list[np.ndarray] = []
    error_arrays: list[np.ndarray] = []

    for index, sample in enumerate(samples):
        gt = _canonical_uvd(sample["uvd"])
        valid = _canonical_valid(sample["uvd_valid_mask"])
        if gt.shape[1] != hands or valid.shape != gt.shape[:2]:
            raise ValueError(
                f"inconsistent GT UVD/validity at sample {index}: {gt.shape}/{valid.shape}"
            )
        point_valid = valid[0].copy()
        point_gt = gt[0].copy()
        point_gt[~point_valid] = np.nan
        gt_points.append(point_gt)
        valid_points.append(point_valid)
        timestamps.append(float(sample.get("metadata", {}).get("timestamp", index)))

        gt_depths = (
            np.asarray(sample["depth_current"], dtype=np.float32),
            np.asarray(sample["depth_future"], dtype=np.float32),
        )
        depth_arrays.extend(gt_depths)
        for label in labels:
            prediction = predictions_by_label[label][index]
            trajectory = _canonical_uvd(prediction["uvd"])
            if trajectory.shape[1] != hands:
                raise ValueError(
                    f"inconsistent predicted hands for {label} sample {index}: {trajectory.shape}"
                )
            point = trajectory[0].copy()
            point[~point_valid] = np.nan
            prediction_points[label].append(point)
            predicted_depths = (
                np.asarray(prediction["depth_current"], dtype=np.float32),
                np.asarray(prediction["depth_future"], dtype=np.float32),
            )
            depth_arrays.extend(predicted_depths)
            error_arrays.extend(
                np.abs(predicted - target)
                for predicted, target in zip(predicted_depths, gt_depths)
            )

    gt_array = np.asarray(gt_points, dtype=np.float32)
    valid_array = np.asarray(valid_points, dtype=np.bool_)
    prediction_arrays = {
        label: np.asarray(prediction_points[label], dtype=np.float32) for label in labels
    }
    all_curves = [gt_array, *prediction_arrays.values()]
    depth_limits = tuple(
        _padded_limits([curve[:, hand, 2] for curve in all_curves])
        for hand in range(hands)
    )
    uv_limits = (
        _padded_limits([curve[..., 0] for curve in all_curves]),
        _padded_limits([curve[..., 1] for curve in all_curves]),
    )
    depth_image_limits = _padded_limits(depth_arrays)
    finite_errors = [array[np.isfinite(array)] for array in error_arrays]
    finite_errors = [array for array in finite_errors if array.size]
    if finite_errors:
        depth_error_limit = max(float(np.concatenate(finite_errors).max()) * 1.05, 1e-6)
    else:
        depth_error_limit = 1.0
    return EpisodeCurveContext(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        gt=gt_array,
        predictions=prediction_arrays,
        valid=valid_array,
        depth_limits=depth_limits,
        uv_limits=uv_limits,
        depth_image_limits=depth_image_limits,
        depth_error_limit=depth_error_limit,
    )
