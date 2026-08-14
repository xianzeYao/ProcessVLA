"""Pure metrics for canonical LIBERO V3 left/right/wrist UVD rollouts.

All UVD inputs use normalized ``(u, v, depth_m)`` values and canonical
``[time, landmark, 3]`` layout.  Landmark order is always left, right, wrist.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .probe_utils import uvd_pixel_scale


LANDMARK_NAMES = ("left", "right", "wrist")

def _strict_integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        qualifier = "positive integer" if positive else "integer"
        raise ValueError(f"{name} must be a {qualifier}, got {value!r}")
    value = int(value)
    if positive and value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _validate_v3_metadata(
    *,
    time_points: int,
    uvd_time: np.ndarray | None,
    uvd_landmark_ids: np.ndarray | None,
) -> None:
    if uvd_time is None and uvd_landmark_ids is None:
        return
    if uvd_time is None or uvd_landmark_ids is None:
        raise ValueError("uvd_time and uvd_landmark_ids must be supplied together")
    expected_count = time_points * len(LANDMARK_NAMES)
    time = np.asarray(uvd_time)
    landmark_ids = np.asarray(uvd_landmark_ids)
    if time.shape != (expected_count,) or landmark_ids.shape != (expected_count,):
        raise ValueError(
            "V3 metadata must be flattened [time, landmark] tokens with "
            f"shape {(expected_count,)}, got {time.shape} and {landmark_ids.shape}"
        )
    if not np.isfinite(time).all():
        raise ValueError("uvd_time must be finite")
    expected_landmarks = np.tile(np.arange(len(LANDMARK_NAMES)), time_points)
    if not np.array_equal(landmark_ids, expected_landmarks):
        raise ValueError("uvd_landmark_ids must be time-major [0, 1, 2] for every block")
    time_blocks = time.reshape(time_points, len(LANDMARK_NAMES))
    block_time = time_blocks[:, 0]
    if not np.array_equal(time_blocks, np.repeat(block_time[:, None], len(LANDMARK_NAMES), axis=1)):
        raise ValueError("each V3 time block must have the same uvd_time")
    if time_points > 1 and not np.all(np.diff(block_time) > 0):
        raise ValueError("V3 uvd_time blocks must be strictly increasing")



def canonicalize_v3_uvd(
    flat: np.ndarray,
    *,
    time_points: int = 4,
    landmarks: int = 3,
    uvd_time: np.ndarray | None = None,
    uvd_landmark_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Reshape validated time-major V3 tokens to canonical ``[T, L, 3]`` UVD."""

    time_points = _strict_integer(time_points, "time_points", positive=True)
    landmarks = _strict_integer(landmarks, "landmarks", positive=True)
    if landmarks != len(LANDMARK_NAMES):
        raise ValueError(f"V3 landmarks must be exactly 3, got {landmarks}")
    flat = np.asarray(flat, dtype=np.float32)
    expected = (time_points * landmarks, 3)
    if flat.shape != expected:
        raise ValueError(f"expected flattened V3 UVD shape {expected}, got {flat.shape}")
    _validate_v3_metadata(
        time_points=time_points,
        uvd_time=uvd_time,
        uvd_landmark_ids=uvd_landmark_ids,
    )
    return flat.reshape(time_points, landmarks, 3)


def align_realized_trace(
    step_uvd: np.ndarray,
    anchor: int,
    offsets: Sequence[int],
    *,
    step_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align one anchor's offsets to a realized trace without tail padding.

    Positions outside the realized episode are explicitly NaN and invalid.
    Finite status and an optional supplied validity mask are retained per
    landmark in-range.
    """

    step_uvd = np.asarray(step_uvd, dtype=np.float32)
    if step_uvd.ndim != 3 or step_uvd.shape[1:] != (len(LANDMARK_NAMES), 3):
        raise ValueError(f"step_uvd must have shape [step, 3, 3], got {step_uvd.shape}")
    anchor = _strict_integer(anchor, "anchor")
    raw_offsets = np.asarray(offsets)
    if raw_offsets.ndim != 1:
        raise ValueError(f"offsets must be one-dimensional, got {raw_offsets.shape}")
    offset_values = np.asarray([_strict_integer(value, "offset") for value in raw_offsets], dtype=np.int64)
    if (offset_values < 0).any():
        raise ValueError("offsets must be non-negative")
    if step_valid is None:
        step_valid_array = np.ones(step_uvd.shape[:2], dtype=np.bool_)
    else:
        step_valid_array = np.asarray(step_valid, dtype=np.bool_)
        if step_valid_array.shape != step_uvd.shape[:2]:
            raise ValueError(
                f"step_valid must have shape [step, 3], got {step_valid_array.shape}"
            )

    target = np.full((len(offset_values), len(LANDMARK_NAMES), 3), np.nan, dtype=np.float32)
    valid = np.zeros(target.shape[:2], dtype=np.bool_)
    for time_index, offset in enumerate(offset_values):
        step_index = anchor + int(offset)
        if not 0 <= step_index < len(step_uvd):
            continue
        target[time_index] = step_uvd[step_index]
        valid[time_index] = step_valid_array[step_index] & np.isfinite(step_uvd[step_index]).all(axis=-1)
    return target, valid


def backproject_uvd(
    uvd: np.ndarray,
    camera_k: np.ndarray,
    image_size: int | tuple[int, int],
) -> np.ndarray:
    """Backproject normalized UVD into camera XYZ, leaving invalid rows NaN.

    ``image_size`` follows the probe convention ``(height, width)``.  Thus U
    is multiplied by ``width - 1`` and V by ``height - 1`` before applying
    ``camera_k``.  Non-finite coordinates and non-positive depths are invalid.
    """

    uvd = np.asarray(uvd, dtype=np.float32)
    if uvd.shape[-1:] != (3,):
        raise ValueError(f"uvd must end in three coordinates, got {uvd.shape}")
    camera_k = np.asarray(camera_k, dtype=np.float64)
    if camera_k.shape != (3, 3) or not np.isfinite(camera_k).all():
        raise ValueError(f"camera_k must be finite [3,3], got {camera_k.shape}")
    try:
        inverse_k = np.linalg.inv(camera_k)
    except np.linalg.LinAlgError as error:
        raise ValueError("camera_k must be invertible") from error

    scale = uvd_pixel_scale(image_size).astype(np.float64)
    output = np.full(uvd.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(uvd).all(axis=-1) & (uvd[..., 2] > 0.0)
    if not valid.any():
        return output
    pixels = np.concatenate(
        (uvd[..., :2].astype(np.float64) * scale, np.ones((*uvd.shape[:-1], 1), dtype=np.float64)),
        axis=-1,
    )
    rays = pixels @ inverse_k.T
    output[valid] = (rays[valid] * uvd[..., 2:3][valid]).astype(np.float32)
    return output


def _validated_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.bool_)
    if prediction.ndim != 3 or prediction.shape[-1] != 3 or prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must share canonical [time, landmark, 3] shape, "
            f"got {prediction.shape} and {target.shape}"
        )
    if prediction.shape[1] != len(LANDMARK_NAMES):
        raise ValueError(f"expected three V3 landmarks, got {prediction.shape[1]}")
    if valid.shape != prediction.shape[:2]:
        raise ValueError(f"valid mask shape mismatch: {valid.shape} vs {prediction.shape[:2]}")
    return prediction, target, valid


def _projection_masks(uvd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projection_valid = np.isfinite(uvd).all(axis=-1) & (uvd[..., 2] > 0.0)
    in_frame = projection_valid & (uvd[..., 0] >= 0.0) & (uvd[..., 0] <= 1.0)
    in_frame &= (uvd[..., 1] >= 0.0) & (uvd[..., 1] <= 1.0)
    return projection_valid, in_frame


def _empty_landmark_metrics(*, projection_valid_count: int, in_frame_count: int) -> dict[str, float | int]:
    return {
        "valid_count": 0,
        "projection_valid_count": projection_valid_count,
        "in_frame_count": in_frame_count,
        "fde_valid_count": 0,
        "uv_ade_px": float("nan"),
        "uv_fde_px": float("nan"),
        "d_mae_mm": float("nan"),
        "delta_d_mae_mm": float("nan"),
        "delta_d_direction_accuracy": float("nan"),
    }


def _landmark_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    uv_scale: np.ndarray,
    depth_dead_zone_m: float,
    projection_valid: np.ndarray,
    in_frame: np.ndarray,
) -> dict[str, float | int]:
    target_valid = np.isfinite(target).all(axis=-1) & (target[:, 2] > 0.0)
    predicted_valid = np.isfinite(prediction).all(axis=-1) & (prediction[:, 2] > 0.0)
    requested = valid & target_valid
    metric_valid = requested & predicted_valid
    projection_valid_count = int((requested & projection_valid).sum())
    in_frame_count = int((requested & in_frame).sum())
    if not metric_valid.any():
        return _empty_landmark_metrics(
            projection_valid_count=projection_valid_count,
            in_frame_count=in_frame_count,
        )

    uv_error = (prediction[:, :2] - target[:, :2]) * uv_scale
    uv_l2 = np.linalg.norm(uv_error, axis=-1)
    depth_error_mm = np.abs(prediction[:, 2] - target[:, 2]) * 1000.0
    fde_valid_count = int(metric_valid[-1])
    uv_fde_px = float(uv_l2[-1]) if fde_valid_count else float("nan")
    adjacent = metric_valid[1:] & metric_valid[:-1]
    target_delta = target[1:, 2] - target[:-1, 2]
    prediction_delta = prediction[1:, 2] - prediction[:-1, 2]
    if adjacent.any():
        delta_d_mae_mm = float(np.abs(prediction_delta[adjacent] - target_delta[adjacent]).mean() * 1000.0)
    else:
        delta_d_mae_mm = float("nan")
    directional = adjacent & (np.abs(target_delta) > depth_dead_zone_m)
    if directional.any():
        direction_accuracy = float(
            (np.sign(prediction_delta[directional]) == np.sign(target_delta[directional])).mean()
        )
    else:
        direction_accuracy = float("nan")
    return {
        "valid_count": int(metric_valid.sum()),
        "projection_valid_count": projection_valid_count,
        "in_frame_count": in_frame_count,
        "fde_valid_count": fde_valid_count,
        "uv_ade_px": float(uv_l2[metric_valid].mean()),
        "uv_fde_px": uv_fde_px,
        "d_mae_mm": float(depth_error_mm[metric_valid].mean()),
        "delta_d_mae_mm": delta_d_mae_mm,
        "delta_d_direction_accuracy": direction_accuracy,
    }


def _aggregate_metrics(per_landmark: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    values = list(per_landmark.values())
    total_valid = sum(int(value["valid_count"]) for value in values)
    aggregate: dict[str, float | int] = {
        "valid_count": total_valid,
        "projection_valid_count": sum(int(value["projection_valid_count"]) for value in values),
        "in_frame_count": sum(int(value["in_frame_count"]) for value in values),
        "fde_valid_count": sum(int(value["fde_valid_count"]) for value in values),
    }
    for metric_name in ("uv_ade_px", "d_mae_mm"):
        if total_valid:
            aggregate[metric_name] = float(
                sum(float(value[metric_name]) * int(value["valid_count"]) for value in values if value["valid_count"])
                / total_valid
            )
        else:
            aggregate[metric_name] = float("nan")
    for metric_name in ("uv_fde_px", "delta_d_mae_mm", "delta_d_direction_accuracy"):
        finite = [float(value[metric_name]) for value in values if np.isfinite(value[metric_name])]
        aggregate[metric_name] = float(np.mean(finite)) if finite else float("nan")
    return aggregate


def _camera_derived_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    metric_valid: np.ndarray,
    *,
    camera_k: np.ndarray | None,
    image_size: int | tuple[int, int],
) -> dict[str, float | int]:
    unavailable = {
        "center_valid_count": 0,
        "center_mae_mm": float("nan"),
        "span_valid_count": 0,
        "span_mae_mm": float("nan"),
        "approach_valid_count": 0,
        "approach_mae_mm": float("nan"),
    }
    if camera_k is None:
        return unavailable
    pred_xyz = backproject_uvd(prediction, camera_k, image_size)
    target_xyz = backproject_uvd(target, camera_k, image_size)
    all_three = metric_valid.all(axis=1)
    if all_three.any():
        center_error = np.linalg.norm(pred_xyz.mean(axis=1) - target_xyz.mean(axis=1), axis=-1)
        center_valid_count = int(all_three.sum())
        center_mae_mm = float(center_error[all_three].mean() * 1000.0)
    else:
        center_valid_count = 0
        center_mae_mm = float("nan")
    span_valid = metric_valid[:, 0] & metric_valid[:, 1]
    if span_valid.any():
        pred_span = np.linalg.norm(pred_xyz[:, 1] - pred_xyz[:, 0], axis=-1)
        target_span = np.linalg.norm(target_xyz[:, 1] - target_xyz[:, 0], axis=-1)
        span_valid_count = int(span_valid.sum())
        span_mae_mm = float(np.abs(pred_span[span_valid] - target_span[span_valid]).mean() * 1000.0)
    else:
        span_valid_count = 0
        span_mae_mm = float("nan")
    approach_valid = all_three[1:] & all_three[:-1]
    if approach_valid.any():
        pred_center = pred_xyz.mean(axis=1)
        target_center = target_xyz.mean(axis=1)
        pred_approach = np.linalg.norm(pred_center[1:] - pred_center[:-1], axis=-1)
        target_approach = np.linalg.norm(target_center[1:] - target_center[:-1], axis=-1)
        approach_valid_count = int(approach_valid.sum())
        approach_mae_mm = float(
            np.abs(pred_approach[approach_valid] - target_approach[approach_valid]).mean() * 1000.0
        )
    else:
        approach_valid_count = 0
        approach_mae_mm = float("nan")
    return {
        "center_valid_count": center_valid_count,
        "center_mae_mm": center_mae_mm,
        "span_valid_count": span_valid_count,
        "span_mae_mm": span_mae_mm,
        "approach_valid_count": approach_valid_count,
        "approach_mae_mm": approach_mae_mm,
    }



def metrics_to_jsonable(value: object) -> object:
    """Convert metric output to strict JSON while preserving invalid gaps as null.

    Internal metric dictionaries use NaN for plotting gaps. Call this adapter
    before writing rollout or summary JSON, then use ``allow_nan=False``.
    """

    if isinstance(value, np.ndarray):
        return metrics_to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return metrics_to_jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): metrics_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metrics_to_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def compute_anchor_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    image_size: int | tuple[int, int],
    depth_dead_zone_m: float = 0.002,
    camera_k: np.ndarray | None = None,
    uvd_time: np.ndarray | None = None,
    uvd_landmark_ids: np.ndarray | None = None,
) -> dict[str, object]:
    """Score one canonical three-landmark prediction horizon.

    UV ADE/FDE are pixels, all depth and camera-space errors are millimetres,
    and the depth direction metric ignores target changes within the supplied
    dead-zone.  The persistence baseline repeats the target at time zero and
    uses the identical validity horizon; it never uses future predictions.
    """

    if depth_dead_zone_m < 0.0:
        raise ValueError(f"depth_dead_zone_m must be non-negative, got {depth_dead_zone_m}")
    prediction, target, valid = _validated_arrays(prediction, target, valid)
    _validate_v3_metadata(
        time_points=prediction.shape[0],
        uvd_time=uvd_time,
        uvd_landmark_ids=uvd_landmark_ids,
    )
    scale = uvd_pixel_scale(image_size)
    projection_valid, in_frame = _projection_masks(prediction)
    per_landmark = {
        name: _landmark_metrics(
            prediction[:, index],
            target[:, index],
            valid[:, index],
            uv_scale=scale,
            depth_dead_zone_m=float(depth_dead_zone_m),
            projection_valid=projection_valid[:, index],
            in_frame=in_frame[:, index],
        )
        for index, name in enumerate(LANDMARK_NAMES)
    }
    target_valid = np.isfinite(target).all(axis=-1) & (target[..., 2] > 0.0)
    predicted_valid = np.isfinite(prediction).all(axis=-1) & (prediction[..., 2] > 0.0)
    metric_valid = valid & target_valid & predicted_valid
    persistence_prediction = np.broadcast_to(target[:1], target.shape).copy()
    persistence_projection, persistence_in_frame = _projection_masks(persistence_prediction)
    persistence = {
        name: _landmark_metrics(
            persistence_prediction[:, index],
            target[:, index],
            valid[:, index],
            uv_scale=scale,
            depth_dead_zone_m=float(depth_dead_zone_m),
            projection_valid=persistence_projection[:, index],
            in_frame=persistence_in_frame[:, index],
        )
        for index, name in enumerate(LANDMARK_NAMES)
    }
    return {
        **per_landmark,
        "aggregate": _aggregate_metrics(per_landmark),
        "persistence": {**persistence, "aggregate": _aggregate_metrics(persistence)},
        "camera": _camera_derived_metrics(
            prediction,
            target,
            metric_valid,
            camera_k=camera_k,
            image_size=image_size,
        ),
    }
