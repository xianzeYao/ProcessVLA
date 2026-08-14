"""Pure utilities for the LIBERO depth/UVD inference probe.

The runner intentionally keeps sampling and metric code independent from
video, websocket, and simulator dependencies so these invariants can be
tested on a CPU-only environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

LATENCY_STAGE_FIELDS = (
    "server_total_ms",
    "preprocess_ms",
    "qwen_backbone_ms",
    "image_token_extract_ms",
    "query_reasoner_ms",
    "depth_current_ms",
    "depth_future_ms",
    "uvd_head_ms",
    "action_expert_ms",
    "action_output_transfer_ms",
    "action_unnormalize_ms",
    "geometry_output_transfer_ms",
    "output_transfer_ms",
)


def build_latency_fields(
    client_roundtrip_ms: float,
    server_timing: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build raw per-request latency fields without aggregate statistics."""

    client_roundtrip_ms = float(client_roundtrip_ms)
    fields: dict[str, float] = {
        # Keep the historical name for existing consumers.
        "latency_ms": client_roundtrip_ms,
        "client_roundtrip_ms": client_roundtrip_ms,
    }
    if server_timing is not None:
        for name in LATENCY_STAGE_FIELDS:
            if name in server_timing:
                fields[name] = float(server_timing[name])
    if "server_total_ms" in fields:
        fields["transport_overhead_ms"] = max(
            client_roundtrip_ms - fields["server_total_ms"], 0.0
        )

    return fields

@dataclass(frozen=True)
class EpisodeRef:
    suite: str
    episode_id: int
    episode_length: int


@dataclass(frozen=True)
class SampleRef:
    suite: str
    episode_id: int
    episode_length: int
    frame_index: int


def allocate_balanced_counts(total: int, suites: Sequence[str]) -> dict[str, int]:
    """Allocate ``total`` samples across suites in input order.

    The first ``total % len(suites)`` suites receive one extra sample.  The
    deterministic order makes a run reproducible and produces 13/13/12/12
    for 50 samples over the four standard LIBERO suites.
    """

    total = int(total)
    suites = list(suites)
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if not suites:
        raise ValueError("at least one suite is required")
    if len(set(suites)) != len(suites):
        raise ValueError(f"suite names must be unique, got {suites}")

    quotient, remainder = divmod(total, len(suites))
    return {
        suite: quotient + int(index < remainder)
        for index, suite in enumerate(suites)
    }


def build_episode_balanced_sample_plan(
    episodes_by_group: Mapping[str, Sequence[EpisodeRef]],
    *,
    samples_per_group: int,
    horizon: int,
    seed: int,
) -> list[SampleRef]:
    """Choose one valid frame from distinct episodes in every group."""

    samples_per_group = int(samples_per_group)
    horizon = int(horizon)
    if samples_per_group < 1:
        raise ValueError(f"samples_per_group must be positive, got {samples_per_group}")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not episodes_by_group:
        raise ValueError("at least one sample group is required")

    rng = np.random.default_rng(int(seed))
    plan: list[SampleRef] = []
    for group, episodes in episodes_by_group.items():
        eligible = [episode for episode in episodes if int(episode.episode_length) - horizon > 0]
        if len(eligible) < samples_per_group:
            raise ValueError(
                f"group {group!r} has only {len(eligible)} distinct episodes with a complete "
                f"horizon={horizon}, but {samples_per_group} distinct episodes are required"
            )
        selected = rng.choice(len(eligible), size=samples_per_group, replace=False)
        for selected_index in selected:
            episode = eligible[int(selected_index)]
            valid_frame_count = int(episode.episode_length) - horizon
            frame_index = int(rng.integers(0, valid_frame_count))
            plan.append(
                SampleRef(str(group), int(episode.episode_id), int(episode.episode_length), frame_index)
            )
    rng.shuffle(plan)
    return plan


def build_sample_plan(
    episodes_by_suite: Mapping[str, Sequence[EpisodeRef]],
    *,
    num_samples: int,
    horizon: int,
    seed: int,
) -> list[SampleRef]:
    """Sample balanced, unique frame references with a complete future window.

    A frame ``t`` is valid when ``t + horizon < episode_length``.  The
    strict inequality matches the dataset convention where both current and
    future observations are real frames and no tail padding is introduced.
    """

    horizon = int(horizon)
    num_samples = int(num_samples)
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    suites = list(episodes_by_suite.keys())
    counts = allocate_balanced_counts(num_samples, suites)
    rng = np.random.default_rng(int(seed))
    plan: list[SampleRef] = []

    for suite in suites:
        candidates: list[SampleRef] = []
        for episode in episodes_by_suite[suite]:
            length = int(episode.episode_length)
            valid_frame_count = length - horizon
            if valid_frame_count <= 0:
                continue
            candidates.extend(
                SampleRef(suite, int(episode.episode_id), length, frame_index)
                for frame_index in range(valid_frame_count)
            )

        requested = counts[suite]
        if requested > len(candidates):
            raise ValueError(
                f"suite {suite!r} has only {len(candidates)} valid frames for "
                f"horizon={horizon}, but {requested} are required"
            )
        if requested:
            selected = rng.choice(len(candidates), size=requested, replace=False)
            plan.extend(candidates[int(index)] for index in selected)

    rng.shuffle(plan)
    return plan


def _finite_mask(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    mask = np.isfinite(prediction).all(axis=-1) & np.isfinite(target).all(axis=-1)
    if valid is not None:
        valid = np.asarray(valid, dtype=np.bool_)
        if valid.shape != prediction.shape[:-1]:
            raise ValueError(f"valid mask shape mismatch: {valid.shape} vs {prediction.shape[:-1]}")
        mask &= valid
    return mask


def uvd_pixel_scale(image_size: int | tuple[int, int]) -> np.ndarray:
    """Return normalized-UVD pixel scales in [u, v] order.

    The repository-wide image_size convention is a scalar or
    (height, width) tuple. Normalized U/V map to pixel-center ranges
    [0, width - 1] and [0, height - 1] respectively.
    """

    if isinstance(image_size, int):
        height = width = int(image_size)
    else:
        height, width = (int(value) for value in image_size)
    if height < 2 or width < 2:
        raise ValueError(f"image size must be at least 2x2, got {(height, width)}")
    return np.asarray([width - 1, height - 1], dtype=np.float32)


def canonicalize_uvd_prediction(
    prediction: np.ndarray,
    *,
    points_per_hand: int,
    hand_count: int,
    token_order: str,
) -> np.ndarray:
    """Convert flattened model UVD tokens to canonical ``[time, hand, 3]``."""

    array = np.asarray(prediction, dtype=np.float32)
    points_per_hand = int(points_per_hand)
    hand_count = int(hand_count)
    expected = points_per_hand * hand_count
    if array.shape != (expected, 3):
        raise ValueError(f"expected flattened UVD shape {(expected, 3)}, got {array.shape}")
    if token_order == "hand_major":
        return array.reshape(hand_count, points_per_hand, 3).transpose(1, 0, 2)
    if token_order == "time_major":
        return array.reshape(points_per_hand, hand_count, 3)
    raise ValueError(f"unknown UVD token order: {token_order!r}")


def _uvd_hand_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    uv_scale: np.ndarray,
) -> dict[str, float | int]:
    finite = np.isfinite(prediction).all(axis=-1) & np.isfinite(target).all(axis=-1)
    mask = np.asarray(valid, dtype=np.bool_) & finite
    if not mask.any():
        return {
            "valid_count": 0,
            "uv_ade_px": float("nan"),
            "uv_fde_px": float("nan"),
            "z_mae_m": float("nan"),
        }
    uv_error = (prediction[..., :2] - target[..., :2]) * uv_scale
    uv_l2 = np.linalg.norm(uv_error, axis=-1)
    last_valid = int(np.flatnonzero(mask)[-1])
    return {
        "valid_count": int(mask.sum()),
        "uv_ade_px": float(uv_l2[mask].mean()),
        "uv_fde_px": float(uv_l2[last_valid]),
        "z_mae_m": float(np.abs(prediction[..., 2] - target[..., 2])[mask].mean()),
    }


def uvd_trajectory_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None,
    *,
    image_size: int | tuple[int, int],
) -> dict[str, object]:
    """Measure canonical UVD trajectories in pixels and metric camera depth."""

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.ndim == 2:
        prediction = prediction[:, None, :]
    if target.ndim == 2:
        target = target[:, None, :]
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(
            f"canonical UVD arrays must share [time,hand,3], got {prediction.shape} and {target.shape}"
        )
    if valid is None:
        valid_array = np.ones(prediction.shape[:2], dtype=np.bool_)
    else:
        valid_array = np.asarray(valid, dtype=np.bool_)
        if valid_array.ndim == 1:
            valid_array = valid_array[:, None]
        if valid_array.shape != prediction.shape[:2]:
            raise ValueError(f"valid mask shape mismatch: {valid_array.shape} vs {prediction.shape[:2]}")

    uv_scale = uvd_pixel_scale(image_size)

    per_hand = {
        f"hand_{hand_index}": _uvd_hand_metrics(
            prediction[:, hand_index], target[:, hand_index], valid_array[:, hand_index], uv_scale=uv_scale
        )
        for hand_index in range(prediction.shape[1])
    }
    valid_hand_metrics = [metrics for metrics in per_hand.values() if int(metrics["valid_count"]) > 0]
    if not valid_hand_metrics:
        return {
            "valid_count": 0,
            "uv_ade_px": float("nan"),
            "uv_fde_px": float("nan"),
            "z_mae_m": float("nan"),
            "per_hand": per_hand,
        }
    total_valid = sum(int(metrics["valid_count"]) for metrics in valid_hand_metrics)
    return {
        "valid_count": total_valid,
        "uv_ade_px": float(
            sum(float(metrics["uv_ade_px"]) * int(metrics["valid_count"]) for metrics in valid_hand_metrics)
            / total_valid
        ),
        "uv_fde_px": float(np.mean([float(metrics["uv_fde_px"]) for metrics in valid_hand_metrics])),
        "z_mae_m": float(
            sum(float(metrics["z_mae_m"]) * int(metrics["valid_count"]) for metrics in valid_hand_metrics)
            / total_valid
        ),
        "per_hand": per_hand,
    }


def uvd_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Return masked UVD errors, including valid start/end point errors."""

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.ndim != 2 or prediction.shape[-1] != 3:
        raise ValueError(f"UVD arrays must have shape [K,3], got {prediction.shape}")
    mask = _finite_mask(prediction, target, valid)
    if not mask.any():
        return {
            "valid_count": 0,
            "uv_mae": float("nan"),
            "depth_mae": float("nan"),
            "uvd_mae": float("nan"),
            "endpoint_start_l2": float("nan"),
            "endpoint_end_l2": float("nan"),
        }

    error = prediction - target
    valid_error = error[mask]
    valid_indices = np.flatnonzero(mask)
    start_index = int(valid_indices[0])
    end_index = int(valid_indices[-1])
    return {
        "valid_count": int(mask.sum()),
        "uv_mae": float(np.abs(valid_error[:, :2]).mean()),
        "depth_mae": float(np.abs(valid_error[:, 2]).mean()),
        "uvd_mae": float(np.abs(valid_error).mean()),
        "endpoint_start_l2": float(np.linalg.norm(error[start_index])),
        "endpoint_end_l2": float(np.linalg.norm(error[end_index])),
    }


def masked_depth_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Return masked metric-depth MAE/RMSE."""

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    mask = np.isfinite(prediction) & np.isfinite(target)
    if valid is not None:
        valid = np.asarray(valid, dtype=np.bool_)
        if valid.shape != prediction.shape:
            raise ValueError(f"valid mask shape mismatch: {valid.shape} vs {prediction.shape}")
        mask &= valid
    mask &= target > 0.0
    if not mask.any():
        return {
            "valid_count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "abs_rel": float("nan"),
            "delta1": float("nan"),
            "smooth_l1": float("nan"),
        }
    error = prediction[mask] - target[mask]
    absolute_error = np.abs(error)
    ratio = np.maximum(
        prediction[mask] / target[mask],
        target[mask] / np.maximum(prediction[mask], np.finfo(np.float32).tiny),
    )
    smooth_l1 = np.where(absolute_error < 1.0, 0.5 * np.square(absolute_error), absolute_error - 0.5)
    return {
        "valid_count": int(mask.sum()),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "abs_rel": float((absolute_error / target[mask]).mean()),
        "delta1": float((ratio < 1.25).mean()),
        "smooth_l1": float(smooth_l1.mean()),
    }


def summarize_latencies(latencies_ms: Sequence[float], warmup_ms: float | None = None) -> dict[str, float | int | None]:
    """Summarize per-request latency in milliseconds.

    If the first value equals ``warmup_ms``, it is treated as the warmup
    request and excluded from aggregate statistics.  This supports both a
    single list containing warmup and measured requests and a measured-only
    list with a separately recorded warmup value.
    """

    values = np.asarray(list(latencies_ms), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("latencies_ms must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("latencies_ms contains non-finite values")
    if warmup_ms is not None and len(values) and np.isclose(values[0], float(warmup_ms)):
        values = values[1:]
    if len(values) == 0:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "warmup_ms": warmup_ms}
    return {
        "count": int(len(values)),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "warmup_ms": None if warmup_ms is None else float(warmup_ms),
    }
