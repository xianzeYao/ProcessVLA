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
    if not mask.any():
        return {"valid_count": 0, "mae": float("nan"), "rmse": float("nan")}
    error = prediction[mask] - target[mask]
    return {
        "valid_count": int(mask.sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
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
