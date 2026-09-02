"""Binary artifact helpers for on-policy RoboCasa representation probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_FEATURE_KEYS = (
    "uvd_hidden",
    "image_hidden_mean",
    "native_hidden_mean",
    "predicted_action",
)
_STATE_KEYS = (
    "state.left_arm",
    "state.right_arm",
    "state.left_hand",
    "state.right_hand",
    "state.waist",
)


def _select_batch(value: Any, batch_index: int, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0 or not 0 <= batch_index < array.shape[0]:
        raise ValueError(
            f"{name} does not contain batch index {batch_index}: shape={array.shape}"
        )
    return array[batch_index]


def build_rollout_feature_record(
    features: Mapping[str, Any],
    observations: Mapping[str, Any],
    *,
    batch_index: int,
    task_index: int | None,
    episode_index: int,
    decision_index: int,
) -> dict[str, Any]:
    missing_features = [key for key in _FEATURE_KEYS if key not in features]
    missing_state = [key for key in _STATE_KEYS if key not in observations]
    if missing_features or missing_state:
        raise KeyError(
            f"missing feature keys={missing_features}; missing state keys={missing_state}"
        )

    record: dict[str, Any] = {
        "task_index": -1 if task_index is None else int(task_index),
        "episode_index": int(episode_index),
        "decision_index": int(decision_index),
        "uvd_hidden": _select_batch(
            features["uvd_hidden"], batch_index, name="uvd_hidden"
        ).astype(np.float16, copy=False),
        "image_hidden_mean": _select_batch(
            features["image_hidden_mean"], batch_index, name="image_hidden_mean"
        ).astype(np.float16, copy=False),
        "native_hidden_mean": _select_batch(
            features["native_hidden_mean"], batch_index, name="native_hidden_mean"
        ).astype(np.float16, copy=False),
        "predicted_action": _select_batch(
            features["predicted_action"], batch_index, name="predicted_action"
        ).astype(np.float32, copy=False),
    }
    state_parts = [
        _select_batch(observations[key], batch_index, name=key).reshape(-1)
        for key in _STATE_KEYS
    ]
    record["state"] = np.concatenate(state_parts).astype(np.float32, copy=False)
    return record


def write_rollout_feature_artifacts(
    output_path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot write an empty rollout feature artifact")

    required = (
        "task_index",
        "episode_index",
        "decision_index",
        "episode_success",
        *_FEATURE_KEYS,
        "state",
    )
    missing_by_record = {
        index: [key for key in required if key not in record]
        for index, record in enumerate(records)
        if any(key not in record for key in required)
    }
    if missing_by_record:
        raise KeyError(f"rollout feature records have missing keys: {missing_by_record}")

    arrays = {key: np.asarray([record[key] for record in records]) for key in required}
    for key in (*_FEATURE_KEYS, "state"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"rollout feature array {key} contains non-finite values")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(output)

    manifest = {
        "schema_version": 1,
        "record_count": len(records),
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "nbytes": int(value.nbytes),
            }
            for key, value in arrays.items()
        },
        "metadata": dict(metadata or {}),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_tmp.replace(manifest_path)
    return manifest
