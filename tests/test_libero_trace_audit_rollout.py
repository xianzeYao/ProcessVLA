from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_rollout import (
    RECORD_VERSION,
    RolloutRecord,
    build_geometry_request,
    complete_anchor_targets,
    derive_inference_seed,
    execute_cadenced_actions,
    load_rollout_record,
    save_rollout_record,
)


def _record() -> RolloutRecord:
    steps, anchors, horizon, height, width = 9, 2, 4, 3, 5
    uvd = np.zeros((anchors, 3, 3, 3), dtype=np.float32)
    uvd[..., 2] = 1.0
    realized = np.zeros((steps, 3, 3), dtype=np.float32)
    realized[..., 2] = 1.0
    return RolloutRecord(
        agent_rgb=np.arange(steps * height * width * 3, dtype=np.uint8).reshape(steps, height, width, 3),
        wrist_rgb=np.full((steps, height, width, 3), 17, dtype=np.uint8),
        agent_depth=np.ones((steps, height, width), dtype=np.float32),
        executed_actions=np.zeros((steps, 7), dtype=np.float32),
        anchor_steps=np.asarray([0, 4], dtype=np.int32),
        predicted_uvd=uvd,
        predicted_uvd_time=np.tile(np.repeat(np.asarray([0.0, 0.5, 1.0], dtype=np.float32), 3), (anchors, 1)),
        predicted_uvd_landmark_ids=np.tile(np.asarray([0, 1, 2] * 3, dtype=np.int64), (anchors, 1)),
        predicted_depth_current=np.ones((anchors, height, width), dtype=np.float32),
        predicted_depth_future=np.full((anchors, height, width), 2.0, dtype=np.float32),
        realized_uvd=realized,
        realized_xyz=np.ones((steps, 3, 3), dtype=np.float32),
        realized_valid=np.ones((steps, 3), dtype=np.bool_),
        realized_in_frame=np.ones((steps, 3), dtype=np.bool_),
        anchor_target_uvd=uvd.copy(),
        anchor_target_valid=np.ones((anchors, 3, 3), dtype=np.bool_),
        dense_depth_current_target=np.ones((anchors, height, width), dtype=np.float32),
        dense_depth_future_target=np.ones((anchors, height, width), dtype=np.float32),
        latency_ms=np.asarray([1.25, 2.5], dtype=np.float64),
        metadata={
            "case": {"suite": "libero_goal", "task_id": 1, "seed": 7},
            "outcome": {"success": False, "end_reason": "max_steps"},
            "action_horizon": horizon,
            "image_size": [height, width],
            "metrics": {"anchor_0": {"aggregate": {"d_mae_mm": None}}},
        },
    )


def test_cadence_excludes_dummy_steps_and_requests_primary_then_wrist_geometry() -> None:
    """Changing chunk cadence, seed derivation, or view order must fail here."""
    primary = np.full((2, 3, 3), 11, dtype=np.uint8)
    wrist = np.full((2, 3, 3), 22, dtype=np.uint8)
    requests: list[dict] = []
    dummy_actions: list[np.ndarray] = []
    executed_steps: list[int] = []

    def observe(step: int) -> dict[str, object]:
        return {"image": [primary + step, wrist + step], "lang": "place the mug"}

    def request(payload: dict) -> dict:
        requests.append(payload)
        return {"actions": np.tile(np.arange(7, dtype=np.float32), (8, 1))}

    trace = execute_cadenced_actions(
        max_steps=17,
        action_horizon=8,
        audit_seed=7,
        dummy_steps=10,
        stabilize=lambda action: dummy_actions.append(np.asarray(action)),
        observe=observe,
        request=request,
        execute=lambda action, step: (executed_steps.append(step) or False),
    )

    assert len(dummy_actions) == 10
    assert trace.anchor_steps.tolist() == [0, 8, 16]
    assert executed_steps == list(range(17))
    assert [payload["inference_seed"] for payload in requests] == [
        derive_inference_seed(7, 0),
        derive_inference_seed(7, 8),
        derive_inference_seed(7, 16),
    ]
    assert all(payload["return_geometry"] is True for payload in requests)
    np.testing.assert_array_equal(requests[0]["examples"][0]["image"][0], primary)
    np.testing.assert_array_equal(requests[0]["examples"][0]["image"][1], wrist)


def test_geometry_request_seed_is_deterministic_and_injective_for_audit_anchors() -> None:
    observation = {"image": [np.zeros((2, 2, 3), dtype=np.uint8)] * 2, "lang": "move"}
    first = build_geometry_request(observation, inference_seed=derive_inference_seed(7, 16))
    second = build_geometry_request(observation, inference_seed=derive_inference_seed(7, 16))

    assert first == second
    assert derive_inference_seed(7, 16) != derive_inference_seed(8, 0)
    assert derive_inference_seed(7, 16) != derive_inference_seed(7, 8)
    assert first["examples"][0]["image"] == observation["image"]


def test_raw_record_round_trip_preserves_exact_array_shapes_and_dtypes(tmp_path: Path) -> None:
    """Changing storage layout or implicit dtype coercion must fail here."""
    path = tmp_path / "rollout"
    record = _record()

    save_rollout_record(path, record)
    loaded = load_rollout_record(path)

    assert RECORD_VERSION == 1
    for field in record.array_field_names():
        original = getattr(record, field)
        restored = getattr(loaded, field)
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        np.testing.assert_array_equal(restored, original)
    assert loaded.metadata == record.metadata
    assert json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["version"] == RECORD_VERSION


def test_load_rejects_corrupt_or_partial_npz_before_resume(tmp_path: Path) -> None:
    path = tmp_path / "rollout"
    save_rollout_record(path, _record())
    npz_path = path.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as source:
        corrupted = {name: source[name] for name in source.files if name != "realized_xyz"}
    np.savez_compressed(npz_path, **corrupted)

    with pytest.raises(ValueError, match="sha256|array keys|realized_xyz"):
        load_rollout_record(path)


def test_anchor_alignment_scores_strict_json_and_keeps_incomplete_tail_invalid() -> None:
    predicted = np.zeros((2, 3, 3, 3), dtype=np.float32)
    predicted[..., 2] = 1.0
    time = np.tile(np.repeat(np.asarray([0.0, 0.5, 1.0], dtype=np.float32), 3), (2, 1))
    landmark_ids = np.tile(np.asarray([0, 1, 2] * 3, dtype=np.int64), (2, 1))
    realized = np.zeros((5, 3, 3), dtype=np.float32)
    realized[..., 2] = 1.0

    targets, valid, metrics = complete_anchor_targets(
        predicted_uvd=predicted,
        predicted_uvd_time=time,
        predicted_uvd_landmark_ids=landmark_ids,
        anchor_steps=np.asarray([0, 4], dtype=np.int32),
        realized_uvd=realized,
        realized_valid=np.ones((5, 3), dtype=np.bool_),
        action_horizon=4,
        image_size=(3, 5),
    )

    assert targets.shape == predicted.shape
    assert valid[0].all()
    assert valid[1, 0].all()
    assert not valid[1, 1:].any()
    assert np.isnan(targets[1, 1:]).all()
    assert metrics["anchor_1"]["aggregate"]["valid_count"] == 3
    json.dumps(metrics, allow_nan=False)
