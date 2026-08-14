from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_rollout import (
    RECORD_VERSION, RolloutRecord, complete_anchor_targets, finalize_state_timeline,
    flip_camera_intrinsics, load_rollout_record, save_rollout_record,
)


def _record() -> RolloutRecord:
    actions, anchors, horizon, h, w = 2, 1, 2, 3, 5
    uvd = np.zeros((anchors, 2, 3, 3), np.float32); uvd[..., 2] = 1
    state = np.zeros((actions + 1, 3, 3), np.float32); state[..., 2] = 1
    return RolloutRecord(
        agent_rgb=np.zeros((actions + 1, h, w, 3), np.uint8), wrist_rgb=np.zeros((actions + 1, h, w, 3), np.uint8), agent_depth=np.ones((actions + 1, h, w), np.float32),
        policy_actions_raw=np.asarray([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]], np.float32), executed_actions=np.asarray([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, -1]], np.float32), anchor_steps=np.asarray([0], np.int32),
        predicted_uvd=uvd, predicted_uvd_time=np.asarray([[0, 0, 0, 1, 1, 1]], np.float32), predicted_uvd_landmark_ids=np.asarray([[0, 1, 2, 0, 1, 2]], np.int64),
        predicted_depth_current=np.ones((1, h, w), np.float32), predicted_depth_future=np.ones((1, h, w), np.float32),
        realized_uvd=state, realized_xyz=np.ones_like(state), realized_valid=np.ones((actions + 1, 3), bool), realized_in_frame=np.ones((actions + 1, 3), bool),
        anchor_target_uvd=state[None, [0, 2]], anchor_target_valid=np.ones((1, 2, 3), bool), dense_depth_current_target=np.ones((1, h, w), np.float32), dense_depth_future_target=np.ones((1, h, w), np.float32), latency_ms=np.asarray([1.0], np.float64), camera_k_agentview_flipped=np.asarray([[10, 0, 2], [0, 11, 1], [0, 0, 1]], np.float32),
        metadata={"case": {"suite": "libero_goal", "task_id": 1, "language": "move", "rank_group": "best", "initial_state_index": 0, "seed": 7, "original_success": True}, "outcome": {"success": False, "end_reason": "max_steps"}, "action_horizon": horizon, "image_size": [h, w], "metrics": {}, "schema": "state_timeline_v2"},
    )


def test_state_timeline_keeps_post_action_terminal_endpoint_and_postprocessed_actions() -> None:
    raw = np.asarray([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]], np.float32)
    states = [np.full((3, 3), i, np.float32) for i in range(3)]
    timeline = finalize_state_timeline(raw, states)
    assert timeline.policy_actions_raw.shape == (2, 7)
    np.testing.assert_array_equal(timeline.executed_actions[:, 6], [1, -1])
    assert timeline.state_count == 3


def test_alignment_uses_state_endpoint_and_invalidates_only_missing_tail() -> None:
    pred = np.zeros((1, 2, 3, 3), np.float32); pred[..., 2] = 1
    states = np.zeros((3, 3, 3), np.float32); states[..., 2] = 1
    target, valid, metrics = complete_anchor_targets(predicted_uvd=pred, predicted_uvd_time=np.asarray([[0, 0, 0, 1, 1, 1]], np.float32), predicted_uvd_landmark_ids=np.asarray([[0, 1, 2, 0, 1, 2]], np.int64), anchor_steps=np.asarray([0], np.int32), realized_uvd=states, realized_valid=np.ones((3, 3), bool), action_horizon=2, image_size=(3, 5), camera_k=np.eye(3, dtype=np.float32))
    assert valid[0, 1].all()  # state index 2 exists after final action
    assert np.isfinite(metrics["anchor_0"]["camera"]["center_mae_mm"])


def test_flipped_rectangular_camera_intrinsics_and_camera_metrics_are_real() -> None:
    k = np.asarray([[10, 0, 3], [0, 11, 2], [0, 0, 1]], np.float32)
    flipped = flip_camera_intrinsics(k, image_size=(7, 13))
    np.testing.assert_allclose(flipped, [[-10, 0, 9], [0, -11, 4], [0, 0, 1]])


def test_generation_manifest_roundtrip_rejects_case_and_config_mismatch(tmp_path: Path) -> None:
    record = _record(); path = tmp_path / "rollout"
    save_rollout_record(path, record)
    loaded = load_rollout_record(path, expected_case=record.metadata["case"], expected_config_identity={"action_horizon": 2, "image_size": [3, 5], "schema": "state_timeline_v2"})
    assert loaded.executed_actions.shape[0] + 1 == loaded.agent_rgb.shape[0]
    with pytest.raises(ValueError, match="expected_case"):
        load_rollout_record(path, expected_case={**record.metadata["case"], "seed": 8})
    with pytest.raises(ValueError, match="expected_config_identity"):
        load_rollout_record(path, expected_config_identity={"action_horizon": 8})


def test_failed_manifest_publish_leaves_previous_generation_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "rollout"; old = _record(); save_rollout_record(path, old)
    import examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_rollout as rollout
    real_replace = rollout.os.replace
    def fail_manifest(source, destination):
        if Path(destination) == path.with_suffix(".manifest.json"): raise OSError("interrupted publish")
        return real_replace(source, destination)
    monkeypatch.setattr(rollout.os, "replace", fail_manifest)
    with pytest.raises(OSError, match="interrupted"):
        save_rollout_record(path, replace(old, latency_ms=np.asarray([2.0], np.float64)))
    assert load_rollout_record(path).latency_ms.tolist() == [1.0]
    assert json.loads(path.with_suffix(".manifest.json").read_text())["version"] == RECORD_VERSION
