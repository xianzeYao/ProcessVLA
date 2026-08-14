from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_selection import AuditCase
from examples.simBenchmarks.CoT.geometry_probe import libero_trace_audit_rollout as rollout
from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_rollout import (
    RECORD_VERSION,
    RolloutRecord,
    _frame,
    _rollout_step_limit,
    collect_rollout,
    complete_anchor_targets,
    execute_cadenced_actions,
    finalize_state_timeline,
    flip_camera_intrinsics,
    load_rollout_record,
    save_rollout_record,
    validate_rollout_record,
)


EXPECTED_CAMERA_CONVENTION = (
    "libero_agentview_display_horizontal_mirror_from_robosuite_opencv"
)


def _case(suite: str = "libero_goal") -> AuditCase:
    return AuditCase(suite, 1, "move", "best", 0, 7, True)


def _record() -> RolloutRecord:
    actions, anchors, horizon, height, width = 2, 1, 2, 3, 5
    predicted_uvd = np.zeros((anchors, 2, 3, 3), np.float32)
    predicted_uvd[..., 2] = 1
    state_uvd = np.zeros((actions + 1, 3, 3), np.float32)
    state_uvd[..., 2] = 1
    case = _case()
    return RolloutRecord(
        agent_rgb=np.zeros((actions + 1, height, width, 3), np.uint8),
        wrist_rgb=np.zeros((actions + 1, height, width, 3), np.uint8),
        agent_depth=np.ones((actions + 1, height, width), np.float32),
        policy_actions_raw=np.asarray(
            [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]], np.float32
        ),
        executed_actions=np.asarray(
            [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, -1]], np.float32
        ),
        anchor_steps=np.asarray([0], np.int32),
        predicted_uvd=predicted_uvd,
        predicted_uvd_time=np.asarray([[0, 0, 0, 1, 1, 1]], np.float32),
        predicted_uvd_landmark_ids=np.asarray([[0, 1, 2, 0, 1, 2]], np.int64),
        predicted_depth_current=np.ones((anchors, height, width), np.float32),
        predicted_depth_future=np.ones((anchors, height, width), np.float32),
        realized_uvd=state_uvd,
        realized_xyz=np.ones_like(state_uvd),
        realized_valid=np.ones((actions + 1, 3), np.bool_),
        realized_in_frame=np.ones((actions + 1, 3), np.bool_),
        anchor_target_uvd=state_uvd[None, [0, 2]],
        anchor_target_valid=np.ones((anchors, 2, 3), np.bool_),
        dense_depth_current_target=np.ones((anchors, height, width), np.float32),
        dense_depth_future_target=np.ones((anchors, height, width), np.float32),
        latency_ms=np.asarray([1.0], np.float64),
        camera_k_agentview_flipped=np.asarray(
            [[-10, 0, 2], [0, 11, 1], [0, 0, 1]], np.float32
        ),
        metadata={
            "case": {
                "suite": case.suite,
                "task_id": case.task_id,
                "language": case.language,
                "rank_group": case.rank_group,
                "initial_state_index": case.initial_state_index,
                "seed": case.seed,
                "original_success": case.original_success,
            },
            "outcome": {"success": False, "end_reason": "max_steps"},
            "action_horizon": horizon,
            "image_size": [height, width],
            "metrics": {},
            "camera_convention": EXPECTED_CAMERA_CONVENTION,
            "schema": "state_timeline_v3",
        },
    )


def _generation_paths(path: Path) -> tuple[Path, Path, dict[str, object]]:
    pointer = json.loads(path.with_suffix(".manifest.json").read_text())
    generation = pointer["generation"]
    npz = path.parent / f"{generation}.npz"
    data = path.parent / f"{generation}.json"
    return npz, data, json.loads(data.read_text())


def _rewrite_payload(data: Path, payload: dict[str, object]) -> None:
    data.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def _rewrite_npz_and_digest(
    npz: Path, data: Path, payload: dict[str, object], arrays: dict[str, np.ndarray]
) -> None:
    np.savez_compressed(npz, **arrays)
    payload["sha256"] = sha256(npz.read_bytes()).hexdigest()
    _rewrite_payload(data, payload)


def test_state_timeline_and_camera_endpoint() -> None:
    raw = np.asarray(
        [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]], np.float32
    )
    assert finalize_state_timeline(raw, [0, 1, 2]).state_count == 3
    predicted = np.zeros((1, 2, 3, 3), np.float32)
    predicted[..., 2] = 1
    _, valid, metrics = complete_anchor_targets(
        predicted_uvd=predicted,
        predicted_uvd_time=np.asarray([[0, 0, 0, 1, 1, 1]], np.float32),
        predicted_uvd_landmark_ids=np.asarray([[0, 1, 2, 0, 1, 2]], np.int64),
        anchor_steps=np.asarray([0], np.int32),
        realized_uvd=np.ones((3, 3, 3), np.float32),
        realized_valid=np.ones((3, 3), np.bool_),
        action_horizon=2,
        image_size=(3, 5),
        camera_k=np.eye(3, dtype=np.float32),
    )
    assert valid[0, 1].all()
    assert np.isfinite(metrics["anchor_0"]["camera"]["center_mae_mm"])
    np.testing.assert_allclose(
        flip_camera_intrinsics(
            np.asarray([[10, 0, 3], [0, 11, 2], [0, 0, 1]], np.float32),
            image_size=(7, 13),
        ),
        [[-10, 0, 9], [0, 11, 2], [0, 0, 1]],
    )


def test_state_timeline_keeps_post_action_terminal_endpoint_and_postprocessed_actions() -> None:
    raw = np.asarray(
        [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]], np.float32
    )
    states = [np.full((3, 3), index, np.float32) for index in range(3)]
    timeline = finalize_state_timeline(raw, states)
    assert timeline.policy_actions_raw.shape == (2, 7)
    np.testing.assert_array_equal(timeline.executed_actions[:, 6], [1, -1])
    assert timeline.state_count == 3


def test_alignment_uses_state_endpoint_and_invalidates_only_missing_tail() -> None:
    predicted = np.zeros((1, 2, 3, 3), np.float32)
    predicted[..., 2] = 1
    states = np.zeros((3, 3, 3), np.float32)
    states[..., 2] = 1
    _, valid, metrics = complete_anchor_targets(
        predicted_uvd=predicted,
        predicted_uvd_time=np.asarray([[0, 0, 0, 1, 1, 1]], np.float32),
        predicted_uvd_landmark_ids=np.asarray([[0, 1, 2, 0, 1, 2]], np.int64),
        anchor_steps=np.asarray([0], np.int32),
        realized_uvd=states,
        realized_valid=np.ones((3, 3), np.bool_),
        action_horizon=2,
        image_size=(3, 5),
        camera_k=np.eye(3, dtype=np.float32),
    )
    assert valid[0, 1].all()
    assert np.isfinite(metrics["anchor_0"]["camera"]["center_mae_mm"])


def test_v3_standard_grid_rounds_to_action_offsets_and_rejects_malformed_time() -> None:
    predicted = np.zeros((1, 4, 3, 3), np.float32)
    predicted[..., 2] = 1
    states = np.zeros((9, 3, 3), np.float32)
    states[..., 0] = np.arange(9, dtype=np.float32)[:, None]
    states[..., 2] = 1
    grid = np.repeat(np.linspace(0.0, 1.0, 4, dtype=np.float32), 3)[None]
    landmark_ids = np.tile(np.arange(3, dtype=np.int64), 4)[None]
    targets, valid, _ = complete_anchor_targets(
        predicted_uvd=predicted,
        predicted_uvd_time=grid,
        predicted_uvd_landmark_ids=landmark_ids,
        anchor_steps=np.asarray([0], np.int32),
        realized_uvd=states,
        realized_valid=np.ones((9, 3), np.bool_),
        action_horizon=8,
        image_size=(3, 5),
    )
    np.testing.assert_array_equal(targets[0, :, 0, 0], [0, 3, 5, 8])
    assert valid.all()

    malformed = grid.copy()
    malformed[0, 3:6] = np.float32(0.31)
    with pytest.raises(ValueError, match="standard.*grid"):
        complete_anchor_targets(
            predicted_uvd=predicted,
            predicted_uvd_time=malformed,
            predicted_uvd_landmark_ids=landmark_ids,
            anchor_steps=np.asarray([0], np.int32),
            realized_uvd=states,
            realized_valid=np.ones((9, 3), np.bool_),
            action_horizon=8,
            image_size=(3, 5),
        )


def test_v3_grid_requires_strictly_unique_action_offsets() -> None:
    predicted = np.zeros((1, 4, 3, 3), np.float32)
    predicted[..., 2] = 1
    grid = np.repeat(np.linspace(0.0, 1.0, 4, dtype=np.float32), 3)[None]
    with pytest.raises(ValueError, match="unique action offsets"):
        complete_anchor_targets(
            predicted_uvd=predicted,
            predicted_uvd_time=grid,
            predicted_uvd_landmark_ids=np.tile(np.arange(3, dtype=np.int64), 4)[None],
            anchor_steps=np.asarray([0], np.int32),
            realized_uvd=np.ones((3, 3, 3), np.float32),
            realized_valid=np.ones((3, 3), np.bool_),
            action_horizon=2,
            image_size=(3, 5),
        )


def test_flipped_rectangular_camera_intrinsics_and_camera_metrics_are_real() -> None:
    camera_k = np.asarray([[10, 0, 3], [0, 11, 2], [0, 0, 1]], np.float32)
    flipped = flip_camera_intrinsics(camera_k, image_size=(7, 13))
    np.testing.assert_allclose(flipped, [[-10, 0, 9], [0, 11, 2], [0, 0, 1]])


def test_frame_converts_raw_sim_depth_to_metric_before_180_flip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_depth = np.asarray(
        [[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]], np.float32
    )
    metric_depth = np.asarray([[[1], [2], [3]], [[4], [5], [6]]], np.float32)
    calls: list[tuple[object, np.ndarray]] = []
    camera_utils = ModuleType("robosuite.utils.camera_utils")

    def get_real_depth_map(sim: object, depth: np.ndarray) -> np.ndarray:
        calls.append((sim, depth))
        return metric_depth

    camera_utils.get_real_depth_map = get_real_depth_map  # type: ignore[attr-defined]
    camera_utils.get_camera_intrinsic_matrix = (  # type: ignore[attr-defined]
        lambda *_: np.asarray([[10, 0, 1], [0, 10, 1], [0, 0, 1]], np.float32)
    )
    camera_utils.get_camera_extrinsic_matrix = (  # type: ignore[attr-defined]
        lambda *_: np.eye(4, dtype=np.float32)
    )
    utils_module = ModuleType("robosuite.utils")
    utils_module.camera_utils = camera_utils  # type: ignore[attr-defined]
    robosuite_module = ModuleType("robosuite")
    robosuite_module.utils = utils_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "robosuite", robosuite_module)
    monkeypatch.setitem(sys.modules, "robosuite.utils", utils_module)
    monkeypatch.setitem(sys.modules, "robosuite.utils.camera_utils", camera_utils)

    class Model:
        def body_name2id(self, name: str) -> int:
            return {
                "gripper0_finger_joint1_tip": 0,
                "gripper0_finger_joint2_tip": 1,
                "gripper0_right_gripper": 2,
            }[name]

    sim = SimpleNamespace(
        model=Model(),
        data=SimpleNamespace(
            body_xpos=np.asarray([[0, 0, 1], [0.01, 0, 1], [0, 0.01, 1]], np.float32)
        ),
    )
    env = SimpleNamespace(sim=sim)
    obs = {
        "agentview_image": np.zeros((2, 3, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((2, 3, 3), np.uint8),
        "agentview_depth": raw_depth,
    }
    frame = _frame(env, obs, 3, None)
    assert len(calls) == 1 and calls[0][0] is sim
    np.testing.assert_array_equal(calls[0][1], raw_depth)
    assert frame["depth"].shape == (2, 3)
    assert frame["depth"].dtype == np.float32
    np.testing.assert_array_equal(frame["depth"], metric_depth[::-1, ::-1, 0])
    np.testing.assert_allclose(frame["uvd"][:, 0], [0.5, 0.45, 0.5], atol=1e-6)
    np.testing.assert_allclose(frame["uvd"][:, 1], [1.0, 1.0, 1.1], atol=1e-6)
    np.testing.assert_allclose(
        frame["k"], [[-10, 0, 1], [0, 10, 1], [0, 0, 1]]
    )

    camera_utils.get_real_depth_map = (  # type: ignore[attr-defined]
        lambda *_: np.repeat(metric_depth, 2, axis=-1)
    )
    with pytest.raises(ValueError, match="metric agentview depth.*singleton channel"):
        _frame(env, obs, 3, None)

    camera_utils.get_real_depth_map = lambda *_: metric_depth  # type: ignore[attr-defined]
    bad_wrist = {**obs, "robot0_eye_in_hand_image": np.zeros((2, 4, 3), np.uint8)}
    with pytest.raises(ValueError, match="wrist RGB.*agentview RGB"):
        _frame(env, bad_wrist, 3, None)


def test_generation_manifest_identity_and_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rollout"
    old = _record()
    save_rollout_record(path, old)
    assert load_rollout_record(path, expected_case=old.metadata["case"]).agent_rgb.shape[0] == 3
    original_files = set(tmp_path.glob("rollout.v3.*"))
    real_replace = rollout.os.replace

    def fail(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == path.with_suffix(".manifest.json"):
            raise OSError("interrupted")
        real_replace(source, destination)

    monkeypatch.setattr(rollout.os, "replace", fail)
    with pytest.raises(OSError, match="interrupted"):
        save_rollout_record(path, replace(old, latency_ms=np.asarray([2.0], np.float64)))
    assert load_rollout_record(path).latency_ms.tolist() == [1.0]
    assert set(tmp_path.glob("rollout.v3.*")) == original_files


def test_generation_manifest_roundtrip_rejects_case_and_config_mismatch(tmp_path: Path) -> None:
    record = _record()
    path = tmp_path / "rollout"
    save_rollout_record(path, record)
    loaded = load_rollout_record(
        path,
        expected_case=record.metadata["case"],
        expected_config_identity={
            "action_horizon": 2,
            "image_size": [3, 5],
            "schema": "state_timeline_v3",
        },
    )
    assert RECORD_VERSION == 3
    assert loaded.metadata["camera_convention"] == EXPECTED_CAMERA_CONVENTION
    assert loaded.metadata["schema"] == "state_timeline_v3"
    assert loaded.executed_actions.shape[0] + 1 == loaded.agent_rgb.shape[0]
    with pytest.raises(ValueError, match="expected_case"):
        load_rollout_record(path, expected_case={**record.metadata["case"], "seed": 8})
    with pytest.raises(ValueError, match="expected_config_identity"):
        load_rollout_record(path, expected_config_identity={"action_horizon": 8})


def test_loader_rejects_v2_manifest_to_force_projection_rerun(tmp_path: Path) -> None:
    path = tmp_path / "rollout"
    save_rollout_record(path, _record())
    manifest = path.with_suffix(".manifest.json")
    pointer = json.loads(manifest.read_text())
    pointer["version"] = 2
    manifest.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported.*manifest"):
        load_rollout_record(path)


def test_failed_manifest_publish_leaves_previous_generation_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rollout"
    old = _record()
    save_rollout_record(path, old)
    original_files = set(tmp_path.glob("rollout.v3.*"))
    real_replace = rollout.os.replace

    def fail_manifest(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == path.with_suffix(".manifest.json"):
            raise OSError("interrupted publish")
        real_replace(source, destination)

    monkeypatch.setattr(rollout.os, "replace", fail_manifest)
    with pytest.raises(OSError, match="interrupted"):
        save_rollout_record(path, replace(old, latency_ms=np.asarray([2.0], np.float64)))
    assert load_rollout_record(path).latency_ms.tolist() == [1.0]
    assert json.loads(path.with_suffix(".manifest.json").read_text())["version"] == RECORD_VERSION
    assert set(tmp_path.glob("rollout.v3.*")) == original_files


def test_loader_rejects_undeclared_npz_keys_and_incomplete_array_manifest(tmp_path: Path) -> None:
    path = tmp_path / "rollout"
    save_rollout_record(path, _record())
    npz, data, payload = _generation_paths(path)
    with np.load(npz, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    arrays["unexpected"] = np.zeros(1, np.float32)
    _rewrite_npz_and_digest(npz, data, payload, arrays)
    with pytest.raises(ValueError, match="array keys"):
        load_rollout_record(path)

    path = tmp_path / "rollout_missing_schema"
    save_rollout_record(path, _record())
    _, data, payload = _generation_paths(path)
    del payload["arrays"]["agent_rgb"]  # type: ignore[index]
    _rewrite_payload(data, payload)
    with pytest.raises(ValueError, match="partial rollout generation"):
        load_rollout_record(path)


@pytest.mark.parametrize("corruption", ["manifest", "payload", "npz"])
def test_corrupt_generation_boundaries_raise_value_error(
    tmp_path: Path, corruption: str
) -> None:
    path = tmp_path / corruption
    save_rollout_record(path, _record())
    manifest = path.with_suffix(".manifest.json")
    npz, data, payload = _generation_paths(path)
    if corruption == "manifest":
        manifest.write_text(json.dumps({"version": RECORD_VERSION}), encoding="utf-8")
    elif corruption == "payload":
        del payload["metadata"]
        _rewrite_payload(data, payload)
    else:
        with np.load(npz, allow_pickle=False) as source:
            arrays = {
                key: np.asarray(source[key]) for key in source.files if key != "agent_rgb"
            }
        _rewrite_npz_and_digest(npz, data, payload, arrays)
    with pytest.raises(ValueError):
        load_rollout_record(path)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda record: replace(
                record,
                predicted_depth_future=np.ones((1, 1, 3, 5), np.float32),
            ),
            "dense depth",
        ),
        (
            lambda record: replace(
                record,
                dense_depth_future_target=np.ones((1, 4, 5), np.float32),
            ),
            "dense depth",
        ),
        (
            lambda record: replace(
                record,
                realized_in_frame=np.ones((3, 3), np.bool_),
                realized_valid=np.zeros((3, 3), np.bool_),
            ),
            "in_frame.*projection valid",
        ),
        (
            lambda record: replace(
                record,
                latency_ms=np.asarray([np.nan], np.float64),
            ),
            "latency",
        ),
        (
            lambda record: replace(
                record,
                camera_k_agentview_flipped=np.eye(3, dtype=np.float32),
            ),
            "camera convention",
        ),
        (
            lambda record: replace(
                record,
                camera_k_agentview_flipped=np.asarray(
                    [[-10, 0, 2], [0, -11, 1], [0, 0, 1]], np.float32
                ),
            ),
            "camera convention",
        ),
        (
            lambda record: replace(
                record,
                metadata={**record.metadata, "schema": "state_timeline_v2"},
            ),
            "metadata schema",
        ),
        (
            lambda record: replace(
                record,
                metadata={
                    key: value
                    for key, value in record.metadata.items()
                    if key != "camera_convention"
                },
            ),
            "metadata schema",
        ),
        (
            lambda record: replace(
                record,
                metadata={**record.metadata, "image_size": [5, 3]},
            ),
            "config metadata",
        ),
        (
            lambda record: replace(
                record,
                metadata={
                    **record.metadata,
                    "case": {**record.metadata["case"], "rank_group": "middle"},
                },
            ),
            "case metadata",
        ),
        (
            lambda record: replace(
                record,
                metadata={
                    **record.metadata,
                    "outcome": {"success": True, "end_reason": "max_steps"},
                },
            ),
            "outcome metadata",
        ),
    ],
)
def test_strict_schema_rejects_invalid_record_arrays_and_metadata(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_rollout_record(mutate(_record()))


class _Env:
    def __init__(self) -> None:
        self.received: list[np.ndarray] = []
        self.closed = False

    def step(self, action: list[float]) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        self.received.append(np.asarray(action, np.float32))
        return _obs(), 0.0, False, {}

    def close(self) -> None:
        self.closed = True


def _obs() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.full((3, 5, 3), 11, np.uint8),
        "robot0_eye_in_hand_image": np.full((3, 5, 3), 22, np.uint8),
        "agentview_depth": np.ones((3, 5), np.float32),
    }


class _Client:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def predict_action(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        actions = np.asarray(
            [[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]]], np.float32
        )
        geometry = {
            "depth_current": np.ones((1, 1, 3, 5), np.float32),
            "depth_future": np.ones((1, 1, 3, 5), np.float32),
            "uvd": np.asarray([[[0.5, 0.5, 1]] * 6], np.float32),
            "uvd_time": np.asarray([[0, 0, 0, 1, 1, 1]], np.float32),
            "uvd_landmark_ids": np.asarray([[0, 1, 2, 0, 1, 2]], np.int64),
        }
        data = {"actions": actions, "geometry": geometry}
        return {"data": data} if len(self.requests) % 2 else data


def _capture(env: _Env, obs: dict[str, np.ndarray], resolution: int) -> dict[str, np.ndarray]:
    return {
        "rgb": np.full((3, 5, 3), 11, np.uint8),
        "wrist": np.full((3, 5, 3), 22, np.uint8),
        "depth": np.ones((3, 5), np.float32),
        "xyz": np.asarray([[1, 0, 0], [2, 0, 0], [3, 0, 0]], np.float32),
        "uvd": np.asarray([[0.2, 0.2, 1], [0.5, 0.2, 1], [0.35, 0.5, 1]], np.float32),
        "valid": np.ones(3, np.bool_),
        "in_frame": np.ones(3, np.bool_),
        "k": np.asarray([[-10, 0, 4], [0, 10, 2], [0, 0, 1]], np.float32),
    }


def test_collect_rollout_fake_env_state_timeline_and_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([1.0, 1.025, 2.0, 2.05])
    monkeypatch.setattr(rollout.time, "perf_counter", lambda: next(ticks))
    env = _Env()
    client = _Client()
    args = SimpleNamespace(
        env_factory=lambda case, resolution: (env, _obs()),
        frame_capture=_capture,
        resolution=5,
        action_horizon=2,
        dummy_steps=10,
        max_steps=4,
    )
    record = collect_rollout(_case(), client, args)
    assert record.anchor_steps.tolist() == [0, 2]
    assert len(record.executed_actions) == 4
    assert record.agent_rgb.shape[0] == 5
    assert record.agent_depth.shape == (5, 3, 5)
    assert record.realized_uvd.shape[0] == 5
    np.testing.assert_array_equal(np.asarray(env.received[10:]), record.executed_actions)
    np.testing.assert_array_equal(record.executed_actions[:, 6], [1, -1, 1, -1])
    np.testing.assert_array_equal(record.policy_actions_raw[:, 6], [0, 1, 0, 1])
    np.testing.assert_allclose(record.latency_ms, [25.0, 50.0])
    assert record.anchor_target_valid[1, 1].all()
    assert np.isfinite(record.metadata["metrics"]["anchor_0"]["camera"]["span_mae_mm"])
    assert client.requests[0]["return_geometry"]
    assert client.requests[0]["inference_seed"] == (7 << 16)
    assert client.requests[1]["inference_seed"] == (7 << 16 | 2)
    np.testing.assert_array_equal(
        client.requests[0]["examples"][0]["image"][0],  # type: ignore[index]
        np.full((3, 5, 3), 11, np.uint8),
    )
    np.testing.assert_array_equal(
        client.requests[0]["examples"][0]["image"][1],  # type: ignore[index]
        np.full((3, 5, 3), 22, np.uint8),
    )
    np.testing.assert_array_equal(record.realized_xyz[0, :, 0], [1, 2, 3])
    assert env.closed


def test_execute_cadence_accepts_direct_and_data_wrapped_responses() -> None:
    responses = iter(
        [
            {"actions": np.zeros((2, 7), np.float32)},
            {"data": {"actions": np.zeros((2, 7), np.float32)}},
        ]
    )
    anchors, raw, _ = execute_cadenced_actions(
        max_steps=3,
        action_horizon=2,
        audit_seed=7,
        dummy_steps=10,
        stabilize=lambda _: None,
        observe=lambda _: {"image": [np.zeros((2, 2, 3), np.uint8)] * 2},
        request=lambda _: next(responses),
        execute=lambda *_: False,
    )
    assert anchors.tolist() == [0, 2]
    assert raw.shape == (3, 7)


def test_suite_specific_max_step_defaults_and_explicit_override() -> None:
    assert {
        suite: _rollout_step_limit(suite, None)
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
    } == {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    assert _rollout_step_limit("libero_spatial", 17) == 17
