from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_rollout import (
    RolloutRecord,
    save_rollout_record,
)
from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_selection import AuditCase
from examples.simBenchmarks.CoT.geometry_probe import run_libero_v3_trace_audit as cli


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _write_logs(root: Path) -> None:
    root.mkdir(parents=True)
    patterns = (
        (True, True, True, True, True),
        (True, True, True, True, False),
        (True, True, True, False, False),
        (True, True, False, False, False),
        (True, False, False, False, False),
        (False, False, False, False, False),
    )
    for gpu, suite in enumerate(SUITES):
        lines = []
        for task_id, outcomes in enumerate(patterns):
            for outcome in outcomes:
                lines.extend((f"Task: {suite} instruction {task_id}", f"Success: {outcome}"))
            lines.append(f"Current task success rate: {sum(outcomes) / len(outcomes):.2f}")
        (root / f"{suite}_gpu{gpu}.worker.log").write_text("\n".join(lines) + "\n")


def _checkpoint(path: Path, contents: bytes = b"checkpoint") -> Path:
    path.write_bytes(contents)
    return path


def _libero_runtime(root: Path) -> tuple[Path, Path]:
    home = root / "LIBERO"
    config = home / "libero"
    config.mkdir(parents=True)
    (config / "config.yaml").write_text("benchmark_root: fixture\n")
    return home, config


def _plan_args(checkpoint: Path, logs: Path, output: Path, *extra: str) -> list[str]:
    return [
        "--phase",
        "plan",
        "--checkpoint",
        str(checkpoint),
        "--source-log-dir",
        str(logs),
        "--output-dir",
        str(output),
        "--dry-run",
        *extra,
    ]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _record(case: AuditCase, config_identity: str) -> RolloutRecord:
    actions, anchors, horizon, height, width = 2, 1, 2, 3, 5
    predicted = np.zeros((anchors, 2, 3, 3), np.float32)
    predicted[..., 2] = 1
    states = np.zeros((actions + 1, 3, 3), np.float32)
    states[..., 2] = 1
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
        predicted_uvd=predicted,
        predicted_uvd_time=np.asarray([[0, 0, 0, 1, 1, 1]], np.float32),
        predicted_uvd_landmark_ids=np.asarray([[0, 1, 2, 0, 1, 2]], np.int64),
        predicted_depth_current=np.ones((anchors, height, width), np.float32),
        predicted_depth_future=np.ones((anchors, height, width), np.float32),
        realized_uvd=states,
        realized_xyz=np.ones_like(states),
        realized_valid=np.ones((actions + 1, 3), np.bool_),
        realized_in_frame=np.ones((actions + 1, 3), np.bool_),
        anchor_target_uvd=states[None, [0, 2]],
        anchor_target_valid=np.ones((anchors, 2, 3), np.bool_),
        dense_depth_current_target=np.ones((anchors, height, width), np.float32),
        dense_depth_future_target=np.ones((anchors, height, width), np.float32),
        latency_ms=np.asarray([1.0], np.float64),
        camera_k_agentview_flipped=np.asarray(
            [[-10, 0, 2], [0, -11, 1], [0, 0, 1]], np.float32
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
            "metrics": {
                "anchor_0": {
                    name: {
                        "uv_ade_px": float(index + 1),
                        "uv_fde_px": float(index + 1.5),
                        "d_mae_mm": float(index + 2),
                        "delta_d_mae_mm": float("nan"),
                        "delta_d_direction_accuracy": 0.75,
                        "valid_count": 6,
                    }
                    for index, name in enumerate(("left", "right", "wrist", "aggregate"))
                } | {
                    "persistence": {
                        "aggregate": {
                            "uv_fde_px": 9.0,
                            "delta_d_direction_accuracy": 0.25,
                        }
                    },
                    "camera": {"center_mae_mm": 4.0, "center_valid_count": 2},
                },
            },
            "schema": "state_timeline_v2",
            "audit_config_identity": config_identity,
        },
    )


def _materialize_raw_and_shards(output: Path, manifest: dict[str, object]) -> None:
    for planned in manifest["cases"]:
        case = AuditCase(**{key: planned[key] for key in cli.CASE_FIELDS})
        save_rollout_record(
            output / "raw" / planned["artifact_id"],
            _record(case, manifest["config_identity"]),
        )
    for suite in manifest["config"]["suite_filter"]:
        cli.execute_worker(output, suite)


def test_dry_run_writes_stable_eighty_case_manifest_without_runtime_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    monkeypatch.setattr(cli, "collect_rollout", lambda *_: pytest.fail("simulator called"))
    monkeypatch.setattr(cli, "_make_policy_client", lambda *_: pytest.fail("server called"))
    monkeypatch.setattr(
        cli, "_apply_runtime_environment", lambda *_: pytest.fail("runtime env applied")
    )

    assert cli.main(_plan_args(checkpoint, logs, output)) == 0
    first = (output / "run_manifest.json").read_bytes()
    assert cli.main(_plan_args(checkpoint, logs, output)) == 0
    assert (output / "run_manifest.json").read_bytes() == first

    selection = _read_json(output / "selection.json")
    manifest = _read_json(output / "run_manifest.json")
    assert len(selection["tasks"]) == 16
    assert len(selection["evaluations"]) == 24
    assert selection["evaluations"][0]["outcomes"] == [True] * 5
    assert manifest["seeds"] == [7, 8, 9, 10, 11]
    assert len(manifest["cases"]) == 80
    ids = [case["artifact_id"] for case in manifest["cases"]]
    assert len(ids) == len(set(ids)) == 80
    assert all(re.fullmatch(r"[a-z0-9_-]+", value) for value in ids)
    assert all(all(token in value for token in ("task-", "rank-", "init-", "seed-")) for value in ids)
    assert "16 selected tasks" in capsys.readouterr().out
    for name in ("raw", "videos", "summaries", "contact_sheets"):
        assert (output / name).is_dir()


def test_max_cases_one_plans_exactly_one_case(tmp_path: Path) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")

    assert cli.main(_plan_args(checkpoint, logs, output, "--max-cases", "1")) == 0

    manifest = _read_json(output / "run_manifest.json")
    assert len(manifest["cases"]) == 1
    assert manifest["expected_artifacts"] == {
        "episodes": 1,
        "videos": 1,
        "rollout_summaries": 1,
        "task_seed_summaries": 0,
        "suite_contact_sheets": 0,
    }
    assert manifest["full_audit"] is False


def test_plan_records_nonempty_runtime_environment_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    home, config = _libero_runtime(tmp_path)
    monkeypatch.setenv("LIBERO_HOME", str(home))
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config))
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")

    cli.main(
        _plan_args(
            _checkpoint(tmp_path / "model.pt"), logs, output, "--max-cases", "1"
        )
    )

    config_payload = _read_json(output / "run_manifest.json")["config"]
    assert config_payload["libero_home"] == str(home.resolve())
    assert config_payload["libero_config_path"] == str(config.resolve())
    assert config_payload["mujoco_gl"] == "osmesa"
    assert config_payload["pyopengl_platform"] == "osmesa"


@pytest.mark.parametrize("phase", ("plan", "run"))
@pytest.mark.parametrize("configured", ("home-only", "config-only"))
def test_plan_and_run_reject_one_sided_runtime_paths_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    configured: str,
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    home, config = _libero_runtime(tmp_path)
    monkeypatch.delenv("LIBERO_HOME", raising=False)
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    monkeypatch.setattr(
        cli,
        "_apply_runtime_environment",
        lambda *_: (_ for _ in ()).throw(RuntimeError("runtime reached")),
    )
    args = [
        "--phase",
        phase,
        "--checkpoint",
        str(_checkpoint(tmp_path / "model.pt")),
        "--source-log-dir",
        str(logs),
        "--output-dir",
        str(output),
        "--max-cases",
        "1",
    ]
    if configured == "home-only":
        args.extend(("--libero-home", str(home)))
    else:
        args.extend(("--libero-config-path", str(config)))

    with pytest.raises(ValueError, match="configured together"):
        cli.main(args)
    assert not output.exists()
    assert not (output / "selection.json").exists()
    assert not (output / "run_manifest.json").exists()


@pytest.mark.parametrize("configured", ("both-none", "both-set"))
def test_plan_allows_runtime_paths_both_unset_or_both_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
) -> None:
    logs, output = tmp_path / "logs", tmp_path / configured
    _write_logs(logs)
    monkeypatch.delenv("LIBERO_HOME", raising=False)
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    extra: tuple[str, ...] = ()
    expected: tuple[str | None, str | None] = (None, None)
    if configured == "both-set":
        home, config = _libero_runtime(tmp_path)
        extra = (
            "--libero-home", str(home),
            "--libero-config-path", str(config),
        )
        expected = (str(home.resolve()), str(config.resolve()))

    cli.main(
        _plan_args(
            _checkpoint(tmp_path / "model.pt"), logs, output, "--max-cases", "1", *extra
        )
    )

    config_payload = _read_json(output / "run_manifest.json")["config"]
    assert (
        config_payload["libero_home"], config_payload["libero_config_path"]
    ) == expected


@pytest.mark.parametrize("missing_key", ("libero_home", "libero_config_path"))
def test_manifest_rejects_one_sided_runtime_paths_after_digest_recomputed(
    tmp_path: Path, missing_key: str,
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    home, config = _libero_runtime(tmp_path)
    cli.main(
        _plan_args(
            _checkpoint(tmp_path / "model.pt"),
            logs,
            output,
            "--max-cases",
            "1",
            "--libero-home",
            str(home),
            "--libero-config-path",
            str(config),
        )
    )
    selection = _read_json(output / "selection.json")
    manifest = _read_json(output / "run_manifest.json")
    manifest["config"][missing_key] = None
    cli._finalize_manifest(manifest)

    with pytest.raises(ValueError, match="configured together"):
        cli._validate_manifest(manifest, selection)


def test_direct_worker_applies_manifest_runtime_environment_before_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    home, config = _libero_runtime(tmp_path)
    cli.main(
        _plan_args(
            checkpoint,
            logs,
            output,
            "--max-cases",
            "1",
            "--libero-home",
            str(home),
            "--libero-config-path",
            str(config),
            "--mujoco-gl",
            "egl",
            "--pyopengl-platform",
            "egl",
        )
    )
    manifest = _read_json(output / "run_manifest.json")
    suite = manifest["cases"][0]["suite"]
    for name in ("LIBERO_HOME", "LIBERO_CONFIG_PATH", "MUJOCO_GL", "PYOPENGL_PLATFORM"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(sys, "path", list(sys.path))
    events: list[str] = []

    def client(_host: str, _port: int) -> object:
        events.append("client")
        assert os.environ["LIBERO_HOME"] == str(home.resolve())
        assert os.environ["LIBERO_CONFIG_PATH"] == str(config.resolve())
        assert os.environ["MUJOCO_GL"] == "egl"
        assert os.environ["PYOPENGL_PLATFORM"] == "egl"
        assert sys.path[0] == str(home.resolve())
        return object()

    def collect(case: AuditCase, _client: object, _args: object) -> RolloutRecord:
        events.append("collect")
        return _record(case, "replaced-by-worker")

    monkeypatch.setattr(cli, "_make_policy_client", client)
    monkeypatch.setattr(cli, "collect_rollout", collect)
    assert cli.main(
        ["--phase", "worker", "--output-dir", str(output), "--worker-suite", suite]
    ) == 0
    assert events == ["client", "collect"]


def test_worker_rejects_runtime_environment_conflict_before_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    home, config = _libero_runtime(tmp_path)
    cli.main(
        _plan_args(
            checkpoint,
            logs,
            output,
            "--max-cases",
            "1",
            "--libero-home",
            str(home),
            "--libero-config-path",
            str(config),
        )
    )
    manifest = _read_json(output / "run_manifest.json")
    suite = manifest["cases"][0]["suite"]
    monkeypatch.setenv("MUJOCO_GL", "osmesa")

    with pytest.raises(ValueError, match="runtime environment conflict"):
        cli.execute_worker(
            output,
            suite,
            client_factory=lambda *_: pytest.fail("client created before conflict"),
        )


def test_existing_manifest_rejects_checkpoint_or_selection_mismatch(tmp_path: Path) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    first = _checkpoint(tmp_path / "first.pt", b"one")
    second = _checkpoint(tmp_path / "second.pt", b"two")
    cli.main(_plan_args(first, logs, output))
    original = (output / "run_manifest.json").read_bytes()

    with pytest.raises(ValueError, match="manifest conflict"):
        cli.main(_plan_args(second, logs, output))
    assert (output / "run_manifest.json").read_bytes() == original

    text = (logs / "libero_goal_gpu2.worker.log").read_text()
    (logs / "libero_goal_gpu2.worker.log").write_text(text.replace("Success: False", "Success: True", 1))
    with pytest.raises(ValueError, match="manifest conflict|logged rate"):
        cli.main(_plan_args(first, logs, output))
    assert (output / "run_manifest.json").read_bytes() == original


def test_checkpoint_identity_ignores_touch_but_rejects_same_size_content_change(
    tmp_path: Path,
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt", b"abcdefgh")
    cli.main(_plan_args(checkpoint, logs, output))
    manifest = _read_json(output / "run_manifest.json")
    original_identity = manifest["config_identity"]
    original_bytes = (output / "run_manifest.json").read_bytes()

    stat = checkpoint.stat()
    os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    cli.main(_plan_args(checkpoint, logs, output))
    assert _read_json(output / "run_manifest.json")["config_identity"] == original_identity
    assert (output / "run_manifest.json").read_bytes() == original_bytes

    checkpoint.write_bytes(b"ABCDEFGH")
    with pytest.raises(ValueError, match="manifest conflict"):
        cli.main(_plan_args(checkpoint, logs, output))


@pytest.mark.parametrize(
    "target,mutate",
    [
        ("run_manifest.json", lambda value: value["config"].__setitem__("resolution", 0)),
        ("run_manifest.json", lambda value: value["ports"].__setitem__("libero_goal", 9999)),
        ("run_manifest.json", lambda value: value["cases"][0].__setitem__("seed", 99)),
        ("selection.json", lambda value: value["tasks"][0].__setitem__("task_id", 99)),
    ],
)
def test_manifest_loader_rejects_any_selection_or_run_tamper(
    tmp_path: Path, target: str, mutate: object
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    cli.main(_plan_args(checkpoint, logs, output, "--max-cases", "1"))
    path = output / target
    value = _read_json(path)
    mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest|selection"):
        cli.execute_worker(output, "libero_10")


def test_strict_schema_rejects_coerced_integer_and_boolean_types(tmp_path: Path) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    cli.main(_plan_args(checkpoint, logs, output, "--max-cases", "1"))

    selection = _read_json(output / "selection.json")
    selection["tasks"][0]["outcome_count"] = float(
        selection["tasks"][0]["outcome_count"]
    )
    with pytest.raises(ValueError, match="selected task"):
        cli._validate_selection(selection)

    selection = _read_json(output / "selection.json")
    manifest = _read_json(output / "run_manifest.json")
    manifest["cases"][0]["seed"] = True
    cli._finalize_manifest(manifest)
    with pytest.raises(ValueError, match="case"):
        cli._validate_manifest(manifest, selection)


def test_schema_versions_reject_boolean_values(tmp_path: Path) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    cli.main(_plan_args(checkpoint, logs, output, "--max-cases", "1"))

    selection = _read_json(output / "selection.json")
    selection["schema_version"] = True
    with pytest.raises(ValueError, match="schema"):
        cli._validate_selection(selection)

    selection = _read_json(output / "selection.json")
    manifest = _read_json(output / "run_manifest.json")
    manifest["schema_version"] = True
    cli._finalize_manifest(manifest)
    with pytest.raises(ValueError, match="version"):
        cli._validate_manifest(manifest, selection)


@pytest.mark.parametrize(
    "extra",
    [
        ("--resolution", "0"), ("--resolution", "1"),
        ("--action-horizon", "0"), ("--dummy-steps", "-1"),
    ],
)
def test_invalid_plan_ranges_fail_before_manifest_write(tmp_path: Path, extra: tuple[str, str]) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")

    with pytest.raises(ValueError):
        cli.main(_plan_args(checkpoint, logs, output, *extra))
    assert not (output / "run_manifest.json").exists()


def test_suite_names_are_canonical_and_cannot_traverse_source_or_shards(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _write_logs(logs)
    with pytest.raises(ValueError, match="canonical"):
        cli.prepare_plan(
            checkpoint=_checkpoint(tmp_path / "model.pt"),
            source_log_dir=logs,
            output_dir=tmp_path / "audit",
            suites=("../x",),
            dry_run=True,
        )


def test_worker_skips_only_valid_identity_and_rebuilds_one_deduplicated_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    cli.main(_plan_args(checkpoint, logs, output, "--max-cases", "1"))
    manifest = _read_json(output / "run_manifest.json")
    planned = manifest["cases"][0]
    case = AuditCase(**{key: planned[key] for key in cli.CASE_FIELDS})
    base = output / "raw" / planned["artifact_id"]
    save_rollout_record(base, _record(case, manifest["config_identity"]))
    calls: list[AuditCase] = []

    def collect(item: AuditCase, _client: object, _args: object) -> RolloutRecord:
        calls.append(item)
        return _record(item, manifest["config_identity"])

    monkeypatch.setattr(cli, "collect_rollout", collect)
    monkeypatch.setattr(cli, "_make_policy_client", lambda *_: object())
    worker_args = [
        "--phase", "worker", "--output-dir", str(output), "--worker-suite", case.suite
    ]
    assert cli.main(worker_args) == 0
    assert calls == []
    pointer = base.with_suffix(".manifest.json")
    pointer.write_text("{broken")
    assert cli.main(worker_args) == 0
    assert calls == [case]
    assert cli.main(worker_args) == 0
    assert calls == [case]
    shard = output / "raw" / "workers" / f"{case.suite}.episodes.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    assert [row["artifact_id"] for row in rows] == [planned["artifact_id"]]


def test_worker_dry_run_never_connects_collects_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    cli.main(_plan_args(checkpoint, logs, output, "--max-cases", "1"))
    manifest = _read_json(output / "run_manifest.json")
    suite = manifest["cases"][0]["suite"]
    monkeypatch.setattr(cli, "_make_policy_client", lambda *_: pytest.fail("connected"))
    monkeypatch.setattr(cli, "collect_rollout", lambda *_: pytest.fail("collected"))
    monkeypatch.setattr(cli, "save_rollout_record", lambda *_: pytest.fail("saved"))
    monkeypatch.setattr(
        cli, "_apply_runtime_environment", lambda *_: pytest.fail("runtime env applied")
    )

    assert cli.main([
        "--phase", "worker", "--output-dir", str(output), "--worker-suite", suite, "--dry-run"
    ]) == 0
    assert "would process 1 cases" in capsys.readouterr().out
    assert not (output / "raw" / "workers" / f"{suite}.episodes.jsonl").exists()


def test_aggregate_dry_run_does_not_load_raw_or_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    cli.main(_plan_args(_checkpoint(tmp_path / "model.pt"), logs, output, "--max-cases", "1"))
    monkeypatch.setattr(cli, "load_rollout_record", lambda *_args, **_kwargs: pytest.fail("loaded raw"))
    monkeypatch.setattr(cli, "render_rollout_video", lambda *_: pytest.fail("rendered"))
    assert cli.main(["--phase", "aggregate", "--output-dir", str(output), "--dry-run"]) == 0


def test_max_cases_worker_with_no_assigned_case_writes_an_empty_shard(tmp_path: Path) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    cli.main(_plan_args(checkpoint, logs, output, "--max-cases", "1"))

    rows = cli.execute_worker(output, "libero_goal")

    assert rows == []
    assert (output / "raw" / "workers" / "libero_goal.episodes.jsonl").read_text() == ""


def test_worker_rejects_port_that_differs_from_manifest(tmp_path: Path) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    cli.main(_plan_args(checkpoint, logs, output, "--max-cases", "1"))

    with pytest.raises(ValueError, match="port.*manifest"):
        cli.execute_worker(output, "libero_spatial", port=7777)


def test_normal_aggregate_requires_validated_worker_shards(tmp_path: Path) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    cli.main(_plan_args(
        _checkpoint(tmp_path / "model.pt"), logs, output, "--max-cases", "1"
    ))
    manifest = _read_json(output / "run_manifest.json")
    _materialize_raw_and_shards(output, manifest)

    summary = cli.aggregate_run(output, render=False, source_mode="worker_shards")

    assert summary["artifact_counts"] == {"episodes": 1}
    assert len((output / "episodes.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "extra", "corrupt", "mismatch", "extra_file"]
)
def test_normal_aggregate_rejects_invalid_worker_shards(
    tmp_path: Path, corruption: str
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    cli.main(_plan_args(
        _checkpoint(tmp_path / "model.pt"), logs, output, "--max-cases", "1"
    ))
    manifest = _read_json(output / "run_manifest.json")
    _materialize_raw_and_shards(output, manifest)
    suite = manifest["cases"][0]["suite"]
    shard = output / "raw" / "workers" / f"{suite}.episodes.jsonl"
    lines = shard.read_text().splitlines()
    if corruption == "missing":
        shard.unlink()
    elif corruption == "duplicate":
        shard.write_text("\n".join([lines[0], lines[0]]) + "\n")
    elif corruption == "extra":
        row = json.loads(lines[0]); row["artifact_id"] = "extra"
        shard.write_text(lines[0] + "\n" + json.dumps(row) + "\n")
    elif corruption == "corrupt":
        shard.write_text("{broken\n")
    elif corruption == "mismatch":
        row = json.loads(lines[0]); row["outcome"]["success"] = not row["outcome"]["success"]
        shard.write_text(json.dumps(row) + "\n")
    else:
        (shard.parent / "unexpected.episodes.jsonl").write_text("")

    with pytest.raises(ValueError, match="shard"):
        cli.aggregate_run(output, render=False, source_mode="worker_shards")


def test_aggregate_validates_counts_deduplicates_and_passes_explicit_rank_to_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    cli.main(_plan_args(checkpoint, logs, output))
    manifest = _read_json(output / "run_manifest.json")
    for planned in manifest["cases"]:
        case = AuditCase(**{key: planned[key] for key in cli.CASE_FIELDS})
        save_rollout_record(
            output / "raw" / planned["artifact_id"],
            _record(case, manifest["config_identity"]),
        )

    seen_ranks: list[list[str]] = []

    def touch(_record: RolloutRecord, path: str | Path, *args: object, **kwargs: object) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"artifact")
        return target

    def sheet(items: list[tuple[Path, str]], path: str | Path) -> Path:
        seen_ranks.append([rank for _, rank in items])
        target = Path(path)
        target.write_bytes(b"sheet")
        return target

    monkeypatch.setattr(cli, "render_rollout_video", touch)
    monkeypatch.setattr(cli, "render_rollout_summary", touch)
    monkeypatch.setattr(cli, "render_task_seed_summary", lambda records, path: touch(records[0], path))
    monkeypatch.setattr(cli, "render_suite_contact_sheet", sheet)

    assert cli.main(["--phase", "aggregate", "--output-dir", str(output), "--render-only"]) == 0

    episodes = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == len({row["artifact_id"] for row in episodes}) == 80
    assert len(list((output / "videos").glob("*.mp4"))) == 80
    assert len(list((output / "summaries").glob("rollout-*.png"))) == 80
    assert len(list((output / "summaries").glob("task-*.png"))) == 16
    assert len(list((output / "contact_sheets").glob("*.png"))) == 4
    assert seen_ranks == [["best", "best", "worst", "worst"]] * 4
    summary = _read_json(output / "summary.json")
    assert summary["artifact_counts"] == manifest["expected_artifacts"]
    assert summary["full_audit"] is True
    assert {group["group_type"] for group in summary["groups"]} >= {
        "landmark", "task", "rank", "suite", "seed"
    }
    metric_paths = {item["metric_path"] for item in summary["observations"]}
    assert {
        "left.uv_fde_px",
        "left.delta_d_direction_accuracy",
        "persistence.aggregate.uv_fde_px",
        "camera.center_mae_mm",
    } <= metric_paths
    assert all({"mean", "std", "count"} <= set(group) for group in summary["groups"])
    episode = summary["episode_table"][0]
    assert episode["success"] is False
    assert episode["outcome_changed"] is True
    assert episode["action_steps"] == 2
    assert episode["state_steps"] == 3
    assert episode["latency_mean_ms"] == 1.0
    assert episode["latency_p95_ms"] == 1.0
    episode_groups = summary["episode_groups"]
    assert {group["group_type"] for group in episode_groups} == {
        "task", "rank_group", "suite", "seed",
    }
    assert {
        "success",
        "outcome_mismatch",
        "action_steps",
        "state_steps",
        "anchor_count",
        "latency_mean_ms",
        "latency_p95_ms",
    } <= {group["metric_name"] for group in episode_groups}
    assert all({"count", "mean", "std"} <= set(group) for group in episode_groups)
    csv_rows = list(csv.DictReader((output / "summary.csv").read_text().splitlines()))
    episode_csv = [row for row in csv_rows if row["row_kind"] == "episode_scalar"]
    assert episode_csv
    assert {row["category"] for row in episode_csv} == {
        "task", "rank_group", "suite", "seed",
    }
    assert {"success", "outcome_mismatch", "action_steps", "latency_mean_ms"} <= {
        row["metric_name"] for row in episode_csv
    }
    assert "NaN" not in (output / "summary.json").read_text()


def test_nonstandard_seeds_do_not_claim_fixed_five_seed_summaries(tmp_path: Path) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")

    cli.main(_plan_args(checkpoint, logs, output, "--seeds", "1,2"))

    expected = _read_json(output / "run_manifest.json")["expected_artifacts"]
    assert expected["episodes"] == 32
    assert expected["task_seed_summaries"] == 0
    assert expected["suite_contact_sheets"] == 0


@pytest.mark.parametrize("home_value", ("empty", "missing"))
def test_launcher_live_preflight_rejects_bad_libero_before_manifest_write(
    tmp_path: Path, home_value: str,
) -> None:
    logs = tmp_path / "logs"
    _write_logs(logs)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _checkpoint(model_dir / "model.pt")
    output = tmp_path / "audit"
    bad_home = tmp_path / "missing-LIBERO"
    script = Path("examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.sh")
    base_env = {
        "PATH": "/usr/bin:/bin",
        "MODEL_DIR": str(model_dir),
        "CKPT_NAME": "model.pt",
        "SOURCE_LOG_DIR": str(logs),
        "OUTPUT_DIR": str(output),
        "POLICY_PYTHON": "/bin/false",
        "SIM_PYTHON": str(
            Path("/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python")
        ),
        "LIBERO_HOME": "" if home_value == "empty" else str(bad_home),
        "LIBERO_CONFIG_PATH": str(bad_home / "libero"),
        "SERVER_READY_TIMEOUT": "1",
    }
    failed = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).parents[1],
        env=base_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode == 2
    assert "LIBERO_HOME" in failed.stderr
    assert not output.exists()

    config = bad_home / "libero"
    config.mkdir(parents=True)
    (config / "config.yaml").write_text("benchmark_root: fixture\n")
    retried = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).parents[1],
        env={
            **base_env,
            "DRY_RUN": "1",
            "LIBERO_HOME": str(bad_home),
            "LIBERO_CONFIG_PATH": str(config),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert retried.returncode == 0, retried.stderr
    assert (output / "run_manifest.json").is_file()


def test_launcher_dry_run_prints_four_suites_and_eighty_cases_without_starting(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    _write_logs(logs)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _checkpoint(model_dir / "model.pt")
    libero_home = tmp_path / "LIBERO home"
    libero_home.mkdir()
    config_path = tmp_path / "config path"
    script = Path("examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.sh")
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).parents[1],
        env={
            "PATH": "/usr/bin:/bin",
            "DEBUG": "release",
            "DRY_RUN": "1",
            "MODEL_DIR": str(model_dir),
            "CKPT_NAME": "model.pt",
            "SOURCE_LOG_DIR": str(logs),
            "OUTPUT_DIR": str(tmp_path / "audit"),
            "POLICY_PYTHON": "/bin/false",
            "SIM_PYTHON": str(Path("/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python")),
            "LIBERO_HOME": str(libero_home),
            "LIBERO_CONFIG_PATH": str(config_path),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("suite=libero_") == 4
    assert "80 cases" in result.stdout
    server_commands = [
        line for line in result.stdout.splitlines()
        if line.startswith("[trace-audit] server command:")
    ]
    assert len(server_commands) == 4
    assert all(" env -u DEBUG CUDA_VISIBLE_DEVICES=" in line for line in server_commands)
    assert "no server or worker was started" in result.stdout
    assert str(libero_home).replace(" ", r"\ ") in result.stdout
    assert str(config_path).replace(" ", r"\ ") in result.stdout
    manifest = _read_json(tmp_path / "audit" / "run_manifest.json")
    assert manifest["config"]["libero_home"] == str(libero_home)
    assert manifest["config"]["libero_config_path"] == str(config_path)
    assert manifest["config"]["mujoco_gl"] == "egl"
    assert manifest["config"]["pyopengl_platform"] == "egl"


def test_launcher_contains_fail_fast_all_server_and_worker_process_safety() -> None:
    script = Path(
        "examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.sh"
    ).read_text()
    assert "wait_for_all_servers" in script
    assert "wait_for_workers_fail_fast" in script
    assert "SERVER_READY" in script
    assert "terminate_group" in script
    assert "LIBERO_HOME" in script and "LIBERO_CONFIG_PATH" in script
    assert "PYOPENGL_PLATFORM" in script and "MUJOCO_GL" in script
    assert script.count("SERVER_CMD=(env -u DEBUG") == 2
    assert "pkill" not in script and "killall" not in script
