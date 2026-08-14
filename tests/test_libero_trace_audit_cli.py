from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess

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
                        "d_mae_mm": float(index + 2),
                        "delta_d_mae_mm": float("nan"),
                    }
                    for index, name in enumerate(("left", "right", "wrist", "aggregate"))
                }
            },
            "schema": "state_timeline_v2",
            "audit_config_identity": config_identity,
        },
    )


def test_dry_run_writes_stable_eighty_case_manifest_without_runtime_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    logs, output = tmp_path / "logs", tmp_path / "audit"
    _write_logs(logs)
    checkpoint = _checkpoint(tmp_path / "model.pt")
    monkeypatch.setattr(cli, "collect_rollout", lambda *_: pytest.fail("simulator called"))
    monkeypatch.setattr(cli, "_make_policy_client", lambda *_: pytest.fail("server called"))

    assert cli.main(_plan_args(checkpoint, logs, output)) == 0
    first = (output / "run_manifest.json").read_bytes()
    assert cli.main(_plan_args(checkpoint, logs, output)) == 0
    assert (output / "run_manifest.json").read_bytes() == first

    selection = _read_json(output / "selection.json")
    manifest = _read_json(output / "run_manifest.json")
    assert len(selection["tasks"]) == 16
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


def test_launcher_dry_run_prints_four_suites_and_eighty_cases_without_starting(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    _write_logs(logs)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _checkpoint(model_dir / "model.pt")
    script = Path("examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.sh")
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).parents[1],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "MODEL_DIR": str(model_dir),
            "CKPT_NAME": "model.pt",
            "SOURCE_LOG_DIR": str(logs),
            "OUTPUT_DIR": str(tmp_path / "audit"),
            "POLICY_PYTHON": "/bin/false",
            "SIM_PYTHON": str(Path("/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python")),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("suite=libero_") == 4
    assert "80 cases" in result.stdout
    assert "no server or worker was started" in result.stdout
