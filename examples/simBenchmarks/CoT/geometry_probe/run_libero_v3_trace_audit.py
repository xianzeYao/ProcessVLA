#!/usr/bin/env python3
"""Plan, resume, aggregate, and render reproducible LIBERO V3 trace audits."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .libero_trace_audit_metrics import metrics_to_jsonable
from .libero_trace_audit_rollout import (
    RolloutRecord,
    collect_rollout,
    load_rollout_record,
    save_rollout_record,
)
from .libero_trace_audit_selection import (
    AuditCase,
    TaskEvaluation,
    build_audit_cases,
    parse_worker_log,
    select_extreme_tasks,
    validate_audit_cases,
)
from .libero_trace_audit_visualization import (
    render_rollout_summary,
    render_rollout_video,
    render_suite_contact_sheet,
    render_task_seed_summary,
)


SCHEMA_VERSION = 1
TOOL_VERSION = "libero-v3-trace-audit-v1"
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_SEEDS = (7, 8, 9, 10, 11)
CASE_FIELDS = (
    "suite",
    "task_id",
    "language",
    "rank_group",
    "initial_state_index",
    "seed",
    "original_success",
)
LANDMARKS = ("left", "right", "wrist", "aggregate")
SELECTION_KEYS = {"schema", "schema_version", "selection_policy", "evaluations", "tasks"}
EVALUATION_KEYS = {"suite", "task_id", "language", "outcomes", "success_rate"}
SELECTED_TASK_KEYS = {
    "suite", "task_id", "language", "rank_group", "success_rate", "outcome_count",
    "initial_state_index", "original_success",
}
MANIFEST_KEYS = {
    "schema", "schema_version", "tool_version", "checkpoint_identity",
    "selection_digest", "config_identity", "manifest_digest", "suites", "seeds",
    "config", "ports", "cases", "expected_artifacts", "full_audit",
    "aggregation_sources",
}
CHECKPOINT_KEYS = {"resolved_path", "exists", "size", "mtime_ns", "content_sha256"}
EXPECTED_KEYS = {
    "episodes", "videos", "rollout_summaries", "task_seed_summaries",
    "suite_contact_sheets",
}
CONFIG_KEYS = {
    "suites", "seeds", "host", "base_port", "ports", "suite_filter", "task_filter",
    "max_cases", "resolution", "action_horizon", "dummy_steps", "unnorm_key",
    "libero_home", "libero_config_path", "mujoco_gl", "pyopengl_platform",
}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _contained(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as error:
        raise ValueError(f"{label} escapes its configured root") from error
    return resolved


def _strict_load(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            metrics_to_jsonable(value),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _digest(value: object) -> str:
    canonical = json.dumps(
        metrics_to_jsonable(value), allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _csv_values(value: str | Sequence[object]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = [str(item) for item in value]
    result = tuple(item.strip() for item in values if item.strip())
    if not result:
        raise ValueError("comma-separated value must not be empty")
    return result


def _integers(value: str | Sequence[object], name: str) -> tuple[int, ...]:
    raw = _csv_values(value)
    try:
        result = tuple(int(item) for item in raw)
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integers") from error
    if len(result) != len(set(result)) or any(item < 0 for item in result):
        raise ValueError(f"{name} must contain unique non-negative integers")
    return result


def _safe_token(value: object) -> str:
    token = re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-_")
    if not token:
        raise ValueError(f"cannot make path-safe token from {value!r}")
    return token


def artifact_id(case: AuditCase | Mapping[str, object]) -> str:
    item = asdict(case) if isinstance(case, AuditCase) else dict(case)
    return "__".join(
        (
            _safe_token(item["suite"]),
            f"task-{int(item['task_id']):03d}",
            f"rank-{_safe_token(item['rank_group'])}",
            f"init-{int(item['initial_state_index']):03d}",
            f"seed-{int(item['seed']):03d}",
        )
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_identity(path: str | Path, *, allow_missing: bool) -> dict[str, object]:
    checkpoint = Path(path).expanduser().resolve(strict=False)
    if not checkpoint.is_file():
        if not allow_missing:
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        return {
            "resolved_path": str(checkpoint), "exists": False, "size": None,
            "mtime_ns": None, "content_sha256": None,
        }
    stat = checkpoint.stat()
    return {
        "resolved_path": str(checkpoint),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "content_sha256": _file_sha256(checkpoint),
    }


def _worker_log(source: Path, suite: str, suite_index: int) -> Path:
    candidates = (
        source / f"{suite}_gpu{suite_index}.worker.log",
        source / f"{suite}.worker.log",
        source / f"{suite}.log",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) == 1:
        return _contained(existing[0], source, "worker log")
    if len(existing) > 1:
        raise ValueError(f"ambiguous worker logs for {suite}: {existing}")
    matches = sorted(source.glob(f"{suite}*.worker.log"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one worker log for {suite} in {source}")
    return _contained(matches[0], source, "worker log")


def _selection_payload(rows: Sequence[TaskEvaluation]) -> dict[str, object]:
    tasks: list[dict[str, object]] = []
    by_suite: dict[str, list[TaskEvaluation]] = {}
    for row in rows:
        by_suite.setdefault(row.suite, []).append(row)
    for suite in sorted(by_suite, key=lambda name: DEFAULT_SUITES.index(name) if name in DEFAULT_SUITES else name):
        selected = select_extreme_tasks(by_suite[suite], count=2)
        for rank in ("best", "worst"):
            for row in selected[rank]:
                desired = rank == "best"
                initial = next((index for index, outcome in enumerate(row.outcomes) if outcome == desired), 0)
                tasks.append(
                    {
                        "suite": suite,
                        "task_id": row.task_id,
                        "language": row.language,
                        "rank_group": rank,
                        "success_rate": row.success_rate,
                        "outcome_count": len(row.outcomes),
                        "initial_state_index": initial,
                        "original_success": row.outcomes[initial],
                    }
                )
    return {
        "schema": "libero-v3-trace-audit-selection",
        "schema_version": SCHEMA_VERSION,
        "selection_policy": "best-first-disjoint-v1",
        "evaluations": [
            {
                "suite": row.suite,
                "task_id": row.task_id,
                "language": row.language,
                "outcomes": list(row.outcomes),
                "success_rate": row.success_rate,
            }
            for row in sorted(
                rows, key=lambda item: (DEFAULT_SUITES.index(item.suite), item.task_id)
            )
        ],
        "tasks": tasks,
    }


def _validate_suite_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a non-empty suite list")
    suites = tuple(value)
    if len(suites) != len(set(suites)) or any(suite not in DEFAULT_SUITES for suite in suites):
        raise ValueError(f"{name} must contain unique canonical standard suites")
    return suites


def _validate_selection(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != SELECTION_KEYS:
        raise ValueError("selection has an invalid exact schema")
    if (
        value["schema"] != "libero-v3-trace-audit-selection"
        or value["schema_version"] != SCHEMA_VERSION
        or value["selection_policy"] != "best-first-disjoint-v1"
        or not isinstance(value["evaluations"], list)
        or not isinstance(value["tasks"], list)
    ):
        raise ValueError("selection has an unsupported schema")
    evaluations: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in value["evaluations"]:
        if not isinstance(row, dict) or set(row) != EVALUATION_KEYS:
            raise ValueError("selection evaluation has an invalid exact schema")
        suite, task_id = row["suite"], row["task_id"]
        outcomes = row["outcomes"]
        if (
            suite not in DEFAULT_SUITES
            or not _is_int(task_id)
            or task_id < 0
            or not isinstance(row["language"], str)
            or not row["language"].strip()
            or not isinstance(outcomes, list)
            or not outcomes
            or any(not isinstance(outcome, bool) for outcome in outcomes)
            or isinstance(row["success_rate"], bool)
            or not isinstance(row["success_rate"], (int, float))
            or not np.isfinite(row["success_rate"])
            or not np.isclose(row["success_rate"], sum(outcomes) / len(outcomes))
        ):
            raise ValueError("selection evaluation values are invalid")
        key = (str(suite), int(task_id))
        if key in evaluations:
            raise ValueError("selection evaluation task ids must be unique")
        evaluations[key] = row
    selected_keys: set[tuple[str, int]] = set()
    for task in value["tasks"]:
        if not isinstance(task, dict) or set(task) != SELECTED_TASK_KEYS:
            raise ValueError("selection selected task has an invalid exact schema")
        suite, task_id = task["suite"], task["task_id"]
        if suite not in DEFAULT_SUITES or not _is_int(task_id) or task_id < 0:
            raise ValueError("selection selected task identity is invalid")
        key = (str(suite), int(task_id))
        row = evaluations.get(key)
        if row is None or key in selected_keys:
            raise ValueError("selection selected task does not have unique parsed provenance")
        initial = task["initial_state_index"]
        if (
            task["rank_group"] not in {"best", "worst"}
            or not isinstance(task["language"], str)
            or not task["language"].strip()
            or isinstance(task["success_rate"], bool)
            or not isinstance(task["success_rate"], (int, float))
            or not np.isfinite(task["success_rate"])
            or not _is_int(task["outcome_count"])
            or task["outcome_count"] < 1
            or not _is_int(initial)
            or not 0 <= initial < len(row["outcomes"])
            or not isinstance(task["original_success"], bool)
            or task["language"] != row["language"]
            or task["success_rate"] != row["success_rate"]
            or task["outcome_count"] != len(row["outcomes"])
            or task["original_success"] is not row["outcomes"][initial]
        ):
            raise ValueError("selection selected task provenance is inconsistent")
        selected_keys.add(key)
    return value


def _manifest_identity_payload(manifest: Mapping[str, object]) -> dict[str, object]:
    checkpoint = manifest["checkpoint_identity"]
    return {
        "schema": manifest["schema"],
        "schema_version": manifest["schema_version"],
        "tool_version": manifest["tool_version"],
        "checkpoint_content_sha256": checkpoint["content_sha256"],
        "selection_digest": manifest["selection_digest"],
        "config": manifest["config"],
        "suites": manifest["suites"],
        "seeds": manifest["seeds"],
        "ports": manifest["ports"],
        "cases": manifest["cases"],
        "expected_artifacts": manifest["expected_artifacts"],
        "full_audit": manifest["full_audit"],
        "aggregation_sources": manifest["aggregation_sources"],
    }


def _finalize_manifest(manifest: dict[str, object]) -> dict[str, object]:
    manifest["config_identity"] = _digest(_manifest_identity_payload(manifest))
    manifest["manifest_digest"] = _digest(
        {key: item for key, item in manifest.items() if key != "manifest_digest"}
    )
    return manifest


def _validate_manifest(value: object, selection: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != MANIFEST_KEYS:
        raise ValueError("run manifest has an invalid exact schema")
    manifest = value
    if (
        manifest["schema"] != "libero-v3-trace-audit-run"
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["tool_version"] != TOOL_VERSION
    ):
        raise ValueError("run manifest version is unsupported")
    checkpoint = manifest["checkpoint_identity"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != CHECKPOINT_KEYS:
        raise ValueError("checkpoint identity has an invalid exact schema")
    exists = checkpoint["exists"]
    digest = checkpoint["content_sha256"]
    if (
        not isinstance(checkpoint["resolved_path"], str)
        or not isinstance(exists, bool)
        or (checkpoint["size"] is not None and (not _is_int(checkpoint["size"]) or checkpoint["size"] < 0))
        or (checkpoint["mtime_ns"] is not None and (not _is_int(checkpoint["mtime_ns"]) or checkpoint["mtime_ns"] < 0))
        or (digest is not None and (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None))
        or (exists and digest is None)
        or (not exists and any(checkpoint[key] is not None for key in ("size", "mtime_ns", "content_sha256")))
    ):
        raise ValueError("checkpoint identity values are invalid")
    if manifest["selection_digest"] != _digest(selection):
        raise ValueError("selection digest does not match selection.json")
    suites = _validate_suite_list(manifest["suites"], "manifest suites")
    seeds = manifest["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not _is_int(seed) or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("manifest seeds are invalid")
    ports = manifest["ports"]
    if (
        not isinstance(ports, dict)
        or set(ports) != set(suites)
        or any(not _is_int(port) or not 1 <= port <= 65535 for port in ports.values())
        or len(set(ports.values())) != len(ports)
    ):
        raise ValueError("manifest ports are invalid")
    config = manifest["config"]
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("manifest config has an invalid exact schema")
    filtered = _validate_suite_list(config["suite_filter"], "suite filter")
    task_filter = config["task_filter"]
    if task_filter is not None and (
        not isinstance(task_filter, list)
        or not task_filter
        or any(not _is_int(task) or task < 0 for task in task_filter)
        or len(task_filter) != len(set(task_filter))
    ):
        raise ValueError("task filter is invalid")
    max_cases = config["max_cases"]
    optional_strings = ("unnorm_key", "libero_home", "libero_config_path")
    if (
        config["suites"] != list(suites)
        or config["seeds"] != seeds
        or config["ports"] != ports
        or any(suite not in suites for suite in filtered)
        or not isinstance(config["host"], str)
        or not config["host"].strip()
        or not _is_int(config["base_port"])
        or config["base_port"] != min(ports.values())
        or ports != {suite: config["base_port"] + index for index, suite in enumerate(suites)}
        or (max_cases is not None and (not _is_int(max_cases) or max_cases < 1))
        or not _is_int(config["resolution"])
        or config["resolution"] < 1
        or not _is_int(config["action_horizon"])
        or config["action_horizon"] < 1
        or not _is_int(config["dummy_steps"])
        or config["dummy_steps"] < 0
        or any(config[key] is not None and not isinstance(config[key], str) for key in optional_strings)
        or not isinstance(config["mujoco_gl"], str)
        or not config["mujoco_gl"]
        or not isinstance(config["pyopengl_platform"], str)
        or not config["pyopengl_platform"]
    ):
        raise ValueError("manifest config values are invalid")
    cases = manifest["cases"]
    if not isinstance(cases, list):
        raise ValueError("manifest cases must be a list")
    selected = {
        (task["suite"], task["task_id"]): task for task in selection["tasks"]
    }
    artifact_ids: set[str] = set()
    for planned in cases:
        if not isinstance(planned, dict) or set(planned) != set(CASE_FIELDS) | {"artifact_id"}:
            raise ValueError("manifest case has an invalid exact schema")
        if (
            planned["suite"] not in DEFAULT_SUITES
            or not _is_int(planned["task_id"])
            or planned["task_id"] < 0
            or not isinstance(planned["language"], str)
            or not planned["language"].strip()
            or planned["rank_group"] not in {"best", "worst"}
            or not _is_int(planned["initial_state_index"])
            or planned["initial_state_index"] < 0
            or not _is_int(planned["seed"])
            or planned["seed"] < 0
            or not isinstance(planned["original_success"], bool)
            or not isinstance(planned["artifact_id"], str)
        ):
            raise ValueError("manifest case values are invalid")
        case = _case_from_plan(planned)
        task = selected.get((case.suite, case.task_id))
        if (
            case.suite not in filtered
            or case.seed not in seeds
            or (task_filter is not None and case.task_id not in task_filter)
            or task is None
            or case.language != task["language"]
            or case.rank_group != task["rank_group"]
            or case.initial_state_index != task["initial_state_index"]
            or case.original_success is not task["original_success"]
            or planned["artifact_id"] in artifact_ids
        ):
            raise ValueError("manifest case provenance is invalid")
        artifact_ids.add(planned["artifact_id"])
    expected = manifest["expected_artifacts"]
    if (
        not isinstance(expected, dict)
        or set(expected) != EXPECTED_KEYS
        or any(not _is_int(count) or count < 0 for count in expected.values())
        or expected != _expected_artifacts(cases, seeds)
    ):
        raise ValueError("expected artifact counts are invalid")
    full = (
        suites == DEFAULT_SUITES
        and tuple(seeds) == DEFAULT_SEEDS
        and filtered == DEFAULT_SUITES
        and task_filter is None
        and max_cases is None
        and len(cases) == 80
    )
    if not isinstance(manifest["full_audit"], bool) or manifest["full_audit"] != full:
        raise ValueError("full_audit flag is invalid")
    if manifest["aggregation_sources"] != {
        "normal": "worker_shards", "render_only": "validated_raw"
    }:
        raise ValueError("aggregation source modes are invalid")
    if manifest["config_identity"] != _digest(_manifest_identity_payload(manifest)):
        raise ValueError("run manifest canonical identity mismatch")
    if manifest["manifest_digest"] != _digest(
        {key: item for key, item in manifest.items() if key != "manifest_digest"}
    ):
        raise ValueError("run manifest integrity digest mismatch")
    return manifest


def _expected_artifacts(cases: Sequence[dict[str, object]], seeds: Sequence[int]) -> dict[str, int]:
    by_task: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for case in cases:
        key = (case["suite"], case["task_id"], case["rank_group"], case["initial_state_index"])
        by_task.setdefault(key, []).append(case)
    complete_tasks = {
        key
        for key, values in by_task.items()
        if tuple(seeds) == DEFAULT_SEEDS
        and len(values) == len(seeds)
        and {int(value["seed"]) for value in values} == set(seeds)
    }
    by_suite: dict[str, list[tuple[object, ...]]] = {}
    for key in complete_tasks:
        by_suite.setdefault(str(key[0]), []).append(key)
    sheets = sum(
        len(values) == 4
        and sum(key[2] == "best" for key in values) == 2
        and sum(key[2] == "worst" for key in values) == 2
        for values in by_suite.values()
    )
    return {
        "episodes": len(cases),
        "videos": len(cases),
        "rollout_summaries": len(cases),
        "task_seed_summaries": len(complete_tasks),
        "suite_contact_sheets": int(sheets),
    }


def prepare_plan(
    *,
    checkpoint: str | Path,
    source_log_dir: str | Path,
    output_dir: str | Path,
    suites: Sequence[str] = DEFAULT_SUITES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    host: str = "127.0.0.1",
    base_port: int = 6694,
    suite_filter: Sequence[str] | None = None,
    task_filter: Sequence[int] | None = None,
    max_cases: int | None = None,
    dry_run: bool = False,
    resolution: int = 256,
    action_horizon: int = 8,
    dummy_steps: int = 10,
    unnorm_key: str | None = None,
    libero_home: str | None = None,
    libero_config_path: str | None = None,
    mujoco_gl: str = "egl",
    pyopengl_platform: str = "egl",
) -> dict[str, object]:
    suites = _validate_suite_list(list(suites), "suites")
    seeds = tuple(seeds)
    if (
        not seeds
        or any(not _is_int(seed) or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("seeds must be unique non-negative integers")
    if (
        not _is_int(base_port)
        or not 1 <= base_port <= 65535
        or base_port + len(suites) - 1 > 65535
    ):
        raise ValueError("base port range is invalid")
    if not _is_int(resolution) or resolution < 1:
        raise ValueError("resolution must be positive")
    if not _is_int(action_horizon) or action_horizon < 1:
        raise ValueError("action horizon must be positive")
    if not _is_int(dummy_steps) or dummy_steps < 0:
        raise ValueError("dummy steps must be non-negative")
    if max_cases is not None and (not _is_int(max_cases) or max_cases < 1):
        raise ValueError("max-cases must be positive")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    selected_suites = (
        _validate_suite_list(list(suite_filter), "suite filter") if suite_filter else suites
    )
    if any(suite not in suites for suite in selected_suites):
        raise ValueError("suite filter must be a subset of suites")
    if task_filter is not None and (
        any(not _is_int(task) or task < 0 for task in task_filter)
        or len(task_filter) != len(set(task_filter))
    ):
        raise ValueError("task filter must contain unique non-negative integers")
    for name, value in (
        ("unnorm_key", unnorm_key),
        ("libero_home", libero_home),
        ("libero_config_path", libero_config_path),
    ):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string or null")
    if not isinstance(mujoco_gl, str) or not mujoco_gl:
        raise ValueError("mujoco_gl must be a non-empty string")
    if not isinstance(pyopengl_platform, str) or not pyopengl_platform:
        raise ValueError("pyopengl_platform must be a non-empty string")
    source = Path(source_log_dir).expanduser().resolve(strict=False)
    if not source.is_dir():
        raise FileNotFoundError(f"source log directory does not exist: {source}")

    all_rows: list[TaskEvaluation] = []
    for index, suite in enumerate(suites):
        all_rows.extend(parse_worker_log(_worker_log(source, suite, index), suite))
    selection = _validate_selection(_selection_payload(all_rows))
    cases = build_audit_cases(all_rows, seeds=seeds, extremes=2)
    if suites == DEFAULT_SUITES and seeds == DEFAULT_SEEDS:
        validate_audit_cases(cases)
    cases = [case for case in cases if case.suite in selected_suites]
    if task_filter is not None:
        wanted = set(task_filter)
        cases = [case for case in cases if case.task_id in wanted]
    if max_cases is not None:
        cases = cases[:max_cases]
    planned_cases = [{**asdict(case), "artifact_id": artifact_id(case)} for case in cases]
    if len({case["artifact_id"] for case in planned_cases}) != len(planned_cases):
        raise ValueError("artifact ids are not unique")

    checkpoint_id = _checkpoint_identity(checkpoint, allow_missing=dry_run)
    selection_digest = _digest(selection)
    ports = {suite: base_port + index for index, suite in enumerate(suites)}
    config = {
        "suites": list(suites),
        "seeds": list(seeds),
        "host": host,
        "base_port": base_port,
        "ports": ports,
        "suite_filter": list(selected_suites),
        "task_filter": list(task_filter) if task_filter is not None else None,
        "max_cases": max_cases,
        "resolution": resolution,
        "action_horizon": action_horizon,
        "dummy_steps": dummy_steps,
        "unnorm_key": unnorm_key,
        "libero_home": libero_home,
        "libero_config_path": libero_config_path,
        "mujoco_gl": mujoco_gl,
        "pyopengl_platform": pyopengl_platform,
    }
    full_audit = (
        suites == DEFAULT_SUITES
        and seeds == DEFAULT_SEEDS
        and selected_suites == DEFAULT_SUITES
        and task_filter is None
        and max_cases is None
        and len(planned_cases) == 80
    )
    manifest = _finalize_manifest({
        "schema": "libero-v3-trace-audit-run",
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "checkpoint_identity": checkpoint_id,
        "selection_digest": selection_digest,
        "config_identity": "",
        "manifest_digest": "",
        "suites": list(suites),
        "seeds": list(seeds),
        "config": config,
        "ports": ports,
        "cases": planned_cases,
        "expected_artifacts": _expected_artifacts(planned_cases, seeds),
        "full_audit": full_audit,
        "aggregation_sources": {
            "normal": "worker_shards", "render_only": "validated_raw",
        },
    })
    _validate_manifest(manifest, selection)
    output = Path(output_dir).expanduser().resolve(strict=False)
    manifest_path = output / "run_manifest.json"
    selection_path = output / "selection.json"
    if manifest_path.exists():
        _, current = _load_manifest(output)
        if current["config_identity"] != manifest["config_identity"]:
            raise ValueError("run manifest conflict: checkpoint, selection, or config differs")
        return current
    if selection_path.exists():
        current_selection = _validate_selection(_strict_load(selection_path))
        if current_selection != selection:
            raise ValueError("run manifest conflict: pre-existing selection.json differs")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(selection_path, selection)
    _atomic_json(manifest_path, manifest)
    for name in ("raw", "raw/workers", "videos", "summaries", "contact_sheets"):
        (output / name).mkdir(parents=True, exist_ok=True)
    return manifest


def _load_manifest(output_dir: str | Path) -> tuple[Path, dict[str, object]]:
    output = Path(output_dir).expanduser().resolve(strict=False)
    selection_path = _contained(output / "selection.json", output, "selection path")
    manifest_path = _contained(output / "run_manifest.json", output, "manifest path")
    if not selection_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("run manifest requires selection.json and run_manifest.json")
    selection = _validate_selection(_strict_load(selection_path))
    payload = _validate_manifest(_strict_load(manifest_path), selection)
    return output, payload


def _case_from_plan(value: Mapping[str, object]) -> AuditCase:
    case = AuditCase(**{key: value[key] for key in CASE_FIELDS})
    if value.get("artifact_id") != artifact_id(case):
        raise ValueError("manifest artifact id does not match its case")
    return case


def _make_policy_client(host: str, port: int) -> object:
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    return WebsocketClientPolicy(host, port)


def _raw_generation(base: Path) -> str:
    pointer = _strict_load(base.with_suffix(".manifest.json"))
    if (
        not isinstance(pointer, dict)
        or set(pointer) != {"version", "generation"}
        or not _is_int(pointer["version"])
        or not isinstance(pointer["generation"], str)
    ):
        raise ValueError("raw generation pointer has an invalid schema")
    return pointer["generation"]


def _episode_row(artifact: str, record: RolloutRecord, raw_generation: str) -> dict[str, object]:
    return {
        "artifact_id": artifact,
        "case": record.metadata["case"],
        "outcome": record.metadata["outcome"],
        "metrics": record.metadata["metrics"],
        "raw_generation": raw_generation,
        "finalized": True,
    }


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    contents = b"".join(
        json.dumps(metrics_to_jsonable(row), allow_nan=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )
    _atomic_bytes(path, contents)


def execute_worker(
    output_dir: str | Path,
    suite: str,
    *,
    host: str | None = None,
    port: int | None = None,
    collector: Callable[[AuditCase, object, object], RolloutRecord] | None = None,
    client_factory: Callable[[str, int], object] | None = None,
) -> list[dict[str, object]]:
    output, manifest = _load_manifest(output_dir)
    if suite not in manifest["suites"]:
        raise ValueError(f"worker suite {suite!r} is not in the run manifest")
    config = manifest["config"]
    expected_identity = {"audit_config_identity": manifest["config_identity"]}
    planned = [case for case in manifest["cases"] if case["suite"] == suite]
    expected_host = str(config["host"])
    expected_port = int(manifest["ports"][suite])
    if host is not None and str(host) != expected_host:
        raise ValueError("worker host differs from run manifest")
    if port is not None and int(port) != expected_port:
        raise ValueError("worker port differs from run manifest")
    if not planned:
        rows: list[dict[str, object]] = []
        _atomic_jsonl(output / "raw" / "workers" / f"{suite}.episodes.jsonl", rows)
        return rows
    records: list[tuple[str, RolloutRecord]] = []
    reruns: list[tuple[Mapping[str, object], AuditCase, Path]] = []
    for value in planned:
        case = _case_from_plan(value)
        base = output / "raw" / str(value["artifact_id"])
        try:
            record = load_rollout_record(
                base,
                expected_case=asdict(case),
                expected_config_identity=expected_identity,
            )
            records.append((str(value["artifact_id"]), record))
        except (FileNotFoundError, ValueError, OSError):
            reruns.append((value, case, base))
    if reruns:
        make_client = client_factory or _make_policy_client
        collect = collector or collect_rollout
        worker_host = expected_host
        worker_port = expected_port
        client = make_client(worker_host, worker_port)
        args = SimpleNamespace(
            resolution=int(config["resolution"]),
            action_horizon=int(config["action_horizon"]),
            dummy_steps=int(config["dummy_steps"]),
            max_steps=None,
            unnorm_key=config.get("unnorm_key"),
        )
        for value, case, base in reruns:
            record = collect(case, client, args)
            record.metadata["audit_config_identity"] = manifest["config_identity"]
            save_rollout_record(base, record)
            verified = load_rollout_record(
                base,
                expected_case=asdict(case),
                expected_config_identity=expected_identity,
            )
            records.append((str(value["artifact_id"]), verified))
    order = {str(value["artifact_id"]): index for index, value in enumerate(planned)}
    rows = [
        _episode_row(name, record, _raw_generation(output / "raw" / name))
        for name, record in sorted(records, key=lambda item: order[item[0]])
    ]
    if len(rows) != len(planned) or len({row["artifact_id"] for row in rows}) != len(planned):
        raise ValueError("worker did not finalize exactly one row per planned artifact")
    _atomic_jsonl(output / "raw" / "workers" / f"{suite}.episodes.jsonl", rows)
    return rows


SHARD_ROW_KEYS = {
    "artifact_id", "case", "outcome", "metrics", "raw_generation", "finalized",
}


def _strict_json_line(line: str, path: Path, line_number: int) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite shard JSON constant {value}")

    try:
        value = json.loads(line, parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"corrupt worker shard {path}:{line_number}") from error
    if not isinstance(value, dict) or set(value) != SHARD_ROW_KEYS:
        raise ValueError(f"worker shard row has an invalid exact schema: {path}:{line_number}")
    return value


def _validated_shard_rows(
    output: Path,
    manifest: Mapping[str, object],
    records: Sequence[tuple[str, RolloutRecord]],
) -> list[dict[str, object]]:
    worker_root = _contained(output / "raw" / "workers", output / "raw", "worker shard root")
    suites = tuple(manifest["config"]["suite_filter"])
    expected_paths = {
        _contained(worker_root / f"{suite}.episodes.jsonl", worker_root, "worker shard").name
        for suite in suites
    }
    actual_paths = {path.name for path in worker_root.glob("*.episodes.jsonl")}
    if actual_paths != expected_paths:
        raise ValueError("worker shard file set is missing or has extras")
    record_map = {artifact: record for artifact, record in records}
    expected_by_suite: dict[str, set[str]] = {suite: set() for suite in suites}
    for planned in manifest["cases"]:
        expected_by_suite[planned["suite"]].add(planned["artifact_id"])
    observed: dict[str, dict[str, object]] = {}
    for suite in suites:
        path = _contained(worker_root / f"{suite}.episodes.jsonl", worker_root, "worker shard")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"cannot read worker shard {path}") from error
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                raise ValueError(f"worker shard contains an empty row: {path}:{line_number}")
            row = _strict_json_line(line, path, line_number)
            artifact = row["artifact_id"]
            if not isinstance(artifact, str) or artifact in observed:
                raise ValueError("worker shard contains a duplicate/invalid artifact id")
            if artifact not in expected_by_suite[suite]:
                raise ValueError("worker shard contains an extra or cross-suite artifact")
            record = record_map.get(artifact)
            if record is None:
                raise ValueError("worker shard references an unplanned raw artifact")
            expected = _episode_row(
                artifact, record, _raw_generation(output / "raw" / artifact)
            )
            if row != metrics_to_jsonable(expected):
                raise ValueError("worker shard row does not match validated raw generation")
            observed[artifact] = row
    expected_ids = {planned["artifact_id"] for planned in manifest["cases"]}
    if set(observed) != expected_ids:
        raise ValueError("worker shards are missing planned artifacts")
    return [observed[planned["artifact_id"]] for planned in manifest["cases"]]


def _walk_metric_scalars(value: object, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], float]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _walk_metric_scalars(value[key], (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_metric_scalars(item, (*path, f"[{index}]"))
    elif isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        scalar = float(value)
        if np.isfinite(scalar):
            yield path, scalar


def _metric_observations(
    artifact: str, record: RolloutRecord
) -> list[dict[str, object]]:
    metrics = record.metadata.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("record metrics must be a mapping")
    case = record.metadata["case"]
    observations: list[dict[str, object]] = []
    for anchor_name, anchor_metrics in metrics.items():
        match = re.fullmatch(r"anchor_(\d+)", str(anchor_name))
        if match is None or not isinstance(anchor_metrics, Mapping):
            continue
        anchor_index = int(match.group(1))
        if not 0 <= anchor_index < len(record.anchor_steps):
            raise ValueError("metric anchor index is outside the rollout cadence")
        for path, scalar in _walk_metric_scalars(anchor_metrics):
            if not path:
                continue
            if path[0] in LANDMARKS:
                landmark, component = path[0], path[0]
            elif path[0] == "persistence" and len(path) > 1 and path[1] in LANDMARKS:
                landmark, component = path[1], "persistence"
            else:
                landmark, component = path[0], ".".join(path[:-1]) or path[0]
            observations.append({
                "artifact_id": artifact,
                "suite": case["suite"],
                "task_id": case["task_id"],
                "rank_group": case["rank_group"],
                "seed": case["seed"],
                "anchor_index": anchor_index,
                "anchor_step": int(record.anchor_steps[anchor_index]),
                "metric_path": ".".join(path),
                "landmark": landmark,
                "component": component,
                "value": scalar,
            })
    return observations


def _summary_groups(observations: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str, str, str], list[float]] = {}
    for item in observations:
        axes = {
            "landmark": str(item["landmark"]),
            "task": f"{item['suite']}/task-{int(item['task_id']):03d}",
            "rank": str(item["rank_group"]),
            "suite": str(item["suite"]),
            "seed": str(item["seed"]),
        }
        for group_type, group_key in axes.items():
            key = (
                group_type, group_key, str(item["metric_path"]),
                str(item["landmark"]), str(item["component"]),
            )
            buckets.setdefault(key, []).append(float(item["value"]))
    groups: list[dict[str, object]] = []
    for (group_type, group_key, metric_path, landmark, component), values in sorted(buckets.items()):
        array = np.asarray(values, dtype=np.float64)
        groups.append({
            "group_type": group_type,
            "group_key": group_key,
            "metric_path": metric_path,
            "landmark": landmark,
            "component": component,
            "count": int(len(array)),
            "mean": float(array.mean()),
            "std": float(array.std()),
        })
    return groups


def _episode_table(records: Sequence[tuple[str, RolloutRecord]]) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for artifact, record in records:
        case, outcome = record.metadata["case"], record.metadata["outcome"]
        latency = np.asarray(record.latency_ms, dtype=np.float64)
        table.append({
            "artifact_id": artifact,
            "suite": case["suite"],
            "task_id": case["task_id"],
            "rank_group": case["rank_group"],
            "seed": case["seed"],
            "original_success": case["original_success"],
            "success": outcome["success"],
            "outcome_changed": bool(outcome["success"] != case["original_success"]),
            "end_reason": outcome["end_reason"],
            "action_steps": int(len(record.executed_actions)),
            "state_steps": int(len(record.agent_rgb)),
            "anchor_count": int(len(record.anchor_steps)),
            "latency_mean_ms": float(latency.mean()),
            "latency_p95_ms": float(np.percentile(latency, 95)),
        })
    return table


def _write_summary_csv(path: Path, groups: Sequence[Mapping[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    fields = (
        "group_type", "group_key", "metric_path", "landmark", "component",
        "count", "mean", "std",
    )
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for group in groups:
        writer.writerow({key: group[key] for key in fields})
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def _render_records(
    output: Path,
    manifest: Mapping[str, object],
    records: Sequence[tuple[str, RolloutRecord]],
) -> dict[str, int]:
    by_task: dict[tuple[str, int, str, int], list[tuple[str, RolloutRecord]]] = {}
    for name, record in records:
        render_rollout_video(record, output / "videos" / f"{name}.mp4")
        render_rollout_summary(record, output / "summaries" / f"rollout-{name}.png")
        case = record.metadata["case"]
        key = (str(case["suite"]), int(case["task_id"]), str(case["rank_group"]), int(case["initial_state_index"]))
        by_task.setdefault(key, []).append((name, record))
    task_paths: dict[tuple[str, int, str, int], Path] = {}
    expected_seeds = [int(seed) for seed in manifest["seeds"]]
    for key, values in sorted(by_task.items()):
        if tuple(expected_seeds) != DEFAULT_SEEDS:
            continue
        if len(values) != len(expected_seeds) or sorted(int(item[1].metadata["case"]["seed"]) for item in values) != sorted(expected_seeds):
            continue
        suite, task_id, rank, initial = key
        path = output / "summaries" / f"task-{_safe_token(suite)}-task-{task_id:03d}-rank-{rank}-init-{initial:03d}.png"
        render_task_seed_summary([record for _, record in values], path)
        task_paths[key] = path
    for suite in manifest["suites"]:
        items = [(key, path) for key, path in task_paths.items() if key[0] == suite]
        if len(items) != 4:
            continue
        items.sort(key=lambda item: (0 if item[0][2] == "best" else 1, item[0][1]))
        render_suite_contact_sheet(
            [(path, key[2]) for key, path in items],
            output / "contact_sheets" / f"{_safe_token(suite)}.png",
        )
    return {
        "episodes": len(records),
        "videos": len(list((output / "videos").glob("*.mp4"))),
        "rollout_summaries": len(list((output / "summaries").glob("rollout-*.png"))),
        "task_seed_summaries": len(list((output / "summaries").glob("task-*.png"))),
        "suite_contact_sheets": len(list((output / "contact_sheets").glob("*.png"))),
    }


def aggregate_run(
    output_dir: str | Path,
    *,
    render: bool = True,
    source_mode: str = "worker_shards",
) -> dict[str, object]:
    output, manifest = _load_manifest(output_dir)
    if source_mode not in {"worker_shards", "validated_raw"}:
        raise ValueError("aggregation source_mode must be worker_shards or validated_raw")
    expected_identity = {"audit_config_identity": manifest["config_identity"]}
    records: list[tuple[str, RolloutRecord]] = []
    for value in manifest["cases"]:
        case = _case_from_plan(value)
        base = _contained(output / "raw" / str(value["artifact_id"]), output / "raw", "raw artifact")
        record = load_rollout_record(
            base,
            expected_case=asdict(case),
            expected_config_identity=expected_identity,
        )
        records.append((str(value["artifact_id"]), record))
    if len(records) != int(manifest["expected_artifacts"]["episodes"]):
        raise ValueError("raw artifact count does not match run manifest")
    if len({name for name, _ in records}) != len(records):
        raise ValueError("duplicate artifact ids in aggregation")
    expected_pointers = {f"{name}.manifest.json" for name, _ in records}
    actual_pointers = {path.name for path in (output / "raw").glob("*.manifest.json")}
    if actual_pointers != expected_pointers:
        raise ValueError("finalized raw pointer count does not match run manifest")
    raw_rows = [
        _episode_row(name, record, _raw_generation(output / "raw" / name))
        for name, record in records
    ]
    if source_mode == "worker_shards":
        episodes = _validated_shard_rows(output, manifest, records)
    else:
        episodes = [metrics_to_jsonable(row) for row in raw_rows]
    _atomic_jsonl(output / "episodes.jsonl", episodes)
    observations = [
        item for artifact, record in records for item in _metric_observations(artifact, record)
    ]
    groups = _summary_groups(observations)
    episode_table = _episode_table(records)
    if render:
        counts = _render_records(output, manifest, records)
    else:
        counts = {"episodes": len(records)}
    expected = dict(manifest["expected_artifacts"])
    if render and counts != expected:
        raise ValueError(f"rendered artifact counts differ: expected {expected}, got {counts}")
    summary = {
        "schema": "libero-v3-trace-audit-summary",
        "schema_version": SCHEMA_VERSION,
        "config_identity": manifest["config_identity"],
        "aggregation_source": source_mode,
        "full_audit": bool(manifest["full_audit"]),
        "expected_artifacts": expected,
        "artifact_counts": counts,
        "episode_table": episode_table,
        "observations": observations,
        "groups": groups,
    }
    _atomic_json(output / "summary.json", summary)
    _write_summary_csv(output / "summary.csv", groups)
    return metrics_to_jsonable(summary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "plan", "worker", "aggregate"), default="run")
    parser.add_argument("--checkpoint")
    parser.add_argument("--source-log-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--host")
    parser.add_argument("--base-port", type=int, default=6694)
    parser.add_argument("--port", type=int)
    parser.add_argument("--suite-filter")
    parser.add_argument("--task-filter")
    parser.add_argument("--worker-suite")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--dummy-steps", type=int, default=10)
    parser.add_argument("--unnorm-key")
    parser.add_argument("--libero-home")
    parser.add_argument("--libero-config-path")
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--pyopengl-platform", default="egl")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.render_only:
        args.phase = "aggregate"
    if args.phase in {"run", "plan"}:
        if not args.checkpoint or not args.source_log_dir:
            raise ValueError("checkpoint and source-log-dir are required for planning")
        suites = _csv_values(args.suites)
        seeds = _integers(args.seeds, "seeds")
        suite_filter = _csv_values(args.suite_filter) if args.suite_filter else None
        task_filter = _integers(args.task_filter, "task-filter") if args.task_filter else None
        manifest = prepare_plan(
            checkpoint=args.checkpoint,
            source_log_dir=args.source_log_dir,
            output_dir=args.output_dir,
            suites=suites,
            seeds=seeds,
            host=args.host or "127.0.0.1",
            base_port=args.base_port,
            suite_filter=suite_filter,
            task_filter=task_filter,
            max_cases=args.max_cases,
            dry_run=args.dry_run,
            resolution=args.resolution,
            action_horizon=args.action_horizon,
            dummy_steps=args.dummy_steps,
            unnorm_key=args.unnorm_key,
            libero_home=args.libero_home,
            libero_config_path=args.libero_config_path,
            mujoco_gl=args.mujoco_gl,
            pyopengl_platform=args.pyopengl_platform,
        )
        selected_tasks = len({(case["suite"], case["task_id"], case["rank_group"]) for case in manifest["cases"]})
        print(f"planned {len(manifest['cases'])} cases from {selected_tasks} selected tasks")
        if args.dry_run or args.phase == "plan":
            print("dry-run: no policy server or simulator was contacted" if args.dry_run else "plan complete")
            return 0
        for suite in manifest["config"]["suite_filter"]:
            execute_worker(args.output_dir, suite)
        aggregate_run(args.output_dir, render=not args.no_render)
        return 0
    if args.phase == "worker":
        if not args.worker_suite:
            raise ValueError("worker phase requires --worker-suite")
        if args.dry_run:
            _, manifest = _load_manifest(args.output_dir)
            if args.worker_suite not in manifest["suites"]:
                raise ValueError("worker suite is not in the run manifest")
            expected_port = int(manifest["ports"][args.worker_suite])
            if args.port is not None and args.port != expected_port:
                raise ValueError("worker port differs from run manifest")
            if args.host is not None and args.host != manifest["config"]["host"]:
                raise ValueError("worker host differs from run manifest")
            count = sum(case["suite"] == args.worker_suite for case in manifest["cases"])
            print(f"dry-run worker suite={args.worker_suite} would process {count} cases")
            return 0
        rows = execute_worker(args.output_dir, args.worker_suite, host=args.host, port=args.port)
        print(f"worker suite={args.worker_suite} finalized {len(rows)} cases")
        return 0
    if args.dry_run:
        _, manifest = _load_manifest(args.output_dir)
        source = "validated_raw" if args.render_only else "worker_shards"
        print(f"dry-run aggregate would validate {len(manifest['cases'])} cases from {source}")
        return 0
    summary = aggregate_run(args.output_dir, render=not args.no_render, source_mode=("validated_raw" if args.render_only else "worker_shards"))
    print(f"aggregate finalized {summary['artifact_counts']['episodes']} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
