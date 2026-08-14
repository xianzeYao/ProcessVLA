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
METRIC_NAMES = ("uv_ade_px", "d_mae_mm", "delta_d_mae_mm")
LANDMARKS = ("left", "right", "wrist", "aggregate")


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


def _checkpoint_identity(path: str | Path, *, allow_missing: bool) -> dict[str, object]:
    checkpoint = Path(path).expanduser().resolve(strict=False)
    if not checkpoint.is_file():
        if not allow_missing:
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        return {"resolved_path": str(checkpoint), "exists": False, "size": None, "mtime_ns": None}
    stat = checkpoint.stat()
    return {
        "resolved_path": str(checkpoint),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _worker_log(source: Path, suite: str, suite_index: int) -> Path:
    candidates = (
        source / f"{suite}_gpu{suite_index}.worker.log",
        source / f"{suite}.worker.log",
        source / f"{suite}.log",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(f"ambiguous worker logs for {suite}: {existing}")
    matches = sorted(source.glob(f"{suite}*.worker.log"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one worker log for {suite} in {source}")
    return matches[0]


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
        "tasks": tasks,
    }


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
) -> dict[str, object]:
    suites = tuple(suites)
    seeds = tuple(int(seed) for seed in seeds)
    if not suites or len(suites) != len(set(suites)):
        raise ValueError("suites must be unique and non-empty")
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique non-negative integers")
    if not 1 <= int(base_port) <= 65535 or int(base_port) + len(suites) - 1 > 65535:
        raise ValueError("base port range is invalid")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max-cases must be positive")
    selected_suites = tuple(suite_filter) if suite_filter else suites
    if not selected_suites or any(suite not in suites for suite in selected_suites):
        raise ValueError("suite filter must be a non-empty subset of suites")
    source = Path(source_log_dir).expanduser().resolve(strict=False)
    if not source.is_dir():
        raise FileNotFoundError(f"source log directory does not exist: {source}")

    all_rows: list[TaskEvaluation] = []
    for index, suite in enumerate(suites):
        all_rows.extend(parse_worker_log(_worker_log(source, suite, index), suite))
    selection = _selection_payload(all_rows)
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
    ports = {suite: int(base_port) + index for index, suite in enumerate(suites)}
    config = {
        "suites": list(suites),
        "seeds": list(seeds),
        "host": host,
        "base_port": int(base_port),
        "ports": ports,
        "suite_filter": list(selected_suites),
        "task_filter": list(task_filter) if task_filter is not None else None,
        "max_cases": max_cases,
        "resolution": int(resolution),
        "action_horizon": int(action_horizon),
        "dummy_steps": int(dummy_steps),
        "unnorm_key": unnorm_key,
    }
    identity = {
        "checkpoint_identity": checkpoint_id,
        "selection_digest": selection_digest,
        "config": config,
        "tool_version": TOOL_VERSION,
    }
    manifest = {
        "schema": "libero-v3-trace-audit-run",
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "checkpoint_identity": checkpoint_id,
        "selection_digest": selection_digest,
        "config_identity": _digest(identity),
        "suites": list(suites),
        "seeds": list(seeds),
        "config": config,
        "ports": ports,
        "cases": planned_cases,
        "expected_artifacts": _expected_artifacts(planned_cases, seeds),
        "full_audit": suites == DEFAULT_SUITES
        and seeds == DEFAULT_SEEDS
        and tuple(selected_suites) == DEFAULT_SUITES
        and task_filter is None
        and max_cases is None
        and len(planned_cases) == 80,
    }
    output = Path(output_dir).expanduser().resolve(strict=False)
    manifest_path = output / "run_manifest.json"
    selection_path = output / "selection.json"
    if manifest_path.exists():
        current = _strict_load(manifest_path)
        if current != manifest:
            raise ValueError("run manifest conflict: checkpoint, selection, or config differs")
        if not selection_path.is_file() or _strict_load(selection_path) != selection:
            raise ValueError("run manifest conflict: selection.json differs")
    else:
        if selection_path.exists() and _strict_load(selection_path) != selection:
            raise ValueError("run manifest conflict: pre-existing selection.json differs")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(selection_path, selection)
        _atomic_json(manifest_path, manifest)
    for name in ("raw", "raw/workers", "videos", "summaries", "contact_sheets"):
        (output / name).mkdir(parents=True, exist_ok=True)
    return manifest


def _load_manifest(output_dir: str | Path) -> tuple[Path, dict[str, object]]:
    output = Path(output_dir).expanduser().resolve(strict=False)
    payload = _strict_load(output / "run_manifest.json")
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "libero-v3-trace-audit-run"
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("cases"), list)
        or not isinstance(payload.get("config_identity"), str)
    ):
        raise ValueError("unsupported or partial run manifest")
    return output, payload


def _case_from_plan(value: Mapping[str, object]) -> AuditCase:
    case = AuditCase(**{key: value[key] for key in CASE_FIELDS})
    if value.get("artifact_id") != artifact_id(case):
        raise ValueError("manifest artifact id does not match its case")
    return case


def _make_policy_client(host: str, port: int) -> object:
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    return WebsocketClientPolicy(host, port)


def _episode_row(artifact: str, record: RolloutRecord) -> dict[str, object]:
    return {
        "artifact_id": artifact,
        "case": record.metadata["case"],
        "outcome": record.metadata["outcome"],
        "metrics": record.metadata["metrics"],
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
    rows = [_episode_row(name, record) for name, record in sorted(records, key=lambda item: order[item[0]])]
    if len(rows) != len(planned) or len({row["artifact_id"] for row in rows}) != len(planned):
        raise ValueError("worker did not finalize exactly one row per planned artifact")
    _atomic_jsonl(output / "raw" / "workers" / f"{suite}.episodes.jsonl", rows)
    return rows


def _metric_observations(record: RolloutRecord) -> Iterable[tuple[str, str, float]]:
    metrics = record.metadata.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("record metrics must be a mapping")
    for anchor in metrics.values():
        if not isinstance(anchor, Mapping):
            continue
        for landmark in LANDMARKS:
            values = anchor.get(landmark)
            if not isinstance(values, Mapping):
                continue
            for metric in METRIC_NAMES:
                value = values.get(metric)
                if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
                    yield landmark, metric, float(value)


def _summary_groups(records: Sequence[tuple[str, RolloutRecord]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str, str], list[float]] = {}
    for _, record in records:
        case = record.metadata["case"]
        axes = {
            "landmark": "all",
            "task": f"{case['suite']}/task-{int(case['task_id']):03d}",
            "rank": str(case["rank_group"]),
            "suite": str(case["suite"]),
            "seed": str(case["seed"]),
        }
        for landmark, metric, value in _metric_observations(record):
            for group_type, group_key in axes.items():
                key = (group_type, group_key if group_type != "landmark" else landmark, landmark, metric)
                buckets.setdefault(key, []).append(value)
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for (group_type, group_key, landmark, metric), values in sorted(buckets.items()):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        item = grouped.setdefault(
            (group_type, group_key, landmark),
            {"group_type": group_type, "group_key": group_key, "landmark": landmark, "metrics": {}},
        )
        item["metrics"][metric] = {
            "count": int(len(finite)),
            "mean": float(np.mean(finite)) if len(finite) else float("nan"),
        }
    return list(grouped.values())


def _write_summary_csv(path: Path, groups: Sequence[Mapping[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=("group_type", "group_key", "landmark", "metric", "count", "mean")
    )
    writer.writeheader()
    for group in groups:
        for metric, value in sorted(group["metrics"].items()):
            writer.writerow(
                {
                    "group_type": group["group_type"],
                    "group_key": group["group_key"],
                    "landmark": group["landmark"],
                    "metric": metric,
                    "count": value["count"],
                    "mean": "" if not np.isfinite(value["mean"]) else f"{value['mean']:.12g}",
                }
            )
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
    sheets = 0
    for suite in manifest["suites"]:
        items = [(key, path) for key, path in task_paths.items() if key[0] == suite]
        if len(items) != 4:
            continue
        items.sort(key=lambda item: (0 if item[0][2] == "best" else 1, item[0][1]))
        render_suite_contact_sheet(
            [(path, key[2]) for key, path in items],
            output / "contact_sheets" / f"{_safe_token(suite)}.png",
        )
        sheets += 1
    return {
        "episodes": len(records),
        "videos": len(list((output / "videos").glob("*.mp4"))),
        "rollout_summaries": len(list((output / "summaries").glob("rollout-*.png"))),
        "task_seed_summaries": len(list((output / "summaries").glob("task-*.png"))),
        "suite_contact_sheets": len(list((output / "contact_sheets").glob("*.png"))),
    }


def aggregate_run(output_dir: str | Path, *, render: bool = True) -> dict[str, object]:
    output, manifest = _load_manifest(output_dir)
    expected_identity = {"audit_config_identity": manifest["config_identity"]}
    records: list[tuple[str, RolloutRecord]] = []
    for value in manifest["cases"]:
        case = _case_from_plan(value)
        record = load_rollout_record(
            output / "raw" / str(value["artifact_id"]),
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
    episodes = [_episode_row(name, record) for name, record in records]
    _atomic_jsonl(output / "episodes.jsonl", episodes)
    groups = _summary_groups(records)
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
        "full_audit": bool(manifest["full_audit"]),
        "expected_artifacts": expected,
        "artifact_counts": counts,
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
        rows = execute_worker(args.output_dir, args.worker_suite, host=args.host, port=args.port)
        print(f"worker suite={args.worker_suite} finalized {len(rows)} cases")
        return 0
    summary = aggregate_run(args.output_dir, render=not args.no_render)
    print(f"aggregate finalized {summary['artifact_counts']['episodes']} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
