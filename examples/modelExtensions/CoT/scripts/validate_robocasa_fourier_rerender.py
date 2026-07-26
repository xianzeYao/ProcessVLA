#!/usr/bin/env python3
"""Validate the complete 24-task StarVLA Fourier RoboCasa rerender corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from starVLA.dataloader.robocasa_fourier_tasks import fourier_dataset_names


def _load_single_validator():
    path = Path(__file__).with_name("validate_robocasa_replay_rgbd.py")
    spec = importlib.util.spec_from_file_location("robocasa_replay_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load single-task validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_fourier_root(root: Path, *, check_all_episodes: bool = False) -> dict[str, Any]:
    validator = _load_single_validator()
    expected = fourier_dataset_names()
    reports: list[dict[str, Any]] = []
    for name in expected:
        task_root = root / name
        try:
            report = validator.validate_task(task_root, check_all_episodes=check_all_episodes)
            if int(report["episodes"]) != 1000:
                raise RuntimeError(f"expected 1000 episodes, got {report['episodes']}")
            report["status"] = "success"
        except Exception as exc:  # noqa: BLE001 - preserve every task failure in the manifest
            report = {"task": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        reports.append(report)
    return {
        "mixture": "fourier_gr1_unified_1000",
        "expected_tasks": expected,
        "total_tasks": len(reports),
        "successful_tasks": sum(report.get("status") == "success" for report in reports),
        "total_episodes": sum(int(report.get("episodes", 0)) for report in reports),
        "tasks": reports,
    }


def write_manifest(root: Path, output: Path, *, check_all_episodes: bool = False, require_complete: bool = False) -> dict[str, Any]:
    manifest = validate_fourier_root(root, check_all_episodes=check_all_episodes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if require_complete and manifest["successful_tasks"] != 24:
        raise RuntimeError(f"Fourier rerender is incomplete: {manifest['successful_tasks']}/24 tasks passed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check-all-episodes", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    output = args.output or args.root / "meta" / "fourier_rerender_manifest.json"
    manifest = write_manifest(args.root, output, check_all_episodes=args.check_all_episodes, require_complete=args.require_complete)
    print(json.dumps({key: manifest[key] for key in ("mixture", "total_tasks", "successful_tasks", "total_episodes")}, indent=2))


if __name__ == "__main__":
    main()
