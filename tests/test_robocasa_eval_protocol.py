import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.simBenchmarks.Robocasa_tabletop.eval_files.robocasa_eval_protocol import (
    build_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "examples/simBenchmarks/Robocasa_tabletop/eval_files/run_multigpu_eval.sh"
)


def _manifest(gpus: list[str]):
    return build_manifest(
        checkpoint="checkpoint.pt",
        gpus=gpus,
        num_episodes=50,
        base_port=18000,
        run_dir=Path("/tmp/robocasa-eval"),
        env_names=[f"suite/task_{index:02d}" for index in range(24)],
        save_video=False,
    )


def test_build_manifest_supports_eight_unique_gpus():
    manifest = _manifest([str(index) for index in range(8)])

    assert manifest["gpus"] == list(range(8))
    assert len(manifest["tasks"]) == 24
    assert [task["worker_id"] for task in manifest["tasks"]] == [
        index % 8 for index in range(24)
    ]
    worker_counts = {
        worker: sum(task["worker_id"] == worker for task in manifest["tasks"])
        for worker in range(8)
    }
    assert worker_counts == {worker: 3 for worker in range(8)}


@pytest.mark.parametrize(
    ("gpus", "message"),
    [
        ([], "between 1 and 24"),
        ([str(index) for index in range(25)], "between 1 and 24"),
        (["0", "1", "1"], "unique"),
    ],
)
def test_build_manifest_rejects_invalid_gpu_lists(gpus, message):
    with pytest.raises(ValueError, match=message):
        _manifest(gpus)


def _run_launcher(tmp_path: Path, *, gpus: str, run_timestamp: str):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    output_dir = tmp_path / "outputs"
    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT": str(checkpoint),
            "GPUS": gpus,
            "DRY_RUN": "1",
            "NUM_EPISODES": "1",
            "OUTPUT_DIR": str(output_dir),
            "RUN_TIMESTAMP": run_timestamp,
            "POLICY_PYTHON": sys.executable,
            "MANIFEST_PYTHON": sys.executable,
            "ROBOCASA_PYTHON": sys.executable,
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, output_dir / "robocasa" / run_timestamp


def test_launcher_dry_run_supports_eight_unique_gpus(tmp_path):
    result, run_dir = _run_launcher(
        tmp_path, gpus="0,1,2,3,4,5,6,7", run_timestamp="eight-gpus"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[robocasa] plan task=") == 24
    manifest = json.loads((run_dir / "manifest.json").read_text())
    worker_ids = [task["worker_id"] for task in manifest["tasks"]]
    assert {worker: worker_ids.count(worker) for worker in range(8)} == {
        worker: 3 for worker in range(8)
    }


def test_launcher_dry_run_records_opt_in_trace_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_CONSISTENCY", "1")
    result, run_dir = _run_launcher(
        tmp_path, gpus="0", run_timestamp="trace-consistency"
    )

    assert result.returncode == 0, result.stderr
    protocol = (run_dir / "protocol.env").read_text()
    assert "TRACE_CONSISTENCY=1" in protocol
    assert "TRACE_ACTION_HORIZON=16" in protocol
    assert (run_dir / "trace_consistency").is_dir()

def test_launcher_rejects_duplicate_gpus(tmp_path):
    result, _ = _run_launcher(
        tmp_path, gpus="0,1,1", run_timestamp="duplicate-gpus"
    )

    assert result.returncode != 0
    assert "unique GPU" in result.stderr


def test_launcher_rejects_empty_gpu_list(tmp_path):
    result, _ = _run_launcher(tmp_path, gpus="", run_timestamp="empty-gpus")

    assert result.returncode != 0
    assert "between 1 and 24" in result.stderr
