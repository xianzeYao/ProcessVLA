import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "examples/modelExtensions/CoT/scripts/compare_cot_v4_timing.py"
)


def write_metrics(path, model_times, losses, memories, start_step=20):
    rows = [
        {
            "step": start_step + index,
            "timing/model": model_time,
            "total_loss": loss,
            "system/gpu_memory_max_allocated_gb": memory,
        }
        for index, (model_time, loss, memory) in enumerate(
            zip(model_times, losses, memories)
        )
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_timing_gate_uses_stable_median_and_max_memory(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "report.json"
    write_metrics(baseline, [1.0, 1.2, 1.1], [3.0] * 3, [10.0, 11.0, 10.5])
    write_metrics(candidate, [1.4, 1.5, 1.6], [3.0] * 3, [12.0, 13.0, 12.5])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["baseline_median_model_s"] == 1.1
    assert report["candidate_median_model_s"] == 1.5
    assert report["model_time_ratio"] == 1.5 / 1.1
    assert report["candidate_max_allocated_gb"] == 13.0
    assert report["passed"] is True


def test_timing_gate_fails_above_ratio_or_on_nonfinite_loss(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "report.json"
    write_metrics(baseline, [1.0], [1.0], [10.0])
    write_metrics(candidate, [1.6], [float("nan")], [12.0])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert json.loads(output.read_text())["passed"] is False
