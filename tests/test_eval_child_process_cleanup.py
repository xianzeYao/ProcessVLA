import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESS_HELPER = (
    REPO_ROOT
    / "examples"
    / "simBenchmarks"
    / "eval_common"
    / "child_process_cleanup.sh"
)


def _run_bash(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", textwrap.dedent(body)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def test_cleanup_does_not_kill_a_reused_pid_with_a_different_start_time():
    result = _run_bash(
        f"""
        set -euo pipefail
        source {PROCESS_HELPER}
        sleep 30 &
        pid=$!
        remember_owned_process "$pid"
        OWNED_PROCESS_START_TIMES["$pid"]="different-start-time"
        cleanup_owned_processes
        kill -0 "$pid"
        kill "$pid"
        wait "$pid" 2>/dev/null || true
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_cleanup_terminates_the_same_process_that_was_registered():
    result = _run_bash(
        f"""
        set -euo pipefail
        source {PROCESS_HELPER}
        sleep 30 &
        pid=$!
        remember_owned_process "$pid"
        cleanup_owned_processes
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            exit 41
        fi
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_forgotten_process_is_not_terminated_during_cleanup():
    result = _run_bash(
        f"""
        set -euo pipefail
        source {PROCESS_HELPER}
        sleep 30 &
        pid=$!
        remember_owned_process "$pid"
        forget_owned_process "$pid"
        cleanup_owned_processes
        kill -0 "$pid"
        kill "$pid"
        wait "$pid" 2>/dev/null || true
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
