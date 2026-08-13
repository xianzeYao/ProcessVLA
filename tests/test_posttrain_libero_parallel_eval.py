import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = (
    REPO_ROOT
    / "examples"
    / "modelExtensions"
    / "CoT"
    / "scripts"
    / "run_posttrain_libero_parallel_eval.sh"
)


class PostTrainingParallelEvalTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        self.sync_dir = self.root / "sync"
        self.sync_dir.mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_launcher(self, name: str, body: str) -> Path:
        path = self.root / name
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + textwrap.dedent(body),
            encoding="utf-8",
        )
        return path

    def _create_required_artifacts(self) -> None:
        checkpoint_dir = self.model_dir / "checkpoints"
        final_dir = self.model_dir / "final_model"
        checkpoint_dir.mkdir()
        final_dir.mkdir()
        (checkpoint_dir / "steps_60000_pytorch_model.pt").write_bytes(b"step")
        (final_dir / "pytorch_model.pt").write_bytes(b"final")

    def _run(self, standard: Path, plus: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "TRAIN_PID": "99999999",
                "MODEL_DIR": str(self.model_dir),
                "EXPECTED_STEP": "60000",
                "LIBERO_EVAL_SCRIPT": str(standard),
                "LIBERO_PLUS_EVAL_SCRIPT": str(plus),
                "POLL_SECONDS": "0.01",
                "GPU_RELEASE_WAIT_SECONDS": "0",
                "FAKE_SYNC_DIR": str(self.sync_dir),
            }
        )
        return subprocess.run(
            ["bash", str(ORCHESTRATOR)],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

    def test_missing_artifacts_launches_nothing(self):
        standard = self._write_launcher(
            "standard.sh", 'touch "$FAKE_SYNC_DIR/standard.started"\n'
        )
        plus = self._write_launcher(
            "plus.sh", 'touch "$FAKE_SYNC_DIR/plus.started"\n'
        )

        result = self._run(standard, plus)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required checkpoint artifact is missing", result.stdout)
        self.assertFalse((self.sync_dir / "standard.started").exists())
        self.assertFalse((self.sync_dir / "plus.started").exists())

    def test_valid_artifacts_launch_both_evaluations_concurrently(self):
        self._create_required_artifacts()
        standard = self._write_launcher(
            "standard.sh",
            """
            printf '%s|%s|%s|%s' "$GPUS" "$NUM_TRIALS_PER_TASK" "$SAVE_VIDEO" "$CKPT" \
                > "$FAKE_SYNC_DIR/standard.env"
            touch "$FAKE_SYNC_DIR/standard.started"
            for _ in {1..100}; do
                if [[ -e "$FAKE_SYNC_DIR/plus.started" ]]; then
                    touch "$FAKE_SYNC_DIR/standard.completed"
                    exit 0
                fi
                sleep 0.02
            done
            exit 41
            """,
        )
        plus = self._write_launcher(
            "plus.sh",
            """
            printf '%s|%s|%s|%s' "$GPUS" "$NUM_TRIALS_PER_TASK" "$SAVE_VIDEO" "$CKPT" \
                > "$FAKE_SYNC_DIR/plus.env"
            touch "$FAKE_SYNC_DIR/plus.started"
            for _ in {1..100}; do
                if [[ -e "$FAKE_SYNC_DIR/standard.started" ]]; then
                    touch "$FAKE_SYNC_DIR/plus.completed"
                    exit 0
                fi
                sleep 0.02
            done
            exit 42
            """,
        )

        result = self._run(standard, plus)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.sync_dir / "standard.completed").exists())
        self.assertTrue((self.sync_dir / "plus.completed").exists())
        final_checkpoint = self.model_dir / "final_model" / "pytorch_model.pt"
        self.assertEqual(
            (self.sync_dir / "standard.env").read_text(encoding="utf-8"),
            f"0,1,2,3|50|0|{final_checkpoint}",
        )
        self.assertEqual(
            (self.sync_dir / "plus.env").read_text(encoding="utf-8"),
            f"0,1,2,3,4,5,6,7|1|0|{final_checkpoint}",
        )

    def test_one_failure_does_not_prevent_other_evaluation_from_finishing(self):
        self._create_required_artifacts()
        standard = self._write_launcher(
            "standard.sh",
            """
            touch "$FAKE_SYNC_DIR/standard.started"
            exit 7
            """,
        )
        plus = self._write_launcher(
            "plus.sh",
            """
            touch "$FAKE_SYNC_DIR/plus.started"
            sleep 0.1
            touch "$FAKE_SYNC_DIR/plus.completed"
            """,
        )

        result = self._run(standard, plus)

        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.sync_dir / "standard.started").exists())
        self.assertTrue((self.sync_dir / "plus.completed").exists())
        self.assertIn("standard LIBERO exit status: 7", result.stdout)
        self.assertIn("LIBERO-plus exit status: 0", result.stdout)


if __name__ == "__main__":
    unittest.main()
