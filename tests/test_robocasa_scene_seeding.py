import importlib
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "examples/simBenchmarks/Robocasa_tabletop/eval_files/run_multigpu_eval.sh"
)
SEED_MODULE = (
    "examples.simBenchmarks.Robocasa_tabletop.eval_files.wrappers."
    "episode_seed_wrapper"
)


class RecordingResetEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.reset_seeds = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        return np.zeros(1, dtype=np.float32), {}


def test_episode_seed_wrapper_assigns_one_reproducible_seed_per_reset():
    try:
        seed_module = importlib.import_module(SEED_MODULE)
    except ModuleNotFoundError:
        seed_module = None
    assert seed_module is not None, "episode seed wrapper is not implemented"

    env = RecordingResetEnv()
    wrapped = seed_module.EpisodeSeedWrapper(
        env,
        eval_seed=7,
        task_index=2,
        env_index=3,
    )

    wrapped.reset()
    wrapped.reset()
    wrapped.reset()

    assert env.reset_seeds == [23007, 23008, 23009]


def test_launcher_dry_run_records_scene_seed_protocol(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    output_dir = tmp_path / "outputs"
    run_timestamp = "scene-seed"
    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT": str(checkpoint),
            "GPUS": "0",
            "DRY_RUN": "1",
            "NUM_EPISODES": "1",
            "OUTPUT_DIR": str(output_dir),
            "RUN_TIMESTAMP": run_timestamp,
            "POLICY_PYTHON": sys.executable,
            "MANIFEST_PYTHON": sys.executable,
            "ROBOCASA_PYTHON": sys.executable,
            "EVAL_SEED": "123",
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

    assert result.returncode == 0, result.stderr
    protocol = (
        output_dir / "robocasa" / run_timestamp / "protocol.env"
    ).read_text(encoding="utf-8")
    assert "EVAL_SEED=123\n" in protocol
    assert "SCENE_SEED_SCHEME=task_env_episode_v1\n" in protocol
