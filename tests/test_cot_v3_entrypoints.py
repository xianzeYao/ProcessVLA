from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "qwen35_gr00t_libero_CoT_v3_q0_depthcond.yaml"
RUN_ID = "qwen35_gr00t_libero_CoT_v3_q0_depthcond_8gpu_bs16"


def test_v3_yaml_selects_triangle_framework_and_dataset() -> None:
    cfg = OmegaConf.load(
        ROOT / "examples/modelExtensions/CoT/configs" / CONFIG_NAME
    )

    assert cfg.run_id == RUN_ID
    assert cfg.framework.name == "QwenGR00TCoTV3"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.state_dim == 7
    assert cfg.framework.action_model.action_horizon == 8
    assert cfg.framework.geometry.include_depth_in_action_condition is True
    assert cfg.framework.geometry.uvd_num_points == 4
    assert cfg.framework.geometry.landmark_count == 3
    assert cfg.framework.geometry.lambda_uvd_temporal == 0.1
    assert cfg.framework.geometry.lambda_uvd_shape == 0.0
    assert "uvd_hand_count" not in cfg.framework.geometry
    assert "lambda_uvd_relative" not in cfg.framework.geometry
    assert cfg.datasets.vla_data.dataset_py == "cot_v3_lerobot_datasets"
    assert cfg.datasets.vla_data.cot_geometry.uvd_num_points == 4
    assert "robocasa" not in str(cfg).lower()


def test_v3_trainer_help_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "starVLA.training.train_starvla_cot_v3",
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config_yaml" in result.stdout


def test_v3_launcher_dry_run_selects_entrypoint_and_forwards_overrides() -> None:
    script = (
        ROOT
        / "examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_libero_CoT_v3.sh"
    )
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "NUM_PROCESSES": "2",
        "MAIN_PROCESS_PORT": "29631",
    }
    result = subprocess.run(
        ["bash", str(script), "--trainer.max_train_steps=7"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "starVLA/training/train_starvla_cot_v3.py" in result.stdout
    assert CONFIG_NAME in result.stdout
    assert f"--run_id {RUN_ID}" in result.stdout
    assert "--num_processes 2" in result.stdout
    assert "--main_process_port 29631" in result.stdout
    assert "--trainer.max_train_steps=7" in result.stdout
