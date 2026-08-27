from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "qwen35_gr00t_robocasa_fourier_CoT_v5_q0_depthcond.yaml"
RUN_ID = "qwen35_gr00t_robocasa_fourier_CoT_v5_q0_depthcond_lrw_8gpu_bs16"


def test_v5_yaml_is_exact_robocasa_q0_depth_lrw_contract():
    cfg = OmegaConf.load(
        ROOT / "examples/modelExtensions/CoT/configs" / CONFIG_NAME
    )

    assert cfg.run_id == RUN_ID
    assert cfg.framework.name == "QwenGR00TCoTV5"
    assert cfg.framework.action_model.action_dim == 29
    assert cfg.framework.action_model.state_dim == 58
    assert cfg.framework.action_model.action_horizon == 16
    assert cfg.framework.action_model.num_target_vision_tokens == 0
    assert cfg.framework.geometry.include_depth_in_action_condition is True
    assert cfg.framework.geometry.depth_query_count == 8
    assert cfg.framework.geometry.uvd_num_points == 6
    assert cfg.framework.geometry.uvd_hand_count == 2
    assert cfg.framework.geometry.landmark_count == 3
    assert cfg.framework.geometry.lambda_uvd_temporal == 0.1
    assert cfg.framework.geometry.lambda_uvd_shape == 0.0
    assert "lambda_uvd_relative" not in cfg.framework.geometry
    assert cfg.datasets.vla_data.dataset_py == "robocasa_v5_lerobot_datasets"
    assert cfg.datasets.vla_data.data_mix == "fourier_gr1_unified_1000"
    assert cfg.datasets.vla_data.cot_geometry.action_horizon == 16
    assert cfg.datasets.vla_data.cot_geometry.uvd_num_points == 6
    assert cfg.trainer.max_train_steps == 100000


def test_v5_trainer_help_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "starVLA.training.train_starvla_cot_v5",
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config_yaml" in result.stdout


def test_v5_launcher_dry_run_isolated_and_forwards_dotlist_overrides():
    script = (
        ROOT
        / "examples/modelExtensions/CoT/scripts/"
        / "run_qwen35_gr00t_robocasa_fourier_CoT_v5.sh"
    )
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--trainer.max_train_steps=7",
            "--datasets.vla_data.num_workers=0",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "NUM_PROCESSES": "4",
            "MAIN_PROCESS_PORT": "29635",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "starVLA/training/train_starvla_cot_v5.py" in result.stdout
    assert CONFIG_NAME in result.stdout
    assert f"--run_id {RUN_ID}" in result.stdout
    assert "--num_processes 4" in result.stdout
    assert "--main_process_port 29635" in result.stdout
    assert "--trainer.max_train_steps=7" in result.stdout
    assert "--datasets.vla_data.num_workers=0" in result.stdout
