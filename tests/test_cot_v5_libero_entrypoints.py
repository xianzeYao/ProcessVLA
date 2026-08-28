from __future__ import annotations

import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "qwen35_gr00t_libero_CoT_v5_q0_nodepthcond.yaml"
RUN_ID = "qwen35_gr00t_libero_CoT_v5_q0_nodepthcond_lrw_8gpu_bs16"


def test_libero_v5_yaml_is_single_hand_q0_without_direct_depth_condition():
    config_path = ROOT / "examples/modelExtensions/CoT/configs" / CONFIG_NAME
    assert config_path.exists(), f"missing LIBERO V5 config: {config_path}"
    cfg = OmegaConf.load(config_path)

    assert cfg.run_id == RUN_ID
    assert cfg.framework.name == "QwenGR00TCoTV5"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.state_dim == 7
    assert cfg.framework.action_model.action_horizon == 8
    assert cfg.framework.action_model.num_target_vision_tokens == 0
    assert cfg.framework.geometry.include_depth_in_action_condition is False
    assert cfg.framework.geometry.depth_query_count == 8
    assert cfg.framework.geometry.uvd_num_points == 4
    assert cfg.framework.geometry.uvd_hand_count == 1
    assert cfg.framework.geometry.landmark_count == 3
    assert cfg.framework.geometry.lambda_depth_current > 0
    assert cfg.framework.geometry.lambda_depth_future > 0
    assert cfg.framework.geometry.lambda_uvd_temporal == 0.1
    assert cfg.framework.geometry.lambda_uvd_shape == 0.0
    assert cfg.datasets.vla_data.dataset_py == "libero_v5_lerobot_datasets"
    assert cfg.datasets.vla_data.data_mix == "libero_cot_all"
    assert cfg.datasets.vla_data.cot_geometry.action_horizon == 8
    assert cfg.datasets.vla_data.cot_geometry.uvd_num_points == 4
    assert cfg.trainer.max_train_steps == 60000


def test_libero_v5_launcher_dry_run_uses_shared_v5_trainer():
    script = (
        ROOT
        / "examples/modelExtensions/CoT/scripts"
        / "run_qwen35_gr00t_libero_CoT_v5.sh"
    )
    assert script.exists(), f"missing LIBERO V5 launcher: {script}"
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
            "MAIN_PROCESS_PORT": "29636",
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
    assert "--main_process_port 29636" in result.stdout
    assert "--trainer.max_train_steps=7" in result.stdout
    assert "--datasets.vla_data.num_workers=0" in result.stdout
