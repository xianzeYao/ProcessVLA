import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "examples/modelExtensions/CoT/configs"
SCRIPT_ROOT = ROOT / "examples/modelExtensions/CoT/scripts"


@pytest.mark.parametrize(
    "filename,horizon,hands,dataset_py,max_steps",
    [
        (
            "qwen35_gr00t_libero_CoT_v4_q0_depthcond_coarse8_local8.yaml",
            8,
            1,
            "cot_v4_lerobot_datasets",
            60000,
        ),
        (
            "qwen35_gr00t_robocasa_fourier_CoT_v4_q0_depthcond_coarse16_local16.yaml",
            16,
            2,
            "robocasa_v4_lerobot_datasets",
            100000,
        ),
    ],
)
def test_v4_yaml_contract(filename, horizon, hands, dataset_py, max_steps):
    cfg = OmegaConf.load(CONFIG_ROOT / filename)

    assert cfg.framework.name == "QwenGR00TCoTV4"
    assert cfg.framework.action_model.action_horizon == horizon
    assert cfg.framework.action_model.num_target_vision_tokens == 0
    assert cfg.framework.geometry.include_depth_in_action_condition is True
    assert cfg.framework.geometry.local_uvd_num_points == horizon
    assert cfg.framework.geometry.coarse_uvd_num_points == horizon
    assert cfg.framework.geometry.coarse_uvd_stride == 2
    assert cfg.framework.geometry.uvd_hand_count == hands
    assert cfg.datasets.vla_data.dataset_py == dataset_py
    assert cfg.datasets.vla_data.cot_geometry.action_horizon == horizon
    assert cfg.datasets.vla_data.cot_geometry.local_uvd_num_points == horizon
    assert cfg.datasets.vla_data.cot_geometry.coarse_uvd_num_points == horizon
    assert cfg.datasets.vla_data.cot_geometry.coarse_uvd_stride == 2
    assert cfg.datasets.vla_data.cot_geometry.terminal_repeat is True
    assert cfg.trainer.max_train_steps == max_steps


@pytest.mark.parametrize(
    "script_name,yaml_name,run_id",
    [
        (
            "run_qwen35_gr00t_libero_CoT_v4.sh",
            "qwen35_gr00t_libero_CoT_v4_q0_depthcond_coarse8_local8.yaml",
            "qwen35_gr00t_libero_CoT_v4_q0_depthcond_coarse8_local8_8gpu_bs16",
        ),
        (
            "run_qwen35_gr00t_robocasa_fourier_CoT_v4.sh",
            "qwen35_gr00t_robocasa_fourier_CoT_v4_q0_depthcond_coarse16_local16.yaml",
            "qwen35_gr00t_robocasa_fourier_CoT_v4_q0_depthcond_coarse16_local16_8gpu_bs16",
        ),
    ],
)
def test_v4_launchers_dry_run_independent_trainer(
    script_name,
    yaml_name,
    run_id,
):
    result = subprocess.run(
        ["bash", str(SCRIPT_ROOT / script_name)],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "NUM_PROCESSES": "8",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "starVLA/training/train_starvla_cot_v4.py" in result.stdout
    assert yaml_name in result.stdout
    assert run_id in result.stdout
    assert "--num_processes 8" in result.stdout
