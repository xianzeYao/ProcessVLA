import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "libero": ("qwen35_gr00t_libero_CoT_v2.yaml", 8, 1, 4, 60000),
    "calvin_ABCD_D": ("qwen35_gr00t_calvin_ABCD_D_CoT_v2.yaml", 8, 1, 4, 80000),
    "calvin_ABC_D": ("qwen35_gr00t_calvin_ABC_D_CoT_v2.yaml", 8, 1, 4, 80000),
    "robocasa_fourier": ("qwen35_gr00t_robocasa_fourier_CoT_v2.yaml", 16, 2, 6, 100000),
}


def test_reverse_full_experiment_is_not_an_active_v2_yaml():
    path = (
        ROOT
        / "examples/modelExtensions/CoT/configs"
        / "qwen35_gr00t_libero_CoT_v2_q0_depthcond_reverse_full_uvd_s4.yaml"
    )
    assert not path.exists()


def test_v2_yamls_keep_fixed_geometry_contract_per_bench():
    config_root = ROOT / "examples/modelExtensions/CoT/configs"
    for _, (filename, horizon, hands, points, max_steps) in CASES.items():
        cfg = OmegaConf.load(config_root / filename)
        assert cfg.framework.name == "QwenGR00TCoTV2"
        assert cfg.framework.action_model.action_horizon == horizon
        assert cfg.framework.geometry.uvd_hand_count == hands
        assert cfg.framework.geometry.uvd_num_points == points
        assert cfg.framework.geometry.full_attention_backend == "sdpa"
        assert "query_layer_count" not in cfg.framework.geometry
        assert cfg.datasets.vla_data.cot_geometry.action_horizon == horizon
        assert cfg.datasets.vla_data.cot_geometry.uvd_num_points == points
        assert cfg.trainer.max_train_steps == max_steps
        assert cfg.trainer.test_diagnostics.enabled is False
        assert cfg.trainer.test_diagnostics.log_token_utilization is True
        assert cfg.trainer.test_diagnostics.log_decoder_reliance is True
        assert cfg.trainer.test_diagnostics.log_uvd_time_metrics is True


def test_v2_scripts_dry_run_the_independent_trainer_and_matching_config():
    scripts = {
        "libero": "run_qwen35_gr00t_libero_CoT_v2.sh",
        "calvin_ABCD_D": "run_qwen35_gr00t_calvin_ABCD_D_CoT_v2.sh",
        "calvin_ABC_D": "run_qwen35_gr00t_calvin_ABC_D_CoT_v2.sh",
        "robocasa_fourier": "run_qwen35_gr00t_robocasa_fourier_CoT_v2.sh",
    }
    script_root = ROOT / "examples/modelExtensions/CoT/scripts"
    env = {**os.environ, "DRY_RUN": "1", "NUM_PROCESSES": "4"}
    for bench, script_name in scripts.items():
        result = subprocess.run(
            ["bash", str(script_root / script_name)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "starVLA/training/train_starvla_cot_v2.py" in result.stdout
        assert CASES[bench][0] in result.stdout
        assert "--num_processes 4" in result.stdout


def test_v2_gradient_probe_cli_exposes_reproducibility_arguments():
    script = ROOT / "examples/modelExtensions/CoT/scripts/probe_qwen35_gr00t_CoT_v2_gradients.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config_yaml" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--sample_indices" in result.stdout
    assert "--qwen_tail_layers" in result.stdout


def test_robocasa_q0_depth_condition_config_changes_only_the_condition_inputs():
    config_root = ROOT / "examples/modelExtensions/CoT/configs"
    baseline_path = config_root / "qwen35_gr00t_robocasa_fourier_CoT_v2_q0.yaml"
    depth_condition_path = (
        config_root / "qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond.yaml"
    )
    assert depth_condition_path.exists(), "depth-condition experiment YAML is missing"

    baseline = OmegaConf.to_container(OmegaConf.load(baseline_path), resolve=True)
    depth_condition = OmegaConf.to_container(
        OmegaConf.load(depth_condition_path),
        resolve=True,
    )

    assert depth_condition["run_id"] == (
        "qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond_8gpu_bs16"
    )
    assert depth_condition["framework"]["geometry"].pop(
        "include_depth_in_action_condition"
    ) is True

    depth_condition["run_id"] = baseline["run_id"]
    assert depth_condition == baseline


def test_libero_q0_depth_condition_config_changes_only_the_condition_inputs():
    config_root = ROOT / "examples/modelExtensions/CoT/configs"
    baseline_path = config_root / "qwen35_gr00t_libero_CoT_v2_q0.yaml"
    depth_condition_path = (
        config_root / "qwen35_gr00t_libero_CoT_v2_q0_depthcond.yaml"
    )
    assert depth_condition_path.exists(), "depth-condition experiment YAML is missing"

    baseline = OmegaConf.to_container(OmegaConf.load(baseline_path), resolve=True)
    depth_condition = OmegaConf.to_container(
        OmegaConf.load(depth_condition_path),
        resolve=True,
    )

    assert depth_condition["run_id"] == (
        "qwen35_gr00t_libero_CoT_v2_q0_depthcond_8gpu_bs16"
    )
    assert depth_condition["framework"]["geometry"].pop(
        "include_depth_in_action_condition"
    ) is True

    depth_condition["run_id"] = baseline["run_id"]
    assert depth_condition == baseline

def test_libero_q32_wrist_depth_config_keeps_depth_out_of_action_expert():
    config_path = (
        ROOT
        / "examples/modelExtensions/CoT/configs"
        / "qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_wristdepth.yaml"
    )
    assert config_path.exists(), f"missing wrist-depth config: {config_path}"
    cfg = OmegaConf.load(config_path)

    assert cfg.run_id == (
        "qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_wristdepth_8gpu_bs16"
    )
    assert cfg.framework.name == "QwenGR00TCoTV2"
    assert cfg.framework.action_model.num_target_vision_tokens == 32
    assert cfg.framework.geometry.include_depth_in_action_condition is False
    assert cfg.framework.geometry.reconstruct_wrist_depth is True
    depth_weights = (
        cfg.framework.geometry.lambda_depth_current,
        cfg.framework.geometry.lambda_depth_future,
        cfg.framework.geometry.lambda_wrist_depth_current,
        cfg.framework.geometry.lambda_wrist_depth_future,
    )
    assert depth_weights == (0.0725,) * 4
    assert sum(depth_weights) == 0.29
    assert cfg.datasets.vla_data.dataset_py == "cot_lerobot_datasets"
    assert cfg.datasets.vla_data.cot_geometry.reconstruct_wrist_depth is True
    assert cfg.trainer.max_train_steps == 60000


def test_libero_q32_wrist_depth_launcher_dry_run_uses_v2_trainer():
    script = (
        ROOT
        / "examples/modelExtensions/CoT/scripts"
        / "run_qwen35_gr00t_libero_CoT_v2_q32_wristdepth.sh"
    )
    assert script.exists(), f"missing wrist-depth launcher: {script}"
    result = subprocess.run(
        ["bash", str(script), "--trainer.max_train_steps=7"],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "NUM_PROCESSES": "8",
            "MAIN_PROCESS_PORT": "29539",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "starVLA/training/train_starvla_cot_v2.py" in result.stdout
    assert (
        "qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_wristdepth.yaml"
        in result.stdout
    )
    assert (
        "--run_id "
        "qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_wristdepth_8gpu_bs16"
        in result.stdout
    )
    assert "--num_processes 8" in result.stdout
    assert "--main_process_port 29539" in result.stdout
    assert "--trainer.max_train_steps=7" in result.stdout
