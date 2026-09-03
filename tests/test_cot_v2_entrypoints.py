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


def test_libero_q32_wrist_depth_future_only_config_and_launcher():
    config_name = (
        "qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_"
        "wristdepth_futureonly_60k_node2.yaml"
    )
    script_name = (
        "run_qwen35_gr00t_libero_CoT_v2_q32_"
        "wristdepth_futureonly_60k_node2.sh"
    )
    config_path = ROOT / "examples/modelExtensions/CoT/configs" / config_name
    script_path = ROOT / "examples/modelExtensions/CoT/scripts" / script_name

    cfg = OmegaConf.load(config_path)
    geometry = cfg.framework.geometry
    assert geometry.enable_current_depth is False
    assert geometry.enable_future_depth is True
    assert geometry.reconstruct_wrist_depth is True
    assert geometry.lambda_depth_current == 0.0
    assert geometry.lambda_wrist_depth_current == 0.0
    assert geometry.lambda_depth_future == 0.145
    assert geometry.lambda_wrist_depth_future == 0.145
    assert cfg.datasets.vla_data.cot_geometry.reconstruct_wrist_depth is True
    assert cfg.run_root_dir == "/data-training/yyf/yxz/outputs/libero"
    assert cfg.framework.qwenvl.base_vlm == "/data-training/yyf/yxz/models/Qwen3.5-4B"
    assert cfg.datasets.vla_data.data_root_dir == (
        "/data-training/yyf/yxz/datasets/libero_rerender"
    )

    result = subprocess.run(
        ["bash", str(script_path), "--trainer.max_train_steps=7"],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "NUM_PROCESSES": "8",
            "MAIN_PROCESS_PORT": "29543",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert config_name in result.stdout
    assert "starVLA/training/train_starvla_cot_v2.py" in result.stdout
    assert "--num_processes 8" in result.stdout
    assert "--main_process_port 29543" in result.stdout
    assert "--trainer.max_train_steps=7" in result.stdout


def test_libero_future_token_sharing_15k_configs_are_matched_and_chainable():
    config_root = ROOT / "examples/modelExtensions/CoT/configs"
    script_root = ROOT / "examples/modelExtensions/CoT/scripts"
    stems = {
        "shared": (
            "qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_"
            "wristdepth_futureonly_shared_gradprobe15k_node2"
        ),
        "separate": (
            "qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_"
            "wristdepth_futureonly_separate_gradprobe15k_node2"
        ),
    }
    configs = {
        name: OmegaConf.to_container(
            OmegaConf.load(config_root / f"{stem}.yaml"), resolve=True
        )
        for name, stem in stems.items()
    }

    for name, cfg in configs.items():
        stem = stems[name]
        geometry = cfg["framework"]["geometry"]
        diagnostics = cfg["trainer"]["test_diagnostics"]
        assert cfg["run_id"] == stems[name]
        assert geometry["enable_current_depth"] is False
        assert geometry["enable_future_depth"] is True
        assert geometry["reconstruct_wrist_depth"] is True
        assert geometry["separate_wrist_future_depth"] is (name == "separate")
        assert geometry["lambda_depth_current"] == 0.0
        assert geometry["lambda_wrist_depth_current"] == 0.0
        assert geometry["lambda_depth_future"] == 0.15
        assert geometry["lambda_wrist_depth_future"] == 0.15
        assert cfg["trainer"]["max_train_steps"] == 15000
        assert cfg["trainer"]["save_interval"] == 5000
        assert diagnostics["enabled"] is True
        assert diagnostics["log_objective_gradients"] is True
        assert diagnostics["objective_gradient_interval"] == 50
        assert diagnostics["module_gradient_interval"] == 50

        result = subprocess.run(
            ["bash", str(script_root / f"run_{stem}.sh")],
            cwd=ROOT,
            env={**os.environ, "DRY_RUN": "1"},
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert f"{stem}.yaml" in result.stdout
        assert "--num_processes 8" in result.stdout

    comparable = {}
    for name, cfg in configs.items():
        cfg.pop("run_id")
        cfg["framework"]["geometry"].pop("separate_wrist_future_depth")
        comparable[name] = cfg
    assert comparable["shared"] == comparable["separate"]

    chain = subprocess.run(
        [
            "bash",
            str(
                script_root
                / "run_qwen35_gr00t_libero_CoT_v2_future_token_sharing_compare15k_node2.sh"
            ),
        ],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert chain.returncode == 0, chain.stderr
    assert chain.stdout.count("train_starvla_cot_v2.py") == 2
    assert stems["shared"] in chain.stdout
    assert stems["separate"] in chain.stdout


def test_robocasa_q32_nodepthcond_single_depth_branch_ablation_configs_and_launchers():
    config_root = ROOT / "examples/modelExtensions/CoT/configs"
    script_root = ROOT / "examples/modelExtensions/CoT/scripts"
    baseline = OmegaConf.to_container(
        OmegaConf.load(config_root / "qwen35_gr00t_robocasa_fourier_CoT_v2.yaml"),
        resolve=True,
    )
    cases = {
        "current": (
            "qwen35_gr00t_robocasa_fourier_CoT_v2_q32_nodepthcond_wo_current_depth.yaml",
            "run_qwen35_gr00t_robocasa_fourier_CoT_v2_q32_nodepthcond_wo_current_depth.sh",
        ),
        "future": (
            "qwen35_gr00t_robocasa_fourier_CoT_v2_q32_nodepthcond_wo_future_depth.yaml",
            "run_qwen35_gr00t_robocasa_fourier_CoT_v2_q32_nodepthcond_wo_future_depth.sh",
        ),
    }

    for disabled_branch, (config_name, script_name) in cases.items():
        config_path = config_root / config_name
        script_path = script_root / script_name
        assert config_path.exists(), f"missing ablation config: {config_path}"
        assert script_path.exists(), f"missing ablation launcher: {script_path}"

        ablation = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        geometry = ablation["framework"]["geometry"]
        assert ablation["framework"]["action_model"]["num_target_vision_tokens"] == 32
        assert geometry["include_depth_in_action_condition"] is False
        assert geometry[f"enable_{disabled_branch}_depth"] is False
        kept_branch = "future" if disabled_branch == "current" else "current"
        assert geometry[f"enable_{kept_branch}_depth"] is True
        assert geometry[f"lambda_depth_{disabled_branch}"] == 0.0
        assert geometry[f"lambda_depth_{kept_branch}"] == baseline["framework"]["geometry"][
            f"lambda_depth_{kept_branch}"
        ]
        expected_run_id = f"{config_name.removesuffix('.yaml')}_8gpu_bs16"
        assert ablation["run_id"] == expected_run_id
        geometry.pop("enable_current_depth")
        geometry.pop("enable_future_depth")
        geometry.pop("include_depth_in_action_condition")
        geometry[f"lambda_depth_{disabled_branch}"] = baseline["framework"]["geometry"][
            f"lambda_depth_{disabled_branch}"
        ]
        ablation["run_id"] = baseline["run_id"]
        assert ablation == baseline

        result = subprocess.run(
            ["bash", str(script_path), "--trainer.max_train_steps=7"],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "NUM_PROCESSES": "8",
                "MAIN_PROCESS_PORT": "29541",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "starVLA/training/train_starvla_cot_v2.py" in result.stdout
        assert config_name in result.stdout
        assert "--num_processes 8" in result.stdout
        assert "--main_process_port 29541" in result.stdout
        assert "--trainer.max_train_steps=7" in result.stdout


def test_libero_wrist_depth_gradprobe_10k_node2_config_and_launcher_are_fixed():
    config_name = (
        "qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_"
        "wristdepth_gradprobe10k_node2.yaml"
    )
    config_path = ROOT / "examples/modelExtensions/CoT/configs" / config_name
    script_path = (
        ROOT
        / "examples/modelExtensions/CoT/scripts"
        / "run_qwen35_gr00t_libero_CoT_v2_q32_wristdepth_gradprobe10k_node2.sh"
    )
    assert config_path.exists(), f"missing fixed 10k gradient-probe config: {config_path}"
    assert script_path.exists(), f"missing fixed 10k gradient-probe launcher: {script_path}"

    cfg = OmegaConf.load(config_path)
    assert cfg.run_id == (
        "qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_"
        "wristdepth_gradprobe10k_node2"
    )
    assert cfg.run_root_dir == "/data-training/yyf/yxz/outputs/libero"
    assert cfg.framework.qwenvl.base_vlm == "/data-training/yyf/yxz/models/Qwen3.5-4B"
    assert cfg.datasets.vla_data.data_root_dir == (
        "/data-training/yyf/yxz/datasets/libero_rerender"
    )
    depth_weights = (
        cfg.framework.geometry.lambda_depth_current,
        cfg.framework.geometry.lambda_depth_future,
        cfg.framework.geometry.lambda_wrist_depth_current,
        cfg.framework.geometry.lambda_wrist_depth_future,
    )
    assert depth_weights == (0.0725,) * 4
    assert cfg.trainer.max_train_steps == 10000
    assert cfg.trainer.num_warmup_steps == 1000
    assert cfg.trainer.save_interval == 5000
    assert cfg.trainer.test_diagnostics.log_objective_gradients is True
    assert cfg.trainer.test_diagnostics.objective_gradient_interval == 50
    assert cfg.trainer.test_diagnostics.module_gradient_interval == 50

    result = subprocess.run(
        ["bash", str(script_path), "--trainer.max_train_steps=7"],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "/data-training/yyf/yxz/envs/train/bin/python" in result.stdout
    assert config_name in result.stdout
    assert "--num_processes 8" in result.stdout
    assert "--trainer.max_train_steps=7" in result.stdout
