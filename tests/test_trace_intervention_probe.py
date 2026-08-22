from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowStepDiagnostics


def _probe_module():
    return importlib.import_module(
        "examples.simBenchmarks.CoT.geometry_probe.trace_intervention_probe"
    )


def _write_materialized_sample(
    path: Path,
    *,
    task_id: str,
    episode_id: int,
    frame_index: int,
    language: str | None = None,
    explicit_task_id: str | None = None,
) -> Path:
    metadata = {
        "suite": task_id,
        "episode_id": episode_id,
        "frame_index": frame_index,
    }
    if language is not None:
        metadata["language"] = language
    if explicit_task_id is not None:
        metadata["task_id"] = explicit_task_id
    np.savez_compressed(
        path,
        image_count=np.asarray(1, dtype=np.int64),
        image_0=np.zeros((4, 4, 3), dtype=np.uint8),
        language=np.asarray("move object"),
        depth_current=np.ones((4, 4), dtype=np.float32),
        depth_future=np.ones((4, 4), dtype=np.float32),
        depth_current_valid=np.ones((4, 4), dtype=np.bool_),
        depth_future_valid=np.ones((4, 4), dtype=np.bool_),
        uvd=np.zeros((2, 2, 3), dtype=np.float32),
        uvd_valid_mask=np.ones((2, 2), dtype=np.bool_),
        uvd_out_of_frame_mask=np.zeros((2, 2), dtype=np.bool_),
        uvd_time=np.asarray([0.0, 1.0], dtype=np.float32),
        uvd_frame_indices=np.asarray([0, 2], dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return path


def test_load_sample_identity_uses_exact_task_before_suite(tmp_path):
    probe = _probe_module()
    first = _write_materialized_sample(
        tmp_path / "first.npz",
        task_id="libero_spatial",
        episode_id=0,
        frame_index=0,
        language="pick up the black bowl",
    )
    second = _write_materialized_sample(
        tmp_path / "second.npz",
        task_id="libero_spatial",
        episode_id=1,
        frame_index=0,
        language="pick up the red mug",
    )
    explicit = _write_materialized_sample(
        tmp_path / "explicit.npz",
        task_id="libero_spatial",
        episode_id=2,
        frame_index=0,
        language="ignored language",
        explicit_task_id="task-17",
    )

    first_id = probe.load_sample_identity(first)["task_id"]
    second_id = probe.load_sample_identity(second)["task_id"]

    assert first_id == "libero_spatial:pick up the black bowl"
    assert second_id == "libero_spatial:pick up the red mug"
    assert first_id != second_id
    assert probe.load_sample_identity(explicit)["task_id"] == "task-17"


def test_build_intervention_batches_pairs_two_samples_from_distinct_tasks(tmp_path):
    probe = _probe_module()
    paths = []
    for task_id in ("a", "b"):
        for sample_index in range(4):
            paths.append(
                _write_materialized_sample(
                    tmp_path / f"{task_id}_{sample_index}.npz",
                    task_id=task_id,
                    episode_id=sample_index,
                    frame_index=sample_index,
                )
            )

    batches = probe.build_intervention_batches(paths, batch_size=4)

    assert len(batches) == 2
    assert sorted(path for batch in batches for path in batch) == sorted(paths)
    for batch in batches:
        task_ids = [probe.load_sample_identity(path)["task_id"] for path in batch]
        assert task_ids.count("a") == 2
        assert task_ids.count("b") == 2


def test_build_intervention_batches_allows_multiple_pairs_per_task(tmp_path):
    probe = _probe_module()
    paths = []
    for task_id in ("a", "b"):
        for sample_index in range(4):
            paths.append(
                _write_materialized_sample(
                    tmp_path / f"{task_id}_{sample_index}.npz",
                    task_id=task_id,
                    episode_id=sample_index,
                    frame_index=sample_index,
                )
            )

    batches = probe.build_intervention_batches(paths, batch_size=8)

    assert len(batches) == 1
    task_ids = [probe.load_sample_identity(path)["task_id"] for path in batches[0]]
    assert task_ids.count("a") == 4
    assert task_ids.count("b") == 4


def test_build_intervention_batches_keeps_odd_task_groups(tmp_path):
    probe = _probe_module()
    paths = []
    for task_id, count in {"a": 3, "b": 3, "c": 2}.items():
        for sample_index in range(count):
            paths.append(
                _write_materialized_sample(
                    tmp_path / f"{task_id}_{sample_index}.npz",
                    task_id=task_id,
                    episode_id=sample_index,
                    frame_index=sample_index,
                )
            )

    batches = probe.build_intervention_batches(paths, batch_size=8)

    assert len(batches) == 1
    assert sorted(batches[0]) == sorted(paths)
    task_ids = [probe.load_sample_identity(path)["task_id"] for path in batches[0]]
    assert task_ids.count("a") == 3
    assert task_ids.count("b") == 3
    assert task_ids.count("c") == 2


@pytest.mark.parametrize(
    ("task_counts", "batch_size", "message"),
    [
        ({"a": 2, "b": 2}, 3, "even"),
        ({"a": 2}, 4, "distinct tasks"),
        ({"a": 1, "b": 3}, 4, "at least two"),
        ({"a": 2, "b": 4}, 4, "full batches"),
    ],
)
def test_build_intervention_batches_rejects_invalid_layouts(
    tmp_path,
    task_counts,
    batch_size,
    message,
):
    probe = _probe_module()
    paths = []
    for task_id, count in task_counts.items():
        for sample_index in range(count):
            paths.append(
                _write_materialized_sample(
                    tmp_path / f"{task_id}_{sample_index}.npz",
                    task_id=task_id,
                    episode_id=sample_index,
                    frame_index=sample_index,
                )
            )

    with pytest.raises(ValueError, match=message):
        probe.build_intervention_batches(paths, batch_size=batch_size)


def test_effect_metrics_report_norm_rms_time_and_dimension_matched_groups():
    probe = _probe_module()
    reference = np.zeros((1, 2, 4), dtype=np.float32)
    alternative = np.asarray([[[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0]]])

    metrics = probe.effect_metrics(
        reference,
        alternative,
        {"first": (0, 2), "second": (2, 4)},
    )

    np.testing.assert_allclose(metrics["l2"], [np.sqrt(30.0)])
    np.testing.assert_allclose(metrics["rms"], [np.sqrt(30.0 / 8.0)])
    np.testing.assert_allclose(metrics["per_time_l2"], [[np.sqrt(5.0), 5.0]])
    np.testing.assert_allclose(metrics["per_time_rms"], [[np.sqrt(5.0 / 4.0), 2.5]])
    np.testing.assert_allclose(metrics["groups"]["first"]["l2"], [np.sqrt(5.0)])
    np.testing.assert_allclose(metrics["groups"]["second"]["l2"], [5.0])
    np.testing.assert_allclose(metrics["groups"]["first"]["rms"], [np.sqrt(5.0 / 4.0)])
    np.testing.assert_allclose(metrics["groups"]["second"]["rms"], [2.5])


def test_effect_metrics_report_relative_velocity_and_cosine():
    probe = _probe_module()
    reference = np.asarray([[[1.0, 0.0], [0.0, 2.0]]])
    alternative = np.asarray([[[0.0, 1.0], [0.0, -2.0]]])

    metrics = probe.effect_metrics(reference, alternative, {"all": (0, 2)})

    np.testing.assert_allclose(metrics["reference_l2"], [np.sqrt(5.0)])
    np.testing.assert_allclose(metrics["relative_l2"], [np.sqrt(18.0 / 5.0)])
    np.testing.assert_allclose(metrics["cosine"], [-0.8])
    np.testing.assert_allclose(metrics["per_time_reference_l2"], [[1.0, 2.0]])
    np.testing.assert_allclose(metrics["per_time_relative_l2"], [[np.sqrt(2.0), 2.0]])
    np.testing.assert_allclose(metrics["per_time_cosine"], [[0.0, -1.0]])
    np.testing.assert_allclose(metrics["groups"]["all"]["cosine"], [-0.8])


def test_effect_metrics_cosine_is_finite_for_zero_vectors():
    probe = _probe_module()
    reference = np.zeros((2, 1, 2), dtype=np.float32)
    alternative = np.asarray([[[0.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32)

    metrics = probe.effect_metrics(reference, alternative, {"all": (0, 2)})

    np.testing.assert_array_equal(metrics["cosine"], [1.0, 0.0])
    assert np.isfinite(metrics["relative_l2"]).all()
    assert np.isfinite(metrics["per_time_cosine"]).all()


def test_action_groups_split_libero_arm_and_hand():
    probe = _probe_module()

    assert probe._action_groups(7) == {"arm": (0, 6), "hand": (6, 7)}


def test_binary_action_flip_metrics_threshold_hand_separately():
    probe = _probe_module()
    reference = np.zeros((1, 3, 7), dtype=np.float32)
    alternative = np.zeros((1, 3, 7), dtype=np.float32)
    reference[0, :, 6] = [0.4, 0.6, 0.5]
    alternative[0, :, 6] = [0.7, 0.6, 0.2]

    metrics = probe.binary_action_flip_metrics(
        reference,
        alternative,
        indices=(6,),
        threshold=0.5,
    )

    np.testing.assert_allclose(metrics["flip_rate"], [2.0 / 3.0])
    np.testing.assert_array_equal(metrics["any_flip"], [1.0])
    np.testing.assert_array_equal(metrics["per_time_flip"], [[1.0, 0.0, 1.0]])


def test_cluster_bootstrap_is_invariant_to_duplicate_frames_within_episode():
    probe = _probe_module()

    compact = probe.cluster_bootstrap_mean_ci(
        [0.0, 0.0, 10.0, 10.0],
        ["a", "a", "b", "b"],
        seed=7,
        resamples=1000,
    )
    duplicated = probe.cluster_bootstrap_mean_ci(
        [0.0] * 20 + [10.0] * 20,
        ["a"] * 20 + ["b"] * 20,
        seed=7,
        resamples=1000,
    )

    assert compact == duplicated
    assert compact == (0.0, 10.0)


def test_write_probe_artifacts_round_trips_jsonl_and_npz(tmp_path):
    probe = _probe_module()

    paths = probe.write_probe_artifacts(
        tmp_path,
        config={"seed": 7, "normalized_actions": True},
        summary={"sample_count": 1},
        per_sample=[{"sample_index": 0, "effect": 1.5}],
        trajectories={"initial_actions": np.zeros((1, 2, 4), dtype=np.float32)},
    )

    assert json.loads(paths["config"].read_text())["seed"] == 7
    assert json.loads(paths["summary"].read_text())["sample_count"] == 1
    assert json.loads(paths["per_sample"].read_text().strip())["effect"] == 1.5
    with np.load(paths["trajectories"], allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["initial_actions"], np.zeros((1, 2, 4)))


def test_write_probe_figures_creates_velocity_final_and_group_views(tmp_path):
    probe = _probe_module()
    effect = {
        "l2": {"count": 4, "mean": 2.0, "median": 2.0, "std": 0.0, "ci95_low": 2.0, "ci95_high": 2.0},
        "rms": {"count": 4, "mean": 1.0, "median": 1.0, "std": 0.0, "ci95_low": 1.0, "ci95_high": 1.0},
        "per_time_l2": [],
        "per_time_rms": [],
        "groups": {
            "arm": {
                "l2": {"count": 4, "mean": 1.5, "median": 1.5, "std": 0.0, "ci95_low": 1.5, "ci95_high": 1.5},
                "rms": {"count": 4, "mean": 0.5, "median": 0.5, "std": 0.0, "ci95_low": 0.5, "ci95_high": 0.5},
                "per_time_l2": [],
                "per_time_rms": [],
                "groups": {},
            }
        },
    }
    summary = {
        "velocity_effect": {"zero_geometry": {"step_0": effect}},
        "final_action_effect": {"zero_geometry": {"all": effect}},
    }

    paths = probe.write_probe_figures(summary, tmp_path)

    assert {path.name for path in paths} == {
        "velocity_effect_heatmap.png",
        "final_action_effect.png",
        "action_group_effect.png",
    }
    assert all(path.stat().st_size > 0 for path in paths)


class _FakeTraceFramework:
    def __init__(
        self,
        *,
        repeat_error=0.0,
        include_state=None,
        state_dim=None,
        action_dim=4,
        norm_stats=None,
    ):
        self.action_model = SimpleNamespace(
            action_horizon=2,
            action_dim=int(action_dim),
            num_inference_timesteps=2,
        )
        if include_state is not None or state_dim is not None:
            self.config = SimpleNamespace(
                datasets=SimpleNamespace(
                    vla_data=SimpleNamespace(include_state=include_state),
                ),
                framework=SimpleNamespace(
                    action_model=SimpleNamespace(state_dim=state_dim),
                ),
            )
        self.repeat_error = float(repeat_error)
        self.calls = []
        self.initial_actions_seen = []
        self.norm_stats = norm_stats

    def to(self, device):
        self.device = str(device)
        return self

    def eval(self):
        self.is_eval = True
        return self

    def predict_action_interventions(
        self,
        examples,
        *,
        variants,
        task_ids,
        initial_actions,
        seed,
        rollout_steps,
    ):
        self.calls.append((len(examples), tuple(variants), tuple(task_ids), seed, rollout_steps))
        self.initial_actions_seen.append(initial_actions)
        batch_size = len(examples)
        shape = (batch_size, 2, self.action_model.action_dim)
        zeros = torch.zeros(shape)
        ones = torch.ones(shape)
        correct_diagnostics = tuple(
            FlowStepDiagnostics(
                step_index=step,
                t_cont=step / 2.0,
                t_discretized=step * 5,
                x_before=zeros.clone(),
                pred_velocity=zeros.clone(),
                x_after=zeros.clone(),
            )
            for step in range(2)
        )
        intervention_diagnostics = tuple(
            FlowStepDiagnostics(
                step_index=step,
                t_cont=step / 2.0,
                t_discretized=step * 5,
                x_before=zeros.clone(),
                pred_velocity=ones.clone(),
                x_after=ones.clone(),
            )
            for step in range(2)
        )
        return {
            "initial_actions": zeros,
            "correct": {
                "actions": zeros,
                "diagnostics": correct_diagnostics,
                "repeat_actions": zeros,
                "repeat_diagnostics": correct_diagnostics,
                "repeat_max_abs_error": self.repeat_error,
            },
            "interventions": {
                name: {
                    "permutation": None,
                    "diagnostic_counterfactual": False,
                    "local_velocities": torch.stack([ones, ones]),
                    "rollouts": {
                        "all": {"actions": ones, "diagnostics": intervention_diagnostics},
                        "step_0": {"actions": ones, "diagnostics": intervention_diagnostics},
                        "step_1": {"actions": ones, "diagnostics": intervention_diagnostics},
                    },
                }
                for name in variants
            },
        }


def test_run_trace_intervention_checkpoint_writes_recomputable_outputs(tmp_path):
    probe = _probe_module()
    samples = []
    for task_id in ("a", "b"):
        for sample_index in range(2):
            samples.append(
                _write_materialized_sample(
                    tmp_path / f"{task_id}_{sample_index}.npz",
                    task_id=task_id,
                    episode_id=sample_index,
                    frame_index=sample_index,
                )
            )
    framework = _FakeTraceFramework()
    loaded = []

    def loader(path):
        loaded.append(path)
        return framework

    summary = probe.run_trace_intervention_checkpoint(
        "checkpoint.pt",
        sample_paths=samples,
        output_dir=tmp_path / "output",
        batch_size=4,
        seed=7,
        device="cpu",
        variants=("zero_geometry",),
        bootstrap_resamples=200,
        framework_loader=loader,
    )

    assert loaded == ["checkpoint.pt"]
    assert framework.calls == [(4, ("zero_geometry",), ("a", "a", "b", "b"), 7, None)]
    assert isinstance(framework.initial_actions_seen[0], torch.Tensor)
    assert framework.initial_actions_seen[0].shape == (4, 2, 4)
    assert summary["sample_count"] == 4
    assert summary["correct_repeat_max_abs_error"] == 0.0
    assert summary["correct_repeat_tolerance"] == 1e-6
    assert summary["correct_repeat_passed"] is True
    assert summary["velocity_effect"]["zero_geometry"]["step_0"]["l2"]["mean"] == pytest.approx(
        np.sqrt(8.0)
    )
    assert summary["velocity_effect"]["zero_geometry"]["step_0"]["l2"]["count"] == 4
    assert summary["final_action_effect"]["zero_geometry"]["all"]["l2"]["mean"] == pytest.approx(
        np.sqrt(8.0)
    )
    assert (tmp_path / "output" / "config.json").exists()
    assert (tmp_path / "output" / "per_sample.jsonl").exists()
    assert (tmp_path / "output" / "trajectories.npz").exists()
    assert (tmp_path / "output" / "summary.json").exists()
    assert (tmp_path / "output" / "figures" / "velocity_effect_heatmap.png").exists()
    assert (tmp_path / "output" / "figures" / "final_action_effect.png").exists()
    assert (tmp_path / "output" / "figures" / "action_group_effect.png").exists()
    with np.load(tmp_path / "output" / "trajectories.npz", allow_pickle=False) as payload:
        np.testing.assert_array_equal(
            payload["correct_repeat_actions"],
            payload["correct_actions"],
        )
    config = json.loads((tmp_path / "output" / "config.json").read_text())
    assert config["dtype"] == "torch.float32"
    assert config["correct_repeat_tolerance"] == 1e-6
    assert config["input_example_keys"] == [
        "image",
        "lang",
        "uvd",
        "uvd_time",
        "uvd_valid_mask",
    ]
    assert config["proprioceptive_state_present"] is False
    assert config["state_conditioning_used"] is False


def test_run_trace_intervention_checkpoint_summarizes_cosine_and_hand_flips(tmp_path):
    probe = _probe_module()
    samples = []
    for task_id in ("a", "b"):
        for sample_index in range(2):
            samples.append(
                _write_materialized_sample(
                    tmp_path / f"{task_id}_{sample_index}.npz",
                    task_id=task_id,
                    episode_id=sample_index,
                    frame_index=sample_index,
                )
            )

    norm_stats = {
        "libero": {
            "action": {
                "q01": [0.0] * 7,
                "q99": [2.0] * 7,
                "mask": [True] * 6 + [False],
            }
        }
    }

    summary = probe.run_trace_intervention_checkpoint(
        "checkpoint.pt",
        sample_paths=samples,
        output_dir=tmp_path / "output",
        batch_size=4,
        seed=7,
        device="cpu",
        variants=("zero_geometry",),
        bootstrap_resamples=20,
        unnorm_key="libero",
        framework_loader=lambda _path: _FakeTraceFramework(action_dim=7, norm_stats=norm_stats),
    )

    velocity = summary["velocity_effect"]["zero_geometry"]["step_0"]
    assert velocity["cosine"]["mean"] == 0.0
    assert np.isfinite(velocity["relative_l2"]["mean"])
    assert velocity["groups"]["arm"]["l2"]["median"] == pytest.approx(np.sqrt(12.0))
    hand = summary["hand_discrete_effect"]["zero_geometry"]["all"]
    assert hand["flip_rate"]["mean"] == 1.0
    assert hand["any_flip"]["mean"] == 1.0
    assert [node["mean"] for node in hand["per_time_flip"]] == [1.0, 1.0]
    physical = summary["final_action_effect_unnormalized"]["zero_geometry"]["all"]
    assert physical["groups"]["arm"]["rms"]["median"] == 1.0
    assert physical["groups"]["hand"]["rms"]["median"] == 1.0
    first_record = json.loads(
        (tmp_path / "output" / "per_sample.jsonl").read_text().splitlines()[0]
    )
    assert first_record["interventions"]["zero_geometry"]["hand_discrete"]["all"] == {
        "flip_rate": 1.0,
        "any_flip": 1.0,
        "per_time_flip": [1.0, 1.0],
    }


def test_run_trace_intervention_checkpoint_rejects_repeat_error_above_tolerance(tmp_path):
    probe = _probe_module()
    samples = []
    for task_id in ("a", "b"):
        for sample_index in range(2):
            samples.append(
                _write_materialized_sample(
                    tmp_path / f"{task_id}_{sample_index}.npz",
                    task_id=task_id,
                    episode_id=sample_index,
                    frame_index=sample_index,
                )
            )

    with pytest.raises(RuntimeError, match="repeat sanity check failed"):
        probe.run_trace_intervention_checkpoint(
            "checkpoint.pt",
            sample_paths=samples,
            output_dir=tmp_path / "output",
            batch_size=4,
            seed=7,
            device="cpu",
            variants=("zero_geometry",),
            bootstrap_resamples=20,
            repeat_tolerance=1e-5,
            framework_loader=lambda _path: _FakeTraceFramework(repeat_error=2e-5),
        )

    assert not (tmp_path / "output" / "summary.json").exists()


def test_run_trace_intervention_checkpoint_rejects_state_contract_mismatch(tmp_path):
    probe = _probe_module()
    samples = []
    for task_id in ("a", "b"):
        for sample_index in range(2):
            samples.append(
                _write_materialized_sample(
                    tmp_path / f"{task_id}_{sample_index}.npz",
                    task_id=task_id,
                    episode_id=sample_index,
                    frame_index=sample_index,
                )
            )

    with pytest.raises(ValueError, match="include_state=True"):
        probe.run_trace_intervention_checkpoint(
            "checkpoint.pt",
            sample_paths=samples,
            output_dir=tmp_path / "output",
            batch_size=4,
            seed=7,
            device="cpu",
            variants=("zero_geometry",),
            bootstrap_resamples=20,
            framework_loader=lambda _path: _FakeTraceFramework(
                include_state=True,
                state_dim=7,
            ),
        )


def test_trace_probe_cli_exposes_checkpoint_sample_and_reproducibility_options():
    cli = importlib.import_module(
        "examples.simBenchmarks.CoT.geometry_probe.run_trace_intervention_probe"
    )

    args = cli.build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--samples-dir",
            "samples",
            "--output-dir",
            "output",
            "--batch-size",
            "8",
            "--seed",
            "9",
            "--device",
            "cuda:2",
            "--repeat-tolerance",
            "1e-5",
            "--unnorm-key",
            "libero",
        ]
    )

    assert args.checkpoint == Path("model.pt")
    assert args.samples_dir == Path("samples")
    assert args.output_dir == Path("output")
    assert args.batch_size == 8
    assert args.seed == 9
    assert args.device == "cuda:2"
    assert args.repeat_tolerance == 1e-5
    assert args.unnorm_key == "libero"


def test_trace_probe_cli_defaults_to_factorized_v3_interventions():
    cli = importlib.import_module(
        "examples.simBenchmarks.CoT.geometry_probe.run_trace_intervention_probe"
    )

    args = cli.build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--samples-dir",
            "samples",
            "--output-dir",
            "output",
        ]
    )

    assert {
        "uvd_within_task_shuffle",
        "uvd_cross_task_swap",
        "current_depth_within_task_shuffle",
        "current_depth_cross_task_swap",
        "future_depth_within_task_shuffle",
        "future_depth_cross_task_swap",
        "depth_within_task_shuffle",
        "depth_cross_task_swap",
    }.issubset(args.variants)
