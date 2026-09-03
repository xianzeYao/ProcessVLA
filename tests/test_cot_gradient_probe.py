import hashlib
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

import starVLA.training.cot_gradient_probe as gradient_probe
from starVLA.training.train_starvla_cot_v2 import CotV2Trainer

from starVLA.training.cot_gradient_probe import (
    extract_v2_objectives,
    measure_objective_gradients,
    select_probe_sample_indices,
    select_shared_parameter_groups,
    sha256_file,
    summarize_probe_batches,
)


def test_gradient_probe_matches_aligned_orthogonal_conflicting_and_combined_vectors():
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    objectives = {
        "action": (parameter[0], 1.0),
        "aligned": (2.0 * parameter[0], 0.5),
        "orthogonal": (parameter[1], 2.0),
        "conflicting": (-parameter[0], 0.25),
    }

    result = measure_objective_gradients(
        objectives,
        {"geometry_tokens": [parameter]},
    )
    group = result["groups"]["geometry_tokens"]

    assert group["objectives"]["action"]["raw_grad_norm"] == pytest.approx(1.0)
    assert group["objectives"]["aligned"]["raw_grad_norm"] == pytest.approx(2.0)
    assert group["objectives"]["aligned"]["cosine_with_action"] == pytest.approx(1.0)
    assert group["objectives"]["orthogonal"]["cosine_with_action"] == pytest.approx(0.0)
    assert group["objectives"]["conflicting"]["cosine_with_action"] == pytest.approx(-1.0)
    assert group["combined_aux"]["weighted_grad_norm"] == pytest.approx(math.sqrt(0.75**2 + 2.0**2))
    assert group["combined_aux"]["cosine_with_action"] == pytest.approx(
        0.75 / math.sqrt(0.75**2 + 2.0**2)
    )


def test_gradient_probe_treats_unused_parameter_gradients_as_zero():
    used = nn.Parameter(torch.tensor([1.0]))
    unused = nn.Parameter(torch.tensor([2.0]))
    objectives = {
        "action": (used.square().sum(), 1.0),
        "aux": (used.sum(), 0.5),
    }

    result = measure_objective_gradients(
        objectives,
        {"geometry_tokens": [used, unused]},
    )

    assert result["groups"]["geometry_tokens"]["objectives"]["action"]["raw_grad_norm"] == 2.0
    assert result["groups"]["geometry_tokens"]["objectives"]["aux"]["raw_grad_norm"] == 1.0


def test_extract_v2_objectives_uses_nested_uvd_relative_weight_without_double_counting():
    scale = nn.Parameter(torch.tensor(1.0))
    output = {
        "action_loss": scale,
        "depth_current_loss": scale,
        "depth_future_loss": scale,
        "uvd_loss": scale,
        "uvd_absolute_loss": scale,
        "uvd_relative_loss": scale,
    }
    model = SimpleNamespace(
        lambda_action=1.0,
        lambda_depth_current=0.14,
        lambda_depth_future=0.15,
        lambda_uvd=0.62,
        lambda_uvd_relative=0.1,
    )

    objectives = extract_v2_objectives(output, model)

    assert set(objectives) == {
        "action",
        "depth_current",
        "depth_future",
        "uvd_absolute",
        "uvd_relative",
    }
    assert objectives["uvd_absolute"][1] == pytest.approx(0.62)
    assert objectives["uvd_relative"][1] == pytest.approx(0.062)


def test_extract_v2_objectives_includes_optional_wrist_depth_losses_and_weights():
    scale = nn.Parameter(torch.tensor(1.0))
    output = {
        "action_loss": scale,
        "depth_current_loss": scale,
        "depth_future_loss": scale,
        "wrist_depth_current_loss": 2.0 * scale,
        "wrist_depth_future_loss": 3.0 * scale,
        "uvd_loss": scale,
        "uvd_absolute_loss": scale,
        "uvd_relative_loss": scale,
    }
    model = SimpleNamespace(
        lambda_action=1.0,
        lambda_depth_current=0.0725,
        lambda_depth_future=0.0725,
        lambda_wrist_depth_current=0.0725,
        lambda_wrist_depth_future=0.0725,
        lambda_uvd=0.62,
        lambda_uvd_relative=0.1,
    )

    objectives = extract_v2_objectives(output, model)

    assert objectives["wrist_depth_current"][1] == pytest.approx(0.0725)
    assert objectives["wrist_depth_future"][1] == pytest.approx(0.0725)
    assert float(objectives["wrist_depth_current"][0]) == pytest.approx(2.0)
    assert float(objectives["wrist_depth_future"][0]) == pytest.approx(3.0)


def test_extract_v2_objectives_accepts_future_only_wrist_depth_loss():
    scale = nn.Parameter(torch.tensor(1.0))
    output = {
        "action_loss": scale,
        "depth_current_loss": 0.0 * scale,
        "depth_future_loss": scale,
        "wrist_depth_future_loss": 3.0 * scale,
        "uvd_loss": scale,
        "uvd_absolute_loss": scale,
        "uvd_relative_loss": scale,
    }
    model = SimpleNamespace(
        lambda_action=1.0,
        lambda_depth_current=0.0,
        lambda_depth_future=0.145,
        lambda_wrist_depth_current=0.0,
        lambda_wrist_depth_future=0.145,
        lambda_uvd=0.62,
        lambda_uvd_relative=0.1,
    )

    objectives = extract_v2_objectives(output, model)

    assert "wrist_depth_current" not in objectives
    assert objectives["wrist_depth_future"][1] == pytest.approx(0.145)
    assert float(objectives["wrist_depth_future"][0]) == pytest.approx(3.0)


def test_paired_depth_token_gradients_report_strength_weighting_and_view_conflict():
    measure = getattr(gradient_probe, "measure_paired_depth_token_gradients", None)
    assert measure is not None, "paired online depth-token gradient probe is missing"

    current_tokens = torch.tensor([[1.0, 2.0]], requires_grad=True)
    future_tokens = torch.tensor([[3.0, 4.0]], requires_grad=True)
    objectives = {
        "depth_current": (current_tokens[0, 0], 0.1),
        "wrist_depth_current": (-2.0 * current_tokens[0, 0], 0.2),
        "depth_future": (future_tokens[0, 0], 0.3),
        "wrist_depth_future": (future_tokens[0, 1], 0.4),
    }

    result = measure(
        objectives,
        {"current": current_tokens, "future": future_tokens},
        distributed=False,
    )

    assert result["depth_gradient/current/main_raw_grad_norm"] == pytest.approx(1.0)
    assert result["depth_gradient/current/main_weighted_grad_norm"] == pytest.approx(0.1)
    assert result["depth_gradient/current/wrist_raw_grad_norm"] == pytest.approx(2.0)
    assert result["depth_gradient/current/wrist_weighted_grad_norm"] == pytest.approx(0.4)
    assert result["depth_gradient/current/wrist_to_main_raw_norm_ratio"] == pytest.approx(2.0)
    assert result["depth_gradient/current/wrist_to_main_weighted_norm_ratio"] == pytest.approx(4.0)
    assert result["depth_gradient/current/main_wrist_cosine"] == pytest.approx(-1.0)
    assert result["depth_gradient/current/cosine_valid"] == 1.0
    assert result["depth_gradient/current/cosine_negative"] == 1.0
    assert result["depth_gradient/current/cosine_below_neg_0_2"] == 1.0

    assert result["depth_gradient/future/main_wrist_cosine"] == pytest.approx(0.0)
    assert result["depth_gradient/future/cosine_valid"] == 1.0
    assert result["depth_gradient/future/cosine_negative"] == 0.0
    assert result["depth_gradient/future/cosine_below_neg_0_2"] == 0.0


def test_paired_depth_token_gradients_skip_disabled_current_wrist_branch():
    current_tokens = torch.empty(1, 0, 2, requires_grad=True)
    future_tokens = torch.tensor([[3.0, 4.0]], requires_grad=True)
    objectives = {
        "depth_current": (future_tokens[0, 0] * 0.0, 0.0),
        "depth_future": (future_tokens[0, 0], 0.145),
        "wrist_depth_future": (future_tokens[0, 1], 0.145),
    }

    result = gradient_probe.measure_paired_depth_token_gradients(
        objectives,
        {"current": current_tokens, "future": future_tokens},
        distributed=False,
    )

    assert not any(key.startswith("depth_gradient/current/") for key in result)
    assert result["depth_gradient/future/main_raw_grad_norm"] == pytest.approx(1.0)
    assert result["depth_gradient/future/wrist_raw_grad_norm"] == pytest.approx(1.0)


def test_paired_depth_token_gradients_measure_independent_view_groups_and_cross_leakage():
    main = torch.tensor([[1.0, 2.0]], requires_grad=True)
    wrist = torch.tensor([[3.0, 4.0]], requires_grad=True)
    objectives = {
        "depth_future": (main[0, 0], 0.15),
        "wrist_depth_future": (2.0 * wrist[0, 1], 0.15),
    }

    result = gradient_probe.measure_paired_depth_token_gradients(
        objectives,
        {"future": main},
        wrist_depth_tokens={"future": wrist},
        distributed=False,
    )

    assert result["depth_gradient/future/shared_token_group"] == 0.0
    assert result["depth_gradient/future/main_raw_grad_norm"] == pytest.approx(1.0)
    assert result["depth_gradient/future/wrist_raw_grad_norm"] == pytest.approx(2.0)
    assert result["depth_gradient/future/main_loss_on_wrist_token_grad_norm"] == 0.0
    assert result["depth_gradient/future/wrist_loss_on_main_token_grad_norm"] == 0.0
    assert result["depth_gradient/future/main_wrist_cosine"] == pytest.approx(0.0)


def test_paired_depth_gradient_metric_root_supports_learnable_query_parameters():
    query = nn.Parameter(torch.tensor([[1.0, 2.0]]))
    objectives = {
        "depth_future": (query[0, 0], 0.15),
        "wrist_depth_future": (-query[0, 0], 0.15),
    }

    result = gradient_probe.measure_paired_depth_token_gradients(
        objectives,
        {"future": query},
        distributed=False,
        metric_root="depth_query_gradient",
    )

    assert result["depth_query_gradient/future/shared_token_group"] == 1.0
    assert result["depth_query_gradient/future/main_wrist_cosine"] == pytest.approx(-1.0)


def _make_online_probe_trainer(*, completed_steps=49, sync_gradients=True):
    trainer = CotV2Trainer.__new__(CotV2Trainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "test_diagnostics": {
                    "enabled": True,
                    "log_objective_gradients": True,
                    "objective_gradient_interval": 50,
                }
            }
        }
    )
    trainer.completed_steps = completed_steps
    trainer.accelerator = SimpleNamespace(
        sync_gradients=sync_gradients,
        unwrap_model=lambda model: model,
    )
    trainer.model = SimpleNamespace(
        lambda_action=1.0,
        lambda_depth_current=0.0725,
        lambda_depth_future=0.0725,
        lambda_wrist_depth_current=0.0725,
        lambda_wrist_depth_future=0.0725,
        lambda_uvd=0.62,
        lambda_uvd_relative=0.1,
    )
    return trainer


def test_online_gradient_probe_schedule_captures_only_synchronized_interval_steps():
    trainer = _make_online_probe_trainer(completed_steps=49, sync_gradients=True)

    assert trainer._forward_diagnostic_kwargs() == {
        "capture_depth_token_gradients": True
    }

    trainer.completed_steps = 48
    assert trainer._forward_diagnostic_kwargs() == {}

    trainer.completed_steps = 49
    trainer.accelerator.sync_gradients = False
    assert trainer._forward_diagnostic_kwargs() == {}


def test_online_gradient_probe_emits_main_wrist_metrics_and_removes_private_tensors():
    trainer = _make_online_probe_trainer()
    trainer._forward_diagnostic_kwargs()
    current = torch.tensor([[1.0, 2.0]], requires_grad=True)
    future = torch.tensor([[3.0, 4.0]], requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=True)
    output = {
        "action_loss": scale,
        "depth_current_loss": current[0, 0],
        "depth_future_loss": future[0, 0],
        "wrist_depth_current_loss": -2.0 * current[0, 0],
        "wrist_depth_future_loss": future[0, 1],
        "uvd_loss": scale,
        "uvd_absolute_loss": scale,
        "uvd_relative_loss": scale,
        "total_loss": scale,
        "_probe_depth_current_tokens": current,
        "_probe_depth_future_tokens": future,
    }

    metrics = trainer._pre_backward_diagnostic_metrics(output)

    assert metrics["diagnostic/objective_gradients_collected"] == 1.0
    assert metrics["depth_gradient/current/main_wrist_cosine"] == pytest.approx(-1.0)
    assert metrics["depth_gradient/future/main_wrist_cosine"] == pytest.approx(0.0)
    assert metrics["timing/objective_gradient_probe"] >= 0.0
    assert "_probe_depth_current_tokens" not in output
    assert "_probe_depth_future_tokens" not in output


def test_online_gradient_probe_records_separate_future_latent_and_query_groups():
    trainer = _make_online_probe_trainer()
    trainer._forward_diagnostic_kwargs()
    main_tokens = torch.tensor([[1.0, 2.0]], requires_grad=True)
    wrist_tokens = torch.tensor([[3.0, 4.0]], requires_grad=True)
    main_query = nn.Parameter(torch.tensor([[5.0, 6.0]]))
    wrist_query = nn.Parameter(torch.tensor([[7.0, 8.0]]))
    scale = torch.tensor(1.0, requires_grad=True)
    main_loss = main_tokens[0, 0] + main_query[0, 0]
    wrist_loss = wrist_tokens[0, 1] + wrist_query[0, 1]
    output = {
        "action_loss": scale,
        "depth_current_loss": scale * 0.0,
        "depth_future_loss": main_loss,
        "wrist_depth_future_loss": wrist_loss,
        "uvd_loss": scale,
        "uvd_absolute_loss": scale,
        "uvd_relative_loss": scale,
        "total_loss": scale + main_loss + wrist_loss,
        "_probe_depth_future_tokens": main_tokens,
        "_probe_wrist_depth_future_tokens": wrist_tokens,
        "_probe_depth_future_query": main_query,
        "_probe_wrist_depth_future_query": wrist_query,
    }

    metrics = trainer._pre_backward_diagnostic_metrics(output)

    assert metrics["depth_gradient/future/shared_token_group"] == 0.0
    assert metrics["depth_query_gradient/future/shared_token_group"] == 0.0
    assert metrics["depth_gradient/future/main_raw_grad_norm"] == pytest.approx(1.0)
    assert metrics["depth_query_gradient/future/wrist_raw_grad_norm"] == pytest.approx(1.0)
    assert not any(key.startswith("_probe_") for key in output)


class _SyntheticV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry_tokens = nn.Linear(2, 2)
        language_model = nn.Module()
        language_model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        core = nn.Module()
        core.language_model = language_model
        inner = nn.Module()
        inner.model = core
        interface = nn.Module()
        interface.model = inner
        self.qwen_vl_interface = interface


def test_select_shared_parameters_uses_geometry_tokens_and_requested_qwen_tail():
    model = _SyntheticV2()

    groups, metadata = select_shared_parameter_groups(model, qwen_tail_layers=2)

    assert len(groups["geometry_tokens"]) == 2
    assert len(groups["qwen_tail"]) == 4
    assert all("layers.0." not in name for name in metadata["qwen_tail"]["parameter_names"])


def test_extract_v2_objectives_rejects_missing_required_loss():
    model = SimpleNamespace(
        lambda_action=1.0,
        lambda_depth_current=0.14,
        lambda_depth_future=0.15,
        lambda_uvd=0.62,
        lambda_uvd_relative=0.1,
    )

    with pytest.raises(KeyError, match="uvd_relative_loss"):
        extract_v2_objectives({"action_loss": torch.tensor(1.0)}, model)


def test_probe_sample_indices_are_seeded_unique_and_overrideable():
    first = select_probe_sample_indices(dataset_length=20, sample_count=6, seed=42)
    second = select_probe_sample_indices(dataset_length=20, sample_count=6, seed=42)
    override = select_probe_sample_indices(
        dataset_length=20,
        sample_count=3,
        seed=99,
        sample_indices="8,3,11",
    )

    assert first == second
    assert len(set(first)) == 6
    assert override == [8, 3, 11]


def test_probe_summary_reports_distribution_and_valid_counts_for_numeric_leaves():
    records = [
        {"groups": {"shared_total": {"combined_aux": {"cosine_with_action": -0.5}}}},
        {"groups": {"shared_total": {"combined_aux": {"cosine_with_action": 0.5}}}},
    ]

    summary = summarize_probe_batches(records)
    stats = summary["groups"]["shared_total"]["combined_aux"]["cosine_with_action"]

    assert stats["mean"] == 0.0
    assert stats["median"] == 0.0
    assert stats["min"] == -0.5
    assert stats["max"] == 0.5
    assert stats["count"] == 2
    assert stats["finite"] is True


def test_checkpoint_content_hash_changes_when_file_content_changes(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"first")
    first = sha256_file(checkpoint)
    checkpoint.write_bytes(b"second")
    second = sha256_file(checkpoint)

    assert first == hashlib.sha256(b"first").hexdigest()
    assert second == hashlib.sha256(b"second").hexdigest()
    assert first != second
