import hashlib
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

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
