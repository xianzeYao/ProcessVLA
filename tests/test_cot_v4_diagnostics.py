import numpy as np
import pytest
import torch
from torch import nn

from starVLA.training.cot_test_diagnostics import (
    collect_batch_valid_ratios,
    collect_module_grad_norms,
    install_module_grad_norm_hooks,
)
from starVLA.training.cot_v4_diagnostics import (
    compute_v4_geometry_metrics,
)


def test_v4_geometry_metrics_cover_coarse_overlap_and_terminal_repeat():
    target = np.full((4, 2, 3), [0.2, 0.2, 1.0], dtype=np.float32)
    coarse = torch.tensor(
        np.full((1, 8, 3), [0.3, 0.3, 1.0], dtype=np.float32)
    )
    local = torch.tensor(
        np.full((1, 8, 3), [0.25, 0.3, 1.0], dtype=np.float32)
    )
    examples = [
        {
            "uvd": target,
            "uvd_valid_mask": np.ones((4, 2), dtype=np.bool_),
            "uvd_frame_indices": np.asarray([9, 9, 9, 9]),
            "uvd_coarse": target,
            "uvd_coarse_valid_mask": np.ones((4, 2), dtype=np.bool_),
            "uvd_coarse_frame_indices": np.asarray([9, 9, 9, 9]),
        }
    ]

    metrics = compute_v4_geometry_metrics(
        {"uvd_coarse": coarse, "uvd": local},
        examples,
        depth_scale=1.0,
        image_size=101,
        hand_count=2,
    )

    assert metrics["uvd_coarse_xy_mae_pixel"] == pytest.approx(10.0)
    assert metrics["uvd_coarse_depth_mae_m"] == pytest.approx(0.0)
    assert metrics["cross_scale/overlap_xy_gap_pixel"] == pytest.approx(5.0)
    assert metrics["data/terminal_repeat_sample_ratio"] == 1.0


def test_v4_batch_valid_ratios_include_coarse_reasons():
    examples = [
        {
            "uvd_coarse_valid_mask": np.asarray([True, False]),
            "uvd_coarse_out_of_frame_mask": np.asarray([False, True]),
            "uvd_coarse_boundary_clamp_mask": np.asarray([True, False]),
        }
    ]

    metrics = collect_batch_valid_ratios(examples)

    assert metrics["data/uvd_coarse_valid_ratio"] == 0.5
    assert metrics["data/uvd_coarse_out_of_frame_ratio"] == 0.5
    assert metrics["data/uvd_coarse_boundary_clamp_ratio"] == 0.5


class _ToyV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry_query = nn.Linear(2, 2, bias=False)
        self.depth_decoder = nn.Linear(2, 2, bias=False)
        self.local_uvd_head = nn.Linear(2, 2, bias=False)
        self.coarse_uvd_head = nn.Linear(2, 2, bias=False)
        self.action_model = nn.Linear(2, 2, bias=False)


def test_v4_gradient_diagnostics_keep_local_and_coarse_heads_separate():
    model = _ToyV4()
    hooks = install_module_grad_norm_hooks(model)
    sum(parameter.sum() for parameter in model.parameters()).backward()

    metrics = collect_module_grad_norms(model, hook_state=hooks)

    assert metrics["grad/local_uvd_head_norm"] > 0.0
    assert metrics["grad/coarse_uvd_head_norm"] > 0.0
    assert "grad/uvd_head_norm" not in metrics
