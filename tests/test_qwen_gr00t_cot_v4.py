import numpy as np
import pytest
import torch
from torch import nn

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import Qwen_GR00T_CoT_V2
from starVLA.model.framework.VLM4A.QwenGR00TCoTV4 import (
    GeometryHiddenSplit,
    Qwen_GR00T_CoT_V4,
    validate_v4_horizon_contract,
)
from starVLA.model.modules.geometric_cot_v4 import GeometryTokenLayout
from starVLA.model.tools import FRAMEWORK_REGISTRY


def make_uninitialized_model(
    *,
    depth_queries=1,
    points=2,
    hands=1,
    include_depth=True,
    hidden_dim=4,
):
    model = Qwen_GR00T_CoT_V4.__new__(Qwen_GR00T_CoT_V4)
    nn.Module.__init__(model)
    model.geometry_layout = GeometryTokenLayout(
        depth_query_count=depth_queries,
        local_uvd_points_per_hand=points,
        coarse_uvd_points_per_hand=points,
        hand_count=hands,
    )
    model.include_depth_in_action_condition = include_depth
    model.lambda_uvd_relative = 0.1
    model.lambda_uvd_coarse_relative = 0.1
    model.local_uvd_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 3),
    )
    model.coarse_uvd_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 3),
    )
    return model


def test_v4_is_registered_and_does_not_subclass_v2():
    assert FRAMEWORK_REGISTRY["QwenGR00TCoTV4"] is Qwen_GR00T_CoT_V4
    assert issubclass(Qwen_GR00T_CoT_V4, Qwen_GR00T)
    assert Qwen_GR00T_CoT_V2 not in Qwen_GR00T_CoT_V4.__mro__


def test_v4_action_condition_is_native_depth_coarse_local():
    model = make_uninitialized_model()
    split = GeometryHiddenSplit(
        native=torch.tensor([[[0.0]]]),
        depth_current=torch.tensor([[[1.0]]]),
        depth_future=torch.tensor([[[2.0]]]),
        uvd_coarse=torch.tensor([[[3.0], [4.0]]]),
        uvd_local=torch.tensor([[[5.0], [6.0]]]),
    )

    condition, mask = model._build_action_condition(
        split,
        native_attention_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    assert condition.flatten().tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]
    assert mask.all()


def test_v4_hidden_split_follows_single_qwen_sequence_order():
    model = make_uninitialized_model(points=2)
    hidden = torch.arange(7, dtype=torch.float32).view(1, 7, 1)

    split = model._split_geometry_hidden(hidden, native_token_count=1)

    assert split.native.flatten().tolist() == [0.0]
    assert split.depth_current.flatten().tolist() == [1.0]
    assert split.depth_future.flatten().tolist() == [2.0]
    assert split.uvd_coarse.flatten().tolist() == [3.0, 4.0]
    assert split.uvd_local.flatten().tolist() == [5.0, 6.0]


def test_v4_has_independent_heads_and_both_receive_gradients():
    model = make_uninitialized_model(hidden_dim=4)
    assert model.local_uvd_head is not model.coarse_uvd_head
    hidden = torch.randn(2, 2, 4, requires_grad=True)

    local = model._predict_local_uvd(hidden)
    coarse = model._predict_coarse_uvd(hidden)
    (local.sum() + coarse.sum()).backward()

    assert all(parameter.grad is not None for parameter in model.local_uvd_head.parameters())
    assert all(parameter.grad is not None for parameter in model.coarse_uvd_head.parameters())


def test_v4_total_loss_uses_independent_local_and_coarse_weights():
    model = make_uninitialized_model()
    model.lambda_action = 1.0
    model.lambda_depth_current = 0.14
    model.lambda_depth_future = 0.15
    model.lambda_uvd = 0.62
    model.lambda_uvd_coarse = 0.2
    values = [torch.tensor(value) for value in [1.0, 2.0, 3.0, 4.0, 5.0]]

    total = model._aggregate_total_loss(*values)

    expected = 1.0 + 0.14 * 2.0 + 0.15 * 3.0 + 0.62 * 4.0 + 0.2 * 5.0
    torch.testing.assert_close(total, torch.tensor(expected))


@pytest.mark.parametrize(
    "model_geometry,data_geometry,message",
    [
        (
            {"local_uvd_num_points": 7, "coarse_uvd_num_points": 8, "coarse_uvd_stride": 2},
            {"action_horizon": 8, "local_uvd_num_points": 8, "coarse_uvd_num_points": 8, "coarse_uvd_stride": 2, "terminal_repeat": True},
            "local",
        ),
        (
            {"local_uvd_num_points": 8, "coarse_uvd_num_points": 8, "coarse_uvd_stride": 1},
            {"action_horizon": 8, "local_uvd_num_points": 8, "coarse_uvd_num_points": 8, "coarse_uvd_stride": 2, "terminal_repeat": True},
            "stride",
        ),
        (
            {"local_uvd_num_points": 8, "coarse_uvd_num_points": 8, "coarse_uvd_stride": 2},
            {"action_horizon": 8, "local_uvd_num_points": 8, "coarse_uvd_num_points": 8, "coarse_uvd_stride": 2, "terminal_repeat": False},
            "terminal_repeat",
        ),
    ],
)
def test_v4_horizon_contract_rejects_model_data_mismatch(
    model_geometry,
    data_geometry,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_v4_horizon_contract(
            action_horizon=8,
            model_geometry=model_geometry,
            data_geometry=data_geometry,
        )


def test_v4_checkpoint_guard_rejects_reverse_full_keys():
    with pytest.raises(RuntimeError, match="reverse/full"):
        Qwen_GR00T_CoT_V4.validate_checkpoint_state_dict(
            {"geometry_tokens.full_uvd_queries": torch.zeros(1)}
        )


def test_v4_checkpoint_guard_accepts_native_v4_keys():
    Qwen_GR00T_CoT_V4.validate_checkpoint_state_dict(
        {
            "geometry_tokens.coarse_queries": torch.zeros(1),
            "geometry_tokens.local_queries": torch.zeros(1),
            "depth_attention_pool.score.weight": torch.zeros(1),
        }
    )
