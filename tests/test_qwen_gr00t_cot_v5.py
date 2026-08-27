from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import (
    GeometryHiddenSplit,
    Qwen_GR00T_CoT_V2,
)
from starVLA.model.framework.VLM4A.QwenGR00TCoTV5 import Qwen_GR00T_CoT_V5
from starVLA.model.modules.cot_losses import uvd_adjacent_relative_loss
from starVLA.model.modules.geometric_cot_v5 import (
    HandConfigurationTokenLayout,
    HandLRWDecoder,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


def make_model(*, points: int = 2, include_depth: bool = True, hidden_dim: int = 4):
    model = Qwen_GR00T_CoT_V5.__new__(Qwen_GR00T_CoT_V5)
    nn.Module.__init__(model)
    model.geometry_layout = HandConfigurationTokenLayout(
        depth_query_count=1,
        uvd_points_per_hand=points,
        hand_count=2,
        landmark_count=3,
    )
    model.include_depth_in_action_condition = include_depth
    model.uvd_hand_count = 2
    model.landmark_count = 3
    model.uvd_track_count = 6
    model.uvd_token_order = "time_major"
    model.lambda_uvd_temporal = 0.1
    model.lambda_uvd_shape = 0.0
    model.uvd_head = HandLRWDecoder(hidden_dim=hidden_dim, landmark_count=3)
    return model


def literal_example() -> dict[str, np.ndarray]:
    uvd = np.asarray(
        [
            [
                [[0.10, 0.20, 1.0], [0.20, 0.20, 1.0], [0.15, 0.35, 1.0]],
                [[0.60, 0.20, 1.1], [0.70, 0.20, 1.1], [0.65, 0.35, 1.1]],
            ],
            [
                [[0.15, 0.20, 1.0], [0.25, 0.20, 1.0], [0.20, 0.35, 1.0]],
                [[0.55, 0.20, 1.1], [0.65, 0.20, 1.1], [0.60, 0.35, 1.1]],
            ],
        ],
        dtype=np.float32,
    )
    return {
        "uvd": uvd,
        "uvd_valid_mask": np.ones((2, 2, 3), dtype=np.bool_),
        "uvd_time": np.asarray([0.0, 1.0], dtype=np.float32),
        "uvd_hand_ids": np.broadcast_to(
            np.asarray([[[0, 0, 0], [1, 1, 1]]], dtype=np.int64),
            (2, 2, 3),
        ).copy(),
        "uvd_landmark_ids": np.broadcast_to(
            np.asarray([[[0, 1, 2], [0, 1, 2]]], dtype=np.int64),
            (2, 2, 3),
        ).copy(),
    }


def test_v5_is_registered_and_keeps_action_condition_at_12_uvd_tokens():
    assert FRAMEWORK_REGISTRY["QwenGR00TCoTV5"] is Qwen_GR00T_CoT_V5
    assert issubclass(Qwen_GR00T_CoT_V5, Qwen_GR00T_CoT_V2)
    model = make_model(points=6, include_depth=True)
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 5, 4),
        depth_current=torch.zeros(1, 8, 4),
        depth_future=torch.zeros(1, 8, 4),
        uvd=torch.zeros(1, 12, 4),
    )

    condition, mask = model._build_action_condition(
        split, native_attention_mask=torch.ones(1, 5, dtype=torch.bool)
    )

    assert condition.shape == (1, 5 + 8 + 8 + 12, 4)
    assert mask.shape == condition.shape[:2]
    assert model._predict_uvd(split.uvd).shape == (1, 36, 3)


def test_v5_packs_36_targets_but_exposes_only_12_qwen_uvd_slots():
    model = make_model(points=6)

    packed = model._prepare_uvd_targets([literal_example()], torch.device("cpu"))

    assert model.geometry_layout.uvd_token_count == 12
    assert model.geometry_layout.output_point_count == 36
    assert packed.target.shape == (1, 36, 3)
    torch.testing.assert_close(
        packed.target[0, :12],
        torch.as_tensor(literal_example()["uvd"].reshape(12, 3)),
    )
    assert not packed.valid[0, 12:].any()


def test_v5_uvd_loss_uses_six_temporal_tracks_and_zero_shape_weight():
    model = make_model(points=2)
    packed = model._prepare_uvd_targets([literal_example()], torch.device("cpu"))
    pred = packed.target.clone()
    pred[0, 7, 0] += 0.5

    losses = model._compute_uvd_losses(pred, packed)
    expected_temporal = uvd_adjacent_relative_loss(
        pred, packed.target, packed.valid, hand_count=6
    )

    torch.testing.assert_close(losses["temporal"], expected_temporal)
    assert losses["shape"] > 0
    torch.testing.assert_close(
        losses["total"], losses["absolute"] + 0.1 * losses["temporal"]
    )


def test_v5_decoder_activation_bounds_uv_and_keeps_positive_depth():
    model = make_model(points=2)
    prediction = model._predict_uvd(torch.randn(2, 4, 4))

    assert prediction.shape == (2, 12, 3)
    assert torch.all((prediction[..., :2] >= 0) & (prediction[..., :2] <= 1))
    assert torch.all(prediction[..., 2] > 0)


def test_v5_forward_reports_temporal_and_shape_components():
    model = make_model(points=2)
    model.lambda_action = 1.0
    model.lambda_depth_current = 0.14
    model.lambda_depth_future = 0.15
    model.lambda_uvd = 0.62
    example = literal_example()
    example.update(
        {
            "depth_current": np.zeros((1, 2, 2), dtype=np.float32),
            "depth_future": np.ones((1, 2, 2), dtype=np.float32),
            "depth_current_valid": np.ones((1, 2, 2), dtype=np.bool_),
            "depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
            "action": np.zeros((16, 29), dtype=np.float32),
        }
    )
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 2, 4),
        depth_current=torch.zeros(1, 1, 4),
        depth_future=torch.zeros(1, 1, 4),
        uvd=torch.zeros(1, 4, 4),
    )
    pred_uvd = torch.as_tensor(example["uvd"].reshape(1, 12, 3)).clone()
    pred_uvd[0, 7, 0] += 0.5
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.ones(1, 2, dtype=torch.long)},
            torch.ones(1, 2, dtype=torch.bool),
        ),
        model,
    )
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)
    model._decode_geometry = MethodType(
        lambda self, hidden, inputs: (
            torch.zeros(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            pred_uvd,
        ),
        model,
    )
    model._build_action_condition = MethodType(
        lambda self, hidden, native_attention_mask: (
            torch.zeros(1, 8, 4),
            torch.ones(1, 8, dtype=torch.bool),
        ),
        model,
    )
    model._action_loss = MethodType(
        lambda self, condition, condition_mask, examples: torch.tensor(2.0), model
    )

    output = model.forward([example])

    assert set(output) == {
        "action_loss",
        "depth_current_loss",
        "depth_future_loss",
        "uvd_loss",
        "uvd_absolute_loss",
        "uvd_temporal_loss",
        "uvd_shape_loss",
        "total_loss",
    }
    torch.testing.assert_close(
        output["uvd_loss"],
        output["uvd_absolute_loss"] + 0.1 * output["uvd_temporal_loss"],
    )


def test_v5_geometry_diagnostics_keep_12_states_and_decode_36_points():
    model = make_model(points=6)
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 2, 4),
        depth_current=torch.zeros(1, 1, 4),
        depth_future=torch.zeros(1, 1, 4),
        uvd=torch.zeros(1, 12, 4),
    )

    prediction = model._predict_uvd(split.uvd)

    assert split.uvd.shape == (1, 12, 4)
    assert prediction.shape == (1, 36, 3)


def test_v5_predict_action_geometry_metadata_matches_all_36_points_once():
    model = make_model(points=6)
    model.config = SimpleNamespace(
        framework=SimpleNamespace(action_model={"state_dim": 0})
    )
    qwen_inputs = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 2, 4),
        depth_current=torch.zeros(1, 1, 4),
        depth_future=torch.zeros(1, 1, 4),
        uvd=torch.zeros(1, 12, 4),
    )
    expected_uvd = torch.arange(108, dtype=torch.bfloat16).reshape(1, 36, 3)
    calls = {"backbone": 0, "decode": 0}

    def run_backbone(self, inputs):
        calls["backbone"] += 1
        return split

    def decode_geometry(self, hidden, inputs, *, timing_callback=None):
        calls["decode"] += 1
        return (
            torch.ones(1, 1, 2, 2),
            torch.full((1, 1, 2, 2), 2.0),
            expected_uvd,
        )

    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            qwen_inputs,
            torch.ones(1, 2, dtype=torch.bool),
        ),
        model,
    )
    model._run_geometry_backbone = MethodType(run_backbone, model)
    model._decode_geometry = MethodType(decode_geometry, model)
    model._build_action_condition = MethodType(
        lambda self, hidden, native_attention_mask: (
            torch.zeros(1, 2, 4),
            torch.ones(1, 2, dtype=torch.bool),
        ),
        model,
    )
    model.action_model = SimpleNamespace(
        predict_action=lambda condition, state, encoder_attention_mask: torch.zeros(
            1, 2, 29
        )
    )

    result = model.predict_action(
        [{"image": [], "lang": "move"}], return_geometry=True
    )

    assert calls == {"backbone": 1, "decode": 1}
    geometry = result["geometry"]
    assert set(geometry) == {
        "depth_current",
        "depth_future",
        "uvd",
        "uvd_time",
        "uvd_hand_ids",
        "uvd_landmark_ids",
    }
    assert geometry["uvd"].shape == (1, 36, 3)
    assert geometry["uvd_time"].shape == (1, 36)
    assert geometry["uvd_hand_ids"].tolist()[0] == [0, 0, 0, 1, 1, 1] * 6
    assert geometry["uvd_landmark_ids"].tolist()[0] == [0, 1, 2, 0, 1, 2] * 6
    torch.testing.assert_close(
        geometry["uvd_time"],
        torch.linspace(0.0, 1.0, 6).repeat_interleave(6).unsqueeze(0),
    )


def test_v5_checkpoint_guard_rejects_v2_decoder_without_landmark_embedding():
    with pytest.raises(RuntimeError, match="uvd_head.landmark_embedding"):
        Qwen_GR00T_CoT_V5.validate_checkpoint_state_dict(
            {
                "geometry_tokens.trajectory_seed": torch.zeros(1),
                "depth_attention_pool.score.weight": torch.zeros(1),
                "uvd_head.0.weight": torch.zeros(1),
            }
        )

    Qwen_GR00T_CoT_V5.validate_checkpoint_state_dict(
        {
            "geometry_tokens.trajectory_seed": torch.zeros(1),
            "depth_attention_pool.score.weight": torch.zeros(1),
            "uvd_head.landmark_embedding.weight": torch.zeros(1),
        }
    )
