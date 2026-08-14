from __future__ import annotations

from types import MethodType
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import (
    GeometryHiddenSplit,
    Qwen_GR00T_CoT_V2,
)
from starVLA.model.framework.VLM4A.QwenGR00TCoTV3 import Qwen_GR00T_CoT_V3
from starVLA.model.modules.geometric_cot_v2 import GeometryTokenLayout
from starVLA.model.modules.geometric_cot_v3 import (
    LandmarkGeometryTokenLayout,
    PackedLandmarkUVDTargets,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


def make_model(*, points: int = 2) -> Qwen_GR00T_CoT_V3:
    model = Qwen_GR00T_CoT_V3.__new__(Qwen_GR00T_CoT_V3)
    model.geometry_layout = LandmarkGeometryTokenLayout(
        depth_query_count=1, uvd_time_points=points, landmark_count=3
    )
    model.landmark_count = 3
    model.uvd_token_order = "time_major"
    model.lambda_uvd_temporal = 0.1
    model.lambda_uvd_shape = 0.0
    return model


def literal_example() -> dict[str, np.ndarray]:
    return {
        "uvd": np.asarray(
            [
                [[0.1, 0.2, 1.0], [0.3, 0.2, 1.0], [0.2, 0.4, 1.0]],
                [[0.2, 0.2, 1.0], [0.4, 0.2, 1.0], [0.3, 0.4, 1.0]],
            ],
            dtype=np.float32,
        ),
        "uvd_valid_mask": np.ones((2, 3), dtype=np.bool_),
        "uvd_time": np.asarray([0.0, 1.0], dtype=np.float32),
        "uvd_landmark_ids": np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
    }


def test_v3_is_independently_registered_and_uses_landmark_layout() -> None:
    assert FRAMEWORK_REGISTRY["QwenGR00TCoTV3"] is Qwen_GR00T_CoT_V3
    assert issubclass(Qwen_GR00T_CoT_V3, Qwen_GR00T_CoT_V2)
    model = make_model(points=4)
    assert model.geometry_layout.uvd_token_count == 12
    assert model.landmark_count == 3
    assert model.uvd_token_order == "time_major"


def test_v3_rejects_non_triangle_config_and_v2_geometry_checkpoint() -> None:
    with pytest.raises(ValueError, match="landmark_count.*exactly 3"):
        LandmarkGeometryTokenLayout(
            depth_query_count=1, uvd_time_points=4, landmark_count=2
        )
    with pytest.raises(RuntimeError, match="landmark_embedding"):
        Qwen_GR00T_CoT_V3.validate_checkpoint_state_dict(
            {
                "geometry_tokens.trajectory_seed": torch.zeros(1),
                "depth_attention_pool.score.weight": torch.zeros(1),
            }
        )


def test_v3_packs_fixed_time_major_landmark_targets() -> None:
    model = make_model(points=3)
    packed = model._prepare_uvd_targets([literal_example()], torch.device("cpu"))

    assert packed.target.shape == (1, 9, 3)
    torch.testing.assert_close(
        packed.target[0, :6],
        torch.as_tensor(literal_example()["uvd"].reshape(6, 3)),
    )
    assert packed.landmark_ids[0].tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_v3_uvd_total_uses_independent_temporal_and_shape_weights() -> None:
    model = make_model(points=2)
    target = torch.as_tensor(literal_example()["uvd"].reshape(1, 6, 3))
    packed = PackedLandmarkUVDTargets(
        target=target,
        valid=torch.ones(1, 6, dtype=torch.bool),
        times=torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.float32),
        landmark_ids=torch.tensor([[0, 1, 2, 0, 1, 2]]),
    )
    pred = target.clone()
    pred[0, 4, 0] += 0.5

    zero_shape = model._compute_uvd_losses(pred, packed)
    assert torch.isclose(
        zero_shape["total"],
        zero_shape["absolute"] + 0.1 * zero_shape["temporal"],
    )

    model.lambda_uvd_shape = 2.0
    weighted_shape = model._compute_uvd_losses(pred, packed)
    assert torch.isclose(
        weighted_shape["total"],
        weighted_shape["absolute"]
        + 0.1 * weighted_shape["temporal"]
        + 2.0 * weighted_shape["shape"],
    )
    assert torch.isclose(
        weighted_shape["total"] - zero_shape["total"],
        2.0 * weighted_shape["shape"],
    )


def test_v3_forward_returns_three_uvd_components_and_existing_losses() -> None:
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
            "action": np.zeros((8, 7), dtype=np.float32),
        }
    )
    qwen_inputs = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 2, 4),
        depth_current=torch.zeros(1, 1, 4),
        depth_future=torch.zeros(1, 1, 4),
        uvd=torch.zeros(1, 6, 4),
    )
    pred_uvd = torch.as_tensor(example["uvd"].reshape(1, 6, 3)).clone()
    pred_uvd[0, 4, 0] += 0.5
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            qwen_inputs,
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
    assert float(output["action_loss"]) == 2.0
    assert float(output["depth_current_loss"]) == 0.0
    assert float(output["depth_future_loss"]) == 0.0
    assert torch.isclose(
        output["uvd_loss"],
        output["uvd_absolute_loss"] + 0.1 * output["uvd_temporal_loss"],
    )
    assert torch.isclose(
        output["total_loss"], 2.0 + 0.62 * output["uvd_loss"]
    )


def test_v3_predict_action_with_return_geometry_reuses_its_action_forward() -> None:
    """A missing geometry decode or a second backbone pass must fail this contract."""
    model = make_model(points=4)
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
    backbone_calls = 0
    decode_calls = 0

    def run_backbone(self, inputs):
        nonlocal backbone_calls
        backbone_calls += 1
        return split

    def decode_geometry(self, hidden, inputs, *, timing_callback=None):
        nonlocal decode_calls
        decode_calls += 1
        return (
            torch.ones(1, 1, 2, 2),
            torch.full((1, 1, 2, 2), 2.0),
            torch.arange(36, dtype=torch.bfloat16).reshape(1, 12, 3),
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
        predict_action=lambda condition, state, encoder_attention_mask: torch.zeros(1, 2, 7)
    )
    example = {"image": [], "lang": "move"}

    result = model.predict_action([example], return_geometry=True)

    assert backbone_calls == 1
    assert decode_calls == 1
    assert set(result["geometry"]) == {
        "depth_current",
        "depth_future",
        "uvd",
        "uvd_time",
        "uvd_landmark_ids",
    }
    assert result["geometry"]["uvd"].shape == (1, 12, 3)
    assert result["geometry"]["uvd_time"].shape == (1, 12)
    assert result["geometry"]["uvd_time"].dtype == torch.float32
    torch.testing.assert_close(
        result["geometry"]["uvd_time"],
        torch.linspace(0.0, 1.0, 4, dtype=torch.float32).repeat_interleave(3).unsqueeze(0),
        rtol=0.0,
        atol=0.0,
    )
    assert result["geometry"]["uvd_landmark_ids"].tolist()[0] == [0, 1, 2] * 4

    action_only = model.predict_action([example])

    assert set(action_only) == {"normalized_actions"}
    assert backbone_calls == 2
    assert decode_calls == 1

def test_v2_return_geometry_has_stable_v3_only_error() -> None:
    model = Qwen_GR00T_CoT_V2.__new__(Qwen_GR00T_CoT_V2)
    model.geometry_layout = GeometryTokenLayout(
        depth_query_count=1,
        uvd_points_per_hand=2,
        hand_count=1,
    )
    model.config = SimpleNamespace(
        framework=SimpleNamespace(action_model={"state_dim": 0})
    )
    calls: list[str] = []

    def unexpected(name: str):
        def fail(*args, **kwargs):
            calls.append(name)
            pytest.fail(f"{name} must not run for a V2 geometry request")

        return fail

    model._build_native_inputs = MethodType(unexpected("preprocess"), model)
    model._run_geometry_backbone = MethodType(unexpected("backbone"), model)
    model._build_action_condition = MethodType(unexpected("action_condition"), model)
    model.action_model = SimpleNamespace(predict_action=unexpected("action_expert"))

    with pytest.raises(ValueError, match="supported only by QwenGR00TCoTV3"):
        model.predict_action([{"image": [], "lang": "move"}], return_geometry=True)

    assert calls == []
