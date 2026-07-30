import numpy as np
import torch
from types import MethodType

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import Qwen_GR00T_CoT_V2
from starVLA.model.modules.geometric_cot_v2 import GeometryTokenLayout, PackedUVDTargets
from starVLA.model.tools import FRAMEWORK_REGISTRY


def make_uninitialized_model(*, depth_queries=2, points=3, hands=2):
    model = Qwen_GR00T_CoT_V2.__new__(Qwen_GR00T_CoT_V2)
    model.geometry_layout = GeometryTokenLayout(
        depth_query_count=depth_queries,
        uvd_points_per_hand=points,
        hand_count=hands,
    )
    return model


def test_v2_is_registered_and_inherits_baseline_without_v1_reasoner():
    assert FRAMEWORK_REGISTRY["QwenGR00TCoTV2"] is Qwen_GR00T_CoT_V2
    assert issubclass(Qwen_GR00T_CoT_V2, Qwen_GR00T)
    assert "Qwen_GR00T_CoT" not in [base.__name__ for base in Qwen_GR00T_CoT_V2.__mro__]


def test_v2_prepares_fixed_time_major_uvd_targets():
    model = make_uninitialized_model()
    examples = [
        {
            "uvd": np.asarray(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
                dtype=np.float32,
            ),
            "uvd_valid_mask": np.asarray([[True, True], [True, True]]),
            "uvd_time": np.asarray([0.0, 0.5], dtype=np.float32),
        }
    ]

    packed = model._prepare_uvd_targets(examples, torch.device("cpu"))

    assert packed.target.shape == (1, 6, 3)
    assert packed.target[0, :4].tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]
    assert packed.valid[0].tolist() == [True, True, True, True, False, False]


def test_v2_uvd_objective_combines_all_point_and_adjacent_relative_losses():
    model = make_uninitialized_model(depth_queries=1, points=3, hands=1)
    model.lambda_uvd_relative = 0.1
    target = torch.zeros(1, 3, 3)
    packed = PackedUVDTargets(
        target=target,
        valid=torch.ones(1, 3, dtype=torch.bool),
        times=torch.tensor([[0.0, 0.5, 1.0]]),
        hand_ids=torch.zeros(1, 3, dtype=torch.long),
    )
    pred = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
    )

    losses = model._compute_uvd_losses(pred, packed)

    assert torch.isclose(losses["absolute"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(losses["relative"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(
        losses["total"],
        torch.tensor((1.0 / 3.0) + 0.1 * (1.0 / 3.0)),
    )


def test_action_condition_contains_all_fixed_uvd_slots_and_excludes_depth_tokens():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1)
    # Sequence values: V0,V1,Dc,Df,U0,U1.
    all_hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    native_mask = torch.tensor([[True, True]])
    split = model._split_geometry_hidden(all_hidden, native_token_count=2)
    condition, condition_mask = model._build_action_condition(
        split,
        native_attention_mask=native_mask,
    )

    assert split.depth_current.flatten().tolist() == [2.0]
    assert split.depth_future.flatten().tolist() == [3.0]
    assert split.uvd.flatten().tolist() == [4.0, 5.0]
    assert condition.flatten().tolist() == [0.0, 1.0, 4.0, 5.0]
    assert condition_mask.tolist() == [[True, True, True, True]]


def test_geometry_backbone_api_cannot_accept_ground_truth_times_or_validity():
    import inspect

    parameters = inspect.signature(Qwen_GR00T_CoT_V2._run_geometry_backbone).parameters

    assert list(parameters) == ["self", "qwen_inputs"]


def test_predict_geometry_does_not_require_ground_truth_uvd_fields():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1)
    qwen_inputs = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )

    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (qwen_inputs, torch.ones(1, 2, dtype=torch.bool)),
        model,
    )
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)
    model._decode_geometry = MethodType(
        lambda self, hidden, inputs: (
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 2, 3),
        ),
        model,
    )

    output = model.predict_geometry([{"image": [], "lang": "move"}])

    assert output["uvd"].shape == (1, 2, 3)
