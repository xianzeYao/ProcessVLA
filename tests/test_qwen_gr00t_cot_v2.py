import numpy as np
import pytest
import torch
from types import MethodType

import starVLA.model.framework.VLM4A.QwenGR00TCoTV2 as cot_v2_module
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import GeometryHiddenSplit, Qwen_GR00T_CoT_V2
from starVLA.model.modules.geometric_cot_v2 import (
    GeometryTokenLayout,
    PackedUVDTargets,
    SharedDepthAttentionPool,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


def make_uninitialized_model(*, depth_queries=2, points=3, hands=2, include_depth=False):
    model = Qwen_GR00T_CoT_V2.__new__(Qwen_GR00T_CoT_V2)
    model.geometry_layout = GeometryTokenLayout(
        depth_query_count=depth_queries,
        uvd_points_per_hand=points,
        hand_count=hands,
    )
    model.include_depth_in_action_condition = include_depth
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


def test_action_condition_includes_both_depth_groups_when_enabled():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    # Sequence values: V0,V1,Dc,Df,U0,U1.
    all_hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    native_mask = torch.tensor([[True, False]])
    split = model._split_geometry_hidden(all_hidden, native_token_count=2)

    condition, condition_mask = model._build_action_condition(
        split,
        native_attention_mask=native_mask,
    )

    assert condition.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert condition_mask.tolist() == [[True, False, True, True, True, True]]


def test_zero_geometry_preserves_correct_shape_and_mask():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )
    native_mask = torch.tensor([[True, False]])

    correct = model._build_intervention_condition(
        split,
        native_attention_mask=native_mask,
        name="correct",
    )
    zero = model._build_intervention_condition(
        split,
        native_attention_mask=native_mask,
        name="zero_geometry",
    )

    assert zero.condition.shape == correct.condition.shape
    assert torch.equal(zero.condition_mask, correct.condition_mask)
    assert zero.condition.flatten().tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_native_only_condition_and_mask_have_matching_lengths():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )

    native_only = model._build_intervention_condition(
        split,
        native_attention_mask=torch.tensor([[True, False]]),
        name="native_only",
    )

    assert native_only.condition.flatten().tolist() == [0.0, 1.0]
    assert native_only.condition_mask.tolist() == [[True, False]]
    assert native_only.condition.shape[:2] == native_only.condition_mask.shape


def test_uvd_only_and_depth_only_are_length_matched_for_depth_conditioned_model():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )
    native_mask = torch.ones(1, 2, dtype=torch.bool)

    uvd_only = model._build_intervention_condition(
        split,
        native_attention_mask=native_mask,
        name="uvd_only",
    )
    depth_only = model._build_intervention_condition(
        split,
        native_attention_mask=native_mask,
        name="depth_only",
    )

    assert uvd_only.condition.flatten().tolist() == [0.0, 1.0, 0.0, 0.0, 4.0, 5.0]
    assert depth_only.condition.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
    assert uvd_only.condition.shape == depth_only.condition.shape == (1, 6, 1)
    assert not uvd_only.diagnostic_counterfactual
    assert not depth_only.diagnostic_counterfactual


def test_depth_only_is_marked_counterfactual_for_q0_model():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=False,
    )
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )

    depth_only = model._build_intervention_condition(
        split,
        native_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        name="depth_only",
    )

    assert depth_only.condition.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert depth_only.condition_mask.tolist() == [[True, True, True, True]]
    assert depth_only.diagnostic_counterfactual


def test_geometry_permutations_obey_within_and_cross_task_constraints():
    task_ids = ["a", "a", "b", "b"]
    generator = torch.Generator().manual_seed(7)

    within = cot_v2_module.build_geometry_permutation(
        task_ids,
        mode="within_task_shuffle",
        generator=generator,
    )
    cross = cot_v2_module.build_geometry_permutation(
        task_ids,
        mode="cross_task_swap",
        generator=torch.Generator().manual_seed(7),
    )

    assert not torch.equal(within, torch.arange(4))
    assert not torch.equal(cross, torch.arange(4))
    assert all(task_ids[index] == task_ids[within[index]] for index in range(4))
    assert all(task_ids[index] != task_ids[cross[index]] for index in range(4))


@pytest.mark.parametrize(
    ("task_ids", "mode", "message"),
    [
        (["a"], "within_task_shuffle", "at least two samples"),
        (["a", "b"], "within_task_shuffle", "at least two samples for task"),
        (["a", "a", "a", "b"], "cross_task_swap", "impossible"),
    ],
)
def test_geometry_permutation_rejects_impossible_batches(task_ids, mode, message):
    with pytest.raises(ValueError, match=message):
        cot_v2_module.build_geometry_permutation(task_ids, mode=mode)


def test_shuffle_moves_the_whole_geometry_bundle_without_moving_native_tokens():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    split = GeometryHiddenSplit(
        native=torch.tensor([[[0.0]], [[10.0]]]),
        depth_current=torch.tensor([[[1.0]], [[11.0]]]),
        depth_future=torch.tensor([[[2.0]], [[12.0]]]),
        uvd=torch.tensor([[[3.0], [4.0]], [[13.0], [14.0]]]),
    )

    shuffled = model._build_intervention_condition(
        split,
        native_attention_mask=torch.ones(2, 1, dtype=torch.bool),
        name="within_task_shuffle",
        permutation=torch.tensor([1, 0]),
    )

    assert shuffled.condition[:, :, 0].tolist() == [
        [0.0, 11.0, 12.0, 13.0, 14.0],
        [10.0, 1.0, 2.0, 3.0, 4.0],
    ]
    assert shuffled.permutation.tolist() == [1, 0]


def test_intervention_rejects_identity_permutation_and_nonfinite_condition():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1, include_depth=True)
    split = GeometryHiddenSplit(
        native=torch.zeros(2, 1, 1),
        depth_current=torch.zeros(2, 1, 1),
        depth_future=torch.zeros(2, 1, 1),
        uvd=torch.zeros(2, 2, 1),
    )

    with pytest.raises(ValueError, match="identity"):
        model._build_intervention_condition(
            split,
            native_attention_mask=torch.ones(2, 1, dtype=torch.bool),
            name="within_task_shuffle",
            permutation=torch.arange(2),
        )

    invalid = GeometryHiddenSplit(
        native=split.native,
        depth_current=split.depth_current,
        depth_future=split.depth_future,
        uvd=torch.full((2, 2, 1), float("nan")),
    )
    with pytest.raises(ValueError, match="finite"):
        model._build_intervention_condition(
            invalid,
            native_attention_mask=torch.ones(2, 1, dtype=torch.bool),
            name="correct",
        )


@pytest.mark.parametrize("value", ["false", 1, None])
def test_action_condition_rejects_non_boolean_depth_condition_flag(value):
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=value,
    )
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )

    with pytest.raises(ValueError, match="include_depth_in_action_condition must be a boolean"):
        model._build_action_condition(split, native_attention_mask=None)


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


def test_old_mean_pooling_v2_checkpoint_is_rejected_explicitly():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1)
    torch.nn.Module.__init__(model)

    with pytest.raises(RuntimeError, match="attention-pooling"):
        model.load_state_dict(
            {"geometry_tokens.current_depth": torch.zeros(1, 1, 1)},
            strict=False,
        )


def test_partial_reload_cannot_bypass_old_v2_checkpoint_rejection(tmp_path):
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1)
    torch.nn.Module.__init__(model)
    model.geometry_tokens = torch.nn.Module()
    model.geometry_tokens.register_parameter(
        "current_depth",
        torch.nn.Parameter(torch.zeros(1, 1, 1)),
    )
    model.depth_attention_pool = SharedDepthAttentionPool(hidden_dim=1)
    checkpoint = tmp_path / "old_mean_pool_v2.pt"
    torch.save(
        {"geometry_tokens.current_depth": torch.ones(1, 1, 1)},
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="attention-pooling"):
        TrainerUtils.load_pretrained_backbones(
            model,
            checkpoint,
            reload_modules="geometry_tokens",
        )


class _CountingDepthDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.queries = []

    def forward(self, image_tokens, *, patch_hw, query, output_hw):
        self.queries.append(query.detach().clone())
        return query[:, :1, None, None].expand(-1, 1, *output_hw)


def make_diagnostic_model():
    model = make_uninitialized_model(depth_queries=2, points=2, hands=1)
    torch.nn.Module.__init__(model)
    model.depth_attention_pool = SharedDepthAttentionPool(hidden_dim=2)
    with torch.no_grad():
        model.depth_attention_pool.score.weight.zero_()
    model.depth_decoder = _CountingDepthDecoder()
    model.depth_output_size = 2
    split = GeometryHiddenSplit(
        native=torch.zeros(2, 2, 2),
        depth_current=torch.tensor(
            [[[1.0, 0.0], [3.0, 0.0]], [[5.0, 0.0], [7.0, 0.0]]]
        ),
        depth_future=torch.tensor(
            [[[10.0, 0.0], [14.0, 0.0]], [[20.0, 0.0], [24.0, 0.0]]]
        ),
        uvd=torch.zeros(2, 2, 2),
    )
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.ones(2, 2, dtype=torch.long)},
            torch.ones(2, 2, dtype=torch.bool),
        ),
        model,
    )
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)
    model._main_image_tokens = MethodType(
        lambda self, native, input_ids: (torch.zeros(2, 1, 2), (1, 1)),
        model,
    )
    model._predict_uvd = MethodType(
        lambda self, tokens: torch.zeros(tokens.shape[0], tokens.shape[1], 3),
        model,
    )
    return model


def test_v2_diagnostics_skip_decoder_interventions_when_disabled():
    model = make_diagnostic_model()

    output = model.predict_geometry_diagnostics(
        [{"image": [], "lang": "move"}, {"image": [], "lang": "move"}],
        include_decoder_interventions=False,
    )

    assert len(model.depth_decoder.queries) == 2
    assert "decoder_interventions" not in output
    assert output["depth_current_pool_weights"].shape == (2, 2)
    assert output["depth_current_tokens"].shape == (2, 2, 2)


def test_v2_diagnostics_reuse_features_for_zero_swap_shuffle_without_parameter_changes():
    model = make_diagnostic_model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    output = model.predict_geometry_diagnostics(
        [{"image": [], "lang": "move"}, {"image": [], "lang": "move"}],
        include_decoder_interventions=True,
    )

    assert len(model.depth_decoder.queries) == 8
    assert set(output["decoder_interventions"]) == {"zero", "swap", "shuffle"}
    assert torch.equal(model.depth_decoder.queries[2], torch.zeros(2, 2))
    assert torch.equal(model.depth_decoder.queries[4], torch.tensor([[12.0, 0.0], [22.0, 0.0]]))
    assert torch.equal(model.depth_decoder.queries[6], torch.tensor([[6.0, 0.0], [2.0, 0.0]]))
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
