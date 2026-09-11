from types import MethodType, SimpleNamespace

import pytest
import torch
from PIL import Image

from starVLA.model.framework.VLM4A.QwenGR00TCoTV2DA3 import (
    Qwen_GR00T_CoT_V2_DA3,
)
from starVLA.training.train_starvla_cot_v2 import CotV2Trainer


def test_da3_forward_uses_exact_loss_weights_without_running_depth_decoder():
    model = Qwen_GR00T_CoT_V2_DA3.__new__(Qwen_GR00T_CoT_V2_DA3)
    torch.nn.Module.__init__(model)
    model.lambda_action = 1.0
    model.lambda_uvd = 0.62
    model.lambda_da3_feature_alignment = 0.15
    model.da3_feature_dim = 4
    model.da3_alignment_projector = torch.nn.Identity()
    model.depth_decoder = SimpleNamespace(
        __call__=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("numerical depth decoder must not run")
        )
    )
    split = SimpleNamespace(
        native=torch.zeros(1, 2, 4),
        depth_future=torch.ones(1, 8, 4),
        uvd=torch.zeros(1, 12, 4),
    )
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.zeros(1, 2, dtype=torch.long)},
            torch.ones(1, 2, dtype=torch.bool),
        ),
        model,
    )
    model._prepare_uvd_targets = MethodType(
        lambda self, examples, device: object(), model
    )
    model._run_geometry_backbone = MethodType(
        lambda self, qwen_inputs: split, model
    )
    model._predict_uvd = MethodType(
        lambda self, tokens: torch.zeros(1, 12, 3), model
    )
    model._compute_uvd_losses = MethodType(
        lambda self, pred, packed: {
            "absolute": torch.tensor(1.5),
            "relative": torch.tensor(0.5),
            "total": torch.tensor(2.0),
        },
        model,
    )
    model._teacher_targets = MethodType(
        lambda self, examples, device: -torch.ones(1, 8, 4), model
    )
    model._build_action_condition = MethodType(
        lambda self, split, native_attention_mask: (
            torch.zeros(1, 14, 4),
            torch.ones(1, 14, dtype=torch.bool),
        ),
        model,
    )
    model._action_loss = MethodType(
        lambda self, condition, condition_mask, examples: torch.tensor(3.0),
        model,
    )

    output = model.forward([{"future_image": Image.new("RGB", (2, 2))}])

    assert output["depth_current_loss"].item() == 0.0
    assert output["depth_future_loss"].item() == 0.0
    assert output["da3_feature_alignment_loss"].item() == pytest.approx(2.0)
    assert output["total_loss"].item() == pytest.approx(
        3.0 + 0.62 * 2.0 + 0.15 * 2.0
    )


def test_da3_predict_action_does_not_load_or_call_teacher():
    class FakeActionModel(torch.nn.Module):
        def predict_action(self, condition, state, encoder_attention_mask=None):
            return torch.zeros(condition.shape[0], 16, 29)

    class ForbiddenTeacher:
        def extract(self, *args, **kwargs):
            raise AssertionError("rollout inference must not touch DA3")

    model = Qwen_GR00T_CoT_V2_DA3.__new__(Qwen_GR00T_CoT_V2_DA3)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        framework=SimpleNamespace(action_model={"state_dim": 0})
    )
    model.action_model = FakeActionModel()
    model._da3_teacher = ForbiddenTeacher()
    split = SimpleNamespace(native=torch.zeros(1, 2, 4), uvd=torch.zeros(1, 12, 4))
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.zeros(1, 2, dtype=torch.long)},
            torch.ones(1, 2, dtype=torch.bool),
        ),
        model,
    )
    model._run_geometry_backbone = MethodType(
        lambda self, qwen_inputs: split, model
    )
    model._build_action_condition = MethodType(
        lambda self, split, native_attention_mask: (
            torch.zeros(1, 14, 4),
            torch.ones(1, 14, dtype=torch.bool),
        ),
        model,
    )

    result = model.predict_action([{"image": [], "lang": "test"}])

    assert result["normalized_actions"].shape == (1, 16, 29)


def test_v2_trainer_logs_raw_and_weighted_da3_alignment_loss():
    trainer = CotV2Trainer.__new__(CotV2Trainer)
    trainer.model = SimpleNamespace(lambda_da3_feature_alignment=0.15)

    metrics, weighted = trainer._extra_objective_metrics(
        {"da3_feature_alignment_loss": torch.tensor(0.4)}
    )

    assert metrics["da3_feature_alignment_loss"] == pytest.approx(0.4)
    assert metrics["weighted_da3_feature_alignment_loss"] == pytest.approx(0.06)
    assert weighted == pytest.approx(0.06)
