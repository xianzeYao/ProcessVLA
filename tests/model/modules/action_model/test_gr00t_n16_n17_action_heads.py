from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import AlternateVLDiT, DiT
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead as LegacyActionHead
from starVLA.model.modules.action_model.GR00T_N16_ActionHeader import N16FlowmatchingActionHead
from starVLA.model.modules.action_model.GR00T_N17_ActionHeader import N17FlowmatchingActionHead

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ROBOCASA_COT_V2_CONFIG = (
    _REPO_ROOT / "examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2.yaml"
)
_ROBOCASA_COT_V2_Q0_CONFIG = (
    _REPO_ROOT / "examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2_q0.yaml"
)


@dataclass
class _AttentionCall:
    encoder_hidden_states: torch.Tensor | None
    encoder_attention_mask: torch.Tensor | None


class _RecordingBlock(nn.Module):
    def __init__(self, calls: list[_AttentionCall]):
        super().__init__()
        self.calls = calls

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        temb=None,
    ):
        self.calls.append(_AttentionCall(encoder_hidden_states, encoder_attention_mask))
        return hidden_states


class _CaptureActionModel(nn.Module):
    def __init__(self, output_dim=8):
        super().__init__()
        self.config = SimpleNamespace(output_dim=output_dim)
        self.hidden_states = None

    def forward(self, hidden_states, **kwargs):
        self.hidden_states = hidden_states
        return torch.zeros(
            *hidden_states.shape[:2],
            self.config.output_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )


class _PassthroughActionEncoder(nn.Module):
    def forward(self, actions, timesteps):
        assert timesteps.shape == (actions.shape[0],)
        return actions


class _ConditionVelocityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(output_dim=2)

    def forward(self, hidden_states, *, encoder_hidden_states, **kwargs):
        condition_mean = encoder_hidden_states.mean(dim=(1, 2), keepdim=True)
        return torch.ones_like(hidden_states) * condition_mean


def _tiny_diagnostic_legacy_head():
    head = LegacyActionHead.__new__(LegacyActionHead)
    nn.Module.__init__(head)
    head.action_horizon = 3
    head.action_dim = 2
    head.num_inference_timesteps = 2
    head.num_timestep_buckets = 10
    head.config = SimpleNamespace(add_pos_embed=False)
    head.action_encoder = _PassthroughActionEncoder()
    head.action_decoder = nn.Identity()
    head.model = _ConditionVelocityModel()
    head.state_encoder = None
    head.future_tokens = None
    return head


class _CaptureStateEncoder(nn.Module):
    def __init__(self, output_dim=8):
        super().__init__()
        self.output_dim = output_dim
        self.state = None

    def forward(self, state, embodiment_id):
        self.state = state
        return torch.zeros(
            state.shape[0],
            1,
            self.output_dim,
            device=state.device,
            dtype=state.dtype,
        )


class _CountingRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.attention_mask = None

    def forward(self, hidden_states, attention_mask=None):
        self.calls += 1
        self.attention_mask = attention_mask
        return hidden_states


def _tiny_full_config(**overrides):
    action_config = {
        "action_dim": 5,
        "state_dim": 4,
        "state_history_length": 1,
        "action_horizon": 3,
        "input_embedding_dim": 8,
        "hidden_size": 8,
        "max_num_embodiments": 2,
        "max_seq_len": 3,
        "add_pos_embed": True,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "num_inference_timesteps": 2,
        "use_vlln": True,
        "backbone_embedding_dim": 6,
        "diffusion_model_cfg": {
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "output_dim": 8,
            "num_layers": 4,
            "dropout": 0.0,
            "final_dropout": False,
            "positional_embeddings": None,
            "cross_attention_dim": 6,
            "attend_text_every_n_blocks": 2,
        },
    }
    action_config.update(overrides)
    return SimpleNamespace(framework=SimpleNamespace(action_model=SimpleNamespace(**action_config)))


def _n16_batch(batch_size=2):
    return {
        "vl_embs": torch.randn(batch_size, 5, 6),
        "actions": torch.randn(batch_size, 3, 5),
        "state": torch.randn(batch_size, 4),
        "encoder_attention_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "image_mask": torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool).expand(batch_size, -1),
    }


def test_alternate_vldit_routes_text_self_image_self():
    model = AlternateVLDiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=4,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        cross_attention_dim=6,
        attend_text_every_n_blocks=2,
    )
    calls: list[_AttentionCall] = []
    model.transformer_blocks = nn.ModuleList([_RecordingBlock(calls) for _ in range(4)])

    hidden_states = torch.randn(2, 3, 8)
    vl_embs = torch.randn(2, 5, 6)
    valid_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    image_mask = torch.tensor([[1, 0, 0, 1, 0], [0, 1, 0, 0, 1]], dtype=torch.bool)

    output = model(
        hidden_states=hidden_states,
        encoder_hidden_states=vl_embs,
        timestep=torch.zeros(2, dtype=torch.long),
        encoder_attention_mask=valid_mask,
        image_mask=image_mask,
    )

    assert output.shape == hidden_states.shape
    assert len(calls) == 4
    assert calls[0].encoder_hidden_states is vl_embs
    assert torch.equal(calls[0].encoder_attention_mask, valid_mask & ~image_mask)
    assert calls[1].encoder_hidden_states is None
    assert calls[1].encoder_attention_mask is None
    assert calls[2].encoder_hidden_states is vl_embs
    assert torch.equal(calls[2].encoder_attention_mask, valid_mask & image_mask)
    assert calls[3].encoder_hidden_states is None
    assert calls[3].encoder_attention_mask is None


def test_n16_uses_one_state_plus_action_tokens_and_one_noise_time_per_item(monkeypatch):
    head = N16FlowmatchingActionHead(_tiny_full_config())
    capture_model = _CaptureActionModel()
    head.model = capture_model
    sample_calls = []

    def _sample_time(batch_size, device, dtype):
        sample_calls.append(batch_size)
        return torch.full((batch_size,), 0.5, device=device, dtype=dtype)

    monkeypatch.setattr(head, "sample_time", _sample_time)
    batch = _n16_batch()
    loss = head(**batch)

    assert loss.ndim == 0
    assert sample_calls == [2]
    assert capture_model.hidden_states.shape == (2, 4, 8)
    assert not hasattr(head, "future_tokens")


def test_n16_action_mask_excludes_padded_dimensions_from_loss(monkeypatch):
    head = N16FlowmatchingActionHead(_tiny_full_config())
    head.model = _CaptureActionModel()
    monkeypatch.setattr(head, "_sample_noise", torch.zeros_like)
    monkeypatch.setattr(
        head,
        "sample_time",
        lambda batch_size, device, dtype: torch.full((batch_size,), 0.5, device=device, dtype=dtype),
    )
    batch = _n16_batch(batch_size=1)
    batch["actions"] = torch.tensor([[[1.0, 2.0, 0.0, 0.0, 0.0]]]).expand(1, 3, -1).clone()
    action_mask = torch.tensor([[[1, 1, 0, 0, 0]]], dtype=torch.bool).expand_as(batch["actions"])

    base_loss = head(**batch, action_mask=action_mask)
    batch["actions"][..., 2:] = 1000.0
    padded_loss = head(**batch, action_mask=action_mask)

    assert torch.equal(base_loss, padded_loss)


def test_n16_requires_image_mask_and_valid_shapes():
    head = N16FlowmatchingActionHead(_tiny_full_config())
    batch = _n16_batch()

    with pytest.raises(ValueError, match="image_mask is required"):
        head(
            batch["vl_embs"],
            batch["actions"],
            state=batch["state"],
            encoder_attention_mask=batch["encoder_attention_mask"],
        )

    with pytest.raises(ValueError, match="action horizon"):
        head(**{**batch, "actions": batch["actions"][:, :2]})

    with pytest.raises(ValueError, match="action width"):
        head(**{**batch, "actions": batch["actions"][..., :4]})


def test_n16_predict_action_returns_configured_chunk_shape():
    head = N16FlowmatchingActionHead(_tiny_full_config())
    batch = _n16_batch()

    actions = head.predict_action(
        batch["vl_embs"],
        state=batch["state"],
        encoder_attention_mask=batch["encoder_attention_mask"],
        image_mask=batch["image_mask"],
    )

    assert actions.shape == (2, 3, 5)


def test_n17_flattens_state_history_and_refines_vlm_features(monkeypatch):
    config = _tiny_full_config(
        state_history_length=2,
        vl_self_attention_cfg={
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_layers": 2,
            "dropout": 0.0,
            "final_dropout": False,
            "positional_embeddings": None,
        },
    )
    head = N17FlowmatchingActionHead(config)
    capture_model = _CaptureActionModel()
    capture_state = _CaptureStateEncoder()
    refinement = _CountingRefinement()
    head.model = capture_model
    head.state_encoder = capture_state
    head.vl_self_attention = refinement
    monkeypatch.setattr(
        head,
        "sample_time",
        lambda batch_size, device, dtype: torch.full((batch_size,), 0.5, device=device, dtype=dtype),
    )
    batch = _n16_batch()
    batch["state"] = torch.randn(2, 2, 4)

    loss = head(**batch)

    assert loss.ndim == 0
    assert capture_state.state.shape == (2, 1, 8)
    assert capture_model.hidden_states.shape == (2, 4, 8)
    assert refinement.calls == 1
    assert refinement.attention_mask.shape == (2, 5)


def test_n17_can_disable_vlm_refinement_and_rejects_unimplemented_rtc():
    head = N17FlowmatchingActionHead(_tiny_full_config(use_vl_self_attention=False))
    assert head.vl_self_attention is None

    with pytest.raises(NotImplementedError, match="RTC"):
        N17FlowmatchingActionHead(_tiny_full_config(rtc_enabled=True))


def test_n16_accepts_official_padded_dimension_aliases():
    config = _tiny_full_config()
    del config.framework.action_model.action_dim
    del config.framework.action_model.state_dim
    config.framework.action_model.max_action_dim = 5
    config.framework.action_model.max_state_dim = 4

    head = N16FlowmatchingActionHead(config)

    assert head.action_dim == 5
    assert head.state_dim == 4


def test_legacy_dit_forward_still_does_not_require_image_mask():
    model = DiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        interleave_self_attention=True,
        cross_attention_dim=6,
    )

    output = model(
        hidden_states=torch.randn(2, 3, 8),
        encoder_hidden_states=torch.randn(2, 5, 6),
        timestep=torch.zeros(2, dtype=torch.long),
        encoder_attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )

    assert output.shape == (2, 3, 8)


def test_n16_rejects_conflicting_vlm_cross_attention_width():
    diffusion_cfg = dict(_tiny_full_config().framework.action_model.diffusion_model_cfg)
    diffusion_cfg["cross_attention_dim"] = 7

    with pytest.raises(ValueError, match="cross_attention_dim"):
        N16FlowmatchingActionHead(_tiny_full_config(diffusion_model_cfg=diffusion_cfg))


@pytest.mark.parametrize("invalid_value", [-1.0, 0.5, float("nan")])
def test_n16_action_mask_requires_finite_binary_values(invalid_value):
    head = N16FlowmatchingActionHead(_tiny_full_config())
    batch = _n16_batch(batch_size=1)
    action_mask = torch.ones_like(batch["actions"])
    action_mask[0, 0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite binary"):
        head(**batch, action_mask=action_mask)


def test_n17_vlm_refinement_ignores_padded_token_values():
    head = N17FlowmatchingActionHead(
        _tiny_full_config(
            vl_self_attention_cfg={
                "num_attention_heads": 2,
                "attention_head_dim": 4,
                "num_layers": 2,
                "dropout": 0.0,
                "final_dropout": False,
                "positional_embeddings": None,
            }
        )
    ).eval()
    valid_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    image_mask = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool)
    base_vlm = torch.randn(1, 5, 6)
    changed_padding_vlm = base_vlm.clone()
    changed_padding_vlm[:, 3:] = 1000.0

    base_features, _, _ = head._normalize_vlm_inputs(base_vlm, valid_mask, image_mask)
    changed_features, _, _ = head._normalize_vlm_inputs(changed_padding_vlm, valid_mask, image_mask)

    assert torch.allclose(base_features[:, :3], changed_features[:, :3], atol=1e-6, rtol=1e-5)


def test_robocasa_cot_v2_q0_config_changes_only_run_id_future_tokens_and_diagnostics():
    with _ROBOCASA_COT_V2_CONFIG.open() as source_file:
        source = yaml.safe_load(source_file)
    with _ROBOCASA_COT_V2_Q0_CONFIG.open() as q0_file:
        q0 = yaml.safe_load(q0_file)

    assert q0["run_id"] == "qwen35_gr00t_robocasa_fourier_CoT_v2_q0_8gpu_bs16"
    assert q0["framework"]["action_model"]["num_target_vision_tokens"] == 0
    assert q0["trainer"]["test_diagnostics"]["enabled"] is True

    normalized_q0 = deepcopy(q0)
    normalized_q0["run_id"] = source["run_id"]
    normalized_q0["framework"]["action_model"]["num_target_vision_tokens"] = source["framework"][
        "action_model"
    ]["num_target_vision_tokens"]
    normalized_q0["trainer"]["test_diagnostics"]["enabled"] = source["trainer"]["test_diagnostics"][
        "enabled"
    ]
    assert normalized_q0 == source


def test_legacy_action_head_zero_future_tokens_uses_only_action_slots(monkeypatch):
    config = _tiny_full_config(
        action_model_type="DiT-B",
        state_dim=0,
        num_target_vision_tokens=0,
    )
    head = LegacyActionHead(config)
    capture_model = _CaptureActionModel(output_dim=8)
    head.model = capture_model
    monkeypatch.setattr(
        head,
        "sample_time",
        lambda batch_size, device, dtype: torch.full((batch_size,), 0.5, device=device, dtype=dtype),
    )
    batch = _n16_batch()

    loss = head(
        batch["vl_embs"],
        batch["actions"],
        state=None,
        encoder_attention_mask=batch["encoder_attention_mask"],
    )

    assert loss.ndim == 0
    assert capture_model.hidden_states.shape == (2, 3, 768)


def test_legacy_action_head_zero_future_tokens_registers_no_empty_parameters():
    config = _tiny_full_config(
        action_model_type="DiT-B",
        state_dim=0,
        num_target_vision_tokens=0,
    )

    head = LegacyActionHead(config)

    assert all(parameter.numel() > 0 for parameter in head.parameters())


def test_legacy_action_head_zero_future_tokens_predicts_with_only_action_slots():
    config = _tiny_full_config(
        action_model_type="DiT-B",
        state_dim=0,
        num_target_vision_tokens=0,
    )
    head = LegacyActionHead(config)
    capture_model = _CaptureActionModel(output_dim=8)
    head.model = capture_model
    batch = _n16_batch()

    actions = head.predict_action(
        batch["vl_embs"],
        state=None,
        encoder_attention_mask=batch["encoder_attention_mask"],
    )

    assert actions.shape == batch["actions"].shape
    assert capture_model.hidden_states.shape == (2, 3, 768)


def test_legacy_action_head_reuses_supplied_initial_actions_without_mutating_them():
    head = _tiny_diagnostic_legacy_head()
    condition = torch.ones(2, 2, 2)
    initial = torch.full((2, 3, 2), 0.25)
    snapshot = initial.clone()

    first = head.predict_action(condition, initial_actions=initial)
    second = head.predict_action(condition, initial_actions=initial)

    assert torch.equal(first, second)
    assert torch.equal(initial, snapshot)
    assert torch.equal(first, torch.full_like(first, 1.25))


def test_legacy_action_head_different_initial_actions_change_output():
    head = _tiny_diagnostic_legacy_head()
    condition = torch.ones(1, 2, 2)

    zeros = head.predict_action(condition, initial_actions=torch.zeros(1, 3, 2))
    ones = head.predict_action(condition, initial_actions=torch.ones(1, 3, 2))

    assert not torch.equal(zeros, ones)


def test_legacy_action_head_returns_detached_flow_diagnostics():
    head = _tiny_diagnostic_legacy_head()

    actions, diagnostics = head.predict_action(
        torch.ones(1, 2, 2),
        initial_actions=torch.zeros(1, 3, 2),
        return_diagnostics=True,
    )

    assert torch.equal(actions, torch.ones_like(actions))
    assert len(diagnostics) == 2
    assert diagnostics[0].step_index == 0
    assert diagnostics[0].t_cont == 0.0
    assert diagnostics[0].t_discretized == 0
    assert torch.equal(diagnostics[0].x_before, torch.zeros(1, 3, 2))
    assert torch.equal(diagnostics[0].pred_velocity, torch.ones(1, 3, 2))
    assert torch.equal(diagnostics[0].x_after, torch.full((1, 3, 2), 0.5))
    assert torch.equal(diagnostics[1].x_before, diagnostics[0].x_after)
    assert all(
        not tensor.requires_grad
        for step in diagnostics
        for tensor in (step.x_before, step.pred_velocity, step.x_after)
    )


def test_legacy_action_head_condition_schedule_changes_only_selected_step():
    head = _tiny_diagnostic_legacy_head()
    correct = torch.ones(1, 2, 2)
    alternative = torch.full((1, 2, 2), 2.0)
    initial = torch.zeros(1, 3, 2)

    baseline = head.predict_action(correct, initial_actions=initial)
    intervened = head.predict_action(
        correct,
        initial_actions=initial,
        condition_schedule=((correct, None), (alternative, None)),
    )

    assert torch.equal(baseline, torch.ones_like(baseline))
    assert torch.equal(intervened, torch.full_like(intervened, 1.5))


def test_legacy_action_head_predict_velocity_uses_requested_state_and_condition():
    head = _tiny_diagnostic_legacy_head()

    velocity = head.predict_velocity(
        torch.zeros(1, 3, 2),
        t_cont=0.5,
        vl_embs=torch.full((1, 2, 3), 3.0),
    )

    assert torch.equal(velocity, torch.full_like(velocity, 3.0))


@pytest.mark.parametrize("bad_shape", [(1, 3, 2), (2, 2, 2), (2, 3, 1)])
def test_legacy_action_head_rejects_wrong_initial_action_shape(bad_shape):
    head = _tiny_diagnostic_legacy_head()

    with pytest.raises(ValueError, match="initial_actions shape"):
        head.predict_action(torch.ones(2, 2, 2), initial_actions=torch.zeros(bad_shape))


def test_legacy_action_head_rejects_invalid_diagnostic_inputs():
    head = _tiny_diagnostic_legacy_head()
    condition = torch.ones(1, 2, 2)

    with pytest.raises(ValueError, match="initial_actions dtype"):
        head.predict_action(condition, initial_actions=torch.zeros(1, 3, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        head.predict_action(condition, initial_actions=torch.full((1, 3, 2), float("nan")))
    with pytest.raises(ValueError, match="condition_schedule has 1 steps, expected 2"):
        head.predict_action(
            condition,
            initial_actions=torch.zeros(1, 3, 2),
            condition_schedule=((condition, None),),
        )
    wrong_hidden_width = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="hidden width"):
        head.predict_action(
            condition,
            initial_actions=torch.zeros(1, 3, 2),
            condition_schedule=(
                (wrong_hidden_width, None),
                (wrong_hidden_width, None),
            ),
        )
