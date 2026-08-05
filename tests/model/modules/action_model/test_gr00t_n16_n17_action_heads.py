from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import AlternateVLDiT, DiT
from starVLA.model.modules.action_model.GR00T_N16_ActionHeader import N16FlowmatchingActionHead
from starVLA.model.modules.action_model.GR00T_N17_ActionHeader import N17FlowmatchingActionHead


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
