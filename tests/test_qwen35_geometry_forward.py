from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.model.modules.qwen35_geometry_forward import (
    forward_qwen35_with_geometry,
    replace_geometry_tail_embeddings,
    run_qwen35_hybrid_text,
)


class RecordingLayer(nn.Module):
    def __init__(self, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.seen_mask = None

    def forward(self, hidden_states, *, attention_mask, **kwargs):
        self.seen_mask = attention_mask
        return hidden_states + 1.0


class FakeRotary(nn.Module):
    def forward(self, hidden_states, position_ids):
        shape = (*hidden_states.shape[:-1], 1)
        return torch.zeros(shape), torch.zeros(shape)


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [RecordingLayer("linear_attention"), RecordingLayer("full_attention")]
        )
        self.norm = nn.Identity()
        self.rotary_emb = FakeRotary()
        self.config = SimpleNamespace(num_hidden_layers=2, _attn_implementation="sdpa")


class FakeCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = FakeLanguageModel()
        self.embedding = nn.Embedding(32, 4)
        self.visual = SimpleNamespace(dtype=torch.float32)
        self.rope_deltas = None

    def get_input_embeddings(self):
        return self.embedding

    def compute_3d_position_ids(self, **kwargs):
        input_ids = kwargs["input_ids"]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1)


def test_replace_geometry_tail_embeddings_preserves_prefix_and_routes_gradients():
    inputs = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4).requires_grad_()
    geometry = torch.full((1, 2, 4), 99.0, requires_grad=True)

    output = replace_geometry_tail_embeddings(inputs, geometry)

    assert output[:, :4].tolist() == inputs[:, :4].tolist()
    assert output[:, 4:].tolist() == geometry.tolist()
    output.sum().backward()
    assert geometry.grad.tolist() == torch.ones_like(geometry).tolist()
    assert inputs.grad[:, 4:].tolist() == torch.zeros_like(inputs[:, 4:]).tolist()


def test_hybrid_text_forward_routes_2d_mask_to_linear_and_4d_mask_to_full():
    language_model = FakeLanguageModel()
    hidden = torch.zeros(1, 5, 4)
    position_ids = torch.arange(5).view(1, 1, 5).expand(3, 1, 5)
    linear_mask = torch.tensor([[0, 1, 1, 1, 1]])
    full_mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)

    output = run_qwen35_hybrid_text(
        language_model,
        inputs_embeds=hidden,
        position_ids=position_ids,
        linear_attention_mask=linear_mask,
        full_attention_mask=full_mask,
    )

    assert torch.equal(language_model.layers[0].seen_mask, linear_mask)
    assert torch.equal(language_model.layers[1].seen_mask, full_mask)
    assert output.tolist() == torch.full_like(hidden, 2.0).tolist()


def test_forward_qwen35_with_geometry_uses_appended_positions_and_returns_all_tokens():
    raw_model = SimpleNamespace(model=FakeCore())
    qwen_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(1, 5, dtype=torch.long),
    }
    geometry = torch.randn(1, 2, 4, requires_grad=True)
    full_mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)

    output = forward_qwen35_with_geometry(
        raw_model,
        qwen_inputs=qwen_inputs,
        geometry_embeddings=geometry,
        full_attention_mask=full_mask,
    )

    assert output.last_hidden_state.shape == (1, 5, 4)
    assert output.position_ids.shape == (3, 1, 5)
    output.last_hidden_state[:, -2:].sum().backward()
    assert geometry.grad is not None


def test_hybrid_forward_rejects_eager_backend_for_boolean_allowed_read_mask():
    language_model = FakeLanguageModel()
    language_model.config._attn_implementation = "eager"

    with pytest.raises(ValueError, match="requires the Qwen text backend to be 'sdpa'"):
        run_qwen35_hybrid_text(
            language_model,
            inputs_embeds=torch.randn(1, 3, 4),
            position_ids=None,
            linear_attention_mask=torch.ones(1, 3, dtype=torch.bool),
            full_attention_mask=torch.ones(1, 1, 3, 3, dtype=torch.bool),
        )
