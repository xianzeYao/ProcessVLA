"""Cache-free Qwen3.5 multimodal forward with geometry-aware hybrid masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class GeometryQwenOutput:
    """Final hidden states and the mRoPE positions used to produce them."""

    last_hidden_state: torch.Tensor
    position_ids: torch.Tensor


def replace_geometry_tail_embeddings(
    input_embeddings: torch.Tensor,
    geometry_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Replace fixed tail placeholders without mutating the embedding output in-place."""

    if input_embeddings.ndim != 3 or geometry_embeddings.ndim != 3:
        raise ValueError(
            "input_embeddings and geometry_embeddings must both have shape [B,S,D], "
            f"got {tuple(input_embeddings.shape)} and {tuple(geometry_embeddings.shape)}"
        )
    if input_embeddings.shape[0] != geometry_embeddings.shape[0]:
        raise ValueError("input_embeddings and geometry_embeddings must share batch size")
    if input_embeddings.shape[2] != geometry_embeddings.shape[2]:
        raise ValueError("input_embeddings and geometry_embeddings must share hidden size")
    geometry_count = int(geometry_embeddings.shape[1])
    if geometry_count < 1 or geometry_count > int(input_embeddings.shape[1]):
        raise ValueError(
            f"geometry token count must be in [1,{input_embeddings.shape[1]}], got {geometry_count}"
        )
    output = input_embeddings.clone()
    output[:, -geometry_count:, :] = geometry_embeddings.to(
        device=output.device,
        dtype=output.dtype,
    )
    return output


def run_qwen35_hybrid_text(
    language_model: torch.nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor | None,
    linear_attention_mask: torch.Tensor | None,
    full_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run Qwen3.5 text layers with separate masks and no generation cache."""

    if inputs_embeds.ndim != 3:
        raise ValueError(f"inputs_embeds must have shape [B,S,D], got {tuple(inputs_embeds.shape)}")
    batch_size, sequence_length = inputs_embeds.shape[:2]
    if linear_attention_mask is not None and tuple(linear_attention_mask.shape) != (batch_size, sequence_length):
        raise ValueError(
            f"linear_attention_mask must have shape {(batch_size, sequence_length)}, "
            f"got {tuple(linear_attention_mask.shape)}"
        )
    expected_full_shape = (batch_size, 1, sequence_length, sequence_length)
    if tuple(full_attention_mask.shape) != expected_full_shape:
        raise ValueError(
            f"full_attention_mask must have shape {expected_full_shape}, got {tuple(full_attention_mask.shape)}"
        )
    backend = getattr(language_model.config, "_attn_implementation", None)
    if backend != "sdpa":
        raise ValueError(
            "geometry boolean allowed-read masking requires the Qwen text backend to be 'sdpa', "
            f"got {backend!r}"
        )

    cache_position = torch.arange(sequence_length, device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(4, batch_size, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
    if position_ids.ndim != 3 or position_ids.shape[1:] != (batch_size, sequence_length):
        raise ValueError(
            "position_ids must have shape [3|4,B,S] or [B,S], "
            f"got {tuple(position_ids.shape)}"
        )
    rotary_position_ids = position_ids[1:] if position_ids.shape[0] == 4 else position_ids

    hidden_states = inputs_embeds
    position_embeddings = language_model.rotary_emb(hidden_states, rotary_position_ids)
    layer_count = int(language_model.config.num_hidden_layers)
    for decoder_layer in language_model.layers[:layer_count]:
        if decoder_layer.layer_type == "linear_attention":
            layer_mask = linear_attention_mask
        elif decoder_layer.layer_type == "full_attention":
            layer_mask = full_attention_mask
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type: {decoder_layer.layer_type!r}")
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=rotary_position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
        )
    return language_model.norm(hidden_states)


def _scatter_image_features(core_model: torch.nn.Module, qwen_inputs: Mapping[str, Any], inputs_embeds: torch.Tensor):
    pixel_values = qwen_inputs.get("pixel_values")
    if pixel_values is None:
        return inputs_embeds
    image_grid_thw = qwen_inputs.get("image_grid_thw")
    image_outputs = core_model.get_image_features(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        return_dict=True,
    )
    image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    image_mask, _ = core_model.get_placeholder_mask(
        qwen_inputs["input_ids"],
        inputs_embeds=inputs_embeds,
        image_features=image_embeds,
    )
    return inputs_embeds.masked_scatter(image_mask, image_embeds)


def _scatter_video_features(core_model: torch.nn.Module, qwen_inputs: Mapping[str, Any], inputs_embeds: torch.Tensor):
    pixel_values_videos = qwen_inputs.get("pixel_values_videos")
    if pixel_values_videos is None:
        return inputs_embeds
    video_grid_thw = qwen_inputs.get("video_grid_thw")
    video_outputs = core_model.get_video_features(
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        return_dict=True,
    )
    video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    _, video_mask = core_model.get_placeholder_mask(
        qwen_inputs["input_ids"],
        inputs_embeds=inputs_embeds,
        video_features=video_embeds,
    )
    return inputs_embeds.masked_scatter(video_mask, video_embeds)


def forward_qwen35_with_geometry(
    qwen_model: torch.nn.Module,
    *,
    qwen_inputs: Mapping[str, Any],
    geometry_embeddings: torch.Tensor,
    full_attention_mask: torch.Tensor,
) -> GeometryQwenOutput:
    """Run one cache-free multimodal Qwen3.5 forward with appended geometry tokens."""

    if "input_ids" not in qwen_inputs or "attention_mask" not in qwen_inputs:
        raise KeyError("qwen_inputs must contain input_ids and attention_mask")
    core_model = qwen_model.model
    input_ids = qwen_inputs["input_ids"]
    attention_mask = qwen_inputs["attention_mask"]
    if input_ids.shape != attention_mask.shape:
        raise ValueError(f"input_ids and attention_mask shapes differ: {input_ids.shape}, {attention_mask.shape}")

    inputs_embeds = core_model.get_input_embeddings()(input_ids)
    inputs_embeds = replace_geometry_tail_embeddings(inputs_embeds, geometry_embeddings)
    inputs_embeds = _scatter_image_features(core_model, qwen_inputs, inputs_embeds)
    inputs_embeds = _scatter_video_features(core_model, qwen_inputs, inputs_embeds)

    position_ids = core_model.compute_3d_position_ids(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=qwen_inputs.get("image_grid_thw"),
        video_grid_thw=qwen_inputs.get("video_grid_thw"),
        attention_mask=attention_mask,
        past_key_values=None,
        mm_token_type_ids=qwen_inputs.get("mm_token_type_ids"),
    )
    hidden_states = run_qwen35_hybrid_text(
        core_model.language_model,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        linear_attention_mask=attention_mask,
        full_attention_mask=full_attention_mask,
    )
    if position_ids is None:
        sequence_length = input_ids.shape[1]
        position_ids = torch.arange(sequence_length, device=input_ids.device).view(1, 1, -1)
        position_ids = position_ids.expand(3, input_ids.shape[0], -1)
    return GeometryQwenOutput(last_hidden_state=hidden_states, position_ids=position_ids)
