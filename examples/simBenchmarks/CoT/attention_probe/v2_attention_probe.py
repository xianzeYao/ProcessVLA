"""Read-only attention collection helpers for QwenGR00TCoTV2.

The collectors attach temporary forward pre-hooks and recompute only the
requested QK softmax rows.  They never replace the model's SDPA processors, so
the action and geometry outputs still come from the checkpoint's normal path.
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def contiguous_token_runs(token_ids: torch.Tensor, token_id: int) -> list[torch.Tensor]:
    """Return contiguous 1-D index runs matching ``token_id``."""

    if token_ids.ndim != 1:
        raise ValueError(f"token_ids must be 1-D, got {tuple(token_ids.shape)}")
    positions = torch.nonzero(token_ids == int(token_id), as_tuple=False).flatten()
    if positions.numel() == 0:
        return []
    boundaries = torch.nonzero(positions[1:] != positions[:-1] + 1, as_tuple=False).flatten() + 1
    return list(torch.tensor_split(positions, boundaries.cpu().tolist()))


def infer_patch_grid(token_count: int) -> tuple[int, int]:
    """Factor a visual-token count into the closest landscape/square grid."""

    token_count = int(token_count)
    if token_count < 1:
        raise ValueError(f"token_count must be positive, got {token_count}")
    height = int(math.sqrt(token_count))
    while token_count % height:
        height -= 1
    return height, token_count // height


def build_condition_groups(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    native_attention_mask: torch.Tensor | None,
    depth_query_count: int,
    uvd_token_count: int,
    include_depth: bool,
) -> dict[str, np.ndarray]:
    """Map one V2 action-condition sequence to named key-index groups."""

    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1-D, got {tuple(input_ids.shape)}")
    native_count = int(input_ids.numel())
    if native_attention_mask is None:
        valid_native = torch.ones(native_count, dtype=torch.bool)
    else:
        valid_native = torch.as_tensor(native_attention_mask, dtype=torch.bool).cpu()
        if tuple(valid_native.shape) != (native_count,):
            raise ValueError(
                f"native_attention_mask must have shape {(native_count,)}, got {tuple(valid_native.shape)}"
            )
    runs = contiguous_token_runs(input_ids.cpu(), image_token_id)
    view_names = ("agentview", "wrist")
    groups: dict[str, np.ndarray] = {}
    image_positions = torch.zeros(native_count, dtype=torch.bool)
    for index, run in enumerate(runs):
        name = view_names[index] if index < len(view_names) else f"image_{index}"
        groups[name] = run.cpu().numpy().astype(np.int64, copy=False)
        image_positions[run.cpu()] = True
    groups["language"] = torch.nonzero(
        valid_native & ~image_positions, as_tuple=False
    ).flatten().numpy()

    cursor = native_count
    if include_depth:
        groups["depth_current"] = np.arange(cursor, cursor + int(depth_query_count))
        cursor += int(depth_query_count)
        groups["depth_future"] = np.arange(cursor, cursor + int(depth_query_count))
        cursor += int(depth_query_count)
    groups["uvd"] = np.arange(cursor, cursor + int(uvd_token_count))
    return groups


def attention_to_patch_map(
    attention: torch.Tensor | np.ndarray,
    *,
    key_indices: Sequence[int] | np.ndarray | torch.Tensor,
    patch_grid: tuple[int, int] | None = None,
) -> np.ndarray:
    """Average selected queries, gather visual keys, and max-normalize a patch map."""

    values = torch.as_tensor(attention).float()
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"attention must resolve to [queries, keys], got {values.shape}")
    indices = torch.as_tensor(key_indices, dtype=torch.long)
    patches = values.mean(dim=0).index_select(0, indices).cpu().numpy()
    grid = infer_patch_grid(len(patches)) if patch_grid is None else tuple(map(int, patch_grid))
    if grid[0] * grid[1] != len(patches):
        raise ValueError(f"patch_grid {grid} does not contain {len(patches)} values")
    maximum = float(np.max(patches)) if patches.size else 0.0
    if maximum > 0.0:
        patches = patches / maximum
    return patches.reshape(grid)


def _selected_mask_rows(
    attention_mask: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    query_count: int,
) -> torch.Tensor:
    mask = attention_mask
    if mask.ndim == 2:
        mask = mask[:, None, None, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    if mask.ndim != 4:
        raise ValueError(f"attention_mask must have 2-4 dimensions, got {mask.ndim}")
    if mask.shape[-2] == query_count:
        mask = mask.index_select(-2, query_indices.to(mask.device))
    elif mask.shape[-2] != 1:
        raise ValueError(
            f"attention_mask query axis must be 1 or {query_count}, got {mask.shape[-2]}"
        )
    return mask


def selected_attention_probabilities(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    query_indices: torch.Tensor | Sequence[int] | None = None,
    attention_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute selected attention rows from projected ``[B,H,S,D]`` Q/K."""

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must both have shape [B,H,S,D]")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(f"incompatible query/key shapes: {query.shape}, {key.shape}")
    if query_indices is None:
        indices = torch.arange(query.shape[-2], device=query.device)
    else:
        indices = torch.as_tensor(query_indices, device=query.device, dtype=torch.long)
    selected_query = query.index_select(-2, indices)
    scores = torch.matmul(selected_query.float(), key.float().transpose(-2, -1))
    scores.mul_(float(scale) if scale is not None else query.shape[-1] ** -0.5)
    if attention_mask is not None:
        mask = _selected_mask_rows(attention_mask, indices, query_count=query.shape[-2])
        mask = mask.to(device=scores.device)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask.float()
    return torch.softmax(scores, dim=-1)


def aggregate_attention_records(records: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack head-averaged records and retain their layer/call identities."""

    if not records:
        raise ValueError("at least one attention record is required")
    attentions = [torch.as_tensor(record["attention"]).float().cpu() for record in records]
    shape = tuple(attentions[0].shape)
    if any(tuple(value.shape) != shape for value in attentions):
        raise ValueError("all attention records must share shape")
    stacked = torch.stack(attentions)
    return {
        "records": stacked,
        "mean": stacked.mean(dim=0),
        "layers": torch.tensor([int(record["layer"]) for record in records]),
        "calls": torch.tensor([int(record.get("call", 0)) for record in records]),
    }


def modality_attention_mass(
    attention: torch.Tensor | np.ndarray,
    groups: Mapping[str, Sequence[int] | np.ndarray | torch.Tensor],
) -> dict[str, float]:
    """Sum mean-query attention by key group and normalize over supplied groups."""

    values = torch.as_tensor(attention).float()
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"attention must resolve to [queries, keys], got {values.shape}")
    key_mass = values.mean(dim=0)
    result: dict[str, float] = {}
    for name, indices in groups.items():
        index = torch.as_tensor(indices, dtype=torch.long)
        result[str(name)] = float(key_mass.index_select(0, index).sum()) if index.numel() else 0.0
    total = sum(result.values())
    if total <= 0.0:
        return {name: 0.0 for name in result}
    return {name: value / total for name, value in result.items()}


def _kwarg_or_arg(args: tuple[Any, ...], kwargs: Mapping[str, Any], name: str, position: int):
    if name in kwargs:
        return kwargs[name]
    return args[position] if len(args) > position else None


@dataclass
class QwenFullAttentionCollector(AbstractContextManager):
    """Collect selected geometry-query rows from every Qwen full-attention layer."""

    language_model: torch.nn.Module
    query_indices: torch.Tensor | Sequence[int]
    records: list[dict[str, Any]] = field(default_factory=list, init=False)
    _handles: list[Any] = field(default_factory=list, init=False)

    def __enter__(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

        requested = torch.as_tensor(self.query_indices, dtype=torch.long)
        for layer_index, layer in enumerate(self.language_model.layers):
            if getattr(layer, "layer_type", None) != "full_attention":
                continue
            module = layer.self_attn

            def capture(attn, args, kwargs, *, index=layer_index):
                hidden = _kwarg_or_arg(args, kwargs, "hidden_states", 0)
                position_embeddings = kwargs["position_embeddings"]
                input_shape = hidden.shape[:-1]
                hidden_shape = (*input_shape, -1, attn.head_dim)
                query, _ = torch.chunk(
                    attn.q_proj(hidden).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1
                )
                query = attn.q_norm(query.view(hidden_shape)).transpose(1, 2)
                key = attn.k_norm(attn.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
                query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
                key = repeat_kv(key, attn.num_key_value_groups)
                probs = selected_attention_probabilities(
                    query,
                    key,
                    query_indices=requested,
                    attention_mask=kwargs.get("attention_mask"),
                    scale=attn.scaling,
                )
                self.records.append(
                    {
                        "layer": index,
                        "call": 0,
                        "attention": probs.mean(dim=1)[0].detach().float().cpu(),
                    }
                )

            self._handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False


@dataclass
class DiTCrossAttentionCollector(AbstractContextManager):
    """Collect action-query to condition attention from DiT cross-attention blocks."""

    dit: torch.nn.Module
    records: list[dict[str, Any]] = field(default_factory=list, init=False)
    _handles: list[Any] = field(default_factory=list, init=False)
    _calls_by_layer: dict[int, int] = field(default_factory=dict, init=False)

    def __enter__(self):
        interleaved = bool(getattr(self.dit.config, "interleave_self_attention", False))
        canonical = bool(getattr(self.dit.config, "use_canonical_forward", True))
        for layer_index, block in enumerate(self.dit.transformer_blocks):
            is_self = layer_index % 2 == 1 and interleaved and canonical
            if is_self:
                continue
            module = block.attn1

            def capture(attn, args, kwargs, *, index=layer_index):
                hidden = _kwarg_or_arg(args, kwargs, "hidden_states", 0)
                encoder = kwargs.get("encoder_hidden_states")
                if encoder is None:
                    return
                batch_size = hidden.shape[0]
                if attn.group_norm is not None:
                    hidden = attn.group_norm(hidden.transpose(1, 2)).transpose(1, 2)
                query = attn.to_q(hidden)
                if attn.norm_cross:
                    encoder = attn.norm_encoder_hidden_states(encoder)
                key = attn.to_k(encoder)
                head_dim = key.shape[-1] // attn.heads
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)
                mask = kwargs.get("attention_mask")
                if mask is not None:
                    mask = attn.prepare_attention_mask(mask, encoder.shape[1], batch_size)
                    mask = mask.view(batch_size, attn.heads, -1, mask.shape[-1])
                probs = selected_attention_probabilities(
                    query,
                    key,
                    attention_mask=mask,
                    scale=getattr(attn, "scale", head_dim**-0.5),
                )
                call = self._calls_by_layer.get(index, 0)
                self._calls_by_layer[index] = call + 1
                self.records.append(
                    {
                        "layer": index,
                        "call": call,
                        "attention": probs.mean(dim=1)[0].detach().float().cpu(),
                    }
                )

            self._handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False
