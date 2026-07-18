
"""Group-causal geometric query reasoner used by QwenGR00TCoT."""

from __future__ import annotations

import math

import torch
from torch import nn


def build_group_causal_attention_mask(group_sizes: list[int], *, device=None) -> torch.Tensor:
    """Return an ``[Q,Q]`` boolean mask whose True entries are allowed reads."""
    if not group_sizes or any(int(size) <= 0 for size in group_sizes):
        raise ValueError(f"group_sizes must contain positive sizes, got {group_sizes}")
    total = sum(int(size) for size in group_sizes)
    allowed = torch.zeros(total, total, dtype=torch.bool, device=device)
    end = 0
    for size in group_sizes:
        start = end
        end += int(size)
        allowed[start:end, :end] = True
    return allowed


class _GeometricQueryLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        query_disallow_mask: torch.Tensor,
        memory_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q = self.self_norm(queries)
        self_out, _ = self.self_attn(q, q, q, attn_mask=query_disallow_mask)
        queries = queries + self_out

        q = self.cross_norm(queries)
        m = self.memory_norm(memory)
        cross_out, _ = self.cross_attn(q, m, m, key_padding_mask=memory_padding_mask)
        queries = queries + cross_out
        return queries + self.ffn(self.ffn_norm(queries))


class GeometricQueryReasoner(nn.Module):
    """Decode current depth, future depth, and temporal UVD query embeddings.

    The V/L tokens are first encoded by Qwen with its native mask. This module
    then applies the approved query-group information flow on the final V/L
    hidden states, avoiding changes to Qwen3.5's hybrid linear/full attention
    internals.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        depth_query_count: int = 8,
        layer_count: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = hidden_dim
        self.depth_query_count = int(depth_query_count)
        self.current_depth_queries = nn.Parameter(torch.randn(1, depth_query_count, hidden_dim) * 0.02)
        self.future_depth_queries = nn.Parameter(torch.randn(1, depth_query_count, hidden_dim) * 0.02)
        self.trajectory_seed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [_GeometricQueryLayer(hidden_dim, num_heads, dropout) for _ in range(layer_count)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        backbone_hidden: torch.Tensor,
        backbone_attention_mask: torch.Tensor | None,
        *,
        horizon: int,
        uvd_num_points: int | None = None,
        uvd_times: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = backbone_hidden.shape[0]
        if uvd_num_points is None:
            uvd_num_points = int(math.floor(0.3 * int(horizon))) + 2
        if uvd_num_points < 2:
            raise ValueError(f"uvd_num_points must be >=2, got {uvd_num_points}")
        output_dtype = backbone_hidden.dtype
        parameter_dtype = self.current_depth_queries.dtype
        memory = backbone_hidden.to(dtype=parameter_dtype)
        current = self.current_depth_queries.expand(batch_size, -1, -1)
        future = self.future_depth_queries.expand(batch_size, -1, -1)
        if uvd_times is None:
            tau = torch.linspace(0.0, 1.0, int(uvd_num_points), device=backbone_hidden.device, dtype=parameter_dtype)
            tau = tau.view(1, -1).expand(batch_size, -1)
        else:
            tau = uvd_times.to(device=backbone_hidden.device, dtype=parameter_dtype)
            if tau.ndim != 2 or tau.shape != (batch_size, int(uvd_num_points)):
                raise ValueError(
                    f"uvd_times must have shape {(batch_size, int(uvd_num_points))}, got {tuple(tau.shape)}"
                )
        trajectory = self.trajectory_seed.expand(batch_size, int(uvd_num_points), -1)
        trajectory = trajectory + self.time_embedding(tau.unsqueeze(-1))
        queries = torch.cat([current, future, trajectory], dim=1)

        allowed = build_group_causal_attention_mask(
            [self.depth_query_count, self.depth_query_count, int(uvd_num_points)],
            device=backbone_hidden.device,
        )
        # MultiheadAttention interprets True entries as blocked.
        disallow = ~allowed
        if backbone_attention_mask is None:
            memory_padding_mask = None
        else:
            memory_padding_mask = ~backbone_attention_mask.to(dtype=torch.bool)
        for layer in self.layers:
            queries = layer(queries, memory, disallow, memory_padding_mask)
        queries = self.output_norm(queries).to(dtype=output_dtype)
        current_end = self.depth_query_count
        future_end = current_end + self.depth_query_count
        return {
            "depth_current_tokens": queries[:, :current_end],
            "depth_future_tokens": queries[:, current_end:future_end],
            "uvd_tokens": queries[:, future_end:],
            "uvd_time": tau.to(dtype=output_dtype),
        }
