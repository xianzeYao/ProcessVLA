
"""Shared multi-stage ConvStack + FiLM metric-depth decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SharedFiLMConvStack(nn.Module):
    """Decode a raster image-token grid with query-conditioned FiLM stages."""

    def __init__(self, hidden_dim: int, features: int = 256, stage_count: int = 3) -> None:
        super().__init__()
        if stage_count < 1:
            raise ValueError(f"stage_count must be positive, got {stage_count}")
        self.hidden_dim = hidden_dim
        self.features = features
        self.in_projection = nn.Conv2d(hidden_dim, features, kernel_size=1)
        self.blocks = nn.ModuleList([_ResidualConvBlock(features) for _ in range(stage_count)])
        self.film = nn.Sequential(
            nn.Linear(hidden_dim, features * 2),
        )
        nn.init.zeros_(self.film[0].weight)
        nn.init.zeros_(self.film[0].bias)
        self.output = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features // 2, 1, kernel_size=1),
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        *,
        patch_hw: tuple[int, int],
        query: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, token_count, hidden_dim = image_tokens.shape
        patch_h, patch_w = map(int, patch_hw)
        if patch_h * patch_w != token_count:
            raise ValueError(f"patch_hw={patch_hw} does not match token_count={token_count}")
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"hidden dim mismatch: got {hidden_dim}, expected {self.hidden_dim}")
        compute_dtype = self.in_projection.weight.dtype
        x = image_tokens.to(dtype=compute_dtype).transpose(1, 2).reshape(batch_size, hidden_dim, patch_h, patch_w)
        x = self.in_projection(x)
        gamma, beta = self.film(query.to(dtype=compute_dtype)).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        for block in self.blocks:
            x = (1.0 + gamma) * x + beta
            x = block(x)
            if x.shape[-2] < output_hw[0] or x.shape[-1] < output_hw[1]:
                x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=output_hw, mode="bilinear", align_corners=False)
        return F.softplus(self.output(x)) + 1e-4
