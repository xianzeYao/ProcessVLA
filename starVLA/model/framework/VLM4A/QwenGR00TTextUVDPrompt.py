"""Prompt-serialized text/UVD ablation wrapper.

This is an additive wrapper around the existing text/UVD scaffold.  It keeps
the baseline QwenGR00T action path, but serializes both optional conditions
into the same user instruction before Qwen's normal processor/tokenizer:

    original instruction -> embodied reasoning -> ordered UVD trace

There is deliberately no extra UVD adapter or appended learned token in this
V1 implementation.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenGR00TTextUVD import Qwen_GR00T_TextUVD
from starVLA.model.tools import FRAMEWORK_REGISTRY


def normalize_condition_mask(
    attention_mask: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    """Convert Qwen's integer mask to the bool mask expected by SDPA."""

    if attention_mask is None:
        return None
    expected_shape = hidden_states.shape[:2]
    if tuple(attention_mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"attention_mask must have shape {tuple(expected_shape)}, "
            f"got {tuple(attention_mask.shape)}"
        )
    return attention_mask.to(device=hidden_states.device, dtype=torch.bool)


def _strip_think_tags(text: str) -> str:
    """Remove outer generation-control tags while preserving their content."""

    text = str(text).strip()
    if text.startswith("<think>"):
        text = text[len("<think>"):]
    if text.endswith("</think>"):
        text = text[:-len("</think>")]
    return text.strip()


def compose_instruction_with_cot(instruction: str, cot_text: str | None) -> str:
    """Append decoded embodied reasoning as plain text."""

    instruction = str(instruction)
    if cot_text is None or not str(cot_text).strip():
        return instruction
    return f"{instruction}\n\nEmbodied reasoning:\n{_strip_think_tags(cot_text)}"


def compose_instruction_with_uvd(
    instruction: str,
    uvd: np.ndarray,
    uvd_valid_mask: np.ndarray | None = None,
) -> str:
    """Append a fixed-K ordered UVD trace as ordinary instruction text.

    U and V are normalized to [0, 1].  Depth is camera-z depth in meters.
    Padded rows stay present to preserve fixed K, but are marked invalid.
    """

    points = np.asarray(uvd, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"uvd must have shape [K,3], got {points.shape}")

    if uvd_valid_mask is None:
        valid = np.ones(points.shape[0], dtype=np.bool_)
    else:
        valid = np.asarray(uvd_valid_mask, dtype=np.bool_)
        if valid.shape != (points.shape[0],):
            raise ValueError(
                f"uvd_valid_mask must have shape [{points.shape[0]}], got {valid.shape}"
            )

    rows = []
    for index, ((u, v, depth), is_valid) in enumerate(zip(points, valid)):
        rows.append(
            f"point_{index}: u={u:.3f}, v={v:.3f}, depth={depth:.3f}m, "
            f"valid={'true' if bool(is_valid) else 'false'}"
        )
    geometry = (
        "Embodied geometry (EEF UVD trace; u,v normalized to [0,1], "
        "depth in meters):\n" + "\n".join(rows)
    )
    return f"{instruction}\n\n{geometry}"


def compose_conditioned_instruction(
    instruction: str,
    cot_text: str | None = None,
    uvd: np.ndarray | None = None,
    uvd_valid_mask: np.ndarray | None = None,
    *,
    use_text: bool = False,
    use_uvd: bool = False,
) -> str:
    """Build one of the four prompts with a stable information order."""

    result = str(instruction)
    if use_text:
        result = compose_instruction_with_cot(result, cot_text)
    if use_uvd:
        if uvd is None:
            raise ValueError("uvd is required when use_uvd=True")
        result = compose_instruction_with_uvd(result, uvd, uvd_valid_mask)
    return result


@FRAMEWORK_REGISTRY.register("QwenGR00TTextUVDPrompt")
class Qwen_GR00T_TextUVDPrompt(Qwen_GR00T_TextUVD):
    """QwenGR00T with prompt-only text/UVD condition ablations."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        # The parent is an existing scaffold that used an adapter-token
        # design. Remove those modules so they cannot add parameters, consume
        # optimizer budget, or accidentally affect the new condition path.
        for name in (
            "uvd_condition_adapter",
            "uvd_position_embedding",
            "uvd_token_count",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _instructions(self, examples: List[dict]) -> list[str]:
        return [
            compose_conditioned_instruction(
                example["lang"],
                cot_text=example.get("cot_text") if self.use_text else None,
                uvd=example.get("uvd") if self.use_uvd else None,
                uvd_valid_mask=example.get("uvd_valid_mask") if self.use_uvd else None,
                use_text=self.use_text,
                use_uvd=self.use_uvd,
            )
            for example in examples
        ]

    def _build_condition(
        self,
        examples: List[dict],
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Text and UVD are already represented by Qwen hidden states. Keep
        # exactly the same [B,L,H] condition and [B,L] mask interface as the
        # baseline action expert.
        return last_hidden, normalize_condition_mask(attention_mask, last_hidden)
