"""Standalone QwenGR00T text/UVD condition ablation wrapper.

The existing QwenGR00T and geometric-CoT frameworks are intentionally left
untouched.  This wrapper keeps the same flow-matching action head and adds
only two optional inputs:

* decoded per-frame embodied text, appended to the Qwen user prompt;
* a fixed-K current-to-near-future UVD trace projected to action-condition
  tokens by a small trainable adapter.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List

import numpy as np
import torch
from torch import nn

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def compose_instruction_with_cot(instruction: str, cot_text: str | None) -> str:
    """Compose the original task and optional decoded embodied reasoning."""

    instruction = str(instruction)
    if cot_text is None or not str(cot_text).strip():
        return instruction
    return f"{instruction}\n\nEmbodied reasoning:\n{str(cot_text).strip()}"


def append_condition_tokens(
    base_condition: torch.Tensor,
    base_mask: torch.Tensor | None,
    extra_tokens: torch.Tensor,
    extra_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Append condition tokens and preserve an explicit validity mask."""

    if base_condition.ndim != 3 or extra_tokens.ndim != 3:
        raise ValueError(
            f"condition tensors must be [B,L,H], got {tuple(base_condition.shape)} and {tuple(extra_tokens.shape)}"
        )
    if base_condition.shape[0] != extra_tokens.shape[0] or base_condition.shape[2] != extra_tokens.shape[2]:
        raise ValueError("base and extra condition batch/hidden dimensions must match")
    if extra_valid.shape != extra_tokens.shape[:2]:
        raise ValueError(
            f"extra_valid must be [B,K]={tuple(extra_tokens.shape[:2])}, got {tuple(extra_valid.shape)}"
        )
    if base_mask is None:
        base_mask = torch.ones(
            base_condition.shape[:2], dtype=torch.bool, device=base_condition.device
        )
    else:
        base_mask = base_mask.to(device=base_condition.device, dtype=torch.bool)
    extra_valid = extra_valid.to(device=base_condition.device, dtype=torch.bool)
    return torch.cat([base_condition, extra_tokens], dim=1), torch.cat([base_mask, extra_valid], dim=1)


@FRAMEWORK_REGISTRY.register("QwenGR00TTextUVD")
class Qwen_GR00T_TextUVD(Qwen_GR00T):
    """QwenGR00T with independent semantic-text and direct-UVD conditions."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.condition_mode = str(self.config.framework.get("condition_mode", "base")).lower()
        if self.condition_mode not in {"base", "text", "uvd", "both"}:
            raise ValueError(f"Unknown condition_mode={self.condition_mode}; expected base/text/uvd/both")
        self.use_text = self.condition_mode in {"text", "both"}
        self.use_uvd = self.condition_mode in {"uvd", "both"}
        self.freeze_qwen = bool(self.config.framework.get("freeze_qwen", True))
        self.uvd_token_count = int(self.config.framework.get("uvd_token_count", 4))
        if self.uvd_token_count < 1:
            raise ValueError(f"uvd_token_count must be positive, got {self.uvd_token_count}")

        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        if self.use_uvd:
            self.uvd_condition_adapter = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.uvd_position_embedding = nn.Parameter(torch.zeros(1, self.uvd_token_count, hidden_dim))
            nn.init.normal_(self.uvd_position_embedding, mean=0.0, std=0.02)

        if self.freeze_qwen:
            self.qwen_vl_interface.eval()
            for parameter in self.qwen_vl_interface.parameters():
                parameter.requires_grad = False

    def _instructions(self, examples: List[dict]) -> list[str]:
        return [
            compose_instruction_with_cot(
                example["lang"],
                example.get("cot_text") if self.use_text else None,
            )
            for example in examples
        ]

    def _run_backbone(self, examples: List[dict]):
        batch_images = [example["image"] for example in examples]
        instructions = self._instructions(examples)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        attention_mask = qwen_inputs.get("attention_mask")
        if self.freeze_qwen:
            self.qwen_vl_interface.eval()
        context = torch.no_grad() if self.freeze_qwen else nullcontext()
        with context:
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        return qwen_inputs, outputs.hidden_states[-1], attention_mask

    def _build_condition(
        self,
        examples: List[dict],
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_uvd:
            return last_hidden, attention_mask

        uvd = torch.as_tensor(
            np.stack([np.asarray(example["uvd"], dtype=np.float32) for example in examples]),
            device=last_hidden.device,
        )
        uvd_time = torch.as_tensor(
            np.stack([np.asarray(example["uvd_time"], dtype=np.float32) for example in examples]),
            device=last_hidden.device,
        )
        uvd_valid = torch.as_tensor(
            np.stack([np.asarray(example["uvd_valid_mask"], dtype=np.bool_) for example in examples]),
            device=last_hidden.device,
        )
        if uvd.shape[1] != self.uvd_token_count:
            raise ValueError(
                f"UVD trace must have K={self.uvd_token_count} points, got {uvd.shape[1]}; "
                "the dataloader must pad short traces before batching"
            )
        adapter_dtype = next(self.uvd_condition_adapter.parameters()).dtype
        adapter_input = torch.cat([uvd, uvd_time.unsqueeze(-1)], dim=-1).to(dtype=adapter_dtype)
        uvd_tokens = self.uvd_condition_adapter(adapter_input)
        uvd_tokens = uvd_tokens + self.uvd_position_embedding.to(dtype=uvd_tokens.dtype)
        uvd_tokens = uvd_tokens.to(dtype=last_hidden.dtype)
        return append_condition_tokens(last_hidden, attention_mask, uvd_tokens, uvd_valid)

    def _action_loss(
        self,
        condition: torch.Tensor,
        condition_mask: torch.Tensor | None,
        examples: List[dict],
    ) -> torch.Tensor:
        action_dtype = next(self.action_model.parameters()).dtype
        actions = torch.as_tensor(
            np.asarray([example["action"] for example in examples]),
            device=condition.device,
            dtype=action_dtype,
        )
        actions_target = actions[:, -self.action_horizon :, :]
        repeated_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        repeated_condition = condition.to(dtype=action_dtype).repeat(repeated_steps, 1, 1)
        repeated_mask = condition_mask.repeat(repeated_steps, 1) if condition_mask is not None else None
        repeated_actions = actions_target.repeat(repeated_steps, 1, 1)
        state = None
        if "state" in examples[0] and self.config.framework.action_model.get("state_dim", 0):
            state = torch.as_tensor(
                np.asarray([example["state"] for example in examples]),
                device=condition.device,
                dtype=condition.dtype,
            )
            state = state[..., : int(self.config.framework.action_model.state_dim)]
            state = state.repeat(repeated_steps, 1, 1)
        return self.action_model(
            repeated_condition,
            repeated_actions,
            state,
            encoder_attention_mask=repeated_mask,
        )

    def forward(self, examples: List[dict] = None, **kwargs) -> dict[str, torch.Tensor]:
        qwen_inputs, last_hidden, attention_mask = self._run_backbone(examples)
        condition, condition_mask = self._build_condition(examples, last_hidden, attention_mask)
        return {"action_loss": self._action_loss(condition, condition_mask, examples)}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        batch_images = [example["image"] for example in examples]
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=self._instructions(examples),
        )
        attention_mask = qwen_inputs.get("attention_mask")
        qwen_inputs = {key: value.to(self.qwen_vl_interface.model.device) if hasattr(value, "to") else value
                       for key, value in qwen_inputs.items()}
        outputs = self.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        if attention_mask is not None:
            attention_mask = qwen_inputs["attention_mask"]
        condition, condition_mask = self._build_condition(examples, last_hidden, attention_mask)
        state = None
        if "state" in examples[0] and self.config.framework.action_model.get("state_dim", 0):
            state = torch.as_tensor(
                np.asarray([example["state"] for example in examples]),
                device=condition.device,
                dtype=condition.dtype,
            )
            state = state[..., : int(self.config.framework.action_model.state_dim)]
        action_dtype = next(self.action_model.parameters()).dtype
        actions = self.action_model.predict_action(
            condition.to(dtype=action_dtype),
            state.to(dtype=action_dtype) if state is not None else None,
            encoder_attention_mask=condition_mask,
        )
        return {"normalized_actions": actions.detach().float().cpu().numpy()}
