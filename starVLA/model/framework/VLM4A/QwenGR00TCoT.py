
"""QwenGR00T V1 with training-only depth/UVD geometric CoT."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.modules.cot_losses import (
    endpoint_depth_uvd_consistency,
    masked_smooth_l1_loss,
    uvd_regression_loss,
)
from starVLA.model.modules.depth_cot_decoder import SharedFiLMConvStack
from starVLA.model.modules.geometric_cot import GeometricQueryReasoner
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


IMAGE_TOKEN_INDEX = 248056


def extract_contiguous_image_token_runs(input_ids: torch.Tensor, image_token_id: int = IMAGE_TOKEN_INDEX) -> list[torch.Tensor]:
    """Return contiguous image-token position runs in one tokenized sample."""
    positions = torch.nonzero(input_ids == int(image_token_id), as_tuple=False).flatten()
    if positions.numel() == 0:
        return []
    split_points = torch.where(positions[1:] != positions[:-1] + 1)[0] + 1
    return list(torch.tensor_split(positions, split_points.tolist()))


def infer_square_patch_hw(token_count: int) -> tuple[int, int]:
    """Infer a raster grid from the actual number of image hidden tokens."""
    token_count = int(token_count)
    side = int(math.isqrt(token_count))
    if side * side == token_count:
        return side, side
    factors = [(d, token_count // d) for d in range(1, side + 1) if token_count % d == 0]
    if not factors:
        raise ValueError(f"cannot infer patch grid from token_count={token_count}")
    return min(factors, key=lambda pair: abs(pair[0] - pair[1]))


def cast_input_to_module_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Match an auxiliary head input to its parameter dtype outside autocast."""
    parameter = next(module.parameters(), None)
    return tensor.to(dtype=parameter.dtype) if parameter is not None else tensor


def append_uvd_condition_mask(
    mask: torch.Tensor | None,
    uvd_valid: int | torch.Tensor,
) -> torch.Tensor | None:
    """Append UVD validity to the native V/L condition mask."""
    if mask is None:
        return None
    if isinstance(uvd_valid, int):
        if uvd_valid < 1:
            raise ValueError(f"uvd_count must be positive, got {uvd_valid}")
        uvd_mask = torch.ones(mask.shape[0], int(uvd_valid), dtype=torch.bool, device=mask.device)
    else:
        uvd_mask = uvd_valid.to(device=mask.device, dtype=torch.bool)
        if uvd_mask.ndim != 2 or uvd_mask.shape[0] != mask.shape[0]:
            raise ValueError(
                f"uvd_valid must have shape [B,K] with B={mask.shape[0]}, got {tuple(uvd_mask.shape)}"
            )
    return torch.cat([mask.to(dtype=torch.bool), uvd_mask], dim=1)


@FRAMEWORK_REGISTRY.register("QwenGR00TCoT")
class Qwen_GR00T_CoT(Qwen_GR00T):
    """QwenGR00T baseline plus geometric query supervision and UVD conditioning."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        geometry = self.config.framework.get("geometry", {})
        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        depth_query_count = int(geometry.get("depth_query_count", 8))
        query_heads = int(geometry.get("query_num_heads", 16))
        if hidden_dim % query_heads != 0:
            query_heads = math.gcd(hidden_dim, query_heads)
        self.geometry_hidden_dim = hidden_dim
        self.depth_query_count = depth_query_count
        self.uvd_num_points = geometry.get("uvd_num_points", None)
        self.geometry_query = GeometricQueryReasoner(
            hidden_dim=hidden_dim,
            num_heads=max(query_heads, 1),
            depth_query_count=depth_query_count,
            layer_count=int(geometry.get("query_layer_count", 2)),
            dropout=float(geometry.get("query_dropout", 0.0)),
        )
        self.depth_decoder = SharedFiLMConvStack(
            hidden_dim=hidden_dim,
            features=int(geometry.get("depth_decoder_features", 256)),
            stage_count=int(geometry.get("depth_decoder_stages", 3)),
        )
        self.uvd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.depth_output_size = int(geometry.get("depth_output_size", geometry.get("image_size", 224)))
        self.uvd_depth_scale = float(geometry.get("uvd_depth_scale", 1.0))
        self.lambda_action = float(geometry.get("lambda_action", 1.0))
        self.lambda_depth_current = float(geometry.get("lambda_depth_current", 0.05))
        self.lambda_depth_future = float(geometry.get("lambda_depth_future", 0.05))
        self.lambda_uvd = float(geometry.get("lambda_uvd", 0.1))
        self.lambda_geometry = float(geometry.get("lambda_geometry", 0.01))

    def _main_image_tokens(self, last_hidden: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        runs = [extract_contiguous_image_token_runs(row) for row in input_ids]
        if not runs or any(len(sample_runs) == 0 for sample_runs in runs):
            raise RuntimeError("Qwen3.5 output contains no image-token span")
        lengths = [int(sample_runs[0].numel()) for sample_runs in runs]
        if len(set(lengths)) != 1:
            raise RuntimeError(f"batched main image token lengths differ: {lengths}")
        positions = torch.stack([sample_runs[0] for sample_runs in runs], dim=0).to(last_hidden.device)
        batch_indices = torch.arange(last_hidden.shape[0], device=last_hidden.device)[:, None]
        tokens = last_hidden[batch_indices, positions]
        return tokens, infer_square_patch_hw(lengths[0])

    def _run_backbone(self, examples: List[dict]):
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        attention_mask = qwen_inputs.get("attention_mask")
        outputs = self.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        return qwen_inputs, last_hidden, attention_mask

    def _predict_uvd(self, tokens: torch.Tensor) -> torch.Tensor:
        raw = self.uvd_head(cast_input_to_module_dtype(tokens, self.uvd_head))
        return torch.cat([torch.sigmoid(raw[..., :2]), F.softplus(raw[..., 2:3])], dim=-1)

    def _prepare_uvd_targets(self, examples: List[dict], device: torch.device):
        default_points = self.uvd_num_points
        if default_points is None:
            default_points = int(math.floor(0.3 * int(self.action_horizon))) + 2
        lengths = [int(np.asarray(example["uvd"]).shape[0]) for example in examples]
        point_count = max(int(default_points), max(lengths, default=1), 2)
        target = torch.zeros(len(examples), point_count, 3, device=device, dtype=torch.float32)
        valid = torch.zeros(len(examples), point_count, device=device, dtype=torch.bool)
        times = torch.zeros(len(examples), point_count, device=device, dtype=torch.float32)
        endpoints = torch.zeros(len(examples), 2, device=device, dtype=torch.long)
        for batch_index, example in enumerate(examples):
            uvd = np.asarray(example["uvd"], dtype=np.float32)
            uvd_valid = np.asarray(example["uvd_valid_mask"], dtype=np.bool_)
            uvd_time = np.asarray(
                example.get("uvd_time", np.linspace(0.0, 1.0, len(uvd), dtype=np.float32)),
                dtype=np.float32,
            )
            count = min(len(uvd), point_count)
            target[batch_index, :count] = torch.as_tensor(uvd[:count], device=device)
            valid[batch_index, :count] = torch.as_tensor(uvd_valid[:count], device=device)
            times[batch_index, :count] = torch.as_tensor(uvd_time[:count], device=device)
            endpoints[batch_index, 1] = max(count - 1, 0)
        return target, valid, times, endpoints

    def _geometry_forward(
        self,
        qwen_inputs,
        last_hidden,
        attention_mask,
        *,
        uvd_times: torch.Tensor | None = None,
        uvd_num_points: int | None = None,
        uvd_valid_mask: torch.Tensor | None = None,
    ):
        horizon = int(self.action_horizon)
        query = self.geometry_query(
            last_hidden,
            attention_mask,
            horizon=horizon,
            uvd_num_points=uvd_num_points if uvd_num_points is not None else self.uvd_num_points,
            uvd_times=uvd_times,
        )
        image_tokens, patch_hw = self._main_image_tokens(last_hidden, qwen_inputs["input_ids"])
        current_query = query["depth_current_tokens"].mean(dim=1)
        future_query = query["depth_future_tokens"].mean(dim=1)
        output_hw = (self.depth_output_size, self.depth_output_size)
        depth_current = self.depth_decoder(image_tokens, patch_hw=patch_hw, query=current_query, output_hw=output_hw)
        depth_future = self.depth_decoder(image_tokens, patch_hw=patch_hw, query=future_query, output_hw=output_hw)
        uvd = self._predict_uvd(query["uvd_tokens"])
        condition = torch.cat([last_hidden, query["uvd_tokens"]], dim=1)
        condition_mask = append_uvd_condition_mask(
            attention_mask,
            uvd_valid_mask if uvd_valid_mask is not None else uvd.shape[1],
        )
        return query, depth_current, depth_future, uvd, condition, condition_mask

    @torch.inference_mode()
    def predict_geometry(self, examples: List[dict]) -> dict[str, torch.Tensor]:
        """Run only the geometry branch for fixed-sample diagnostics."""
        qwen_inputs, last_hidden, attention_mask = self._run_backbone(examples)
        uvd_target, uvd_valid, uvd_times, _ = self._prepare_uvd_targets(examples, last_hidden.device)
        _, depth_current, depth_future, uvd, _, _ = self._geometry_forward(
            qwen_inputs,
            last_hidden,
            attention_mask,
            uvd_times=uvd_times,
            uvd_num_points=uvd_target.shape[1],
            uvd_valid_mask=uvd_valid,
        )
        return {
            "depth_current": depth_current,
            "depth_future": depth_future,
            "uvd": uvd,
        }

    def _action_loss(self, condition: torch.Tensor, condition_mask: torch.Tensor | None, examples: List[dict]) -> torch.Tensor:
        action_dtype = next(self.action_model.parameters()).dtype
        actions = torch.as_tensor(np.asarray([example["action"] for example in examples]), device=condition.device, dtype=action_dtype)
        actions_target = actions[:, -self.action_horizon :, :]
        repeated_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        repeated_condition = condition.to(dtype=action_dtype).repeat(repeated_steps, 1, 1)
        repeated_mask = condition_mask.repeat(repeated_steps, 1) if condition_mask is not None else None
        repeated_actions = actions_target.repeat(repeated_steps, 1, 1)
        state = None
        if "state" in examples[0] and self.config.framework.action_model.get("state_dim", 0):
            state = torch.as_tensor(np.asarray([example["state"] for example in examples]), device=condition.device, dtype=condition.dtype)
            state = state[..., : int(self.config.framework.action_model.state_dim)]
            state = state.repeat(repeated_steps, 1, 1)
        return self.action_model(repeated_condition, repeated_actions, state, encoder_attention_mask=repeated_mask)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict[str, torch.Tensor]:
        qwen_inputs, last_hidden, attention_mask = self._run_backbone(examples)
        device = last_hidden.device
        uvd_target, uvd_valid, uvd_times, uvd_endpoints = self._prepare_uvd_targets(examples, device)
        query, depth_current, depth_future, uvd, condition, condition_mask = self._geometry_forward(
            qwen_inputs,
            last_hidden,
            attention_mask,
            uvd_times=uvd_times,
            uvd_num_points=uvd_target.shape[1],
            uvd_valid_mask=uvd_valid,
        )
        action_loss = self._action_loss(condition, condition_mask, examples)
        depth_current_target = torch.as_tensor(np.stack([x["depth_current"] for x in examples]), device=device)
        depth_future_target = torch.as_tensor(np.stack([x["depth_future"] for x in examples]), device=device)
        depth_current_valid = torch.as_tensor(np.stack([x["depth_current_valid"] for x in examples]), device=device)
        depth_future_valid = torch.as_tensor(np.stack([x["depth_future_valid"] for x in examples]), device=device)
        depth_current_loss = masked_smooth_l1_loss(depth_current, depth_current_target, depth_current_valid)
        depth_future_loss = masked_smooth_l1_loss(depth_future, depth_future_target, depth_future_valid)
        uvd_loss = uvd_regression_loss(uvd, uvd_target, uvd_valid)
        geometry_loss = endpoint_depth_uvd_consistency(
            depth_current,
            depth_future,
            uvd,
            uvd_valid,
            depth_scale=self.uvd_depth_scale,
            endpoint_indices=uvd_endpoints,
        )
        total_loss = (
            self.lambda_action * action_loss
            + self.lambda_depth_current * depth_current_loss
            + self.lambda_depth_future * depth_future_loss
            + self.lambda_uvd * uvd_loss
            + self.lambda_geometry * geometry_loss
        )
        return {
            "action_loss": action_loss,
            "depth_current_loss": depth_current_loss,
            "depth_future_loss": depth_future_loss,
            "uvd_loss": uvd_loss,
            "geometry_loss": geometry_loss,
            "total_loss": total_loss,
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        attention_mask = qwen_inputs.get("attention_mask")
        outputs = self.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        query = self.geometry_query(
            last_hidden,
            attention_mask,
            horizon=int(self.action_horizon),
            uvd_num_points=self.uvd_num_points,
        )
        condition = torch.cat([last_hidden, query["uvd_tokens"]], dim=1)
        condition_mask = append_uvd_condition_mask(attention_mask, query["uvd_tokens"].shape[1])
        state = None
        if "state" in examples[0] and self.config.framework.action_model.get("state_dim", 0):
            state = torch.as_tensor(np.asarray([example["state"] for example in examples]), device=condition.device, dtype=condition.dtype)
            state = state[..., : int(self.config.framework.action_model.state_dim)]
        action_dtype = next(self.action_model.parameters()).dtype
        condition = condition.to(dtype=action_dtype)
        if state is not None:
            state = state.to(dtype=action_dtype)
        actions = self.action_model.predict_action(condition, state, encoder_attention_mask=condition_mask)
        return {"normalized_actions": actions.detach().float().cpu().numpy()}
