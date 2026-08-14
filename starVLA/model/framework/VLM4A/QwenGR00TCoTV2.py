"""QwenGR00T V2 with geometry query tokens inside Qwen3.5."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.modules.cot_losses import (
    aggregate_cot_total_loss,
    masked_smooth_l1_loss,
    uvd_adjacent_relative_loss,
    uvd_regression_loss,
)
from starVLA.model.modules.depth_cot_decoder import SharedFiLMConvStack
from starVLA.model.modules.geometric_cot_v2 import (
    GeometryTokenEmbedding,
    GeometryTokenLayout,
    PackedUVDTargets,
    SharedDepthAttentionPool,
    append_geometry_slots,
    build_depth_summary_interventions,
    build_geometry_full_attention_mask,
    pack_uvd_targets_time_major,
)
from starVLA.model.modules.qwen35_geometry_forward import forward_qwen35_with_geometry
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass(frozen=True)
class GeometryHiddenSplit:
    """Final hidden-state groups emitted by the V2 Qwen sequence."""

    native: torch.Tensor
    depth_current: torch.Tensor
    depth_future: torch.Tensor
    uvd: torch.Tensor


def _require_boolean_option(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def _extract_contiguous_runs(input_ids: torch.Tensor, token_id: int) -> list[torch.Tensor]:
    positions = torch.nonzero(input_ids == int(token_id), as_tuple=False).flatten()
    if positions.numel() == 0:
        return []
    split_points = torch.where(positions[1:] != positions[:-1] + 1)[0] + 1
    return list(torch.tensor_split(positions, split_points.tolist()))


def _infer_patch_hw(token_count: int) -> tuple[int, int]:
    token_count = int(token_count)
    side = int(math.isqrt(token_count))
    if side * side == token_count:
        return side, side
    factors = [(value, token_count // value) for value in range(1, side + 1) if token_count % value == 0]
    if not factors:
        raise ValueError(f"cannot infer patch grid from token_count={token_count}")
    return min(factors, key=lambda pair: abs(pair[0] - pair[1]))


def _cast_to_module_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    parameter = next(module.parameters(), None)
    return tensor.to(dtype=parameter.dtype) if parameter is not None else tensor


@FRAMEWORK_REGISTRY.register("QwenGR00TCoTV2")
class Qwen_GR00T_CoT_V2(Qwen_GR00T):
    """Directly insert supervised depth/UVD latent tokens into Qwen3.5."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        geometry = self.config.framework.get("geometry", {})
        self.include_depth_in_action_condition = _require_boolean_option(
            geometry.get("include_depth_in_action_condition", False),
            name="include_depth_in_action_condition",
        )
        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        points_per_hand = geometry.get("uvd_num_points", None)
        if points_per_hand is None:
            points_per_hand = int(math.floor(0.3 * int(self.action_horizon))) + 2
        self.geometry_layout = GeometryTokenLayout(
            depth_query_count=int(geometry.get("depth_query_count", 8)),
            uvd_points_per_hand=int(points_per_hand),
            hand_count=int(geometry.get("uvd_hand_count", 1)),
        )
        self.uvd_hand_count = int(self.geometry_layout.hand_count)
        self.uvd_token_order = "time_major"
        self.geometry_tokens = GeometryTokenEmbedding(hidden_dim=hidden_dim, layout=self.geometry_layout)
        self.depth_attention_pool = SharedDepthAttentionPool(hidden_dim=hidden_dim)
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
        self.lambda_depth_current = float(geometry.get("lambda_depth_current", 0.14))
        self.lambda_depth_future = float(geometry.get("lambda_depth_future", 0.15))
        self.lambda_uvd = float(geometry.get("lambda_uvd", 0.62))
        self.lambda_uvd_relative = float(geometry.get("lambda_uvd_relative", 0.1))

        backend = str(geometry.get("full_attention_backend", "sdpa"))
        if backend != "sdpa":
            raise ValueError(f"full_attention_backend must be 'sdpa', got {backend!r}")
        self.full_attention_backend = backend
        language_model = self.qwen_vl_interface.model.model.language_model
        language_model.config._attn_implementation = backend

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        placeholder_id = tokenizer.pad_token_id
        if placeholder_id is None:
            placeholder_id = tokenizer.eos_token_id
        if placeholder_id is None:
            raise ValueError("Qwen tokenizer must define pad_token_id or eos_token_id for geometry placeholders")
        self.geometry_placeholder_token_id = int(placeholder_id)

    @staticmethod
    def validate_checkpoint_state_dict(state_dict) -> None:
        """Reject old mean-pooled V2 checkpoints before full or partial loading."""

        keys = tuple(str(key) for key in state_dict)
        has_v2_geometry = any(
            key.startswith("geometry_tokens.") or ".geometry_tokens." in key
            for key in keys
        )
        has_attention_pool = any(
            key.startswith("depth_attention_pool.") or ".depth_attention_pool." in key
            for key in keys
        )
        if has_v2_geometry and not has_attention_pool:
            raise RuntimeError(
                "This checkpoint predates the V2 shared depth attention-pooling module. "
                "Mean-pooling V2 checkpoints are intentionally incompatible; start a new "
                "V2 run or load a checkpoint containing depth_attention_pool parameters."
            )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.validate_checkpoint_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def geometry_query(self) -> nn.Module:
        """Compatibility alias for existing module-gradient diagnostics."""

        return self.geometry_tokens

    def _trajectory_point_count(self) -> int:
        return int(self.geometry_layout.uvd_points_per_hand)

    def _prepare_uvd_targets(self, examples: List[dict], device: torch.device) -> PackedUVDTargets:
        return pack_uvd_targets_time_major(examples, self.geometry_layout, device=device)

    def _compute_uvd_losses(
        self,
        pred: torch.Tensor,
        packed: PackedUVDTargets,
    ) -> dict[str, torch.Tensor]:
        absolute = uvd_regression_loss(pred, packed.target, packed.valid)
        relative = uvd_adjacent_relative_loss(
            pred,
            packed.target,
            packed.valid,
            hand_count=self.geometry_layout.hand_count,
        )
        return {
            "absolute": absolute,
            "relative": relative,
            "total": absolute + self.lambda_uvd_relative * relative,
        }

    def _split_geometry_hidden(
        self,
        last_hidden: torch.Tensor,
        *,
        native_token_count: int,
    ) -> GeometryHiddenSplit:
        expected = int(native_token_count) + self.geometry_layout.geometry_token_count
        if int(last_hidden.shape[1]) != expected:
            raise ValueError(f"hidden sequence has {last_hidden.shape[1]} tokens, expected {expected}")
        slices = self.geometry_layout.sequence_slices(native_token_count)
        return GeometryHiddenSplit(
            native=last_hidden[:, slices.native],
            depth_current=last_hidden[:, slices.depth_current],
            depth_future=last_hidden[:, slices.depth_future],
            uvd=last_hidden[:, slices.uvd],
        )

    def _build_action_condition(
        self,
        split: GeometryHiddenSplit,
        *,
        native_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        include_depth = _require_boolean_option(
            self.include_depth_in_action_condition,
            name="include_depth_in_action_condition",
        )
        geometry_condition = []
        if include_depth:
            geometry_condition.extend([split.depth_current, split.depth_future])
        geometry_condition.append(split.uvd)
        condition = torch.cat([split.native, *geometry_condition], dim=1)
        if native_attention_mask is None:
            return condition, None
        native_attention_mask = native_attention_mask.to(device=condition.device, dtype=torch.bool)
        geometry_attention_mask = torch.ones(
            condition.shape[0],
            sum(tokens.shape[1] for tokens in geometry_condition),
            device=condition.device,
            dtype=torch.bool,
        )
        return condition, torch.cat([native_attention_mask, geometry_attention_mask], dim=1)

    def _build_native_inputs(self, examples: List[dict], *, inference: bool) -> tuple[dict, torch.Tensor]:
        if inference:
            batch_images = [to_pil_preserve(example["image"]) for example in examples]
            train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
            if train_obs_image_size:
                batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        else:
            batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        native_attention_mask = qwen_inputs.get("attention_mask")
        if native_attention_mask is not None:
            native_attention_mask = native_attention_mask.to(dtype=torch.bool)
        return qwen_inputs, native_attention_mask

    def _run_geometry_backbone(
        self,
        qwen_inputs: dict,
    ) -> GeometryHiddenSplit:
        native_token_count = int(qwen_inputs["input_ids"].shape[1])
        geometry_embeddings = self.geometry_tokens(
            batch_size=int(qwen_inputs["input_ids"].shape[0]),
        )
        appended_inputs = append_geometry_slots(
            qwen_inputs,
            self.geometry_layout,
            placeholder_token_id=self.geometry_placeholder_token_id,
        )
        full_attention_mask = build_geometry_full_attention_mask(
            appended_inputs["attention_mask"],
            self.geometry_layout,
        )
        output = forward_qwen35_with_geometry(
            self.qwen_vl_interface.model,
            qwen_inputs=appended_inputs,
            geometry_embeddings=geometry_embeddings,
            full_attention_mask=full_attention_mask,
        )
        return self._split_geometry_hidden(
            output.last_hidden_state,
            native_token_count=native_token_count,
        )

    def _main_image_tokens(
        self,
        native_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        image_token_id = int(self.qwen_vl_interface.model.config.image_token_id)
        runs = [_extract_contiguous_runs(row, image_token_id) for row in input_ids]
        if not runs or any(len(sample_runs) == 0 for sample_runs in runs):
            raise RuntimeError("Qwen3.5 output contains no image-token span")
        lengths = [int(sample_runs[0].numel()) for sample_runs in runs]
        if len(set(lengths)) != 1:
            raise RuntimeError(f"batched main image token lengths differ: {lengths}")
        positions = torch.stack([sample_runs[0] for sample_runs in runs], dim=0).to(native_hidden.device)
        batch_indices = torch.arange(native_hidden.shape[0], device=native_hidden.device)[:, None]
        return native_hidden[batch_indices, positions], _infer_patch_hw(lengths[0])

    def _predict_uvd(self, tokens: torch.Tensor) -> torch.Tensor:
        raw = self.uvd_head(_cast_to_module_dtype(tokens, self.uvd_head))
        return torch.cat([torch.sigmoid(raw[..., :2]), F.softplus(raw[..., 2:3])], dim=-1)

    def _pool_depth_summaries(
        self,
        split: GeometryHiddenSplit,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_summary, current_weights = self.depth_attention_pool(split.depth_current)
        future_summary, future_weights = self.depth_attention_pool(split.depth_future)
        return current_summary, future_summary, current_weights, future_weights

    def _decode_depth_summaries(
        self,
        image_tokens: torch.Tensor,
        *,
        patch_hw: tuple[int, int],
        current_summary: torch.Tensor,
        future_summary: torch.Tensor,
        timing_callback: Callable[[str, Callable[[], Any]], Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def timed(name: str, fn: Callable[[], Any]) -> Any:
            return timing_callback(name, fn) if timing_callback is not None else fn()

        output_hw = (self.depth_output_size, self.depth_output_size)
        depth_current = timed(
            "depth_current_ms",
            lambda: self.depth_decoder(
                image_tokens,
                patch_hw=patch_hw,
                query=current_summary,
                output_hw=output_hw,
            ),
        )
        depth_future = timed(
            "depth_future_ms",
            lambda: self.depth_decoder(
                image_tokens,
                patch_hw=patch_hw,
                query=future_summary,
                output_hw=output_hw,
            ),
        )
        return depth_current, depth_future

    def _decode_geometry(
        self,
        split: GeometryHiddenSplit,
        qwen_inputs: dict,
        *,
        timing_callback: Callable[[str, Callable[[], Any]], Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def timed(name: str, fn: Callable[[], Any]) -> Any:
            return timing_callback(name, fn) if timing_callback is not None else fn()

        image_tokens, patch_hw = timed(
            "image_token_extract_ms",
            lambda: self._main_image_tokens(split.native, qwen_inputs["input_ids"]),
        )
        current_summary, future_summary, _, _ = self._pool_depth_summaries(split)
        depth_current, depth_future = self._decode_depth_summaries(
            image_tokens,
            patch_hw=patch_hw,
            current_summary=current_summary,
            future_summary=future_summary,
            timing_callback=timing_callback,
        )
        uvd = timed("uvd_head_ms", lambda: self._predict_uvd(split.uvd))
        return depth_current, depth_future, uvd

    def _action_loss(
        self,
        condition: torch.Tensor,
        condition_mask: torch.Tensor | None,
        examples: List[dict],
    ) -> torch.Tensor:
        actions = torch.as_tensor(
            np.asarray([example["action"] for example in examples]),
            device=condition.device,
            dtype=condition.dtype,
        )
        actions_target = actions[:, -self.action_horizon :, :]
        repeated_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        repeated_condition = condition.repeat(repeated_steps, 1, 1)
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
        with torch.autocast("cuda", dtype=torch.float32):
            return self.action_model(
                repeated_condition,
                repeated_actions,
                state,
                encoder_attention_mask=repeated_mask,
            )

    def forward(self, examples: List[dict] = None, **kwargs) -> dict[str, torch.Tensor]:
        qwen_inputs, native_attention_mask = self._build_native_inputs(examples, inference=False)
        packed = self._prepare_uvd_targets(examples, qwen_inputs["input_ids"].device)
        split = self._run_geometry_backbone(qwen_inputs)
        depth_current, depth_future, uvd = self._decode_geometry(split, qwen_inputs)
        condition, condition_mask = self._build_action_condition(
            split,
            native_attention_mask=native_attention_mask,
        )
        action_loss = self._action_loss(condition, condition_mask, examples)
        device = split.native.device
        depth_current_target = torch.as_tensor(np.stack([x["depth_current"] for x in examples]), device=device)
        depth_future_target = torch.as_tensor(np.stack([x["depth_future"] for x in examples]), device=device)
        depth_current_valid = torch.as_tensor(np.stack([x["depth_current_valid"] for x in examples]), device=device)
        depth_future_valid = torch.as_tensor(np.stack([x["depth_future_valid"] for x in examples]), device=device)
        depth_current_loss = masked_smooth_l1_loss(depth_current, depth_current_target, depth_current_valid)
        depth_future_loss = masked_smooth_l1_loss(depth_future, depth_future_target, depth_future_valid)
        uvd_losses = self._compute_uvd_losses(uvd, packed)
        uvd_loss = uvd_losses["total"]
        total_loss = aggregate_cot_total_loss(
            action_loss,
            depth_current_loss,
            depth_future_loss,
            uvd_loss,
            lambda_action=self.lambda_action,
            lambda_depth_current=self.lambda_depth_current,
            lambda_depth_future=self.lambda_depth_future,
            lambda_uvd=self.lambda_uvd,
        )
        return {
            "action_loss": action_loss,
            "depth_current_loss": depth_current_loss,
            "depth_future_loss": depth_future_loss,
            "uvd_loss": uvd_loss,
            "uvd_absolute_loss": uvd_losses["absolute"],
            "uvd_relative_loss": uvd_losses["relative"],
            "total_loss": total_loss,
        }

    @torch.inference_mode()
    def predict_geometry(self, examples: List[dict]) -> dict[str, torch.Tensor]:
        if not isinstance(examples, list):
            examples = [examples]
        qwen_inputs, _ = self._build_native_inputs(examples, inference=True)
        split = self._run_geometry_backbone(qwen_inputs)
        depth_current, depth_future, uvd = self._decode_geometry(split, qwen_inputs)
        return {"depth_current": depth_current, "depth_future": depth_future, "uvd": uvd}

    @torch.inference_mode()
    def predict_geometry_diagnostics(
        self,
        examples: List[dict],
        *,
        include_decoder_interventions: bool = False,
    ) -> dict[str, Any]:
        """Decode geometry once and optionally perturb only the depth summaries."""

        if not isinstance(examples, list):
            examples = [examples]
        qwen_inputs, _ = self._build_native_inputs(examples, inference=True)
        split = self._run_geometry_backbone(qwen_inputs)
        image_tokens, patch_hw = self._main_image_tokens(split.native, qwen_inputs["input_ids"])
        current_summary, future_summary, current_weights, future_weights = self._pool_depth_summaries(split)
        depth_current, depth_future = self._decode_depth_summaries(
            image_tokens,
            patch_hw=patch_hw,
            current_summary=current_summary,
            future_summary=future_summary,
        )
        output: dict[str, Any] = {
            "depth_current": depth_current,
            "depth_future": depth_future,
            "uvd": self._predict_uvd(split.uvd),
            "depth_current_tokens": split.depth_current,
            "depth_future_tokens": split.depth_future,
            "uvd_tokens": split.uvd,
            "depth_current_pool_weights": current_weights,
            "depth_future_pool_weights": future_weights,
        }
        if include_decoder_interventions:
            interventions = {}
            variants = build_depth_summary_interventions(current_summary, future_summary)
            for name, (variant_current, variant_future) in variants.items():
                if name == "normal":
                    continue
                variant_depth_current, variant_depth_future = self._decode_depth_summaries(
                    image_tokens,
                    patch_hw=patch_hw,
                    current_summary=variant_current,
                    future_summary=variant_future,
                )
                interventions[name] = {
                    "depth_current": variant_depth_current,
                    "depth_future": variant_depth_future,
                }
            output["decoder_interventions"] = interventions
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        timing_callback = kwargs.pop("timing_callback", None)
        return_geometry = _require_boolean_option(
            kwargs.pop("return_geometry", False),
            name="return_geometry",
        )
        if return_geometry and not hasattr(self.geometry_layout, "uvd_time_points"):
            raise ValueError(
                "return_geometry is supported only by QwenGR00TCoTV3 landmark layouts"
            )
        timing: dict[str, float] = {}
        def timed(name: str, fn: Callable[[], Any]) -> Any:
            return timing_callback(name, fn) if timing_callback is not None else fn()

        preprocess_start = time.perf_counter()
        qwen_inputs, native_attention_mask = self._build_native_inputs(examples, inference=True)
        timing["preprocess_ms"] = (time.perf_counter() - preprocess_start) * 1000.0

        def run_qwen():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self._run_geometry_backbone(qwen_inputs)

        split = timed("qwen_backbone_ms", run_qwen)
        condition, condition_mask = self._build_action_condition(
            split,
            native_attention_mask=native_attention_mask,
        )
        state = None
        if "state" in examples[0] and self.config.framework.action_model.get("state_dim", 0):
            state = torch.as_tensor(
                np.asarray([example["state"] for example in examples]),
                device=condition.device,
                dtype=condition.dtype,
            )
            state = state[..., : int(self.config.framework.action_model.state_dim)]

        def run_action():
            with torch.autocast("cuda", dtype=torch.float32):
                return self.action_model.predict_action(
                    condition,
                    state,
                    encoder_attention_mask=condition_mask,
                )

        actions = timed("action_expert_ms", run_action)
        output_start = time.perf_counter()
        normalized_actions = actions.detach().float().cpu().numpy()
        timing["output_transfer_ms"] = (time.perf_counter() - output_start) * 1000.0
        result = {"normalized_actions": normalized_actions}
        if return_geometry:
            depth_current, depth_future, uvd = self._decode_geometry(
                split,
                qwen_inputs,
                timing_callback=timing_callback,
            )
            time_points = int(self.geometry_layout.uvd_time_points)
            landmark_count = int(self.geometry_layout.landmark_count)
            uvd_time = torch.linspace(
                0.0,
                1.0,
                time_points,
                device=uvd.device,
                dtype=torch.float32,
            ).repeat_interleave(landmark_count)
            uvd_landmark_ids = torch.arange(
                landmark_count,
                device=uvd.device,
                dtype=torch.long,
            ).repeat(time_points)
            result["geometry"] = {
                "depth_current": depth_current,
                "depth_future": depth_future,
                "uvd": uvd,
                "uvd_time": uvd_time.unsqueeze(0).expand(uvd.shape[0], -1),
                "uvd_landmark_ids": uvd_landmark_ids.unsqueeze(0).expand(uvd.shape[0], -1),
            }
        if timing_callback is not None:
            result["timing"] = timing
        return result
