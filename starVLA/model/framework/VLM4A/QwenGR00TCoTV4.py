"""Standalone QwenGR00TCoTV4 with forward coarse-to-local UVD reasoning."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.modules.cot_losses import (
    masked_smooth_l1_loss,
    uvd_adjacent_relative_loss,
    uvd_regression_loss,
)
from starVLA.model.modules.depth_cot_decoder import SharedFiLMConvStack
from starVLA.model.modules.geometric_cot_v4 import (
    GeometryTokenEmbedding,
    GeometryTokenLayout,
    PackedUVDTargets,
    SharedDepthAttentionPool,
    append_geometry_slots,
    build_depth_summary_interventions,
    build_geometry_full_attention_mask,
    pack_coarse_uvd_targets_time_major,
    pack_local_uvd_targets_time_major,
)
from starVLA.model.modules.qwen35_geometry_forward import (
    forward_qwen35_with_geometry,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass(frozen=True)
class GeometryHiddenSplit:
    """Hidden groups from one [native, depth, coarse, local] Qwen pass."""

    native: torch.Tensor
    depth_current: torch.Tensor
    depth_future: torch.Tensor
    uvd_coarse: torch.Tensor
    uvd_local: torch.Tensor


@dataclass(frozen=True)
class GeometryActionCondition:
    name: str
    condition: torch.Tensor
    condition_mask: torch.Tensor | None
    permutation: torch.Tensor | None = None
    diagnostic_counterfactual: bool = False


def _require_boolean_option(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def _option(mapping: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    return mapping.get(name, default) if hasattr(mapping, "get") else default


def validate_v4_horizon_contract(
    *,
    action_horizon: int,
    model_geometry: Mapping[str, Any] | Any,
    data_geometry: Mapping[str, Any] | Any,
) -> None:
    """Fail fast unless model and data implement the same fixed V4 horizons."""

    horizon = int(action_horizon)
    model_local = int(_option(model_geometry, "local_uvd_num_points", -1))
    model_coarse = int(_option(model_geometry, "coarse_uvd_num_points", -1))
    model_stride = int(_option(model_geometry, "coarse_uvd_stride", -1))
    if model_local != horizon:
        raise ValueError(
            f"model local_uvd_num_points must equal action_horizon "
            f"({model_local} != {horizon})"
        )
    if model_coarse != horizon:
        raise ValueError(
            f"model coarse_uvd_num_points must equal action_horizon "
            f"({model_coarse} != {horizon})"
        )
    if model_stride != 2:
        raise ValueError(f"model coarse stride must be 2, got {model_stride}")

    data_horizon = int(_option(data_geometry, "action_horizon", -1))
    data_local = int(_option(data_geometry, "local_uvd_num_points", -1))
    data_coarse = int(_option(data_geometry, "coarse_uvd_num_points", -1))
    data_stride = int(_option(data_geometry, "coarse_uvd_stride", -1))
    terminal_repeat = _option(data_geometry, "terminal_repeat", None)
    if data_horizon != horizon:
        raise ValueError(
            f"data action_horizon must equal model action_horizon "
            f"({data_horizon} != {horizon})"
        )
    if data_local != horizon:
        raise ValueError(
            f"data local_uvd_num_points must equal action_horizon "
            f"({data_local} != {horizon})"
        )
    if data_coarse != horizon:
        raise ValueError(
            f"data coarse_uvd_num_points must equal action_horizon "
            f"({data_coarse} != {horizon})"
        )
    if data_stride != 2:
        raise ValueError(f"data coarse stride must be 2, got {data_stride}")
    if terminal_repeat is not True:
        raise ValueError("data terminal_repeat must be true for V4")


def _extract_contiguous_runs(
    input_ids: torch.Tensor,
    token_id: int,
) -> list[torch.Tensor]:
    positions = torch.nonzero(
        input_ids == int(token_id),
        as_tuple=False,
    ).flatten()
    if positions.numel() == 0:
        return []
    split_points = torch.where(
        positions[1:] != positions[:-1] + 1
    )[0] + 1
    return list(torch.tensor_split(positions, split_points.tolist()))


def _infer_patch_hw(token_count: int) -> tuple[int, int]:
    token_count = int(token_count)
    side = int(math.isqrt(token_count))
    if side * side == token_count:
        return side, side
    factors = [
        (value, token_count // value)
        for value in range(1, side + 1)
        if token_count % value == 0
    ]
    if not factors:
        raise ValueError(
            f"cannot infer patch grid from token_count={token_count}"
        )
    return min(factors, key=lambda pair: abs(pair[0] - pair[1]))


def _cast_to_module_dtype(
    tensor: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    parameter = next(module.parameters(), None)
    return (
        tensor.to(dtype=parameter.dtype)
        if parameter is not None
        else tensor
    )


@FRAMEWORK_REGISTRY.register("QwenGR00TCoTV4")
class Qwen_GR00T_CoT_V4(Qwen_GR00T):
    """One-pass Qwen backbone with independent coarse and local UVD heads."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        geometry = self.config.framework.get("geometry", {})
        data_geometry = self.config.datasets.vla_data.get("cot_geometry", {})
        validate_v4_horizon_contract(
            action_horizon=int(self.action_horizon),
            model_geometry=geometry,
            data_geometry=data_geometry,
        )
        self.include_depth_in_action_condition = _require_boolean_option(
            geometry.get("include_depth_in_action_condition", True),
            name="include_depth_in_action_condition",
        )
        hidden_dim = int(
            self.qwen_vl_interface.model.config.hidden_size
        )
        self.geometry_layout = GeometryTokenLayout(
            depth_query_count=int(geometry.get("depth_query_count", 8)),
            local_uvd_points_per_hand=int(
                geometry["local_uvd_num_points"]
            ),
            coarse_uvd_points_per_hand=int(
                geometry["coarse_uvd_num_points"]
            ),
            hand_count=int(geometry.get("uvd_hand_count", 1)),
        )
        self.uvd_hand_count = int(self.geometry_layout.hand_count)
        self.uvd_token_order = "time_major"
        self.geometry_tokens = GeometryTokenEmbedding(
            hidden_dim=hidden_dim,
            layout=self.geometry_layout,
        )
        self.depth_attention_pool = SharedDepthAttentionPool(
            hidden_dim=hidden_dim
        )
        self.depth_decoder = SharedFiLMConvStack(
            hidden_dim=hidden_dim,
            features=int(geometry.get("depth_decoder_features", 256)),
            stage_count=int(geometry.get("depth_decoder_stages", 3)),
        )
        self.local_uvd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.coarse_uvd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.depth_output_size = int(
            geometry.get(
                "depth_output_size",
                geometry.get("image_size", 224),
            )
        )
        self.uvd_depth_scale = float(
            geometry.get("uvd_depth_scale", 1.0)
        )
        self.lambda_action = float(geometry.get("lambda_action", 1.0))
        self.lambda_depth_current = float(
            geometry.get("lambda_depth_current", 0.14)
        )
        self.lambda_depth_future = float(
            geometry.get("lambda_depth_future", 0.15)
        )
        self.lambda_uvd = float(geometry.get("lambda_uvd", 0.62))
        self.lambda_uvd_relative = float(
            geometry.get("lambda_uvd_relative", 0.1)
        )
        self.lambda_uvd_coarse = float(
            geometry.get("lambda_uvd_coarse", 0.2)
        )
        self.lambda_uvd_coarse_relative = float(
            geometry.get("lambda_uvd_coarse_relative", 0.1)
        )

        backend = str(
            geometry.get("full_attention_backend", "sdpa")
        )
        if backend != "sdpa":
            raise ValueError(
                f"full_attention_backend must be 'sdpa', got {backend!r}"
            )
        self.full_attention_backend = backend
        language_model = (
            self.qwen_vl_interface.model.model.language_model
        )
        language_model.config._attn_implementation = backend

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        placeholder_id = tokenizer.pad_token_id
        if placeholder_id is None:
            placeholder_id = tokenizer.eos_token_id
        if placeholder_id is None:
            raise ValueError(
                "Qwen tokenizer must define pad_token_id or eos_token_id "
                "for geometry placeholders"
            )
        self.geometry_placeholder_token_id = int(placeholder_id)

    @staticmethod
    def validate_checkpoint_state_dict(state_dict) -> None:
        keys = tuple(str(key).lower() for key in state_dict)
        reverse_markers = ("uvd_full", "full_uvd", "reverse_full")
        if any(
            marker in key
            for key in keys
            for marker in reverse_markers
        ):
            raise RuntimeError(
                "V4 rejects reverse/full trajectory checkpoint keys; "
                "start from the clean V2/base checkpoint"
            )
        geometry_keys = [
            key for key in keys if "geometry_tokens." in key
        ]
        if geometry_keys:
            has_coarse = any(
                "coarse_queries" in key for key in geometry_keys
            )
            has_local = any(
                "local_queries" in key for key in geometry_keys
            )
            if has_coarse != has_local:
                raise RuntimeError(
                    "V4 geometry checkpoints must contain both coarse and "
                    "local query groups"
                )

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        self.validate_checkpoint_state_dict(state_dict)
        return super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

    @property
    def geometry_query(self) -> nn.Module:
        return self.geometry_tokens

    def _trajectory_point_count(self) -> int:
        return int(self.geometry_layout.local_uvd_points_per_hand)

    def _prepare_local_uvd_targets(
        self,
        examples: List[dict],
        device: torch.device,
    ) -> PackedUVDTargets:
        return pack_local_uvd_targets_time_major(
            examples,
            self.geometry_layout,
            device=device,
        )

    def _prepare_coarse_uvd_targets(
        self,
        examples: List[dict],
        device: torch.device,
    ) -> PackedUVDTargets:
        return pack_coarse_uvd_targets_time_major(
            examples,
            self.geometry_layout,
            device=device,
        )

    def _compute_uvd_losses(
        self,
        pred: torch.Tensor,
        packed: PackedUVDTargets,
        *,
        relative_weight: float,
    ) -> dict[str, torch.Tensor]:
        absolute = uvd_regression_loss(
            pred,
            packed.target,
            packed.valid,
        )
        relative = uvd_adjacent_relative_loss(
            pred,
            packed.target,
            packed.valid,
            hand_count=self.geometry_layout.hand_count,
        )
        return {
            "absolute": absolute,
            "relative": relative,
            "total": absolute + float(relative_weight) * relative,
        }

    def _split_geometry_hidden(
        self,
        last_hidden: torch.Tensor,
        *,
        native_token_count: int,
    ) -> GeometryHiddenSplit:
        expected = (
            int(native_token_count)
            + self.geometry_layout.geometry_token_count
        )
        if int(last_hidden.shape[1]) != expected:
            raise ValueError(
                f"hidden sequence has {last_hidden.shape[1]} tokens, "
                f"expected {expected}"
            )
        slices = self.geometry_layout.sequence_slices(
            native_token_count
        )
        return GeometryHiddenSplit(
            native=last_hidden[:, slices.native],
            depth_current=last_hidden[:, slices.depth_current],
            depth_future=last_hidden[:, slices.depth_future],
            uvd_coarse=last_hidden[:, slices.uvd_coarse],
            uvd_local=last_hidden[:, slices.uvd_local],
        )

    def _build_action_condition(
        self,
        split: GeometryHiddenSplit,
        *,
        native_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        geometry_groups = []
        if _require_boolean_option(
            self.include_depth_in_action_condition,
            name="include_depth_in_action_condition",
        ):
            geometry_groups.extend(
                [split.depth_current, split.depth_future]
            )
        geometry_groups.extend([split.uvd_coarse, split.uvd_local])
        condition = torch.cat(
            [split.native, *geometry_groups],
            dim=1,
        )
        if native_attention_mask is None:
            return condition, None
        native_mask = native_attention_mask.to(
            device=condition.device,
            dtype=torch.bool,
        )
        geometry_mask = torch.ones(
            condition.shape[0],
            sum(group.shape[1] for group in geometry_groups),
            device=condition.device,
            dtype=torch.bool,
        )
        return condition, torch.cat(
            [native_mask, geometry_mask],
            dim=1,
        )

    @staticmethod
    def _validate_intervention_split(
        split: GeometryHiddenSplit,
    ) -> int:
        tensors = {
            "native": split.native,
            "depth_current": split.depth_current,
            "depth_future": split.depth_future,
            "uvd_coarse": split.uvd_coarse,
            "uvd_local": split.uvd_local,
        }
        if split.native.ndim != 3:
            raise ValueError("native geometry split must be rank 3")
        batch_size = int(split.native.shape[0])
        hidden_size = int(split.native.shape[2])
        for name, tensor in tensors.items():
            if tensor.ndim != 3:
                raise ValueError(
                    f"{name} geometry split must be rank 3"
                )
            if (
                int(tensor.shape[0]) != batch_size
                or int(tensor.shape[2]) != hidden_size
            ):
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} is incompatible "
                    f"with native {tuple(split.native.shape)}"
                )
            if (
                tensor.device != split.native.device
                or tensor.dtype != split.native.dtype
            ):
                raise ValueError(
                    f"{name} device/dtype must match native"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{name} must contain only finite values"
                )
        return batch_size

    def _build_intervention_condition(
        self,
        split: GeometryHiddenSplit,
        *,
        native_attention_mask: torch.Tensor | None,
        name: str,
        permutation: torch.Tensor | None = None,
    ) -> GeometryActionCondition:
        batch_size = self._validate_intervention_split(split)
        supported = {
            "correct",
            "native_only",
            "zero_geometry",
            "depth_only",
            "coarse_only",
            "local_only",
            "coarse+local",
            "uvd_only",
        }
        if name not in supported:
            raise ValueError(
                f"unknown V4 geometry intervention {name!r}"
            )
        if permutation is not None:
            raise ValueError(
                f"{name} does not accept a donor permutation"
            )
        if name == "native_only":
            condition = split.native
            mask = (
                None
                if native_attention_mask is None
                else native_attention_mask.to(
                    device=condition.device,
                    dtype=torch.bool,
                )
            )
            return GeometryActionCondition(name, condition, mask)
        if (
            name == "depth_only"
            and not self.include_depth_in_action_condition
        ):
            raise ValueError(
                "depth_only requires a trained direct depth condition"
            )

        zeros = {
            "depth_current": torch.zeros_like(split.depth_current),
            "depth_future": torch.zeros_like(split.depth_future),
            "uvd_coarse": torch.zeros_like(split.uvd_coarse),
            "uvd_local": torch.zeros_like(split.uvd_local),
        }
        keep = {
            "correct": {
                "depth_current",
                "depth_future",
                "uvd_coarse",
                "uvd_local",
            },
            "zero_geometry": set(),
            "depth_only": {"depth_current", "depth_future"},
            "coarse_only": {"uvd_coarse"},
            "local_only": {"uvd_local"},
            "coarse+local": {"uvd_coarse", "uvd_local"},
            "uvd_only": {"uvd_coarse", "uvd_local"},
        }[name]
        variant = GeometryHiddenSplit(
            native=split.native,
            depth_current=(
                split.depth_current
                if "depth_current" in keep
                else zeros["depth_current"]
            ),
            depth_future=(
                split.depth_future
                if "depth_future" in keep
                else zeros["depth_future"]
            ),
            uvd_coarse=(
                split.uvd_coarse
                if "uvd_coarse" in keep
                else zeros["uvd_coarse"]
            ),
            uvd_local=(
                split.uvd_local
                if "uvd_local" in keep
                else zeros["uvd_local"]
            ),
        )
        condition, mask = self._build_action_condition(
            variant,
            native_attention_mask=native_attention_mask,
        )
        if mask is not None and mask.shape != condition.shape[:2]:
            raise ValueError(
                f"{name} condition mask does not match condition"
            )
        return GeometryActionCondition(name, condition, mask)

    def _build_native_inputs(
        self,
        examples: List[dict],
        *,
        inference: bool,
    ) -> tuple[dict, torch.Tensor | None]:
        if inference:
            batch_images = [
                to_pil_preserve(example["image"])
                for example in examples
            ]
            train_size = getattr(
                self.config.datasets.vla_data,
                "obs_image_size",
                None,
            )
            if train_size:
                batch_images = resize_images(
                    batch_images,
                    target_size=train_size,
                )
        else:
            batch_images = [
                example["image"] for example in examples
            ]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=[example["lang"] for example in examples],
        )
        native_mask = qwen_inputs.get("attention_mask")
        if native_mask is not None:
            native_mask = native_mask.to(dtype=torch.bool)
        return qwen_inputs, native_mask

    def _run_geometry_backbone(
        self,
        qwen_inputs: dict,
        *,
        local_targets: PackedUVDTargets | None = None,
        coarse_targets: PackedUVDTargets | None = None,
    ) -> GeometryHiddenSplit:
        native_token_count = int(
            qwen_inputs["input_ids"].shape[1]
        )
        geometry_embeddings = self.geometry_tokens(
            batch_size=int(qwen_inputs["input_ids"].shape[0]),
            coarse_uvd_times=(
                None if coarse_targets is None else coarse_targets.times
            ),
            coarse_uvd_hand_ids=(
                None if coarse_targets is None else coarse_targets.hand_ids
            ),
            local_uvd_times=(
                None if local_targets is None else local_targets.times
            ),
            local_uvd_hand_ids=(
                None if local_targets is None else local_targets.hand_ids
            ),
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
        image_token_id = int(
            self.qwen_vl_interface.model.config.image_token_id
        )
        runs = [
            _extract_contiguous_runs(row, image_token_id)
            for row in input_ids
        ]
        if not runs or any(
            len(sample_runs) == 0 for sample_runs in runs
        ):
            raise RuntimeError(
                "Qwen3.5 output contains no image-token span"
            )
        lengths = [
            int(sample_runs[0].numel())
            for sample_runs in runs
        ]
        if len(set(lengths)) != 1:
            raise RuntimeError(
                f"batched main image token lengths differ: {lengths}"
            )
        positions = torch.stack(
            [sample_runs[0] for sample_runs in runs],
            dim=0,
        ).to(native_hidden.device)
        batch_indices = torch.arange(
            native_hidden.shape[0],
            device=native_hidden.device,
        )[:, None]
        return (
            native_hidden[batch_indices, positions],
            _infer_patch_hw(lengths[0]),
        )

    @staticmethod
    def _predict_with_head(
        tokens: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        raw = head(_cast_to_module_dtype(tokens, head))
        return torch.cat(
            [
                torch.sigmoid(raw[..., :2]),
                F.softplus(raw[..., 2:3]),
            ],
            dim=-1,
        )

    def _predict_local_uvd(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self._predict_with_head(tokens, self.local_uvd_head)

    def _predict_coarse_uvd(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self._predict_with_head(tokens, self.coarse_uvd_head)

    def _pool_depth_summaries(
        self,
        split: GeometryHiddenSplit,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_summary, current_weights = (
            self.depth_attention_pool(split.depth_current)
        )
        future_summary, future_weights = (
            self.depth_attention_pool(split.depth_future)
        )
        return (
            current_summary,
            future_summary,
            current_weights,
            future_weights,
        )

    def _decode_depth_summaries(
        self,
        image_tokens: torch.Tensor,
        *,
        patch_hw: tuple[int, int],
        current_summary: torch.Tensor,
        future_summary: torch.Tensor,
        timing_callback: Callable[
            [str, Callable[[], Any]],
            Any,
        ]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def timed(name: str, fn: Callable[[], Any]) -> Any:
            return (
                timing_callback(name, fn)
                if timing_callback is not None
                else fn()
            )

        output_hw = (
            self.depth_output_size,
            self.depth_output_size,
        )
        current = timed(
            "depth_current_ms",
            lambda: self.depth_decoder(
                image_tokens,
                patch_hw=patch_hw,
                query=current_summary,
                output_hw=output_hw,
            ),
        )
        future = timed(
            "depth_future_ms",
            lambda: self.depth_decoder(
                image_tokens,
                patch_hw=patch_hw,
                query=future_summary,
                output_hw=output_hw,
            ),
        )
        return current, future

    def _decode_geometry(
        self,
        split: GeometryHiddenSplit,
        qwen_inputs: dict,
        *,
        timing_callback: Callable[
            [str, Callable[[], Any]],
            Any,
        ]
        | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        def timed(name: str, fn: Callable[[], Any]) -> Any:
            return (
                timing_callback(name, fn)
                if timing_callback is not None
                else fn()
            )

        image_tokens, patch_hw = timed(
            "image_token_extract_ms",
            lambda: self._main_image_tokens(
                split.native,
                qwen_inputs["input_ids"],
            ),
        )
        current_summary, future_summary, _, _ = (
            self._pool_depth_summaries(split)
        )
        depth_current, depth_future = self._decode_depth_summaries(
            image_tokens,
            patch_hw=patch_hw,
            current_summary=current_summary,
            future_summary=future_summary,
            timing_callback=timing_callback,
        )
        coarse = timed(
            "uvd_coarse_head_ms",
            lambda: self._predict_coarse_uvd(split.uvd_coarse),
        )
        local = timed(
            "uvd_head_ms",
            lambda: self._predict_local_uvd(split.uvd_local),
        )
        return depth_current, depth_future, coarse, local

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
        repeated_steps = int(
            self.config.framework.action_model.get(
                "repeated_diffusion_steps",
                4,
            )
        )
        repeated_condition = condition.repeat(
            repeated_steps,
            1,
            1,
        )
        repeated_mask = (
            condition_mask.repeat(repeated_steps, 1)
            if condition_mask is not None
            else None
        )
        repeated_actions = actions_target.repeat(
            repeated_steps,
            1,
            1,
        )
        state = None
        if (
            "state" in examples[0]
            and self.config.framework.action_model.get(
                "state_dim",
                0,
            )
        ):
            state = torch.as_tensor(
                np.asarray([example["state"] for example in examples]),
                device=condition.device,
                dtype=condition.dtype,
            )
            state = state[
                ...,
                : int(
                    self.config.framework.action_model.state_dim
                ),
            ].repeat(repeated_steps, 1, 1)

        def call():
            return self.action_model(
                repeated_condition,
                repeated_actions,
                state,
                encoder_attention_mask=repeated_mask,
            )

        if condition.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float32):
                return call()
        return call()

    def _aggregate_total_loss(
        self,
        action_loss: torch.Tensor,
        depth_current_loss: torch.Tensor,
        depth_future_loss: torch.Tensor,
        local_uvd_loss: torch.Tensor,
        coarse_uvd_loss: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.lambda_action * action_loss
            + self.lambda_depth_current * depth_current_loss
            + self.lambda_depth_future * depth_future_loss
            + self.lambda_uvd * local_uvd_loss
            + self.lambda_uvd_coarse * coarse_uvd_loss
        )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        qwen_inputs, native_mask = self._build_native_inputs(
            examples,
            inference=False,
        )
        device = qwen_inputs["input_ids"].device
        local_targets = self._prepare_local_uvd_targets(
            examples,
            device,
        )
        coarse_targets = self._prepare_coarse_uvd_targets(
            examples,
            device,
        )
        split = self._run_geometry_backbone(
            qwen_inputs,
            local_targets=local_targets,
            coarse_targets=coarse_targets,
        )
        (
            depth_current,
            depth_future,
            coarse_uvd,
            local_uvd,
        ) = self._decode_geometry(split, qwen_inputs)
        condition, condition_mask = self._build_action_condition(
            split,
            native_attention_mask=native_mask,
        )
        action_loss = self._action_loss(
            condition,
            condition_mask,
            examples,
        )
        target_device = split.native.device
        depth_current_target = torch.as_tensor(
            np.stack([x["depth_current"] for x in examples]),
            device=target_device,
        )
        depth_future_target = torch.as_tensor(
            np.stack([x["depth_future"] for x in examples]),
            device=target_device,
        )
        depth_current_valid = torch.as_tensor(
            np.stack([x["depth_current_valid"] for x in examples]),
            device=target_device,
        )
        depth_future_valid = torch.as_tensor(
            np.stack([x["depth_future_valid"] for x in examples]),
            device=target_device,
        )
        depth_current_loss = masked_smooth_l1_loss(
            depth_current,
            depth_current_target,
            depth_current_valid,
        )
        depth_future_loss = masked_smooth_l1_loss(
            depth_future,
            depth_future_target,
            depth_future_valid,
        )
        local_losses = self._compute_uvd_losses(
            local_uvd,
            local_targets,
            relative_weight=self.lambda_uvd_relative,
        )
        coarse_losses = self._compute_uvd_losses(
            coarse_uvd,
            coarse_targets,
            relative_weight=self.lambda_uvd_coarse_relative,
        )
        total_loss = self._aggregate_total_loss(
            action_loss,
            depth_current_loss,
            depth_future_loss,
            local_losses["total"],
            coarse_losses["total"],
        )
        return {
            "action_loss": action_loss,
            "depth_current_loss": depth_current_loss,
            "depth_future_loss": depth_future_loss,
            "uvd_loss": local_losses["total"],
            "uvd_absolute_loss": local_losses["absolute"],
            "uvd_relative_loss": local_losses["relative"],
            "uvd_coarse_loss": coarse_losses["total"],
            "uvd_coarse_absolute_loss": coarse_losses["absolute"],
            "uvd_coarse_relative_loss": coarse_losses["relative"],
            "total_loss": total_loss,
        }

    @torch.inference_mode()
    def predict_geometry(
        self,
        examples: List[dict],
    ) -> dict[str, torch.Tensor]:
        if not isinstance(examples, list):
            examples = [examples]
        qwen_inputs, _ = self._build_native_inputs(
            examples,
            inference=True,
        )
        split = self._run_geometry_backbone(qwen_inputs)
        current, future, coarse, local = self._decode_geometry(
            split,
            qwen_inputs,
        )
        return {
            "depth_current": current,
            "depth_future": future,
            "uvd_coarse": coarse,
            "uvd": local,
        }

    @torch.inference_mode()
    def predict_geometry_diagnostics(
        self,
        examples: List[dict],
        *,
        include_decoder_interventions: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(examples, list):
            examples = [examples]
        qwen_inputs, _ = self._build_native_inputs(
            examples,
            inference=True,
        )
        split = self._run_geometry_backbone(qwen_inputs)
        image_tokens, patch_hw = self._main_image_tokens(
            split.native,
            qwen_inputs["input_ids"],
        )
        (
            current_summary,
            future_summary,
            current_weights,
            future_weights,
        ) = self._pool_depth_summaries(split)
        current, future = self._decode_depth_summaries(
            image_tokens,
            patch_hw=patch_hw,
            current_summary=current_summary,
            future_summary=future_summary,
        )
        output: dict[str, Any] = {
            "depth_current": current,
            "depth_future": future,
            "uvd_coarse": self._predict_coarse_uvd(
                split.uvd_coarse
            ),
            "uvd": self._predict_local_uvd(split.uvd_local),
            "depth_current_tokens": split.depth_current,
            "depth_future_tokens": split.depth_future,
            "uvd_coarse_tokens": split.uvd_coarse,
            "uvd_tokens": split.uvd_local,
            "depth_current_pool_weights": current_weights,
            "depth_future_pool_weights": future_weights,
        }
        if include_decoder_interventions:
            interventions = {}
            for name, (
                variant_current,
                variant_future,
            ) in build_depth_summary_interventions(
                current_summary,
                future_summary,
            ).items():
                if name == "normal":
                    continue
                depth_current, depth_future = (
                    self._decode_depth_summaries(
                        image_tokens,
                        patch_hw=patch_hw,
                        current_summary=variant_current,
                        future_summary=variant_future,
                    )
                )
                interventions[name] = {
                    "depth_current": depth_current,
                    "depth_future": depth_future,
                }
            output["decoder_interventions"] = interventions
        return output

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs,
    ) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        timing_callback = kwargs.pop("timing_callback", None)
        return_geometry = _require_boolean_option(
            kwargs.pop("return_geometry", False),
            name="return_geometry",
        )
        timing: dict[str, float] = {}

        def timed(name: str, fn: Callable[[], Any]) -> Any:
            return (
                timing_callback(name, fn)
                if timing_callback is not None
                else fn()
            )

        preprocess_start = time.perf_counter()
        qwen_inputs, native_mask = self._build_native_inputs(
            examples,
            inference=True,
        )
        timing["preprocess_ms"] = (
            time.perf_counter() - preprocess_start
        ) * 1000.0

        def run_qwen():
            if qwen_inputs["input_ids"].device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return self._run_geometry_backbone(qwen_inputs)
            return self._run_geometry_backbone(qwen_inputs)

        split = timed("qwen_backbone_ms", run_qwen)
        condition, condition_mask = self._build_action_condition(
            split,
            native_attention_mask=native_mask,
        )
        state = None
        if (
            "state" in examples[0]
            and self.config.framework.action_model.get(
                "state_dim",
                0,
            )
        ):
            state = torch.as_tensor(
                np.asarray([example["state"] for example in examples]),
                device=condition.device,
                dtype=condition.dtype,
            )
            state = state[
                ...,
                : int(
                    self.config.framework.action_model.state_dim
                ),
            ]

        def run_action():
            if condition.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float32):
                    return self.action_model.predict_action(
                        condition,
                        state,
                        encoder_attention_mask=condition_mask,
                    )
            return self.action_model.predict_action(
                condition,
                state,
                encoder_attention_mask=condition_mask,
            )

        actions = timed("action_expert_ms", run_action)
        output_start = time.perf_counter()
        result = {
            "normalized_actions": (
                actions.detach().float().cpu().numpy()
            )
        }
        timing["output_transfer_ms"] = (
            time.perf_counter() - output_start
        ) * 1000.0
        if return_geometry:
            current, future, coarse, local = self._decode_geometry(
                split,
                qwen_inputs,
                timing_callback=timing_callback,
            )
            result["geometry"] = {
                "depth_current": current,
                "depth_future": future,
                "uvd_coarse": coarse,
                "uvd": local,
            }
        if timing_callback is not None:
            result["timing"] = timing
        return result
