"""RoboCasa CoT V5 with decoder-expanded bilateral hand configurations."""

from __future__ import annotations

import time
from typing import Any, Callable, List

import numpy as np
import torch
from torch import nn

from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import (
    GeometryHiddenSplit,
    Qwen_GR00T_CoT_V2,
    _require_boolean_option,
)
from starVLA.model.modules.cot_losses import (
    aggregate_cot_total_loss,
    masked_smooth_l1_loss,
    uvd_adjacent_relative_loss,
    uvd_regression_loss,
    uvd_triangle_shape_loss,
)
from starVLA.model.modules.geometric_cot_v5 import (
    GeometryTokenEmbedding,
    HandConfigurationTokenLayout,
    HandLRWDecoder,
    PackedHandLRWTargets,
    build_time_major_hand_landmark_ids,
    pack_hand_lrw_targets_time_major,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenGR00TCoTV5")
class Qwen_GR00T_CoT_V5(Qwen_GR00T_CoT_V2):
    """Decode thumb/index/wrist for both hands from 12 Qwen trajectory states."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        geometry = self.config.framework.get("geometry", {})
        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        points_per_hand = geometry.get("uvd_num_points", None)
        if points_per_hand is None:
            points_per_hand = self._trajectory_point_count()
        hand_count = int(geometry.get("uvd_hand_count", 2))
        landmark_count = int(geometry.get("landmark_count", 3))
        self.geometry_layout = HandConfigurationTokenLayout(
            depth_query_count=int(geometry.get("depth_query_count", 8)),
            uvd_points_per_hand=int(points_per_hand),
            hand_count=hand_count,
            landmark_count=landmark_count,
        )
        self.uvd_hand_count = int(self.geometry_layout.hand_count)
        self.landmark_count = int(self.geometry_layout.landmark_count)
        self.uvd_track_count = self.uvd_hand_count * self.landmark_count
        self.uvd_token_order = "time_major"
        self.geometry_tokens = GeometryTokenEmbedding(
            hidden_dim=hidden_dim,
            layout=self.geometry_layout,
        )
        self.uvd_head = HandLRWDecoder(
            hidden_dim=hidden_dim,
            landmark_count=self.landmark_count,
        )
        self.lambda_uvd_temporal = float(
            geometry.get(
                "lambda_uvd_temporal",
                geometry.get("lambda_uvd_relative", 0.1),
            )
        )
        self.lambda_uvd_shape = float(geometry.get("lambda_uvd_shape", 0.0))

    @staticmethod
    def validate_checkpoint_state_dict(state_dict) -> None:
        Qwen_GR00T_CoT_V2.validate_checkpoint_state_dict(state_dict)
        keys = tuple(str(key) for key in state_dict)
        has_geometry_or_uvd = any(
            key.startswith("geometry_tokens.")
            or ".geometry_tokens." in key
            or key.startswith("uvd_head.")
            or ".uvd_head." in key
            for key in keys
        )
        has_landmark_embedding = any(
            key.startswith("uvd_head.landmark_embedding.")
            or ".uvd_head.landmark_embedding." in key
            for key in keys
        )
        if has_geometry_or_uvd and not has_landmark_embedding:
            raise RuntimeError(
                "QwenGR00TCoTV5 checkpoints must contain "
                "uvd_head.landmark_embedding; V2/V3/V4 decoder checkpoints are "
                "intentionally incompatible."
            )

    def _trajectory_point_count(self) -> int:
        layout = getattr(self, "geometry_layout", None)
        if isinstance(layout, HandConfigurationTokenLayout):
            return int(layout.uvd_points_per_hand)
        return super()._trajectory_point_count()

    def _prepare_uvd_targets(
        self,
        examples: List[dict],
        device: torch.device,
    ) -> PackedHandLRWTargets:
        return pack_hand_lrw_targets_time_major(
            examples,
            self.geometry_layout,
            device=device,
        )

    def _compute_uvd_losses(
        self,
        pred: torch.Tensor,
        packed: PackedHandLRWTargets,
    ) -> dict[str, torch.Tensor]:
        absolute = uvd_regression_loss(pred, packed.target, packed.valid)
        temporal = uvd_adjacent_relative_loss(
            pred,
            packed.target,
            packed.valid,
            hand_count=self.uvd_track_count,
        )
        shape = uvd_triangle_shape_loss(pred, packed.target, packed.valid)
        return {
            "absolute": absolute,
            "temporal": temporal,
            "shape": shape,
            "total": (
                absolute
                + self.lambda_uvd_temporal * temporal
                + self.lambda_uvd_shape * shape
            ),
        }

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        qwen_inputs, native_attention_mask = self._build_native_inputs(
            examples,
            inference=False,
        )
        packed = self._prepare_uvd_targets(
            examples,
            qwen_inputs["input_ids"].device,
        )
        split = self._run_geometry_backbone(qwen_inputs)
        depth_current, depth_future, uvd = self._decode_geometry(
            split, qwen_inputs
        )
        condition, condition_mask = self._build_action_condition(
            split,
            native_attention_mask=native_attention_mask,
        )
        action_loss = self._action_loss(
            condition, condition_mask, examples
        )

        device = split.native.device
        depth_current_target = torch.as_tensor(
            np.stack([example["depth_current"] for example in examples]),
            device=device,
        )
        depth_future_target = torch.as_tensor(
            np.stack([example["depth_future"] for example in examples]),
            device=device,
        )
        depth_current_valid = torch.as_tensor(
            np.stack([example["depth_current_valid"] for example in examples]),
            device=device,
        )
        depth_future_valid = torch.as_tensor(
            np.stack([example["depth_future_valid"] for example in examples]),
            device=device,
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
            "uvd_temporal_loss": uvd_losses["temporal"],
            "uvd_shape_loss": uvd_losses["shape"],
            "total_loss": total_loss,
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        timing_callback = kwargs.pop("timing_callback", None)
        return_geometry = _require_boolean_option(
            kwargs.pop("return_geometry", False),
            name="return_geometry",
        )
        timing: dict[str, float] = {}

        def timed(name: str, fn: Callable[[], Any]) -> Any:
            return timing_callback(name, fn) if timing_callback is not None else fn()

        preprocess_start = time.perf_counter()
        qwen_inputs, native_attention_mask = self._build_native_inputs(
            examples, inference=True
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
            native_attention_mask=native_attention_mask,
        )
        state = None
        if "state" in examples[0] and self.config.framework.action_model.get(
            "state_dim", 0
        ):
            state = torch.as_tensor(
                np.asarray([example["state"] for example in examples]),
                device=condition.device,
                dtype=condition.dtype,
            )
            state = state[..., : int(self.config.framework.action_model.state_dim)]

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
        normalized_actions = actions.detach().float().cpu().numpy()
        timing["output_transfer_ms"] = (
            time.perf_counter() - output_start
        ) * 1000.0
        result = {"normalized_actions": normalized_actions}
        if return_geometry:
            depth_current, depth_future, uvd = self._decode_geometry(
                split,
                qwen_inputs,
                timing_callback=timing_callback,
            )
            expected_points = int(self.geometry_layout.output_point_count)
            if int(uvd.shape[1]) != expected_points:
                raise RuntimeError(
                    f"V5 decoder returned {uvd.shape[1]} points, expected "
                    f"{expected_points}"
                )
            points_per_time = self.uvd_track_count
            uvd_time = torch.linspace(
                0.0,
                1.0,
                int(self.geometry_layout.uvd_points_per_hand),
                device=uvd.device,
                dtype=torch.float32,
            ).repeat_interleave(points_per_time)
            hand_ids, landmark_ids = build_time_major_hand_landmark_ids(
                self.geometry_layout,
                device=uvd.device,
            )
            result["geometry"] = {
                "depth_current": depth_current,
                "depth_future": depth_future,
                "uvd": uvd,
                "uvd_time": uvd_time.unsqueeze(0).expand(uvd.shape[0], -1),
                "uvd_hand_ids": hand_ids.unsqueeze(0).expand(uvd.shape[0], -1),
                "uvd_landmark_ids": landmark_ids.unsqueeze(0).expand(
                    uvd.shape[0], -1
                ),
            }
        if timing_callback is not None:
            result["timing"] = timing
        return result


__all__ = ["Qwen_GR00T_CoT_V5"]
