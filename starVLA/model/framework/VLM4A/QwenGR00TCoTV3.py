"""QwenGR00T V3 with time-major LIBERO gripper landmarks."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import (
    GeometryHiddenSplit,
    Qwen_GR00T_CoT_V2,
)
from starVLA.model.modules.cot_losses import (
    aggregate_cot_total_loss,
    masked_smooth_l1_loss,
    uvd_adjacent_relative_loss,
    uvd_regression_loss,
    uvd_triangle_shape_loss,
)
from starVLA.model.modules.geometric_cot_v3 import (
    LandmarkGeometryTokenEmbedding,
    LandmarkGeometryTokenLayout,
    PackedLandmarkUVDTargets,
    append_geometry_slots,
    build_landmark_geometry_full_attention_mask,
    pack_landmark_uvd_targets_time_major,
)
from starVLA.model.modules.qwen35_geometry_forward import (
    forward_qwen35_with_geometry,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenGR00TCoTV3")
class Qwen_GR00T_CoT_V3(Qwen_GR00T_CoT_V2):
    """Predict three LIBERO landmarks at every sampled trajectory time."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        geometry = self.config.framework.get("geometry", {})
        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        time_points = geometry.get("uvd_num_points", None)
        if time_points is None:
            time_points = self._trajectory_point_count()
        self.geometry_layout = LandmarkGeometryTokenLayout(
            depth_query_count=int(geometry.get("depth_query_count", 8)),
            uvd_time_points=int(time_points),
            landmark_count=int(geometry.get("landmark_count", 3)),
        )
        self.landmark_count = int(self.geometry_layout.landmark_count)
        self.uvd_token_order = "time_major"
        self.geometry_tokens = LandmarkGeometryTokenEmbedding(
            hidden_dim=hidden_dim,
            layout=self.geometry_layout,
        )
        self.lambda_uvd_temporal = float(
            geometry.get("lambda_uvd_temporal", geometry.get("lambda_uvd_relative", 0.1))
        )
        self.lambda_uvd_shape = float(geometry.get("lambda_uvd_shape", 0.0))

    @staticmethod
    def validate_checkpoint_state_dict(state_dict) -> None:
        Qwen_GR00T_CoT_V2.validate_checkpoint_state_dict(state_dict)
        keys = tuple(str(key) for key in state_dict)
        has_geometry = any(
            key.startswith("geometry_tokens.") or ".geometry_tokens." in key
            for key in keys
        )
        has_landmark_embedding = any(
            key.startswith("geometry_tokens.landmark_embedding.")
            or ".geometry_tokens.landmark_embedding." in key
            for key in keys
        )
        if has_geometry and not has_landmark_embedding:
            raise RuntimeError(
                "QwenGR00TCoTV3 checkpoints must contain geometry_tokens.landmark_embedding; "
                "V2 geometry checkpoints are intentionally incompatible."
            )

    def _trajectory_point_count(self) -> int:
        layout = self.geometry_layout
        if isinstance(layout, LandmarkGeometryTokenLayout):
            return int(layout.uvd_time_points)
        return super()._trajectory_point_count()

    def _prepare_uvd_targets(
        self,
        examples: List[dict],
        device: torch.device,
    ) -> PackedLandmarkUVDTargets:
        return pack_landmark_uvd_targets_time_major(
            examples,
            self.geometry_layout,
            device=device,
        )

    def _compute_uvd_losses(
        self,
        pred: torch.Tensor,
        packed: PackedLandmarkUVDTargets,
    ) -> dict[str, torch.Tensor]:
        absolute = uvd_regression_loss(pred, packed.target, packed.valid)
        temporal = uvd_adjacent_relative_loss(
            pred,
            packed.target,
            packed.valid,
            hand_count=self.geometry_layout.landmark_count,
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
        full_attention_mask = build_landmark_geometry_full_attention_mask(
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
        depth_current, depth_future, uvd = self._decode_geometry(split, qwen_inputs)
        condition, condition_mask = self._build_action_condition(
            split,
            native_attention_mask=native_attention_mask,
        )
        action_loss = self._action_loss(condition, condition_mask, examples)

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


__all__ = ["Qwen_GR00T_CoT_V3"]
