"""V2 geometry tokens aligned to frozen DA3 future-image features."""

from __future__ import annotations

from typing import Any, List

import torch
from torch import nn

from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import (
    GeometryHiddenSplit,
    Qwen_GR00T_CoT_V2,
    _cast_to_module_dtype,
    _require_boolean_option,
)
from starVLA.model.modules.da3_feature_alignment import (
    FrozenDA3FeatureTeacher,
    cosine_feature_alignment_loss,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


def validate_da3_experiment(
    alignment: Any,
    *,
    future_query_count: int,
    enable_current_depth: bool,
    enable_future_tokens: bool,
    reconstruct_future_depth: bool,
    include_depth_in_action_condition: bool,
) -> None:
    """Validate the fixed RQ3 feature-alignment experiment contract."""

    if not _require_boolean_option(alignment.get("enabled", False), name="DA3 enabled"):
        raise ValueError("DA3 feature alignment must be enabled")
    weight = float(alignment.get("loss_weight", 0.15))
    if weight < 0.0:
        raise ValueError(f"DA3 alignment loss weight must be non-negative, got {weight}")
    if int(alignment.get("selected_layer", -1)) != 23:
        raise ValueError("DA3 feature alignment must supervise layer 23")
    if int(alignment.get("feature_dim", -1)) != 2048:
        raise ValueError("DA3 layer-23 feature dimension must be 2048")
    pooling_grid = tuple(int(value) for value in alignment.get("pooling_grid", ()))
    if len(pooling_grid) != 2 or pooling_grid[0] * pooling_grid[1] != 8:
        raise ValueError("DA3 pooling grid must contain exactly eight cells")
    if int(future_query_count) != 8:
        raise ValueError("DA3 feature alignment requires eight future latent queries")
    if enable_current_depth:
        raise ValueError("current-depth tokens must be disabled for DA3 feature alignment")
    if not enable_future_tokens:
        raise ValueError("eight future latent query positions must be enabled")
    if reconstruct_future_depth:
        raise ValueError("numerical future-depth reconstruction must be disabled")
    if include_depth_in_action_condition:
        raise ValueError("future latent tokens must not enter the Action Expert directly")


@FRAMEWORK_REGISTRY.register("QwenGR00TCoTV2DA3")
class Qwen_GR00T_CoT_V2_DA3(Qwen_GR00T_CoT_V2):
    """Feature-alignment-only V2 variant for the RoboCasa RQ3 experiment."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        geometry = self.config.framework.get("geometry", {})
        alignment = geometry.get("da3_feature_alignment", {})
        reconstruct_future_depth = _require_boolean_option(
            geometry.get("reconstruct_future_depth", False),
            name="reconstruct_future_depth",
        )
        validate_da3_experiment(
            alignment,
            future_query_count=self.geometry_layout.depth_query_count,
            enable_current_depth=self.geometry_layout.enable_current_depth,
            enable_future_tokens=self.geometry_layout.enable_future_depth,
            reconstruct_future_depth=reconstruct_future_depth,
            include_depth_in_action_condition=self.include_depth_in_action_condition,
        )
        if self.lambda_depth_current != 0.0 or self.lambda_depth_future != 0.0:
            raise ValueError(
                "numerical current/future depth loss weights must both be zero for DA3 alignment"
            )

        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        self.da3_feature_dim = int(alignment.get("feature_dim", 2048))
        self.lambda_da3_feature_alignment = float(
            alignment.get("loss_weight", 0.15)
        )
        self.da3_alignment_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.da3_feature_dim),
            nn.GELU(),
            nn.Linear(self.da3_feature_dim, self.da3_feature_dim),
        )

        # The V2 constructor creates numerical depth modules. This isolated
        # variant removes them from parameters/state_dict and never executes them.
        self.depth_attention_pool = None
        self.depth_decoder = None
        self.wrist_depth_decoder = None
        self._da3_teacher = FrozenDA3FeatureTeacher(
            model_path=alignment.get("model_path", ""),
            source_path=alignment.get("source_path", ""),
            selected_layer=int(alignment.get("selected_layer", 23)),
            feature_dim=self.da3_feature_dim,
            image_size=int(alignment.get("image_size", 224)),
            pool_hw=tuple(int(value) for value in alignment.get("pooling_grid", (2, 4))),
        )

    @staticmethod
    def validate_checkpoint_state_dict(state_dict) -> None:
        """DA3 checkpoints intentionally contain no V2 depth decoder/pool."""

    def _decode_geometry(
        self,
        split: GeometryHiddenSplit,
        qwen_inputs: dict,
        *,
        timing_callback=None,
    ) -> tuple[None, None, torch.Tensor]:
        if timing_callback is None:
            uvd = self._predict_uvd(split.uvd)
        else:
            uvd = timing_callback("uvd_head_ms", lambda: self._predict_uvd(split.uvd))
        return None, None, uvd

    def _future_images(self, examples: List[dict]) -> list[Any]:
        missing = [index for index, example in enumerate(examples) if "future_image" not in example]
        if missing:
            raise KeyError(
                "DA3 feature alignment requires one future_image per training example; "
                f"missing batch indices={missing}"
            )
        return [example["future_image"] for example in examples]

    def _teacher_targets(
        self,
        examples: List[dict],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return self._da3_teacher.extract(self._future_images(examples), device=device)

    def forward(
        self,
        examples: List[dict] = None,
        *,
        capture_depth_token_gradients: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        qwen_inputs, native_attention_mask = self._build_native_inputs(
            examples, inference=False
        )
        packed = self._prepare_uvd_targets(examples, qwen_inputs["input_ids"].device)
        split = self._run_geometry_backbone(qwen_inputs)
        uvd = self._predict_uvd(split.uvd)
        uvd_losses = self._compute_uvd_losses(uvd, packed)

        student_features = self.da3_alignment_projector(
            _cast_to_module_dtype(split.depth_future, self.da3_alignment_projector)
        )
        expected_student = (
            len(examples),
            8,
            self.da3_feature_dim,
        )
        if tuple(student_features.shape) != expected_student:
            raise ValueError(
                f"projected DA3 student shape {tuple(student_features.shape)}, "
                f"expected {expected_student}"
            )
        teacher_features = self._teacher_targets(examples, device=split.native.device)
        da3_loss = cosine_feature_alignment_loss(student_features, teacher_features)

        condition, condition_mask = self._build_action_condition(
            split,
            native_attention_mask=native_attention_mask,
        )
        action_loss = self._action_loss(condition, condition_mask, examples)
        uvd_loss = uvd_losses["total"]
        total_loss = (
            self.lambda_action * action_loss
            + self.lambda_uvd * uvd_loss
            + self.lambda_da3_feature_alignment * da3_loss
        )
        zero_depth_loss = action_loss.new_zeros(())
        output = {
            "action_loss": action_loss,
            "depth_current_loss": zero_depth_loss,
            "depth_future_loss": zero_depth_loss,
            "uvd_loss": uvd_loss,
            "uvd_absolute_loss": uvd_losses["absolute"],
            "uvd_relative_loss": uvd_losses["relative"],
            "da3_feature_alignment_loss": da3_loss,
            "total_loss": total_loss,
        }
        if capture_depth_token_gradients:
            output["_probe_depth_future_tokens"] = split.depth_future
            output["_probe_wrist_depth_future_tokens"] = split.depth_future
            output["_probe_depth_future_query"] = (
                self.geometry_tokens.future_depth_queries
            )
            output["_probe_wrist_depth_future_query"] = (
                self.geometry_tokens.future_depth_queries
            )
        return output
