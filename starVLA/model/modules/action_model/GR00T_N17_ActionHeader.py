"""GR00T N1.7-style flow-matching action head.

N1.7 retains the N1.6 state/action DiT path and adds an optional VLM
self-attention refinement stage after VLLN. State history is flattened into
one state token by the shared N1.6 implementation.
"""

from __future__ import annotations

from torch import nn

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import (
    SelfAttentionTransformer,
)
from starVLA.model.modules.action_model.GR00T_N16_ActionHeader import (
    FlowmatchingActionHeadConfig,
    N16FlowmatchingActionHead,
    _config_dict,
    _config_get,
)

__all__ = [
    "FlowmatchingActionHead",
    "FlowmatchingActionHeadConfig",
    "N17FlowmatchingActionHead",
    "get_action_model",
]


class N17FlowmatchingActionHead(N16FlowmatchingActionHead):
    """N1.7 extension with state history and refined VLM conditioning."""

    def __init__(self, full_config):
        config = full_config.framework.action_model
        rtc_enabled = bool(_config_get(config, "rtc_enabled", False) or _config_get(config, "use_rtc", False))
        if rtc_enabled or _config_get(config, "rtc_config", None) is not None:
            raise NotImplementedError("RTC is not implemented by the StarVLA N1.7 action head")

        super().__init__(full_config)
        self.use_vl_self_attention = bool(_config_get(config, "use_vl_self_attention", True))
        if not self.use_vl_self_attention:
            self.vl_refinement_in = nn.Identity()
            self.vl_self_attention = None
            self.vl_refinement_out = nn.Identity()
            return

        refinement_cfg = {
            "num_attention_heads": 32,
            "attention_head_dim": 64,
            "output_dim": 2048,
            "num_layers": 4,
            "dropout": 0.1,
            "attention_bias": True,
            "activation_fn": "gelu-approximate",
            "upcast_attention": False,
            "final_dropout": True,
            "positional_embeddings": None,
        }
        supplied_cfg = _config_dict(_config_get(config, "vl_self_attention_cfg", None))
        requested_width = supplied_cfg.pop("hidden_size", None)
        refinement_cfg.update(supplied_cfg)
        refinement_width = int(refinement_cfg["num_attention_heads"]) * int(
            refinement_cfg["attention_head_dim"]
        )
        if requested_width is not None and int(requested_width) != refinement_width:
            raise ValueError(
                "vl_self_attention hidden_size must equal num_attention_heads * attention_head_dim "
                f"({refinement_width}), got {requested_width}"
            )
        refinement_cfg["output_dim"] = refinement_width

        self.vl_refinement_in = (
            nn.Identity()
            if refinement_width == self.backbone_embedding_dim
            else nn.Linear(self.backbone_embedding_dim, refinement_width)
        )
        self.vl_self_attention = SelfAttentionTransformer(**refinement_cfg)
        self.vl_refinement_out = (
            nn.Identity()
            if refinement_width == self.backbone_embedding_dim
            else nn.Linear(refinement_width, self.backbone_embedding_dim)
        )

    def process_vlm_features(self, vl_embs, encoder_attention_mask=None):
        vl_embs = self.vlln(vl_embs)
        if self.vl_self_attention is None:
            return vl_embs
        vl_embs = self.vl_refinement_in(vl_embs)
        vl_embs = self.vl_self_attention(vl_embs, attention_mask=encoder_attention_mask)
        return self.vl_refinement_out(vl_embs)


FlowmatchingActionHead = N17FlowmatchingActionHead


def get_action_model(config=None):
    return N17FlowmatchingActionHead(full_config=config)
