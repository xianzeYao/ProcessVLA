"""GR00T N1.6-style flow-matching action head.

Unlike the legacy StarVLA N1.5-derived head, this head has no future query
tokens and performs no internal batch repeat. Its DiT input is exactly one
state token followed by ``action_horizon`` noisy-action tokens.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions import Beta
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import AlternateVLDiT
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    CategorySpecificMLP,
    FlowmatchingActionHeadConfig,
    MultiEmbodimentActionEncoder,
)

__all__ = [
    "FlowmatchingActionHead",
    "FlowmatchingActionHeadConfig",
    "N16FlowmatchingActionHead",
    "get_action_model",
]


_MISSING = object()


def _config_get(config: Any, name: str, default: Any = _MISSING):
    if isinstance(config, dict):
        if name in config:
            return config[name]
    elif hasattr(config, name):
        return getattr(config, name)
    if default is _MISSING:
        raise ValueError(f"action-model config is missing required field {name!r}")
    return default


def _config_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError("diffusion_model_cfg must be a mapping")


def _config_get_first(config: Any, names: tuple[str, ...], default: Any = _MISSING):
    for name in names:
        if isinstance(config, dict) and name in config:
            value = config[name]
        elif not isinstance(config, dict) and hasattr(config, name):
            value = getattr(config, name)
        else:
            continue
        if value is not None:
            return value
    if default is _MISSING:
        joined_names = " or ".join(repr(name) for name in names)
        raise ValueError(f"action-model config is missing required field {joined_names}")
    return default


class N16FlowmatchingActionHead(nn.Module):
    """Configurable N1.6-style AlternateVLDiT action head."""

    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config
        self.config = config

        diffusion_cfg = {
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "output_dim": 1024,
            "num_layers": 32,
            "dropout": 0.2,
            "attention_bias": True,
            "activation_fn": "gelu-approximate",
            "upcast_attention": False,
            "norm_type": "ada_norm",
            "norm_elementwise_affine": False,
            "final_dropout": True,
            "positional_embeddings": None,
        }
        diffusion_cfg.update(_config_dict(_config_get(config, "diffusion_model_cfg", None)))

        self.backbone_embedding_dim = int(
            _config_get(config, "backbone_embedding_dim", diffusion_cfg.get("cross_attention_dim", 1536))
        )
        cross_attention_dim = int(diffusion_cfg.get("cross_attention_dim", self.backbone_embedding_dim))
        if cross_attention_dim != self.backbone_embedding_dim:
            raise ValueError(
                "diffusion_model_cfg.cross_attention_dim must equal backbone_embedding_dim "
                f"({self.backbone_embedding_dim}), got {cross_attention_dim}"
            )
        diffusion_cfg["cross_attention_dim"] = self.backbone_embedding_dim
        attend_text_every_n_blocks = int(
            diffusion_cfg.pop(
                "attend_text_every_n_blocks",
                _config_get(config, "attend_text_every_n_blocks", 2),
            )
        )

        model_width = int(diffusion_cfg["num_attention_heads"]) * int(diffusion_cfg["attention_head_dim"])
        self.input_embedding_dim = int(_config_get(config, "input_embedding_dim", model_width))
        if self.input_embedding_dim != model_width:
            raise ValueError(
                "input_embedding_dim must equal num_attention_heads * attention_head_dim "
                f"({model_width}), got {self.input_embedding_dim}"
            )

        self.model = AlternateVLDiT(
            attend_text_every_n_blocks=attend_text_every_n_blocks,
            **diffusion_cfg,
        )
        self.action_horizon = int(_config_get(config, "action_horizon"))
        self.action_dim = int(_config_get_first(config, ("action_dim", "max_action_dim")))
        self.state_dim = int(_config_get_first(config, ("state_dim", "max_state_dim")))
        self.state_history_length = int(
            _config_get_first(config, ("state_history_length", "state_horizon"), default=1)
        )
        self.hidden_size = int(_config_get(config, "hidden_size", 1024))
        self.max_num_embodiments = int(
            _config_get_first(config, ("max_num_embodiments", "num_embodiments"), default=32)
        )
        self.num_inference_timesteps = int(_config_get(config, "num_inference_timesteps", 4))
        self.num_timestep_buckets = int(_config_get(config, "num_timestep_buckets", 1000))

        if self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        if self.state_dim <= 0 or self.state_history_length <= 0:
            raise ValueError("state_dim and state_history_length must be positive")
        if self.num_inference_timesteps <= 0:
            raise ValueError("num_inference_timesteps must be positive")

        self.vlln = (
            nn.LayerNorm(self.backbone_embedding_dim)
            if bool(_config_get(config, "use_vlln", True))
            else nn.Identity()
        )
        self.state_encoder = CategorySpecificMLP(
            num_categories=self.max_num_embodiments,
            input_dim=self.state_dim * self.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=self.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=self.max_num_embodiments,
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.add_pos_embed = bool(_config_get(config, "add_pos_embed", True))
        if self.add_pos_embed:
            max_seq_len = int(_config_get(config, "max_seq_len", self.action_horizon))
            if max_seq_len < self.action_horizon:
                raise ValueError("max_seq_len must cover action_horizon")
            self.position_embedding = nn.Embedding(max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.noise_s = float(_config_get(config, "noise_s", 0.999))
        self.beta_dist = Beta(
            float(_config_get(config, "noise_beta_alpha", 1.5)),
            float(_config_get(config, "noise_beta_beta", 1.0)),
        )

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
        return (1.0 - sample) * self.noise_s

    @staticmethod
    def _sample_noise(actions):
        return torch.randn_like(actions)

    def _normalize_embodiment_id(self, embodiment_id, batch_size, device):
        if embodiment_id is None:
            embodiment_id = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            embodiment_id = torch.as_tensor(embodiment_id, device=device, dtype=torch.long)
        if embodiment_id.shape != (batch_size,):
            raise ValueError(f"embodiment_id must have shape ({batch_size},)")
        if torch.any(embodiment_id < 0) or torch.any(embodiment_id >= self.max_num_embodiments):
            raise ValueError(f"embodiment_id values must be in [0, {self.max_num_embodiments})")
        return embodiment_id

    def _normalize_state(self, state, batch_size):
        if state is None:
            raise ValueError("state is required for the N1.6 action head")
        if state.shape[0] != batch_size:
            raise ValueError(f"state batch size must be {batch_size}")
        if state.ndim == 2:
            if self.state_history_length != 1 or state.shape[1] != self.state_dim:
                expected = self.state_dim * self.state_history_length
                if state.shape[1] != expected:
                    raise ValueError(f"flattened state width must be {expected}")
            state = state.reshape(batch_size, 1, -1)
        elif state.ndim == 3:
            expected = (batch_size, self.state_history_length, self.state_dim)
            if tuple(state.shape) != expected:
                raise ValueError(f"state history must have shape {expected}, got {tuple(state.shape)}")
            state = state.reshape(batch_size, 1, -1)
        else:
            raise ValueError("state must have shape [B, D] or [B, history, D]")
        return state

    def _normalize_vlm_inputs(self, vl_embs, encoder_attention_mask, image_mask):
        if vl_embs.ndim != 3 or vl_embs.shape[-1] != self.backbone_embedding_dim:
            raise ValueError(
                f"vl_embs must have shape [B, L, {self.backbone_embedding_dim}], got {tuple(vl_embs.shape)}"
            )
        expected_mask_shape = vl_embs.shape[:2]
        if image_mask is None:
            raise ValueError("image_mask is required")
        if image_mask.shape != expected_mask_shape:
            raise ValueError(f"image_mask must have shape {tuple(expected_mask_shape)}")
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(expected_mask_shape, device=vl_embs.device, dtype=torch.bool)
        elif encoder_attention_mask.shape != expected_mask_shape:
            raise ValueError(f"encoder_attention_mask must have shape {tuple(expected_mask_shape)}")
        encoder_attention_mask = encoder_attention_mask.to(device=vl_embs.device, dtype=torch.bool)
        image_mask = image_mask.to(device=vl_embs.device, dtype=torch.bool)
        return (
            self.process_vlm_features(vl_embs, encoder_attention_mask=encoder_attention_mask),
            encoder_attention_mask,
            image_mask,
        )

    def _normalize_actions(self, actions, batch_size):
        if actions.ndim != 3 or actions.shape[0] != batch_size:
            raise ValueError("actions must have shape [B, action_horizon, action_dim]")
        if actions.shape[1] != self.action_horizon:
            raise ValueError(f"action horizon must be {self.action_horizon}, got {actions.shape[1]}")
        if actions.shape[2] != self.action_dim:
            raise ValueError(f"action width must be {self.action_dim}, got {actions.shape[2]}")

    def _normalize_action_mask(self, action_mask, actions):
        if action_mask is None:
            return torch.ones_like(actions)
        action_mask = torch.as_tensor(action_mask, device=actions.device)
        if action_mask.shape == (actions.shape[0], self.action_dim):
            action_mask = action_mask[:, None, :].expand_as(actions)
        elif action_mask.shape != actions.shape:
            raise ValueError(
                f"action_mask must have shape {tuple(actions.shape)} or "
                f"({actions.shape[0]}, {self.action_dim})"
            )
        action_mask = action_mask.to(dtype=actions.dtype)
        if not torch.isfinite(action_mask).all() or not ((action_mask == 0) | (action_mask == 1)).all():
            raise ValueError("action_mask values must be finite binary values")
        if action_mask.sum() <= 0:
            raise ValueError("action_mask must select at least one action element")
        return action_mask

    def process_vlm_features(self, vl_embs, encoder_attention_mask=None):
        return self.vlln(vl_embs)

    def _encode_state(self, state, embodiment_id):
        return self.state_encoder(state, embodiment_id)

    def _build_action_model_input(self, noisy_actions, state, timestep, embodiment_id):
        action_features = self.action_encoder(noisy_actions, timestep, embodiment_id)
        if self.add_pos_embed:
            position_ids = torch.arange(self.action_horizon, device=noisy_actions.device)
            action_features = action_features + self.position_embedding(position_ids)[None]
        state_features = self._encode_state(state, embodiment_id)
        return torch.cat((state_features, action_features), dim=1)

    def _discretize_time(self, time):
        return (time * self.num_timestep_buckets).long().clamp(max=self.num_timestep_buckets - 1)

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        image_mask=None,
        embodiment_id=None,
        action_mask=None,
    ):
        batch_size = vl_embs.shape[0]
        self._normalize_actions(actions, batch_size)
        state = self._normalize_state(state, batch_size).to(device=actions.device, dtype=actions.dtype)
        embodiment_id = self._normalize_embodiment_id(embodiment_id, batch_size, actions.device)
        vl_embs, encoder_attention_mask, image_mask = self._normalize_vlm_inputs(
            vl_embs, encoder_attention_mask, image_mask
        )
        action_mask = self._normalize_action_mask(action_mask, actions)

        noise = self._sample_noise(actions)
        time = self.sample_time(batch_size, device=actions.device, dtype=actions.dtype)
        broadcast_time = time[:, None, None]
        noisy_actions = (1.0 - broadcast_time) * noise + broadcast_time * actions
        target_velocity = actions - noise
        timestep = self._discretize_time(time)

        action_model_input = self._build_action_model_input(
            noisy_actions, state, timestep, embodiment_id
        )
        model_output = self.model(
            hidden_states=action_model_input,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            image_mask=image_mask,
            timestep=timestep,
        )
        pred_velocity = self.action_decoder(model_output, embodiment_id)[:, -self.action_horizon :]
        squared_error = (pred_velocity - target_velocity).square() * action_mask
        return squared_error.sum() / action_mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        image_mask=None,
        embodiment_id=None,
        action_mask=None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        state = self._normalize_state(state, batch_size).to(device=vl_embs.device, dtype=vl_embs.dtype)
        embodiment_id = self._normalize_embodiment_id(embodiment_id, batch_size, vl_embs.device)
        vl_embs, encoder_attention_mask, image_mask = self._normalize_vlm_inputs(
            vl_embs, encoder_attention_mask, image_mask
        )
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=vl_embs.device,
            dtype=vl_embs.dtype,
        )
        action_mask = self._normalize_action_mask(action_mask, actions)
        actions = actions * action_mask
        dt = 1.0 / self.num_inference_timesteps

        for step in range(self.num_inference_timesteps):
            time = torch.full(
                (batch_size,),
                step / float(self.num_inference_timesteps),
                device=actions.device,
                dtype=actions.dtype,
            )
            timestep = self._discretize_time(time)
            action_model_input = self._build_action_model_input(actions, state, timestep, embodiment_id)
            model_output = self.model(
                hidden_states=action_model_input,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                image_mask=image_mask,
                timestep=timestep,
            )
            pred_velocity = self.action_decoder(model_output, embodiment_id)[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity * action_mask
        return actions * action_mask

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


FlowmatchingActionHead = N16FlowmatchingActionHead


def get_action_model(config=None):
    return N16FlowmatchingActionHead(full_config=config)
