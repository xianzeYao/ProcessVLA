# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025].
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
# Action repeat is inspired by CogACT


from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Whether to tune the diffusion model."})
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


@dataclass(frozen=True)
class FlowStepDiagnostics:
    """Detached tensors recorded around one Euler integration step."""

    step_index: int
    t_cont: float
    t_discretized: int
    x_before: torch.Tensor
    pred_velocity: torch.Tensor
    x_after: torch.Tensor


class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config

        # ------------------------------------------------------------------
        # DiT architecture selection
        #   action_model_type: "DiT-B" | "DiT-L"
        #     DiT-B → input_embedding_dim=768,  heads=12, head_dim=64
        #     DiT-L → input_embedding_dim=1536, heads=32, head_dim=48
        #   diffusion_model_cfg overrides/extends the base DiT shape.
        #   In particular, diffusion_model_cfg.cross_attention_dim MUST be
        #   set by the framework to match the VLM hidden size BEFORE calling
        #   get_action_model(), e.g.:
        #       cfg.framework.action_model.diffusion_model_cfg.cross_attention_dim
        #           = vlm.model.config.hidden_size
        # ------------------------------------------------------------------
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)

        # ------------------------------------------------------------------
        # Action horizon (chunk length sent to the DiT)
        #   Single source of truth: `action_horizon` (e.g. 8).
        #   Legacy YAMLs that only provide `future_action_window_size` are
        #   normalised to `action_horizon` upstream by
        #   `share_tools.apply_config_compat`, so this code never touches
        #   the legacy alias.
        # ------------------------------------------------------------------
        self.action_horizon = int(config.action_horizon)

        # ------------------------------------------------------------------
        # Action / state dimensions
        #   action_dim: DoF of the robot action (e.g. 7 for 6-DoF + gripper)
        #   state_dim:  proprioception dimension; set to 0/None to disable
        #               the state_encoder branch entirely.
        # ------------------------------------------------------------------
        self.action_dim = config.action_dim

        # ------------------------------------------------------------------
        # Inference denoising steps
        #   num_inference_timesteps: Euler steps during predict_action().
        #   Typically 4–10; fewer = faster but less accurate.
        # ------------------------------------------------------------------
        self.num_inference_timesteps = config.num_inference_timesteps

        # ------------------------------------------------------------------
        # hidden_size: intermediate MLP width for state_encoder / action_decoder.
        #   Decoupled from input_embedding_dim so you can use a smaller hidden
        #   for the MLP without changing the DiT latent size.
        # ------------------------------------------------------------------
        self.hidden_size = config.hidden_size

        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # ------------------------------------------------------------------
        # future_tokens: learnable query tokens prepended before the action
        #   sequence so the DiT has dedicated "planning" slots.
        #   num_target_vision_tokens controls how many such tokens are added.
        # ------------------------------------------------------------------
        if config.num_target_vision_tokens > 0:
            self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        else:
            self.future_tokens = None

        # ------------------------------------------------------------------
        # Positional embedding over the action sequence
        #   add_pos_embed: whether to add sinusoidal-style learned PE
        #   max_seq_len:   max supported action sequence length
        # ------------------------------------------------------------------
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # ------------------------------------------------------------------
        # Flow-matching noise schedule (Beta distribution)
        #   noise_beta_alpha / noise_beta_beta: Beta(α, β) shape params.
        #   noise_s: upper-clip of the sampled value so t ∈ [0, noise_s].
        #   num_timestep_buckets: discretise continuous t into N buckets for
        #     the timestep encoder inside DiT.
        # ------------------------------------------------------------------
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(
        self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None
    ):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, action_dim)
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        sa_features = [action_features]
        if self.future_tokens is not None:
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_features.insert(0, future_tokens)
        if state_features is not None:
            sa_features.insert(0, state_features)
        sa_embs = torch.cat(sa_features, dim=1) if len(sa_features) > 1 else action_features

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def _predict_velocity_from_features(
        self,
        actions: torch.Tensor,
        *,
        timesteps_tensor: torch.Tensor,
        vl_embs: torch.Tensor,
        state_features: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        action_features = self.action_encoder(actions, timesteps_tensor)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_features = [action_features]
        if self.future_tokens is not None:
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_features.insert(0, future_tokens)
        if state_features is not None:
            sa_features.insert(0, state_features)
        sa_embs = torch.cat(sa_features, dim=1) if len(sa_features) > 1 else action_features

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timesteps_tensor,
        )
        pred = self.action_decoder(model_output)
        return pred[:, -self.action_horizon :]

    def _validate_diagnostic_condition(
        self,
        condition: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        batch_size: int,
        reference: torch.Tensor,
        name: str,
    ) -> None:
        if condition.ndim != 3 or int(condition.shape[0]) != batch_size:
            raise ValueError(
                f"{name} must have shape [B, tokens, hidden] with B={batch_size}, "
                f"got {tuple(condition.shape)}"
            )
        if int(condition.shape[2]) != int(reference.shape[2]):
            raise ValueError(
                f"{name} hidden width {int(condition.shape[2])} does not match "
                f"reference hidden width {int(reference.shape[2])}"
            )
        if condition.device != reference.device:
            raise ValueError(f"{name} device {condition.device} does not match {reference.device}")
        if condition.dtype != reference.dtype:
            raise ValueError(f"{name} dtype {condition.dtype} does not match {reference.dtype}")
        if not torch.isfinite(condition).all():
            raise ValueError(f"{name} must contain only finite values")
        if attention_mask is None:
            return
        expected_mask_shape = tuple(condition.shape[:2])
        if tuple(attention_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"{name} attention mask shape {tuple(attention_mask.shape)} "
                f"does not match {expected_mask_shape}"
            )
        if attention_mask.device != condition.device:
            raise ValueError(
                f"{name} attention mask device {attention_mask.device} "
                f"does not match {condition.device}"
            )
        if attention_mask.dtype != torch.bool:
            raise ValueError(f"{name} attention mask must have dtype bool")

    @torch.no_grad()
    def predict_velocity(
        self,
        actions: torch.Tensor,
        *,
        t_cont: float,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the action velocity at an explicitly supplied flow state."""

        expected_shape = (int(vl_embs.shape[0]), self.action_horizon, self.action_dim)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(f"actions shape {tuple(actions.shape)} does not match {expected_shape}")
        if actions.device != vl_embs.device:
            raise ValueError(f"actions device {actions.device} does not match {vl_embs.device}")
        if actions.dtype != vl_embs.dtype:
            raise ValueError(f"actions dtype {actions.dtype} does not match {vl_embs.dtype}")
        if not torch.isfinite(actions).all():
            raise ValueError("actions must contain only finite values")
        t_cont = float(t_cont)
        if not 0.0 <= t_cont <= 1.0:
            raise ValueError(f"t_cont must be in [0, 1], got {t_cont}")
        self._validate_diagnostic_condition(
            vl_embs,
            encoder_attention_mask,
            batch_size=expected_shape[0],
            reference=vl_embs,
            name="vl_embs",
        )
        state_features = self.state_encoder(state) if state is not None else None
        t_discretized = int(t_cont * self.num_timestep_buckets)
        timesteps_tensor = torch.full(
            size=(expected_shape[0],),
            fill_value=t_discretized,
            device=actions.device,
        )
        velocity = self._predict_velocity_from_features(
            actions,
            timesteps_tensor=timesteps_tensor,
            vl_embs=vl_embs,
            state_features=state_features,
            encoder_attention_mask=encoder_attention_mask,
        )
        if not torch.isfinite(velocity).all():
            raise ValueError("predicted velocity must contain only finite values")
        return velocity

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        *,
        initial_actions: torch.Tensor | None = None,
        condition_schedule: Sequence[tuple[torch.Tensor, torch.Tensor | None]] | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[FlowStepDiagnostics, ...]]:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        expected_shape = (batch_size, self.action_horizon, self.action_dim)
        diagnostic_mode = initial_actions is not None or condition_schedule is not None or return_diagnostics
        if initial_actions is None:
            actions = torch.randn(
                size=expected_shape,
                dtype=vl_embs.dtype,
                device=device,
            )
        else:
            if tuple(initial_actions.shape) != expected_shape:
                raise ValueError(
                    f"initial_actions shape {tuple(initial_actions.shape)} does not match {expected_shape}"
                )
            if initial_actions.device != device:
                raise ValueError(
                    f"initial_actions device {initial_actions.device} does not match {device}"
                )
            if initial_actions.dtype != vl_embs.dtype:
                raise ValueError(
                    f"initial_actions dtype {initial_actions.dtype} does not match {vl_embs.dtype}"
                )
            if not torch.isfinite(initial_actions).all():
                raise ValueError("initial_actions must contain only finite values")
            actions = initial_actions.clone()

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None
        if condition_schedule is not None:
            if len(condition_schedule) != num_steps:
                raise ValueError(
                    f"condition_schedule has {len(condition_schedule)} steps, expected {num_steps}"
                )
            for step_index, (step_condition, step_mask) in enumerate(condition_schedule):
                self._validate_diagnostic_condition(
                    step_condition,
                    step_mask,
                    batch_size=batch_size,
                    reference=vl_embs,
                    name=f"condition_schedule[{step_index}]",
                )
        if diagnostic_mode:
            self._validate_diagnostic_condition(
                vl_embs,
                encoder_attention_mask,
                batch_size=batch_size,
                reference=vl_embs,
                name="vl_embs",
            )
        diagnostics: list[FlowStepDiagnostics] = []

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)
            step_condition, step_mask = (
                condition_schedule[t]
                if condition_schedule is not None
                else (vl_embs, encoder_attention_mask)
            )

            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            x_before = actions
            pred_velocity = self._predict_velocity_from_features(
                actions,
                timesteps_tensor=timesteps_tensor,
                vl_embs=step_condition,
                state_features=state_features,
                encoder_attention_mask=step_mask,
            )
            if diagnostic_mode and not torch.isfinite(pred_velocity).all():
                raise ValueError(f"predicted velocity at step {t} must contain only finite values")

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
            if diagnostic_mode and not torch.isfinite(actions).all():
                raise ValueError(f"actions after step {t} must contain only finite values")
            if return_diagnostics:
                diagnostics.append(
                    FlowStepDiagnostics(
                        step_index=t,
                        t_cont=t_cont,
                        t_discretized=t_discretized,
                        x_before=x_before.detach().clone(),
                        pred_velocity=pred_velocity.detach().clone(),
                        x_after=actions.detach().clone(),
                    )
                )
        if return_diagnostics:
            return actions, tuple(diagnostics)
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHead(full_config=config)


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass
