# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 2.0.
"""Policy server wrapper.

Encapsulates a ``baseframework`` instance plus a training-time action
un-normalizer.  Action-only clients retain the historical response shape;
optional auxiliary fields returned by a framework are forwarded as NumPy
arrays for diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor


class PolicyServerWrapper:
    """Wrap a framework for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg
        action_model_cfg = model_cfg["framework"]["action_model"]
        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )

        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}
        _, namespace = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(namespace.keys())

        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata sent at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
        }
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the framework and un-normalize actions.

        The framework may optionally return an auxiliary ``geometry`` mapping.
        Existing action-only callers receive the same ``{"actions": ...}``
        payload as before.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        out = self._predict_with_scoped_seed(examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])
        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        result: Dict[str, Any] = {"actions": unnorm}
        auxiliary = out.get("geometry")
        if auxiliary is not None:
            result["geometry"] = {
                key: self._to_numpy(value)
                for key, value in auxiliary.items()
            }
        return result

    def _predict_with_scoped_seed(self, examples: List[dict], **kwargs) -> Dict[str, Any]:
        inference_seed = kwargs.pop("inference_seed", None)
        if inference_seed is None:
            return self._framework.predict_action(examples=examples, **kwargs)

        device = self._framework_device()
        device_index = device.index if device.index is not None else 0
        devices = [device_index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.default_generator.manual_seed(inference_seed)
            if device.type == "cuda":
                with torch.cuda.device(device_index):
                    torch.cuda.manual_seed(inference_seed)
            return self._framework.predict_action(examples=examples, **kwargs)

    def _framework_device(self) -> torch.device:
        device = getattr(self._framework, "device", None)
        if device is not None:
            return torch.device(device)
        try:
            return next(self._framework.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu().numpy() if value.is_floating_point() else value.detach().cpu().numpy()
        return np.asarray(value)
