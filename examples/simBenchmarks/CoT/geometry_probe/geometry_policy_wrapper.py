"""Diagnostic policy wrapper with one-forward action + geometry inference."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch
import time

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.dataloader.gr00t_lerobot.cot_geometry import sample_real_uvd_indices
from starVLA.training.trainer_utils.trainer_tools import resize_images


class _StageTimer:
    """Collect GPU event timings with one final synchronization."""

    def __init__(self) -> None:
        self._events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._cpu_timings: dict[str, float] = {}

    def time_gpu(self, name: str, fn):
        if not torch.cuda.is_available():
            start = time.perf_counter()
            result = fn()
            self._cpu_timings[name] = (time.perf_counter() - start) * 1000.0
            return result
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = fn()
        end_event.record()
        self._events[name] = (start_event, end_event)
        return result

    def finish(self) -> dict[str, float]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings = dict(self._cpu_timings)
        timings.update(
            {
                name: float(start.elapsed_time(end))
                for name, (start, end) in self._events.items()
            }
        )
        return timings



class GeometryProbePolicyWrapper(PolicyServerWrapper):
    """Expose optional CoT depth/UVD predictions without changing eval clients."""

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(super().metadata)
        metadata["supports_geometry"] = True
        return metadata

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        return_geometry = bool(kwargs.pop("return_geometry", False))
        if not return_geometry:
            return super().predict_action(examples, unnorm_key=unnorm_key, **kwargs)

        if type(examples) is not list:
            examples = [examples]
        server_start = time.perf_counter()
        stage_timer = _StageTimer()
        framework = self._framework
        required = ("qwen_vl_interface", "geometry_query", "_geometry_forward")
        if any(not hasattr(framework, name) for name in required):
            raise TypeError(
                "return_geometry=True requires the QwenGR00TCoT framework; "
                f"got {type(framework).__name__}"
            )

        preprocess_start = time.perf_counter()
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        train_obs_image_size = getattr(framework.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = framework.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        attention_mask = qwen_inputs.get("attention_mask")
        outputs = stage_timer.time_gpu(
            "qwen_backbone_ms",
            lambda: framework.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            ),
        )
        last_hidden = outputs.hidden_states[-1]

        horizon = int(framework.action_horizon)
        uvd_num_points = framework.uvd_num_points
        if uvd_num_points is None:
            uvd_num_points = int(np.floor(0.3 * horizon)) + 2
        uvd_indices = sample_real_uvd_indices(0, horizon, int(uvd_num_points))
        uvd_times = torch.as_tensor(
            np.broadcast_to(uvd_indices.astype(np.float32) / float(horizon), (len(examples), len(uvd_indices))).copy(),
            device=last_hidden.device,
        )
        _, depth_current, depth_future, uvd, condition, condition_mask = framework._geometry_forward(
            qwen_inputs,
            last_hidden,
            attention_mask,
            uvd_times=uvd_times,
            uvd_num_points=int(uvd_num_points),
            timing_callback=stage_timer.time_gpu,
        )

        state = None
        action_model_cfg = framework.config.framework.action_model
        if "state" in examples[0] and action_model_cfg.get("state_dim", 0):
            state = torch.as_tensor(
                np.asarray([example["state"] for example in examples]),
                device=condition.device,
                dtype=condition.dtype,
            )
            state = state[..., : int(action_model_cfg.state_dim)]

        action_dtype = next(framework.action_model.parameters()).dtype
        condition = condition.to(dtype=action_dtype)
        if state is not None:
            state = state.to(dtype=action_dtype)
        def run_action_expert():
            return framework.action_model.predict_action(
                condition, state, encoder_attention_mask=condition_mask
            )

        actions = stage_timer.time_gpu("action_expert_ms", run_action_expert)

        action_output_transfer_start = time.perf_counter()
        normalized = actions.detach().float().cpu().numpy()
        action_output_transfer_ms = (time.perf_counter() - action_output_transfer_start) * 1000.0
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    "return_geometry=True requires unnorm_key for a multi-dataset checkpoint; "
                    f"available={self._available_unnorm_keys}"
                )
        action_unnormalize_start = time.perf_counter()
        proc = self._get_processor(effective_key)
        unnorm = np.stack(
            [proc.unapply_actions(normalized[index]) for index in range(normalized.shape[0])],
            axis=0,
        )
        action_unnormalize_ms = (time.perf_counter() - action_unnormalize_start) * 1000.0
        geometry_output_transfer_start = time.perf_counter()
        geometry = {
            "depth_current": depth_current.detach().float().cpu().numpy(),
            "depth_future": depth_future.detach().float().cpu().numpy(),
            "uvd": uvd.detach().float().cpu().numpy(),
            "uvd_time": uvd_times.detach().float().cpu().numpy(),
        }
        geometry_output_transfer_ms = (time.perf_counter() - geometry_output_transfer_start) * 1000.0
        output_transfer_ms = action_output_transfer_ms + geometry_output_transfer_ms
        timing = stage_timer.finish()
        timing["preprocess_ms"] = preprocess_ms
        timing["output_transfer_ms"] = output_transfer_ms
        timing["action_output_transfer_ms"] = action_output_transfer_ms
        timing["geometry_output_transfer_ms"] = geometry_output_transfer_ms
        timing["action_unnormalize_ms"] = action_unnormalize_ms
        timing["server_total_ms"] = (time.perf_counter() - server_start) * 1000.0
        return {"actions": unnorm, "geometry": geometry, "timing": timing}
