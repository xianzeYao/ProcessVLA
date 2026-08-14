"""Policy server with optional stage-level inference timing."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, List, Optional

import numpy as np

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from examples.simBenchmarks.CoT.geometry_probe.geometry_policy_wrapper import _StageTimer


class TimedPolicyServerWrapper(PolicyServerWrapper):
    """Keep the normal policy path, adding timing only when requested."""

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        return_timing = bool(kwargs.pop("return_timing", False))
        if not return_timing:
            return super().predict_action(examples, unnorm_key=unnorm_key, **kwargs)

        server_start = time.perf_counter()
        stage_timer = _StageTimer()
        output = self._predict_with_scoped_seed(
            examples,
            timing_callback=stage_timer.time_gpu,
            **kwargs,
        )
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)

        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    "timed predict_action requires unnorm_key for a multi-dataset checkpoint; "
                    f"available={self._available_unnorm_keys}"
                )

        unnormalize_start = time.perf_counter()
        proc = self._get_processor(effective_key)
        actions = np.stack(
            [proc.unapply_actions(normalized[index]) for index in range(normalized.shape[0])],
            axis=0,
        )
        action_unnormalize_ms = (time.perf_counter() - unnormalize_start) * 1000.0

        timing = stage_timer.finish()
        timing.update(output.get("timing", {}))
        timing["action_unnormalize_ms"] = action_unnormalize_ms
        timing["server_total_ms"] = (time.perf_counter() - server_start) * 1000.0
        return {"actions": actions, "timing": timing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--port", type=int, default=10094)
    parser.add_argument("--idle_timeout", type=int, default=3600)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    wrapper = TimedPolicyServerWrapper(ckpt_path=args.ckpt_path, device="cuda")
    logging.info("Starting timed policy server: metadata=%s", wrapper.metadata)
    WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
