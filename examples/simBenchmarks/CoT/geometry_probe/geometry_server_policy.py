"""Start the geometry-enabled websocket policy server."""

from __future__ import annotations

import argparse
import logging

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from examples.simBenchmarks.CoT.geometry_probe.geometry_policy_wrapper import (
    GeometryProbePolicyWrapper,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--idle_timeout", type=int, default=1800)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_argparser().parse_args()
    wrapper = GeometryProbePolicyWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
    )
    logging.info("Starting geometry probe server: metadata=%s", wrapper.metadata)
    WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
