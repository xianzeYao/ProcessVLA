"""Measure action-only inference latency on a fixed random LIBERO sample plan."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.simBenchmarks.CoT.geometry_probe.dataset_probe import LiberoRerenderStore
from examples.simBenchmarks.CoT.geometry_probe.probe_utils import (
    build_latency_fields,
    build_sample_plan,
    summarize_latencies,
)


DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=["libero", "libero_plus"], default="libero")
    parser.add_argument("--dataset-root", default="/root/data/yxz/datasets/libero_rerender")
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--video-backend",
        default="torchvision_av",
        choices=["torchvision_av", "decord", "pyav"],
    )
    parser.add_argument("--video-thread-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10094)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--keep-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def run(args: argparse.Namespace) -> dict[str, Any]:
    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    if not suites:
        raise ValueError("--suites must contain at least one suite")

    store = LiberoRerenderStore(
        args.dataset_root,
        suites,
        image_size=args.image_size,
        horizon=args.horizon,
        video_backend=args.video_backend,
        video_backend_kwargs={"thread_count": int(args.video_thread_count)},
    )
    plan = build_sample_plan(
        store.episode_refs(),
        num_samples=args.num_samples,
        horizon=args.horizon,
        seed=args.seed,
    )
    config = {
        "bench": args.bench,
        "dataset_root": str(args.dataset_root),
        "suites": suites,
        "num_samples": int(args.num_samples),
        "horizon": int(args.horizon),
        "image_size": int(args.image_size),
        "video_backend": args.video_backend,
        "video_thread_count": int(args.video_thread_count),
        "seed": int(args.seed),
        "action_model_inference_timesteps": 4,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_plan.json").write_text(
        json.dumps([item.__dict__ for item in plan], indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps({"config": config, "plan": [item.__dict__ for item in plan]}, indent=2))
        return {"config": config, "samples": len(plan), "dry_run": True}

    client = WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    latencies: list[float] = []
    actions: list[np.ndarray] = []
    jsonl_path = output_dir / "samples.jsonl"

    try:
        for sample_index, ref in enumerate(plan):
            sample = store.load_sample(ref)
            example = dict(sample["example"])
            # Match the standard LIBERO eval contract: no proprioceptive state.
            example.pop("state", None)
            request = {"examples": [example], "return_timing": True}
            if args.unnorm_key is not None:
                request["unnorm_key"] = args.unnorm_key

            start = time.perf_counter()
            response = client.predict_action(request)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            data = response.get("data", response)
            if "actions" not in data:
                raise KeyError(f"actions missing from response; keys={list(data)}")
            action = np.asarray(data["actions"], dtype=np.float32)
            if action.ndim != 3 or action.shape[0] != 1:
                raise ValueError(f"expected actions with shape [1,T,D], got {action.shape}")

            actions.append(action[0])
            latency_fields = build_latency_fields(elapsed_ms, data.get("timing"))
            latencies.append(float(elapsed_ms))
            row = {
                **sample["metadata"],
                "sample_index": int(sample_index),
                **latency_fields,
                "action_shape": list(action.shape[1:]),
                "warmup": sample_index == 0,
            }
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=_json_default) + "\n")

            logging.info(
                "sample=%d/%d suite=%s episode=%d frame=%d latency=%.2fms",
                sample_index + 1,
                len(plan),
                ref.suite,
                ref.episode_id,
                ref.frame_index,
                elapsed_ms,
            )
    finally:
        client.close()

    np.savez_compressed(output_dir / "actions.npz", actions=np.stack(actions, axis=0))
    latency = summarize_latencies(
        latencies,
        warmup_ms=None if args.keep_warmup else (latencies[0] if latencies else None),
    )
    summary = {
        "config": config,
        "server_metadata": server_metadata,
        "samples": len(plan),
        "latency": latency,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    (output_dir / "server_metadata.json").write_text(
        json.dumps(server_metadata, indent=2, default=_json_default), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(parse_args())
