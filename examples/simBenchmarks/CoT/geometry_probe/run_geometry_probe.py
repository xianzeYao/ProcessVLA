"""Run a 50-sample LIBERO depth/UVD decode probe."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.simBenchmarks.CoT.geometry_probe.dataset_probe import LiberoRerenderStore
from examples.simBenchmarks.CoT.geometry_probe.probe_utils import (
    build_latency_fields,
    build_sample_plan,
    masked_depth_metrics,
    summarize_latencies,
    uvd_metrics,
)
from examples.simBenchmarks.CoT.geometry_probe.visualization import (
    save_sample_bundle,
    save_sample_figure,
    save_summary,
)


DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=["libero", "libero_plus"], default="libero")
    parser.add_argument("--gt-mode", choices=["dataset", "rollout"], default="dataset")
    parser.add_argument("--dataset-root", default="/root/data/yxz/datasets/libero_rerender")
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--uvd-num-points", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--uvd-depth-scale", type=float, default=1.0)
    parser.add_argument("--video-backend", default="torchvision_av", choices=["torchvision_av", "decord", "pyav"])
    parser.add_argument("--video-thread-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="/root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v1_geometry_probe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-warmup", action="store_true", help="include warmup in latency aggregates")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _squeeze_geometry(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[0] != 1:
        raise ValueError(f"geometry field {name!r} must have batch size 1, got {array.shape}")
    array = array[0]
    if name.startswith("depth") and array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return array


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    keys = sorted({key for row in rows for key in row})
    output: dict[str, float | int] = {}
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                values.append(float(value))
        if values:
            output[key] = float(np.mean(values))
    return output


def _build_request(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "examples": [sample["example"]],
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": 10,
        "return_geometry": True,
    }
    if args.unnorm_key is not None:
        payload["unnorm_key"] = args.unnorm_key
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.gt_mode != "dataset":
        raise NotImplementedError(
            "GT_MODE=rollout is reserved for the simulator runner; this first probe uses dataset-aligned GT."
        )
    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    if not suites:
        raise ValueError("--suites must contain at least one suite")

    store = LiberoRerenderStore(
        args.dataset_root,
        suites,
        image_size=args.image_size,
        horizon=args.horizon,
        uvd_num_points=args.uvd_num_points,
        uvd_depth_scale=args.uvd_depth_scale,
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
        "gt_mode": args.gt_mode,
        "dataset_root": str(args.dataset_root),
        "suites": suites,
        "num_samples": int(args.num_samples),
        "horizon": int(args.horizon),
        "uvd_num_points": int(args.uvd_num_points or math.floor(0.3 * args.horizon) + 2),
        "image_size": int(args.image_size),
        "video_backend": args.video_backend,
        "video_thread_count": int(args.video_thread_count),
        "seed": int(args.seed),
    }
    if args.dry_run:
        print(json.dumps({"config": config, "plan": [item.__dict__ for item in plan]}, indent=2))
        return {"config": config, "samples": len(plan), "dry_run": True}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "sample_plan.json"
    plan_path.write_text(json.dumps([item.__dict__ for item in plan], indent=2), encoding="utf-8")
    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    logging.info("Connected to geometry server: %s", metadata)
    latency_values: list[float] = []
    metric_rows: list[dict[str, Any]] = []
    suite_counts: dict[str, int] = {suite: 0 for suite in suites}
    jsonl_path = output_dir / "samples.jsonl"

    try:
        for sample_index, ref in enumerate(plan):
            sample = store.load_sample(ref)
            request = _build_request(sample, args)
            start = time.perf_counter()
            response = client.predict_action(request)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            data = response.get("data", response)
            if "geometry" not in data:
                raise KeyError(f"geometry output missing; response keys={list(data)}")
            latency_fields = build_latency_fields(
                elapsed_ms,
                data.get("timing"),
            )
            geometry = data["geometry"]
            prediction = {
                "depth_current": _squeeze_geometry(geometry["depth_current"], name="depth_current"),
                "depth_future": _squeeze_geometry(geometry["depth_future"], name="depth_future"),
                "uvd": _squeeze_geometry(geometry["uvd"], name="uvd"),
            }
            metrics = {
                "depth_current": masked_depth_metrics(
                    prediction["depth_current"], sample["depth_current"], sample["depth_current_valid"]
                ),
                "depth_future": masked_depth_metrics(
                    prediction["depth_future"], sample["depth_future"], sample["depth_future_valid"]
                ),
                "uvd": uvd_metrics(prediction["uvd"], sample["uvd"], sample["uvd_valid_mask"]),
            }
            flat_metrics = {
                **{f"depth_current_{key}": value for key, value in metrics["depth_current"].items()},
                **{f"depth_future_{key}": value for key, value in metrics["depth_future"].items()},
                **{f"uvd_{key}": value for key, value in metrics["uvd"].items()},
            }
            bundle_path = save_sample_bundle(
                output_dir,
                sample_index,
                sample=sample,
                prediction=prediction,
                metrics=metrics,
                latency_ms=elapsed_ms,
                latency_fields=latency_fields,
            )
            figure_path = save_sample_figure(
                output_dir / f"sample_{sample_index:04d}.png",
                sample=sample,
                prediction=prediction,
                metrics=metrics,
                latency_ms=elapsed_ms,
            )
            latency_values.append(float(elapsed_ms))
            metric_rows.append(flat_metrics)
            suite_counts[ref.suite] += 1
            row = {
                **sample["metadata"],
                "sample_index": sample_index,
                **latency_fields,
                "warmup": sample_index == 0,
                "bundle_path": str(bundle_path),
                "figure_path": str(figure_path),
                "metrics": flat_metrics,
            }
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=_json_default) + "\n")
            logging.info(
                "sample=%d/%d suite=%s episode=%d frame=%d latency=%.2fms uvd_mae=%.6f",
                sample_index + 1,
                len(plan),
                ref.suite,
                ref.episode_id,
                ref.frame_index,
                elapsed_ms,
                flat_metrics.get("uvd_uvd_mae", float("nan")),
            )
    finally:
        client.close()

    warmup = latency_values[0] if latency_values else None
    latency_summary = summarize_latencies(
        latency_values,
        warmup_ms=None if args.keep_warmup else warmup,
    )
    summary = {
        "config": config,
        "server_metadata": metadata,
        "suite_counts": suite_counts,
        "samples": len(plan),
        "latency": latency_summary,
        "metrics": _mean_metrics(metric_rows),
    }
    save_summary(
        output_dir / "summary.json",
        config=config,
        latency=latency_summary,
        metrics={"mean": _mean_metrics(metric_rows), "suite_counts": suite_counts},
        samples=len(plan),
    )
    (output_dir / "server_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(parse_args())
