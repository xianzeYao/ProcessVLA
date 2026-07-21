"""Bench-backed rollout probe for action-conditioned geometry."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.simBenchmarks.CoT.geometry_probe.probe_utils import (
    build_latency_fields,
    masked_depth_metrics,
    summarize_latencies,
    uvd_metrics,
)
from examples.simBenchmarks.CoT.geometry_probe.visualization import save_sample_bundle, save_sample_figure, save_summary
from examples.simBenchmarks.CoT.geometry_probe.sim_geometry_utils import resize_depth, sample_real_uvd_indices, transform_uvd_to_model_space


@dataclass(frozen=True)
class RolloutCandidate:
    suite: str
    task_id: int
    task_name: str
    language: str
    hdf5_path: Path
    demo_id: str
    frame_index: int
    episode_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=["libero", "libero_plus"], default="libero")
    parser.add_argument("--hdf5-root", default="/root/data/yxz/datasets/libero_original_hdf5")
    parser.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--uvd-num-points", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _allocate(total: int, suites: list[str]) -> dict[str, int]:
    q, r = divmod(int(total), len(suites))
    return {suite: q + int(i < r) for i, suite in enumerate(suites)}


def _hdf5_path(root: Path, suite: str, task_name: str) -> Path:
    path = root / suite / f"{task_name}_demo.hdf5"
    if path.exists():
        return path
    matches = list(root.glob(f"{suite}/*{task_name}*_demo.hdf5"))
    if matches:
        return matches[0]
    raise FileNotFoundError(path)


def _candidate_plan(args: argparse.Namespace) -> list[RolloutCandidate]:
    from libero.libero import benchmark

    suites = [x.strip() for x in args.suites.split(",") if x.strip()]
    counts = _allocate(args.num_samples, suites)
    rng = np.random.default_rng(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    plan: list[RolloutCandidate] = []
    for suite in suites:
        task_suite = benchmark_dict[suite]()
        candidates: list[RolloutCandidate] = []
        for task_id in range(int(task_suite.n_tasks)):
            task = task_suite.get_task(task_id)
            path = _hdf5_path(Path(args.hdf5_root), suite, task.name)
            with h5py.File(path, "r") as handle:
                for demo_id in sorted(handle["data"].keys(), key=lambda x: int(x.split("_")[-1])):
                    length = int(handle[f"data/{demo_id}/states"].shape[0])
                    for frame_index in range(max(0, length - int(args.horizon))):
                        candidates.append(
                            RolloutCandidate(
                                suite=suite,
                                task_id=task_id,
                                task_name=task.name,
                                language=str(task.language),
                                hdf5_path=path,
                                demo_id=demo_id,
                                frame_index=frame_index,
                                episode_length=length,
                            )
                        )
        requested = counts[suite]
        if requested > len(candidates):
            raise ValueError(f"suite {suite} has only {len(candidates)} valid rollout frames; need {requested}")
        if requested:
            selected = rng.choice(len(candidates), size=requested, replace=False)
            plan.extend(candidates[int(i)] for i in selected)
    rng.shuffle(plan)
    return plan


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(1.0 - float(quat[3] ** 2), 0.0))
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / denominator


def _metric_depth(env: Any, raw_depth: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_depth, dtype=np.float32)
    finite = raw[np.isfinite(raw)]
    if finite.size and float(finite.min()) >= -1e-6 and float(finite.max()) <= 1.000001:
        from robosuite.utils import camera_utils

        return np.asarray(camera_utils.get_real_depth_map(env.sim, raw), dtype=np.float32)
    return raw


def _camera_eef_uvd(env: Any, eef_world: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    from robosuite.utils import camera_utils
    from robosuite.utils.transform_utils import pose_inv

    k = camera_utils.get_camera_intrinsic_matrix(env.sim, "agentview", resolution, resolution)
    camera_pose_world = camera_utils.get_camera_extrinsic_matrix(env.sim, "agentview")
    world_to_camera = pose_inv(camera_pose_world)
    homogeneous = np.concatenate(
        [np.asarray(eef_world, dtype=np.float32), np.ones((len(eef_world), 1), dtype=np.float32)], axis=1
    )
    camera_xyz = (world_to_camera @ homogeneous.T).T[:, :3]
    projected = (k @ camera_xyz.T).T
    safe_z = np.where(np.abs(projected[:, 2]) > 1e-8, projected[:, 2], 1.0)
    uvd = np.concatenate([projected[:, :2] / safe_z[:, None], camera_xyz[:, 2:3]], axis=1)
    uvd[:, 0] = (resolution - 1) - uvd[:, 0]
    uvd[:, 1] = (resolution - 1) - uvd[:, 1]
    valid = (
        np.isfinite(uvd).all(axis=1)
        & (uvd[:, 2] > 0.0)
        & (uvd[:, 0] >= 0.0)
        & (uvd[:, 0] < resolution)
        & (uvd[:, 1] >= 0.0)
        & (uvd[:, 1] < resolution)
    )
    return uvd, valid


def _observation(env: Any, obs: dict[str, Any], resolution: int, image_size: int) -> dict[str, Any]:
    primary = np.ascontiguousarray(np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1])
    wrist = np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)[::-1, ::-1])
    raw_depth = np.asarray(obs["agentview_depth"])
    depth = np.ascontiguousarray(_metric_depth(env, raw_depth)[::-1, ::-1])
    depth_valid = np.isfinite(depth) & (depth > 0.0)
    depth, depth_valid = resize_depth(depth, depth_valid, (image_size, image_size))
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )
    return {
        "example": {"image": [primary, wrist], "lang": "", "state": state},
        "rgb": primary,
        "wrist_rgb": wrist,
        "depth": depth.astype(np.float32),
        "depth_valid": depth_valid.astype(np.bool_),
        "eef_world": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
    }


def _load_state(candidate: RolloutCandidate) -> np.ndarray:
    with h5py.File(candidate.hdf5_path, "r") as handle:
        return np.asarray(handle[f"data/{candidate.demo_id}/states"][candidate.frame_index], dtype=np.float32)


def _build_request(observation: dict[str, Any], language: str, args: argparse.Namespace) -> dict[str, Any]:
    example = dict(observation["example"])
    example["lang"] = language
    payload: dict[str, Any] = {
        "examples": [example],
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": 10,
        "return_geometry": True,
    }
    if args.unnorm_key is not None:
        payload["unnorm_key"] = args.unnorm_key
    return payload


def _squeeze_geometry(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)[0]
    if name.startswith("depth") and array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return array


def run(args: argparse.Namespace) -> dict[str, Any]:
    # These environment variables must be set before importing libero/mujoco.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suites = [x.strip() for x in args.suites.split(",") if x.strip()]
    plan = _candidate_plan(args)
    config = {
        "bench": args.bench,
        "gt_mode": "rollout",
        "suites": suites,
        "num_samples": int(args.num_samples),
        "horizon": int(args.horizon),
        "uvd_num_points": int(args.uvd_num_points or math.floor(0.3 * args.horizon) + 2),
        "resolution": int(args.resolution),
        "image_size": int(args.image_size),
        "seed": int(args.seed),
        "hdf5_root": str(args.hdf5_root),
    }
    if args.dry_run:
        print(json.dumps({"config": config, "plan": [item.__dict__ for item in plan]}, indent=2, default=str))
        return config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    latency_values: list[float] = []
    metric_rows: list[dict[str, float | int]] = []
    jsonl_path = output_dir / "samples.jsonl"
    suite_counts: dict[str, int] = {suite: 0 for suite in suites}

    try:
        for sample_index, candidate in enumerate(plan):
            task_bddl = Path(get_libero_path("bddl_files")) / candidate.suite / f"{candidate.task_name}.bddl"
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=args.resolution,
                camera_widths=args.resolution,
                camera_depths=True,
            )
            try:
                env.seed(args.seed + sample_index)
                obs = env.regenerate_obs_from_state(_load_state(candidate))
                current = _observation(env, obs, args.resolution, args.image_size)
                current["example"]["lang"] = candidate.language
                request = _build_request(current, candidate.language, args)
                start = time.perf_counter()
                response = client.predict_action(request)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                data = response.get("data", response)
                latency_fields = build_latency_fields(elapsed_ms, data.get("timing"))
                geometry = data["geometry"]
                prediction = {
                    "depth_current": _squeeze_geometry(geometry["depth_current"], "depth_current"),
                    "depth_future": _squeeze_geometry(geometry["depth_future"], "depth_future"),
                    "uvd": _squeeze_geometry(geometry["uvd"], "uvd"),
                }
                action_chunk = np.asarray(data["actions"], dtype=np.float32)[0]
                eef_positions = [current["eef_world"]]
                done = False
                for action in action_chunk[: args.horizon]:
                    action = np.asarray(action, dtype=np.float32).reshape(-1)
                    if len(action) != 7:
                        raise ValueError(f"expected 7D action, got {action.shape}")
                    libero_action = np.concatenate([action[:6], [1.0 - 2.0 * float(action[6] > 0.5)]])
                    obs, _, done, _ = env.step(libero_action.tolist())
                    eef_positions.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
                    if done:
                        break
                if len(eef_positions) != args.horizon + 1:
                    logging.warning("Skipping early-done sample %s/%s", candidate.suite, candidate.demo_id)
                    continue
                future = _observation(env, obs, args.resolution, args.image_size)
                raw_uvd, raw_valid = _camera_eef_uvd(env, np.stack(eef_positions), args.resolution)
                indices = sample_real_uvd_indices(0, args.horizon, int(config["uvd_num_points"]))
                uvd_gt = transform_uvd_to_model_space(
                    raw_uvd[indices],
                    source_width=args.resolution,
                    source_height=args.resolution,
                    target_width=args.image_size,
                    target_height=args.image_size,
                    depth_scale=1.0,
                )
                uvd_valid = raw_valid[indices]
                uvd_time = indices.astype(np.float32) / float(args.horizon)
                sample = {
                    "example": current["example"],
                    "rgb": current["rgb"],
                    "wrist_rgb": current["wrist_rgb"],
                    "depth_current": current["depth"],
                    "depth_future": future["depth"],
                    "depth_current_valid": current["depth_valid"],
                    "depth_future_valid": future["depth_valid"],
                    "uvd": uvd_gt,
                    "uvd_valid_mask": uvd_valid,
                    "uvd_time": uvd_time,
                    "metadata": {
                        "suite": candidate.suite,
                        "task_id": candidate.task_id,
                        "demo_id": candidate.demo_id,
                        "frame_index": candidate.frame_index,
                        "future_frame": "after_predicted_action_chunk",
                        "language": candidate.language,
                        "done": bool(done),
                    },
                }
                metrics = {
                    "depth_current": masked_depth_metrics(prediction["depth_current"], sample["depth_current"], sample["depth_current_valid"]),
                    "depth_future": masked_depth_metrics(prediction["depth_future"], sample["depth_future"], sample["depth_future_valid"]),
                    "uvd": uvd_metrics(prediction["uvd"], sample["uvd"], sample["uvd_valid_mask"]),
                }
                flat = {
                    **{f"depth_current_{k}": v for k, v in metrics["depth_current"].items()},
                    **{f"depth_future_{k}": v for k, v in metrics["depth_future"].items()},
                    **{f"uvd_{k}": v for k, v in metrics["uvd"].items()},
                }
                bundle = save_sample_bundle(
                    output_dir,
                    sample_index,
                    sample=sample,
                    prediction=prediction,
                    metrics=metrics,
                    latency_ms=elapsed_ms,
                    latency_fields=latency_fields,
                )
                figure = save_sample_figure(output_dir / f"sample_{sample_index:04d}.png", sample=sample, prediction=prediction, metrics=metrics, latency_ms=elapsed_ms)
                latency_values.append(elapsed_ms)
                metric_rows.append(flat)
                suite_counts[candidate.suite] += 1
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                **sample["metadata"],
                                "sample_index": sample_index,
                                **latency_fields,
                                "metrics": flat,
                                "bundle_path": str(bundle),
                                "figure_path": str(figure),
                            },
                            default=str,
                        )
                        + "\n"
                    )
            finally:
                env.close()
    finally:
        client.close()

    if len(latency_values) != int(args.num_samples):
        logging.warning("requested %d rollout samples but completed %d valid samples", int(args.num_samples), len(latency_values))
    latency = summarize_latencies(latency_values, warmup_ms=latency_values[0] if latency_values else None)
    metric_keys = sorted({key for row in metric_rows for key in row})
    means = {
        key: float(np.nanmean([row[key] for row in metric_rows if key in row]))
        for key in metric_keys
    }
    save_summary(output_dir / "summary.json", config=config, latency=latency, metrics={"mean": means, "suite_counts": suite_counts}, samples=len(latency_values))
    (output_dir / "server_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"config": config, "latency": latency, "metrics": means, "suite_counts": suite_counts}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(parse_args())
