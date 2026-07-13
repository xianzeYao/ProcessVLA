#!/usr/bin/env python3
"""
Export metric depth from original LIBERO HDF5 demonstrations by restoring saved simulator states.

Original LIBERO demo HDF5 files contain per-step flattened MuJoCo states:

  data/demo_x/states: (T, state_dim)

This script restores each state directly with LIBERO's OffScreenRenderEnv and
renders RGB + metric depth.  Unlike action replay, this does not depend on
gripper convention, controller integration, or contact determinism.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Must be set before importing mujoco / robosuite.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore original LIBERO HDF5 simulator states and export RGB/depth."
    )
    parser.add_argument("--hdf5", required=True, help="Path to original LIBERO *_demo.hdf5 file.")
    parser.add_argument("--demo-id", default="demo_0", help="Demo group name, e.g. demo_0.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--cameras", nargs="+", default=["agentview", "robot0_eye_in_hand"])
    parser.add_argument("--max-steps", type=int, default=0, help="<=0 means all states.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--save-npy", action="store_true", help="Save per-frame metric depth .npy files.")
    parser.add_argument("--save-rgb", action="store_true", help="Save per-frame RGB png files.")
    parser.add_argument("--make-video", action="store_true", default=True)
    parser.add_argument("--no-video", dest="make_video", action="store_false")
    parser.add_argument("--near", type=float, default=None, help="Depth value mapped to black in videos.")
    parser.add_argument("--far", type=float, default=None, help="Depth value mapped to white in videos.")
    parser.add_argument("--invert-depth", action="store_true", help="Make near objects white in videos.")
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Rotate rendered RGB/depth 180 degrees for quick comparison with LeRobot train-view convention.",
    )
    return parser.parse_args()


def squeeze_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth


def convert_depth_to_meters(sim: Any, raw_depth: np.ndarray) -> tuple[np.ndarray, str]:
    raw = np.asarray(raw_depth, dtype=np.float32)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return raw, "empty_or_nonfinite"
    if float(finite.min()) < -1e-6 or float(finite.max()) > 1.000001:
        return raw, "assumed_metric_raw_depth_not_in_0_1"
    try:
        from robosuite.utils import camera_utils as CU

        if hasattr(CU, "get_real_depth_map"):
            return CU.get_real_depth_map(sim, raw), "robosuite.camera_utils.get_real_depth_map"
    except Exception:
        pass
    extent = float(sim.model.stat.extent)
    near = float(sim.model.vis.map.znear) * extent
    far = float(sim.model.vis.map.zfar) * extent
    metric = near / (1.0 - raw * (1.0 - near / far))
    return metric.astype(np.float32), "mujoco_near_far_formula"


def depth_stats(depth: np.ndarray) -> dict[str, Any]:
    depth = np.asarray(depth)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return {"shape": list(depth.shape), "dtype": str(depth.dtype), "finite_count": 0}
    return {
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "finite_count": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "p01": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
    }


def depth_to_gray(depth: np.ndarray, near: float, far: float, invert: bool) -> np.ndarray:
    d = squeeze_depth(depth).astype(np.float32)
    gray = np.clip((d - near) / max(far - near, 1e-6), 0.0, 1.0)
    if invert:
        gray = 1.0 - gray
    gray_u8 = (gray * 255.0).astype(np.uint8)
    return np.repeat(gray_u8[..., None], 3, axis=-1)


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, min(360, pil.width), 24], fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255))
    return np.asarray(pil)




def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def write_h264(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )


def normalize_bddl_path(bddl_name: str) -> str:
    """Use the local LIBERO bddl path when HDF5 attrs contain legacy/chiliocosm paths."""
    if not bddl_name:
        return bddl_name
    p = Path(bddl_name)
    parts = p.parts
    if "bddl_files" in parts:
        idx = parts.index("bddl_files")
        rel = Path(*parts[idx + 1 :])
        try:
            from libero.libero import get_libero_path

            candidate = Path(get_libero_path("bddl_files")) / rel
            if candidate.exists():
                return str(candidate)
        except Exception:
            pass
        candidate = Path("/root/data/yxz/benchmarks/LIBERO/libero/libero/bddl_files") / rel
        if candidate.exists():
            return str(candidate)
        return str(Path(*parts[idx:]))
    return bddl_name


def env_kwargs_from_hdf5_attrs(attrs: dict[str, Any], cameras: list[str], resolution: int) -> dict[str, Any]:
    env_args_raw = attrs.get("env_args")
    if isinstance(env_args_raw, bytes):
        env_args_raw = env_args_raw.decode("utf-8")
    env_args = json.loads(env_args_raw)
    env_kwargs = dict(env_args.get("env_kwargs", {}))
    # Current LIBERO ControlEnv reconstructs controller_configs from the
    # top-level controller argument, then passes it into the underlying task.
    # Original HDF5 env_args may already contain controller_configs, which would
    # otherwise be passed twice.
    env_kwargs.pop("controller_configs", None)
    env_kwargs.setdefault("controller", "OSC_POSE")

    # Prefer the local attr path because it matches this checkout.  Some older
    # env_args store a legacy "chiliocosm/..." bddl path.
    bddl_file_name = attrs.get("bddl_file_name") or env_kwargs.get("bddl_file_name") or env_args.get("bddl_file")
    if isinstance(bddl_file_name, bytes):
        bddl_file_name = bddl_file_name.decode("utf-8")
    env_kwargs["bddl_file_name"] = normalize_bddl_path(str(bddl_file_name))

    env_kwargs.update(
        {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_depths": True,
            "camera_names": cameras,
            "camera_heights": resolution,
            "camera_widths": resolution,
            "camera_segmentations": None,
        }
    )
    return env_kwargs


def read_hdf5_metadata(hdf5_path: Path, demo_id: str) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{hdf5_path} has no 'data' group")
        if demo_id not in f["data"]:
            demos = sorted(f["data"].keys())
            raise KeyError(f"{demo_id!r} not found. Available demos: {demos[:10]}...")
        attrs = dict(f["data"].attrs)
        demo = f[f"data/{demo_id}"]
        if "states" not in demo:
            raise KeyError(f"{hdf5_path}:{demo_id} has no states dataset")
        states = demo["states"][()]
        demo_info = {
            "keys": list(demo.keys()),
            "num_samples_attr": int(demo.attrs["num_samples"]) if "num_samples" in demo.attrs else None,
            "states_shape": list(states.shape),
        }
        for key in ["actions", "dones", "rewards", "robot_states"]:
            if key in demo:
                demo_info[f"{key}_shape"] = list(demo[key].shape)
        return attrs, states, demo_info


def main() -> None:
    args = parse_args()
    hdf5_path = Path(args.hdf5)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    attrs, states, demo_info = read_hdf5_metadata(hdf5_path, args.demo_id)
    if args.max_steps and args.max_steps > 0:
        states = states[: args.max_steps]
    frame_indices = list(range(0, len(states), max(1, int(args.stride))))

    from libero.libero.envs import OffScreenRenderEnv

    env_kwargs = env_kwargs_from_hdf5_attrs(attrs, args.cameras, args.resolution)
    env = OffScreenRenderEnv(**env_kwargs)

    per_camera_rgbs: dict[str, list[np.ndarray]] = {cam: [] for cam in args.cameras}
    per_camera_depths: dict[str, list[np.ndarray]] = {cam: [] for cam in args.cameras}
    per_camera_stats: dict[str, list[dict[str, Any]]] = {cam: [] for cam in args.cameras}
    conversion_methods: dict[str, str] = {}
    success_values: list[bool] = []

    try:
        for out_i, state_i in enumerate(frame_indices):
            obs = env.regenerate_obs_from_state(states[state_i])
            success_values.append(bool(env.check_success()))
            for cam in args.cameras:
                rgb_key = f"{cam}_image"
                depth_key = f"{cam}_depth"
                if rgb_key not in obs or depth_key not in obs:
                    raise KeyError(f"Missing {rgb_key} or {depth_key}; obs keys={sorted(obs.keys())}")
                rgb = np.asarray(obs[rgb_key], dtype=np.uint8)
                metric, method = convert_depth_to_meters(env.sim, obs[depth_key])
                metric = squeeze_depth(metric).astype(np.float32)
                if args.flip:
                    rgb = np.ascontiguousarray(rgb[::-1, ::-1])
                    metric = np.ascontiguousarray(metric[::-1, ::-1])
                conversion_methods[cam] = method
                per_camera_rgbs[cam].append(rgb)
                per_camera_depths[cam].append(metric)
                per_camera_stats[cam].append(depth_stats(metric))

                cam_dir = out_dir / cam
                cam_dir.mkdir(parents=True, exist_ok=True)
                if args.save_rgb:
                    Image.fromarray(rgb).save(cam_dir / f"frame_{state_i:06d}_rgb.png")
                if args.save_npy:
                    np.save(cam_dir / f"frame_{state_i:06d}_depth_m.npy", metric)
    finally:
        env.close()

    video_meta: dict[str, Any] = {}
    if args.make_video:
        for cam in args.cameras:
            depths = per_camera_depths[cam]
            rgbs = per_camera_rgbs[cam]
            finite = np.concatenate([d[np.isfinite(d)].reshape(-1) for d in depths if np.any(np.isfinite(d))])
            if finite.size == 0:
                raise ValueError(f"No finite depth values for {cam}")
            near = float(args.near) if args.near is not None else float(np.percentile(finite, 1))
            far = float(args.far) if args.far is not None else float(np.percentile(finite, 99))
            frames = []
            for idx, rgb, depth in zip(frame_indices, rgbs, depths):
                gray = depth_to_gray(depth, near, far, args.invert_depth)
                if gray.shape[:2] != rgb.shape[:2]:
                    gray = cv2.resize(gray, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                left = add_label(rgb, f"{cam} restored RGB")
                right = add_label(gray, "metric depth gray")
                frames.append(np.concatenate([left, right], axis=1))
            video_path = out_dir / f"{cam}_hdf5_state_rgb_depth.mp4"
            write_h264(video_path, frames, args.fps / max(1, int(args.stride)))
            video_meta[cam] = {
                "path": str(video_path),
                "num_frames": len(frames),
                "depth_gray_near": near,
                "depth_gray_far": far,
            }

    serializable_attrs = {}
    for k, v in attrs.items():
        serializable_attrs[k] = v.decode("utf-8") if isinstance(v, bytes) else v
    summary = {
        "hdf5": str(hdf5_path),
        "demo_id": args.demo_id,
        "output_dir": str(out_dir),
        "hdf5_data_attrs": serializable_attrs,
        "demo_info": demo_info,
        "env_kwargs_used": env_kwargs,
        "resolution": args.resolution,
        "cameras": args.cameras,
        "stride": args.stride,
        "max_steps": args.max_steps,
        "frame_indices": frame_indices,
        "num_exported_frames": len(frame_indices),
        "success_any": bool(any(success_values)),
        "success_last": bool(success_values[-1]) if success_values else None,
        "success_count": int(sum(success_values)),
        "depth_conversion_methods": conversion_methods,
        "depth_stats_first": {cam: per_camera_stats[cam][0] for cam in args.cameras if per_camera_stats[cam]},
        "depth_stats_last": {cam: per_camera_stats[cam][-1] for cam in args.cameras if per_camera_stats[cam]},
        "videos": video_meta,
        "notes": [
            "Depth is rendered by restoring original HDF5 MuJoCo states, not by replaying actions.",
            "Metric conversion uses robosuite.camera_utils.get_real_depth_map when available.",
        ],
    }
    summary_path = out_dir / "summary.json"
    summary = to_jsonable(summary)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
