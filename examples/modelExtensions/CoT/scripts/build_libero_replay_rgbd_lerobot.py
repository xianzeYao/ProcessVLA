#!/usr/bin/env python3
"""Build a LeRobot-style LIBERO replay-RGBD dataset from HDF5 simulator states.

This keeps the source LeRobot / IPEC tabular fields (actions, robot state,
task metadata, timestamps) but replaces the main RGB videos with RGB rendered
from the matched HDF5 MuJoCo states and adds metric depth + camera parameters.

The rendered RGB/depth are rotated by 180 degrees by default so their visual
convention matches the local LeRobot / IPEC training videos observed in this
workspace. Camera intrinsics are saved after this rotation as ``*_K``; these
keys project directly into the stored rotated images.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_libero_hdf5_state_depth import (  # noqa: E402
    convert_depth_to_meters,
    env_kwargs_from_hdf5_attrs,
    squeeze_depth,
    write_h264,
)


CAMERAS = ("agentview", "robot0_eye_in_hand")
CAMERA_TO_RGB_KEY = {
    "agentview": "observation.images.image",
    "robot0_eye_in_hand": "observation.images.wrist_image",
}
CAMERA_TO_DEPTH_KEY = {
    "agentview": "observation.depth.image_m",
    "robot0_eye_in_hand": "observation.depth.wrist_m",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create replay RGB + metric depth LeRobot-style LIBERO dataset.")
    ap.add_argument("--base-lerobot-root", required=True, help="Source LeRobot/IPEC dataset root.")
    ap.add_argument("--mapping", required=True, help="JSONL mapping produced by build_libero_lerobot_hdf5_mapping.py.")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--max-episodes", type=int, default=0, help="Debug only. <=0 means all mapping rows.")
    ap.add_argument("--flip", dest="flip", action="store_true", help="Rotate rendered RGB/depth by 180 degrees (default).")
    ap.add_argument("--no-flip", dest="flip", action="store_false", help="Keep raw robosuite image orientation.")
    ap.set_defaults(flip=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-visual-sites", action="store_true", help="Keep robosuite gripper visual/debug sites in replay renders.")
    ap.add_argument("--symlink-extra-videos", action="store_true", default=True)
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def episode_parquet_path(root: Path, episode_index: int) -> Path:
    p = root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
    if p.exists():
        return p
    matches = list((root / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet"))
    if matches:
        return matches[0]
    raise FileNotFoundError(p)


def chunk_dir(kind: str, root: Path, episode_index: int, key: str | None = None) -> Path:
    base = root / kind / f"chunk-{episode_index // 1000:03d}"
    return base / key if key else base


def output_episode_parquet_path(root: Path, episode_index: int) -> Path:
    return root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def load_states_for_row(h5: h5py.File, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    states = h5[f"data/{row['demo_id']}/states"]
    indices = row.get("hdf5_indices")
    if indices is None:
        indices_arr = np.arange(int(row["hdf5_start"]), int(row["hdf5_end_exclusive"]), dtype=np.int64)
    else:
        indices_arr = np.asarray(indices, dtype=np.int64)
    return states[indices_arr], indices_arr


def flip180(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array[::-1, ::-1])


def intrinsic_for_rot180(k: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a K that maps raw projected pixels into the 180-degree-rotated image.

    For LIBERO's OpenGL image convention plus the stored array flip, the effective
    projection into the stored LeRobot image mirrors only the horizontal axis.
    """
    out = np.asarray(k, dtype=np.float32).copy()
    # LIBERO/robosuite RGB arrays use the OpenGL vertical convention. The
    # stored LeRobot-compatible image is array-flipped by 180 degrees, so
    # the OpenGL v coordinate is already the final image row; only u is
    # mirrored in the pixel projection.
    out[0, 0] = -out[0, 0]
    out[0, 2] = (width - 1) - out[0, 2]
    return out


def camera_npz_payload(
    k_rot_arrays: dict[str, list[np.ndarray] | np.ndarray],
    t_arrays: dict[str, list[np.ndarray] | np.ndarray],
) -> dict[str, np.ndarray]:
    """Build the compact camera-parameter payload for stored rot180 images."""
    return {
        "agentview_K": np.stack(k_rot_arrays["agentview"], axis=0).astype(np.float32),
        "agentview_T_world_camera": np.stack(t_arrays["agentview"], axis=0).astype(np.float32),
        "wrist_K": np.stack(k_rot_arrays["robot0_eye_in_hand"], axis=0).astype(np.float32),
        "wrist_T_world_camera": np.stack(t_arrays["robot0_eye_in_hand"], axis=0).astype(np.float32),
    }


def replay_state_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """Build the canonical 8-D LeRobot state from a replayed simulator observation."""
    from robosuite.utils.transform_utils import quat2axisangle

    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(-1)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if eef_pos.size != 3 or eef_quat.size != 4 or gripper_qpos.size != 2:
        raise ValueError(
            "Expected robot0_eef_pos[3], robot0_eef_quat[4], "
            f"robot0_gripper_qpos[2], got {eef_pos.size}, {eef_quat.size}, {gripper_qpos.size}"
        )
    axis_angle = np.asarray(quat2axisangle(eef_quat), dtype=np.float32).reshape(-1)
    if axis_angle.size != 3:
        raise ValueError(f"Expected axis-angle[3], got {axis_angle.size}")
    return np.concatenate([eef_pos, axis_angle, gripper_qpos]).astype(np.float32, copy=False)


def project_world_to_pixel(k: np.ndarray, t_world_camera: np.ndarray, p_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a world-frame 3D point into pixel coordinates.

    k can be raw K or K_rot180. If K_rot180 is used, returned uv is already in
    the stored 180-degree-rotated image coordinates.
    """
    p = np.asarray([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float64)
    t_camera_world = np.linalg.inv(np.asarray(t_world_camera, dtype=np.float64))
    p_camera = t_camera_world @ p
    uvw = np.asarray(k, dtype=np.float64) @ p_camera[:3]
    uv = np.asarray([uvw[0] / uvw[2], uvw[1] / uvw[2]], dtype=np.float32)
    return uv, p_camera[:3].astype(np.float32)


def hide_visual_sites(env: Any) -> list[str]:
    """Hide robosuite gripper visual/debug sites from rendered RGB/depth.

    LIBERO / robosuite Panda grippers expose semi-transparent red/green sites
    such as gripper0_grip_site and gripper0_grip_site_cylinder. They are useful
    for debugging but should not be baked into training RGB/depth.
    """
    model = env.sim.model
    hidden: list[str] = []
    exact_names = {
        "gripper0_ft_frame",
        "gripper0_grip_site",
        "gripper0_grip_site_cylinder",
    }
    for site_id in range(model.nsite):
        try:
            name = model.site_id2name(site_id)
        except Exception:
            name = ""
        should_hide = name in exact_names or (name.startswith("gripper0_") and ("site" in name or "ft_frame" in name))
        if should_hide and float(model.site_rgba[site_id, 3]) != 0.0:
            model.site_rgba[site_id, 3] = 0.0
            hidden.append(name)
    return hidden


def ensure_clean_output(root: Path, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists. Pass --overwrite to replace generated files.")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def copy_static_metadata(base_root: Path, out_root: Path, rows: list[dict[str, Any]], fps: float, resolution: int, flip: bool) -> None:
    meta_out = out_root / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_root / "meta" / "tasks.jsonl", meta_out / "tasks.jsonl")

    selected = {int(r["episode_index"]) for r in rows}
    for name in ("episodes.jsonl", "episodes_stats.jsonl"):
        src = base_root / "meta" / name
        if not src.exists():
            continue
        filtered = []
        for row in read_jsonl(src):
            if int(row["episode_index"]) in selected:
                filtered.append(row)
        write_jsonl(meta_out / name, filtered)

    if (base_root / "README.md").exists():
        shutil.copy2(base_root / "README.md", out_root / "README.source.md")

    info = json.loads((base_root / "meta" / "info.json").read_text(encoding="utf-8"))
    info["total_episodes"] = len(rows)
    info["total_frames"] = int(sum(int(r["lerobot_length"]) for r in rows))
    info["total_videos"] = len(rows) * sum(1 for v in info["features"].values() if v.get("dtype") == "video")
    info["splits"] = {"train": f"0:{len(rows)}"}
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["depth_path"] = "depth/chunk-{episode_chunk:03d}/{depth_key}/episode_{episode_index:06d}.npz"
    info["camera_path"] = "camera/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz"
    info["source_dataset"] = str(base_root)
    info["replay_rgbd"] = {
        "rgb_source": "HDF5 MuJoCo state replay",
        "depth_source": "HDF5 MuJoCo state replay converted with robosuite camera_utils.get_real_depth_map when available",
        "stored_image_transform": "rot180" if flip else "none",
        "depth_unit": "meters",
        "depth_dtype_on_disk": "float16",
        "camera_extrinsic_key": "T_world_camera",
        "camera_intrinsic_note": "K matches the stored image orientation; --flip selects the LeRobot-compatible stored image convention.",
        "state_source": "HDF5 MuJoCo state replay",
        "state_layout": "[robot0_eef_pos(3), quat2axisangle(robot0_eef_quat)(3), robot0_gripper_qpos(2)]",
        "hidden_visual_sites": "gripper visual/debug sites hidden",
    }

    video_info = {
        "video.height": resolution,
        "video.width": resolution,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }
    for key in CAMERA_TO_RGB_KEY.values():
        info["features"][key]["shape"] = [resolution, resolution, 3]
        info["features"][key]["info"] = dict(video_info)

    info["features"].update(
        {
            "observation.depth.image_m_path": {"dtype": "string", "shape": [1], "names": None},
            "observation.depth.wrist_m_path": {"dtype": "string", "shape": [1], "names": None},
            "observation.camera.params_path": {"dtype": "string", "shape": [1], "names": None},
            "source.hdf5_path": {"dtype": "string", "shape": [1], "names": None},
            "source.hdf5_demo_id": {"dtype": "string", "shape": [1], "names": None},
            "source.hdf5_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    (meta_out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def symlink_extra_video_dirs(base_root: Path, out_root: Path) -> None:
    src_chunk_root = base_root / "videos"
    if not src_chunk_root.exists():
        return
    for src in src_chunk_root.glob("chunk-*/*"):
        if not src.is_dir():
            continue
        key = src.name
        if key in set(CAMERA_TO_RGB_KEY.values()):
            continue
        dst = out_root / "videos" / src.parent.name / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        dst.symlink_to(src, target_is_directory=True)


def group_by_hdf5(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("accepted", False):
            raise ValueError(f"Mapping row is not accepted: episode={row.get('episode_index')}")
        grouped[row["hdf5"]].append(row)
    return grouped


def render_one_episode(
    env: Any,
    camera_utils: Any,
    h5: h5py.File,
    row: dict[str, Any],
    base_root: Path,
    out_root: Path,
    resolution: int,
    fps: float,
    flip: bool,
) -> dict[str, Any]:
    episode_index = int(row["episode_index"])
    states, hdf5_indices = load_states_for_row(h5, row)

    src_df = pd.read_parquet(episode_parquet_path(base_root, episode_index))
    if len(src_df) != len(states):
        raise ValueError(
            f"episode {episode_index}: parquet length {len(src_df)} != mapped states length {len(states)}"
        )

    rgb_frames = {cam: [] for cam in CAMERAS}
    depth_arrays = {cam: [] for cam in CAMERAS}
    k_rot_arrays = {cam: [] for cam in CAMERAS}
    t_arrays = {cam: [] for cam in CAMERAS}
    replay_states = []

    for state in states:
        obs = env.regenerate_obs_from_state(state)
        replay_states.append(replay_state_from_obs(obs))
        for cam in CAMERAS:
            rgb = np.asarray(obs[f"{cam}_image"], dtype=np.uint8)
            metric, _ = convert_depth_to_meters(env.sim, obs[f"{cam}_depth"])
            metric = squeeze_depth(metric).astype(np.float32)

            k = np.asarray(camera_utils.get_camera_intrinsic_matrix(env.sim, cam, resolution, resolution), dtype=np.float32)
            t_world_camera = np.asarray(camera_utils.get_camera_extrinsic_matrix(env.sim, cam), dtype=np.float32)
            k_rot = intrinsic_for_rot180(k, resolution, resolution)
            if flip:
                rgb = flip180(rgb)
                metric = flip180(metric)

            rgb_frames[cam].append(rgb)
            depth_arrays[cam].append(metric.astype(np.float16))
            k_rot_arrays[cam].append(k_rot if flip else k)
            t_arrays[cam].append(t_world_camera)

    for cam in CAMERAS:
        video_key = CAMERA_TO_RGB_KEY[cam]
        video_dir = chunk_dir("videos", out_root, episode_index, video_key)
        video_dir.mkdir(parents=True, exist_ok=True)
        write_h264(video_dir / f"episode_{episode_index:06d}.mp4", rgb_frames[cam], fps)

    depth_rel_paths = {}
    for cam in CAMERAS:
        depth_key = CAMERA_TO_DEPTH_KEY[cam]
        depth_dir = chunk_dir("depth", out_root, episode_index, depth_key)
        depth_dir.mkdir(parents=True, exist_ok=True)
        rel = Path("depth") / f"chunk-{episode_index // 1000:03d}" / depth_key / f"episode_{episode_index:06d}.npz"
        np.savez_compressed(out_root / rel, depth_m=np.stack(depth_arrays[cam], axis=0))
        depth_rel_paths[cam] = rel.as_posix()

    camera_rel = Path("camera") / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.npz"
    (out_root / camera_rel).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_root / camera_rel, **camera_npz_payload(k_rot_arrays, t_arrays))

    out_df = src_df.copy()
    out_df["observation.depth.image_m_path"] = depth_rel_paths["agentview"]
    out_df["observation.depth.wrist_m_path"] = depth_rel_paths["robot0_eye_in_hand"]
    out_df["observation.camera.params_path"] = camera_rel.as_posix()
    out_df["source.hdf5_path"] = row["hdf5"]
    out_df["source.hdf5_demo_id"] = row["demo_id"]
    out_df["source.hdf5_index"] = hdf5_indices.astype(np.int64)
    out_df["observation.state"] = list(np.stack(replay_states, axis=0).astype(np.float32))
    for obsolete in (
        "observation.eef.agentview_uv",
        "observation.eef.wrist_uv",
        "observation.eef.agentview_depth_m",
        "observation.eef.wrist_depth_m",
    ):
        if obsolete in out_df.columns:
            out_df = out_df.drop(columns=[obsolete])

    out_parquet = output_episode_parquet_path(out_root, episode_index)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)

    return {
        "episode_index": episode_index,
        "length": int(len(states)),
        "hdf5": row["hdf5"],
        "demo_id": row["demo_id"],
        "hdf5_indices_first": int(hdf5_indices[0]),
        "hdf5_indices_last": int(hdf5_indices[-1]),
        "rgb_videos": {
            cam: (Path("videos") / f"chunk-{episode_index // 1000:03d}" / CAMERA_TO_RGB_KEY[cam] / f"episode_{episode_index:06d}.mp4").as_posix()
            for cam in CAMERAS
        },
        "depth_npz": depth_rel_paths,
        "camera_npz": camera_rel.as_posix(),
    }


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_lerobot_root)
    out_root = Path(args.output_root)
    mapping = Path(args.mapping)

    ensure_clean_output(out_root, args.overwrite)
    rows = read_jsonl(mapping)
    if args.max_episodes > 0:
        rows = rows[: args.max_episodes]
    rows = sorted(rows, key=lambda r: int(r["episode_index"]))

    copy_static_metadata(base_root, out_root, rows, args.fps, args.resolution, args.flip)
    if args.symlink_extra_videos:
        symlink_extra_video_dirs(base_root, out_root)

    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU

    grouped = group_by_hdf5(rows)
    source_rows: list[dict[str, Any]] = []
    for group_i, (hdf5_path, hdf5_rows) in enumerate(grouped.items(), start=1):
        print(f"[render] hdf5 {group_i}/{len(grouped)} {Path(hdf5_path).name} episodes={len(hdf5_rows)}", flush=True)
        with h5py.File(hdf5_path, "r") as h5:
            env_kwargs = env_kwargs_from_hdf5_attrs(dict(h5["data"].attrs), list(CAMERAS), args.resolution)
            env = OffScreenRenderEnv(**env_kwargs)
            hidden_sites = [] if args.keep_visual_sites else hide_visual_sites(env)
            if hidden_sites:
                print(f"  [hide-sites] {hidden_sites}", flush=True)
            try:
                for row in sorted(hdf5_rows, key=lambda r: int(r["episode_index"])):
                    print(f"  [episode] {int(row['episode_index']):06d} {row['demo_id']} len={row['lerobot_length']}", flush=True)
                    source_rows.append(
                        render_one_episode(
                            env=env,
                            camera_utils=CU,
                            h5=h5,
                            row=row,
                            base_root=base_root,
                            out_root=out_root,
                            resolution=args.resolution,
                            fps=args.fps,
                            flip=args.flip,
                        )
                    )
            finally:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()

    write_jsonl(out_root / "meta" / "source_mapping.jsonl", source_rows)
    summary = {
        "status": "success",
        "base_lerobot_root": str(base_root),
        "mapping": str(mapping),
        "output_root": str(out_root),
        "num_episodes": len(rows),
        "num_frames": int(sum(r["length"] for r in source_rows)),
        "resolution": args.resolution,
        "fps": args.fps,
        "stored_image_transform": "rot180" if args.flip else "none",
        "hidden_visual_sites": "kept" if args.keep_visual_sites else "hidden",
        "source_mapping": "meta/source_mapping.jsonl",
    }
    (out_root / "meta" / "replay_rgbd_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
