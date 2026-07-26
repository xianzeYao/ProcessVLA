#!/usr/bin/env python3
"""Rerender a GR1 RoboCasa HDF5 task into a LeRobot-compatible RGB-D dataset.

The source LeRobot task supplies the original 44-D state/action trajectories and
annotations.  The matching raw HDF5 task supplies MuJoCo states and XML models;
we replay those states to create a synchronized ``egoview`` RGB video, metric
depth, camera matrices, and left/right wrist world positions for CoT-UVD supervision.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import numpy as np
import pandas as pd

from robocasa_rerender_geometry import (
    DIAL_RENDER_HEIGHT,
    DIAL_RENDER_WIDTH,
    apply_image_affine,
    transformed_agentview_intrinsic,
)
from robocasa_replay_points import (
    bilateral_eef_world_positions,
    bilateral_pinch_center_world_positions,
    bilateral_robot_eef_site_world_positions,
    bilateral_thumb_index_pinch_world_positions,
)


CAMERA = "egoview"
RGB_KEY = "observation.images.ego_view"


def build_episode_instruction_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Load the pure task metadata helper without importing StarVLA's package init."""
    module_path = Path(__file__).resolve().parents[4] / "starVLA" / "dataloader" / "robocasa_fourier_tasks.py"
    spec = importlib.util.spec_from_file_location("robocasa_fourier_tasks_for_replay", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Fourier task metadata helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Register the dynamically loaded module before executing it. This is
    # required by dataclasses/type resolution on some Python versions.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_episode_instruction_table(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-lerobot-root", required=True)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--robocasa-repo", required=True, help="Checkout containing robocasa/ and playback_dataset.py.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-episodes", type=int, default=0, help="Debug limit; <= 0 renders all selected episodes.")
    parser.add_argument("--episode-start", type=int, default=0, help="Inclusive LeRobot episode index for a shard.")
    parser.add_argument("--episode-end", type=int, default=0, help="Exclusive LeRobot episode index for a shard; <=0 means through the end.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Initialize metadata and sidecars without rendering episodes.")
    parser.add_argument("--skip-prepare", action="store_true", help="Write only this shard; output metadata must already exist.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def episode_path(root: Path, episode_index: int) -> Path:
    matches = list((root / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one parquet for episode {episode_index}, found {matches}")
    return matches[0]


def build_mapping(base_root: Path, hdf5_path: Path) -> list[dict[str, Any]]:
    """Map random LeRobot ordering using the official ``trajectory_id``.

    HDF5 control actions are 24-D whereas released training actions are 44-D,
    so action bytes are not comparable. ``trajectory_id`` retains the original
    demo number (``...-00577`` maps to ``data/demo_577``); lengths are checked.
    """
    episode_meta = {int(row["episode_index"]): row for row in read_jsonl(base_root / "meta" / "episodes.jsonl")}
    with h5py.File(hdf5_path, "r") as h5:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for parquet in sorted((base_root / "data").glob("chunk-*/episode_*.parquet")):
            episode_index = int(parquet.stem.rsplit("_", 1)[-1])
            if episode_index not in episode_meta:
                raise RuntimeError(f"episode {episode_index} missing from meta/episodes.jsonl")
            meta = episode_meta[episode_index]
            trajectory_id = str(meta["trajectory_id"])
            try:
                demo_id = f"demo_{int(trajectory_id.rsplit('-', 1)[1])}"
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"Cannot parse raw demo number from trajectory_id={trajectory_id!r}") from exc
            if demo_id not in h5["data"]:
                raise RuntimeError(f"episode {episode_index} maps to missing HDF5 {demo_id}")
            length = len(pd.read_parquet(parquet))
            h5_length = int(h5[f"data/{demo_id}/states"].shape[0])
            meta_length = int(meta["length"])
            if length != h5_length or length != meta_length:
                raise RuntimeError(f"Length verification failed for episode {episode_index}/{demo_id}: parquet={length}, meta={meta_length}, HDF5={h5_length}")
            if demo_id in seen:
                raise RuntimeError(f"HDF5 demo {demo_id} matched more than once")
            seen.add(demo_id)
            rows.append({"episode_index": episode_index, "demo_id": demo_id, "length": length, "trajectory_id": trajectory_id})
    return sorted(rows, key=lambda row: row["episode_index"])

def write_h264(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=2,
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def convert_depth_to_meters(sim: Any, raw_depth: np.ndarray) -> np.ndarray:
    from robosuite.utils import camera_utils

    raw = np.asarray(raw_depth, dtype=np.float32)
    if raw.size and float(np.nanmin(raw)) >= -1e-6 and float(np.nanmax(raw)) <= 1.000001:
        if hasattr(camera_utils, "get_real_depth_map"):
            return np.asarray(camera_utils.get_real_depth_map(sim, raw), dtype=np.float32)
        extent = float(sim.model.stat.extent)
        near, far = float(sim.model.vis.map.znear) * extent, float(sim.model.vis.map.zfar) * extent
        return near / (1.0 - raw * (1.0 - near / far))
    return raw


def prepare_output(base_root: Path, out_root: Path, rows: list[dict[str, Any]], *, fps: float, resolution: int, overwrite: bool) -> dict[int, int]:
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to replace it")
        shutil.rmtree(out_root)
    (out_root / "meta").mkdir(parents=True)
    selected = {int(row["episode_index"]) for row in rows}
    source_episodes = read_jsonl(base_root / "meta" / "episodes.jsonl")
    selected_episodes = [row for row in source_episodes if int(row["episode_index"]) in selected]
    tasks, episode_to_task = build_episode_instruction_table(selected_episodes)
    write_jsonl(out_root / "meta" / "tasks.jsonl", tasks)
    for name in ("episodes.jsonl", "episodes_stats.jsonl", "modality.json"):
        src = base_root / "meta" / name
        if src.exists():
            if name.startswith("episodes"):
                output_rows = [row for row in read_jsonl(src) if int(row["episode_index"]) in selected]
                if name == "episodes.jsonl":
                    for row in output_rows:
                        row["tasks"] = [tasks[episode_to_task[int(row["episode_index"])]]["task"]]
                write_jsonl(out_root / "meta" / name, output_rows)
            else:
                shutil.copy2(src, out_root / "meta" / name)
    info = json.loads((base_root / "meta" / "info.json").read_text(encoding="utf-8"))
    info["total_episodes"] = len(rows)
    info["total_frames"] = int(sum(row["length"] for row in rows))
    info["total_tasks"] = len(tasks)
    info["total_videos"] = len(rows)
    info["splits"] = {"train": f"0:{len(rows)}"}
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/observation.images.ego_view/episode_{episode_index:06d}.mp4"
    info["depth_path"] = "depth/chunk-{episode_chunk:03d}/observation.depth.ego_view_m/episode_{episode_index:06d}.npz"
    info["camera_path"] = "camera/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz"
    info["source_dataset"] = str(base_root)
    info["source_hdf5"] = "mapped from official trajectory_id to original HDF5 demo ID; lengths validated; see meta/source_mapping.jsonl"
    info["language_source"] = "Teleop episodes.remarks canonicalized with official X-Embodiment-Sim unlocked_waist prefix"
    info["replay_rgbd"] = {
        "camera": CAMERA,
        "rgb_source": "MuJoCo HDF5-state replay",
        "depth_unit": "meters",
        "depth_dtype_on_disk": "float16",
        "raw_render_size": [DIAL_RENDER_HEIGHT, DIAL_RENDER_WIDTH],
        "stored_image_transform": "dial_vertical_flip_crop_pad_resize",
        "stored_image_size": [resolution, resolution],
        "camera_intrinsic_transform": "same_affine_as_rgb_depth",
        "wrist_frame": "world",
        "wrist_order": ["left", "right"],
        "wrist_world_keys": ["observation.left_robot_eef_pos", "observation.right_robot_eef_pos"],
        "eef_ablation_world_keys": ["observation.left_robot_eef_pos", "observation.right_robot_eef_pos"],
        "grip_site_world_keys": ["observation.left_eef_pos", "observation.right_eef_pos"],
        "uvd_world_keys": ["observation.left_thumb_index_pinch_pos", "observation.right_thumb_index_pinch_pos"],
        "virtual_pinch_world_keys": ["observation.left_pinch_pos", "observation.right_pinch_pos"],
        "thumb_index_pinch_world_keys": ["observation.left_thumb_index_pinch_pos", "observation.right_thumb_index_pinch_pos"],
        "uvd_point_definition": "midpoint of DIAL gripper0_{left,right}_{L,R}_thumb_distal_link and index_intermediate_link body positions",
        "point_definitions": {
            "robot_eef_site": "robot0_left_eef_site and robot0_right_eef_site when present; current Teleop XML falls back on the coincident robot0_r_wrist_site for right",
            "grip_site": "gripper0_{left,right}_grip_site via robosuite eef_site_id",
            "pinch_center": "mean of robot0_{left,right}_pinch_spheres_0..3",
            "thumb_index_pinch": "midpoint of DIAL gripper0_{left,right}_{L,R}_thumb_distal_link and index_intermediate_link body positions",
        },
    }
    info["features"][RGB_KEY]["shape"] = [resolution, resolution, 3]
    info["features"][RGB_KEY]["video_info"] = {"video.height": resolution, "video.width": resolution, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": fps, "video.channels": 3, "has_audio": False}
    info["features"].update({
        "observation.depth.image_m_path": {"dtype": "string", "shape": [1], "names": None},
        "observation.camera.params_path": {"dtype": "string", "shape": [1], "names": None},
        "observation.left_eef_pos": {"dtype": "float32", "shape": [3], "names": None},
        "observation.right_eef_pos": {"dtype": "float32", "shape": [3], "names": None},
        "observation.left_robot_eef_pos": {"dtype": "float32", "shape": [3], "names": None},
        "observation.right_robot_eef_pos": {"dtype": "float32", "shape": [3], "names": None},
        "observation.left_pinch_pos": {"dtype": "float32", "shape": [3], "names": None},
        "observation.right_pinch_pos": {"dtype": "float32", "shape": [3], "names": None},
        "observation.left_thumb_index_pinch_pos": {"dtype": "float32", "shape": [3], "names": None},
        "observation.right_thumb_index_pinch_pos": {"dtype": "float32", "shape": [3], "names": None},
        "source.hdf5_demo_id": {"dtype": "string", "shape": [1], "names": None},
    })
    (out_root / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return episode_to_task


def render_episode(
    env: Any,
    camera_utils: Any,
    h5: h5py.File,
    row: dict[str, Any],
    base_root: Path,
    out_root: Path,
    resolution: int,
    fps: float,
    episode_to_task: dict[int, int],
    reset_to: Any,
) -> dict[str, Any]:
    episode_index, demo_id = int(row["episode_index"]), row["demo_id"]
    src_df = pd.read_parquet(episode_path(base_root, episode_index))
    demo = h5[f"data/{demo_id}"]
    states = np.asarray(demo["states"])
    if len(states) != len(src_df):
        raise RuntimeError(f"Length mismatch: episode {episode_index}={len(src_df)}, {demo_id}={len(states)}")
    reset_to(env, {"model": demo.attrs["model_file"], "ep_meta": demo.attrs.get("ep_meta"), "states": states[0]})
    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    left_eef_positions: list[np.ndarray] = []
    right_eef_positions: list[np.ndarray] = []
    left_robot_eef_positions: list[np.ndarray] = []
    right_robot_eef_positions: list[np.ndarray] = []
    left_pinch_positions: list[np.ndarray] = []
    right_pinch_positions: list[np.ndarray] = []
    left_thumb_index_pinch_positions: list[np.ndarray] = []
    right_thumb_index_pinch_positions: list[np.ndarray] = []
    k_frames: list[np.ndarray] = []
    t_frames: list[np.ndarray] = []
    for state in states:
        reset_to(env, {"states": state})
        env._get_observations(force_update=True)
        rgb, depth = env.sim.render(
            height=DIAL_RENDER_HEIGHT,
            width=DIAL_RENDER_WIDTH,
            camera_name=CAMERA,
            depth=True,
        )
        rgb = apply_image_affine(np.asarray(rgb, dtype=np.uint8), output_size=resolution)
        depth = np.asarray(convert_depth_to_meters(env.sim, depth), dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = apply_image_affine(depth, output_size=resolution)
        k = transformed_agentview_intrinsic(camera_utils, env.sim, CAMERA, output_size=resolution)
        rgb_frames.append(rgb)
        depth_frames.append(np.asarray(depth, dtype=np.float16))
        eef_world = bilateral_eef_world_positions(env)
        left_eef_positions.append(eef_world[0])
        right_eef_positions.append(eef_world[1])
        robot_eef_world = bilateral_robot_eef_site_world_positions(env)
        left_robot_eef_positions.append(robot_eef_world[0])
        right_robot_eef_positions.append(robot_eef_world[1])
        pinch_world = bilateral_pinch_center_world_positions(env)
        left_pinch_positions.append(pinch_world[0])
        right_pinch_positions.append(pinch_world[1])
        thumb_index_pinch_world = bilateral_thumb_index_pinch_world_positions(env)
        left_thumb_index_pinch_positions.append(thumb_index_pinch_world[0])
        right_thumb_index_pinch_positions.append(thumb_index_pinch_world[1])
        k_frames.append(k)
        t_frames.append(np.asarray(camera_utils.get_camera_extrinsic_matrix(env.sim, CAMERA), dtype=np.float32))
    chunk = episode_index // 1000
    video_rel = Path("videos") / f"chunk-{chunk:03d}" / RGB_KEY / f"episode_{episode_index:06d}.mp4"
    depth_rel = Path("depth") / f"chunk-{chunk:03d}" / "observation.depth.ego_view_m" / f"episode_{episode_index:06d}.npz"
    camera_rel = Path("camera") / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.npz"
    write_h264(out_root / video_rel, rgb_frames, fps)
    (out_root / depth_rel).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_root / depth_rel, depth_m=np.stack(depth_frames))
    (out_root / camera_rel).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_root / camera_rel, agentview_K=np.stack(k_frames), agentview_T_world_camera=np.stack(t_frames))
    out_df = src_df.copy()
    out_df["observation.depth.image_m_path"] = depth_rel.as_posix()
    out_df["observation.camera.params_path"] = camera_rel.as_posix()
    out_df["observation.left_eef_pos"] = list(np.stack(left_eef_positions))
    out_df["observation.right_eef_pos"] = list(np.stack(right_eef_positions))
    out_df["observation.left_robot_eef_pos"] = list(np.stack(left_robot_eef_positions))
    out_df["observation.right_robot_eef_pos"] = list(np.stack(right_robot_eef_positions))
    out_df["observation.left_pinch_pos"] = list(np.stack(left_pinch_positions))
    out_df["observation.right_pinch_pos"] = list(np.stack(right_pinch_positions))
    out_df["observation.left_thumb_index_pinch_pos"] = list(np.stack(left_thumb_index_pinch_positions))
    out_df["observation.right_thumb_index_pinch_pos"] = list(np.stack(right_thumb_index_pinch_positions))
    out_df["source.hdf5_demo_id"] = demo_id
    language_key = "annotation.human.coarse_action"
    if language_key not in out_df.columns:
        raise RuntimeError(f"source episode {episode_index} is missing {language_key}")
    out_df[language_key] = int(episode_to_task[episode_index])
    out_parquet = out_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    return {"episode_index": episode_index, "length": len(states), "hdf5": str(h5.filename), "demo_id": demo_id, "rgb_video": video_rel.as_posix(), "depth_npz": depth_rel.as_posix(), "camera_npz": camera_rel.as_posix()}


def main() -> None:
    args = parse_args()
    base_root, hdf5_path, out_root, repo = map(Path, (args.base_lerobot_root, args.hdf5, args.output_root, args.robocasa_repo))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    full_mapping = build_mapping(base_root, hdf5_path)
    end = args.episode_end if args.episode_end > 0 else len(full_mapping)
    mapping = [row for row in full_mapping if args.episode_start <= int(row["episode_index"]) < end]
    if args.max_episodes > 0:
        mapping = mapping[:args.max_episodes]
    if not mapping:
        raise RuntimeError("No episodes selected; check --episode-start/--episode-end")
    if args.prepare_only and args.skip_prepare:
        raise ValueError("--prepare-only and --skip-prepare are mutually exclusive")
    if args.prepare_only:
        prepare_output(
            base_root,
            out_root,
            mapping,
            fps=args.fps,
            resolution=args.resolution,
            overwrite=args.overwrite,
        )
        print(f"[prepared] {out_root} episodes={len(mapping)}", flush=True)
        return
    if args.skip_prepare:
        if not (out_root / "meta" / "info.json").exists():
            raise FileNotFoundError("--skip-prepare requires an already initialized output root")
        _, episode_to_task = build_episode_instruction_table(
            read_jsonl(base_root / "meta" / "episodes.jsonl")
        )
    else:
        episode_to_task = prepare_output(
            base_root,
            out_root,
            mapping,
            fps=args.fps,
            resolution=args.resolution,
            overwrite=args.overwrite,
        )
    from argparse import Namespace
    from robocasa.scripts.playback_dataset import make_env_from_args, reset_to as robocasa_reset_to
    from robosuite.utils import camera_utils
    from robocasa_book_filter import make_book_safe_reset_to

    env = make_env_from_args(Namespace(dataset=str(hdf5_path), use_abs_actions=False, render=False, verbose=False))
    reset_to = make_book_safe_reset_to(robocasa_reset_to)
    result = []
    try:
        with h5py.File(hdf5_path, "r") as h5:
            for i, row in enumerate(mapping, 1):
                print(f"[render {i}/{len(mapping)}] episode={row['episode_index']:06d} demo={row['demo_id']}", flush=True)
                result.append(
                    render_episode(
                        env,
                        camera_utils,
                        h5,
                        row,
                        base_root,
                        out_root,
                        args.resolution,
                        args.fps,
                        episode_to_task,
                        reset_to,
                    )
                )
    finally:
        env.close()
    shard_name = f"source_mapping_{mapping[0]['episode_index']:06d}_{mapping[-1]['episode_index']:06d}.jsonl"
    write_jsonl(out_root / "meta" / shard_name, result)
    (out_root / "meta" / f"replay_rgbd_summary_{mapping[0]['episode_index']:06d}_{mapping[-1]['episode_index']:06d}.json").write_text(json.dumps({"status": "success", "num_episodes": len(result), "num_frames": sum(r["length"] for r in result), "resolution": args.resolution, "fps": args.fps}, indent=2) + "\n")


if __name__ == "__main__":
    main()
