#!/usr/bin/env python3
"""Replay complete CALVIN episodes and write native RGB-D + UVD outputs.

This is intentionally CALVIN-native.  Each source frame provides the exact
``robot_obs`` and ``scene_obs`` used to reset PyBullet; the source actions are
copied unchanged for alignment.  RGB and depth come from the fresh CALVIN
render.  UVD is computed from the current frame's PyBullet view/projection
matrices by projecting the EEF world position.

The output contains one HDF5 file and one full comparison video per episode.
The video has original RGB, rerendered RGB, RGB difference, UV overlay, and the
same four panels for metric depth.  No LIBERO image flip or camera convention
is used here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import h5py
import hydra
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf


STATIC_SIZE = (200, 200)
GRIPPER_SIZE = (84, 84)
DEPTH_RANGES = {"static": (3.5, 6.3), "gripper": (0.05, 1.2)}
DEPTH_DIFF_RANGES = {"static": 0.30, "gripper": 0.12}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--config-split", choices=("auto", "validation", "training"), default="auto")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-comparison", action="store_true", help="Skip comparison MP4/JPEG files for large runs.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_config_dir(dataset_root: Path, config_split: str) -> Path:
    candidates = ("validation", "training") if config_split == "auto" else (config_split,)
    for split in candidates:
        config_dir = dataset_root / split
        if (config_dir / ".hydra" / "merged_config.yaml").exists():
            return config_dir
    checked = ", ".join(str(dataset_root / x / ".hydra" / "merged_config.yaml") for x in candidates)
    raise FileNotFoundError("missing CALVIN config; checked: " + checked)


def select_complete_episodes(dataset_root: Path, start: int, count: int) -> List[Tuple[int, int, int]]:
    training = dataset_root / "training"
    boundaries = np.asarray(np.load(training / "ep_start_end_ids.npy"), dtype=np.int64)
    frame_ids = {int(path.stem.split("_")[-1]) for path in training.glob("episode_*.npz")}
    selected: List[Tuple[int, int, int]] = []
    for index, (begin, end) in enumerate(boundaries.tolist()):
        if index < start:
            continue
        expected = int(end - begin + 1)
        present = sum(int(frame_id) in frame_ids for frame_id in range(int(begin), int(end) + 1))
        if present != expected:
            continue
        selected.append((index, int(begin), int(end)))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError("only found %d complete episodes; requested %d" % (len(selected), count))
    return selected


def bullet_matrix(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(4, 4).T


def camera_calibration(camera: Any) -> Dict[str, np.ndarray]:
    view_values = camera.viewMatrix if hasattr(camera, "viewMatrix") else camera.view_matrix
    projection_values = camera.projectionMatrix if hasattr(camera, "projectionMatrix") else camera.projection_matrix
    view_opengl = bullet_matrix(view_values)
    projection = bullet_matrix(projection_values)
    width = int(camera.width)
    height = int(camera.height)

    # Convert PyBullet/OpenGL camera coordinates (+Y up, -Z forward) to the
    # conventional CV coordinates (+Y down, +Z forward) used by UVD/K/T.
    opengl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    t_cv_world = opengl_to_cv @ view_opengl
    t_world_camera = np.linalg.inv(t_cv_world).astype(np.float32)
    fx = float(projection[0, 0]) * width / 2.0
    fy = float(projection[1, 1]) * height / 2.0
    cx = (1.0 - float(projection[0, 2])) * width / 2.0
    cy = (1.0 + float(projection[1, 2])) * height / 2.0
    k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return {
        "view_opengl": view_opengl,
        "projection": projection,
        "t_world_camera": t_world_camera,
        "k": k,
        "width": np.asarray(width, dtype=np.int32),
        "height": np.asarray(height, dtype=np.int32),
    }


def project_world_to_uvd(
    view_opengl: np.ndarray,
    projection: np.ndarray,
    point_world: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    point = np.concatenate([np.asarray(point_world, dtype=np.float32).reshape(3), np.ones(1, dtype=np.float32)])
    camera = np.asarray(view_opengl, dtype=np.float32) @ point
    clip = np.asarray(projection, dtype=np.float32) @ camera
    if abs(float(clip[3])) < 1e-8:
        return np.full(2, np.nan, dtype=np.float32), np.full(3, np.nan, dtype=np.float32), False
    ndc = clip[:3] / clip[3]
    uv = np.asarray([(ndc[0] + 1.0) * width / 2.0, (1.0 - ndc[1]) * height / 2.0], dtype=np.float32)
    # In PyBullet/OpenGL, camera-space -Z is forward.  UVD stores positive Z.
    uvd = np.asarray([uv[0], uv[1], -camera[2]], dtype=np.float32)
    valid = bool(np.isfinite(uvd).all() and uvd[2] > 0 and 0 <= uv[0] < width and 0 <= uv[1] < height)
    return uv, uvd, valid


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = np.ascontiguousarray(image).copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1] - 1, 260), 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def resize_rgb(image: np.ndarray, size: int = 200) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    return np.ascontiguousarray(cv2.resize(np.asarray(image, dtype=np.uint8), (size, size), interpolation=cv2.INTER_NEAREST))


def depth_visual(depth: np.ndarray, camera: str) -> np.ndarray:
    low, high = DEPTH_RANGES[camera]
    normalized = np.clip((np.asarray(depth, dtype=np.float32) - low) / (high - low), 0.0, 1.0)
    bgr = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return bgr[:, :, ::-1]


def diff_visual(left: np.ndarray, right: np.ndarray, depth: bool, camera: str) -> np.ndarray:
    if depth:
        scale = DEPTH_DIFF_RANGES[camera]
        value = np.clip(np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)) / scale, 0.0, 1.0)
        bgr = cv2.applyColorMap((value * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
        return bgr[:, :, ::-1]
    value = np.abs(np.asarray(left, dtype=np.int16) - np.asarray(right, dtype=np.int16)).mean(axis=2)
    value = np.clip(value * 4.0, 0.0, 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(value, cv2.COLORMAP_INFERNO)
    return bgr[:, :, ::-1]


def overlay_uv(image: np.ndarray, uv: np.ndarray, valid: bool, size: int = 200) -> np.ndarray:
    out = resize_rgb(image, size)
    if valid and np.isfinite(uv).all():
        x = int(round(float(uv[0]) * size / image.shape[1]))
        y = int(round(float(uv[1]) * size / image.shape[0]))
        cv2.circle(out, (x, y), 5, (255, 0, 0), 2)
        cv2.circle(out, (x, y), 1, (255, 255, 255), -1)
    return out


def comparison_frame(original: Dict[str, np.ndarray], rendered: Dict[str, Any], frame_id: int) -> np.ndarray:
    cells: List[List[np.ndarray]] = []
    for camera in ("static", "gripper"):
        prefix = "rgb_" + camera
        original_rgb = original[prefix]
        rendered_rgb = rendered[prefix]
        cells.append([
            label(resize_rgb(original_rgb), "original RGB"),
            label(resize_rgb(rendered_rgb), "rerender RGB"),
            label(resize_rgb(diff_visual(original_rgb, rendered_rgb, False, camera)), "RGB abs diff"),
            label(overlay_uv(rendered_rgb, rendered["uv_" + camera], rendered["uv_valid_" + camera]), "rerender + EEF UV"),
        ])
        prefix = "depth_" + camera
        original_depth = original[prefix]
        rendered_depth = rendered[prefix]
        cells.append([
            label(resize_rgb(depth_visual(original_depth, camera)), "original depth"),
            label(resize_rgb(depth_visual(rendered_depth, camera)), "rerender depth"),
            label(resize_rgb(diff_visual(original_depth, rendered_depth, True, camera)), "depth abs diff"),
            label(overlay_uv(depth_visual(rendered_depth, camera), rendered["uv_" + camera], rendered["uv_valid_" + camera]), "depth + EEF UV"),
        ])
    rows = [np.concatenate(row, axis=1) for row in cells]
    canvas = np.concatenate(rows, axis=0)
    return label(canvas, "frame %d | top static, bottom wrist" % frame_id)


class EpisodeWriter:
    def __init__(self, path: Path, length: int) -> None:
        self.file = h5py.File(str(path), "w")
        self.index = 0
        self.datasets: Dict[str, h5py.Dataset] = {}
        specs = {
            "frame_id": ((length,), np.int64),
            "actions": ((length, 7), np.float32),
            "rel_actions": ((length, 7), np.float32),
            "robot_obs": ((length, 15), np.float32),
            "scene_obs": ((length, 24), np.float32),
            "rgb_static": ((length, 200, 200, 3), np.uint8),
            "rgb_gripper": ((length, 84, 84, 3), np.uint8),
            "depth_static_m": ((length, 200, 200), np.float32),
            "depth_gripper_m": ((length, 84, 84), np.float32),
            "uv_static": ((length, 2), np.float32),
            "uv_gripper": ((length, 2), np.float32),
            "uvd_static": ((length, 3), np.float32),
            "uvd_gripper": ((length, 3), np.float32),
            "uv_valid_static": ((length,), np.bool_),
            "uv_valid_gripper": ((length,), np.bool_),
            "camera_K_static": ((length, 3, 3), np.float32),
            "camera_K_gripper": ((length, 3, 3), np.float32),
            "camera_T_world_camera_static": ((length, 4, 4), np.float32),
            "camera_T_world_camera_gripper": ((length, 4, 4), np.float32),
            "camera_view_static": ((length, 4, 4), np.float32),
            "camera_view_gripper": ((length, 4, 4), np.float32),
            "camera_projection_static": ((length, 4, 4), np.float32),
            "camera_projection_gripper": ((length, 4, 4), np.float32),
        }
        for name, (shape, dtype) in specs.items():
            self.datasets[name] = self.file.create_dataset(name, shape=shape, dtype=dtype, chunks=(1,) + shape[1:], compression="lzf", shuffle=True)

    def append(self, values: Dict[str, Any]) -> None:
        for name, dataset in self.datasets.items():
            dataset[self.index] = values[name]
        self.index += 1

    def close(self) -> None:
        self.file.attrs["num_frames"] = self.index
        self.file.attrs["uv_definition"] = "EEF robot_obs[:3] projected by current Calvin PyBullet view/projection matrices"
        self.file.attrs["camera_coordinate_convention"] = "CV: +X right, +Y down, +Z forward; T_world_camera is camera-to-world"
        self.file.close()


def make_env(config_dir: Path) -> Any:
    config = OmegaConf.load(str(config_dir / ".hydra" / "merged_config.yaml"))
    config.env.use_egl = False
    config.env.show_gui = False
    if "tactile" in config.env.cameras:
        config.env.cameras.pop("tactile")
    return hydra.utils.instantiate(config.env, show_gui=False, use_vr=False, use_scene_info=True)


def render_episode(env: Any, raw_dir: Path, output_dir: Path, episode_index: int, begin: int, end: int, fps: float, write_comparison: bool = True) -> Dict[str, Any]:
    length = end - begin + 1
    h5_path = output_dir / ("episode_%03d.h5" % episode_index)
    video_path = output_dir / ("episode_%03d_comparison.mp4" % episode_index)
    writer = EpisodeWriter(h5_path, length)
    video = imageio.get_writer(str(video_path), fps=fps, codec="libx264", quality=5, macro_block_size=1, ffmpeg_params=["-preset", "ultrafast", "-crf", "23"]) if write_comparison else None
    contacts: List[np.ndarray] = []
    contact_offsets = set(np.linspace(0, length - 1, 9, dtype=np.int64).tolist())
    static_camera = next(cam for cam in env.cameras if cam.name == "static")
    gripper_camera = next(cam for cam in env.cameras if cam.name == "gripper")
    try:
        for offset, frame_id in enumerate(range(begin, end + 1)):
            source_path = raw_dir / ("episode_%07d.npz" % frame_id)
            with np.load(source_path, allow_pickle=False) as source:
                original = {key: np.asarray(source[key]) for key in ("rgb_static", "rgb_gripper", "depth_static", "depth_gripper")}
                robot_obs = np.asarray(source["robot_obs"], dtype=np.float32)
                scene_obs = np.asarray(source["scene_obs"], dtype=np.float32)
                env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                rgb_obs, depth_obs = env.get_camera_obs()
                static_cal = camera_calibration(static_camera)
                gripper_cal = camera_calibration(gripper_camera)
                _, uvd_static, valid_static = project_world_to_uvd(static_cal["view_opengl"], static_cal["projection"], robot_obs[:3], 200, 200)
                _, uvd_gripper, valid_gripper = project_world_to_uvd(gripper_cal["view_opengl"], gripper_cal["projection"], robot_obs[:3], 84, 84)
                rendered = {
                    "rgb_static": np.asarray(rgb_obs["rgb_static"], dtype=np.uint8),
                    "rgb_gripper": np.asarray(rgb_obs["rgb_gripper"], dtype=np.uint8),
                    "depth_static": np.asarray(depth_obs["depth_static"], dtype=np.float32),
                    "depth_gripper": np.asarray(depth_obs["depth_gripper"], dtype=np.float32),
                    "uv_static": uvd_static[:2], "uv_gripper": uvd_gripper[:2],
                    "uv_valid_static": valid_static, "uv_valid_gripper": valid_gripper,
                    "depth_static_m": np.asarray(depth_obs["depth_static"], dtype=np.float32),
                    "depth_gripper_m": np.asarray(depth_obs["depth_gripper"], dtype=np.float32),
                }
                values = {
                    "frame_id": frame_id,
                    "actions": np.asarray(source["actions"], dtype=np.float32),
                    "rel_actions": np.asarray(source["rel_actions"], dtype=np.float32),
                    "robot_obs": robot_obs, "scene_obs": scene_obs,
                    "rgb_static": rendered["rgb_static"], "rgb_gripper": rendered["rgb_gripper"],
                    "depth_static_m": rendered["depth_static_m"], "depth_gripper_m": rendered["depth_gripper_m"],
                    "uv_static": rendered["uv_static"], "uv_gripper": rendered["uv_gripper"],
                    "uvd_static": uvd_static, "uvd_gripper": uvd_gripper,
                    "uv_valid_static": valid_static, "uv_valid_gripper": valid_gripper,
                }
                for camera, cal in (("static", static_cal), ("gripper", gripper_cal)):
                    values["camera_K_" + camera] = cal["k"]
                    values["camera_T_world_camera_" + camera] = cal["t_world_camera"]
                    values["camera_view_" + camera] = cal["view_opengl"]
                    values["camera_projection_" + camera] = cal["projection"]
                writer.append(values)
                if write_comparison:
                    canvas = comparison_frame(original, rendered, frame_id)
                    video.append_data(canvas)
                    if offset in contact_offsets:
                        contacts.append(canvas)
            if offset % 500 == 0:
                print("[episode %03d] %d/%d" % (episode_index, offset + 1, length), flush=True)
    finally:
        if video is not None:
            video.close()
        writer.close()
    if write_comparison:
        contact = np.concatenate([np.concatenate(contacts[i:i + 3], axis=1) for i in range(0, len(contacts), 3)], axis=0)
        imageio.imwrite(output_dir / ("episode_%03d_contact.jpg" % episode_index), contact)
    return {"episode_index": episode_index, "begin": begin, "end": end, "frames": length, "h5": h5_path.name, "video": video_path.name if write_comparison else None}


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    raw_dir = dataset_root / "training"
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError("output exists; pass --overwrite: " + str(args.output_root))
        import shutil
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_dir = resolve_config_dir(dataset_root, args.config_split)
    selected = select_complete_episodes(dataset_root, args.start_episode, args.num_episodes)
    print("selected episodes:", selected, flush=True)
    env = make_env(config_dir)
    summaries: List[Dict[str, Any]] = []
    try:
        for episode_index, begin, end in selected:
            summaries.append(render_episode(env, raw_dir, args.output_root, episode_index, begin, end, args.fps, write_comparison=not args.no_comparison))
    finally:
        env.close()
    summary = {"status": "success", "source_dataset": str(dataset_root), "config_dir": str(config_dir), "episodes": summaries, "uv_point": "robot_obs[:3] EEF world position", "output_root": str(args.output_root), "comparison_visualizations": not args.no_comparison}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
