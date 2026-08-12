#!/usr/bin/env python3
"""Preview physical LIBERO gripper triangles on existing agentview videos."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from starVLA.gripper_triangle import (
    LANDMARK_BODY_NAMES,
    project_world_to_agentview_uvd as shared_project_world_to_agentview_uvd,
)


POINT_BODY_NAMES = LANDMARK_BODY_NAMES


def select_evenly_spaced_episode_ids(total_episodes: int, count: int) -> list[int]:
    """Select deterministic episode IDs spanning the complete suite."""

    total_episodes = int(total_episodes)
    count = int(count)
    if total_episodes < 1:
        raise ValueError(f"total_episodes must be positive, got {total_episodes}")
    if count < 1 or count > total_episodes:
        raise ValueError(
            f"count must be positive and no larger than total_episodes, got "
            f"count={count}, total_episodes={total_episodes}"
        )
    return np.rint(np.linspace(0, total_episodes - 1, count)).astype(np.int64).tolist()


def project_world_points(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    world_from_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project batched world-frame points through the shared sidecar helper."""

    coordinate_limit = int(np.iinfo(np.int32).max)
    uvd, valid, _ = shared_project_world_to_agentview_uvd(
        points_world,
        intrinsics,
        world_from_camera,
        width=coordinate_limit,
        height=coordinate_limit,
    )
    return (
        uvd[..., :2].astype(np.float64),
        uvd[..., 2].astype(np.float64),
        valid,
    )


def triangle_geometry(points_world: np.ndarray) -> dict[str, Any]:
    """Measure finger separation and 3D triangle area for `[L, R, W]` points."""

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (3, 3):
        raise ValueError(f"points_world must have shape [T,3,3], got {points.shape}")
    left_to_right = points[:, 1] - points[:, 0]
    left_to_wrist = points[:, 2] - points[:, 0]
    return {
        "finger_distance_m": np.linalg.norm(left_to_right, axis=-1),
        "triangle_area_m2": 0.5 * np.linalg.norm(
            np.cross(left_to_right, left_to_wrist), axis=-1
        ),
    }


def draw_triangle_overlay(
    image: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    *,
    frame_index: int,
    finger_distance_m: float,
    triangle_area_m2: float,
) -> np.ndarray:
    """Draw one `[L, R, W]` projected triangle on an RGB frame."""

    import cv2

    source = np.asarray(image)
    points = np.asarray(uv, dtype=np.float64)
    point_valid = np.asarray(valid, dtype=np.bool_)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError(f"image must be uint8 RGB [H,W,3], got {source.shape}/{source.dtype}")
    if points.shape != (3, 2) or point_valid.shape != (3,):
        raise ValueError(f"uv/valid must have shape [3,2]/[3], got {points.shape}/{point_valid.shape}")

    output = source.copy()
    height, width = output.shape[:2]
    integer_points: list[tuple[int, int] | None] = []
    for point, is_valid in zip(points, point_valid):
        if not is_valid or not np.isfinite(point).all():
            integer_points.append(None)
        else:
            integer_points.append((int(np.rint(point[0])), int(np.rint(point[1]))))

    edge_colors = ((130, 245, 255), (255, 160, 230), (255, 235, 110))
    for edge_index, (start_index, end_index) in enumerate(((0, 1), (1, 2), (2, 0))):
        start = integer_points[start_index]
        end = integer_points[end_index]
        if start is None or end is None:
            continue
        clipped, clipped_start, clipped_end = cv2.clipLine((0, 0, width, height), start, end)
        if clipped:
            cv2.line(output, clipped_start, clipped_end, edge_colors[edge_index], 2, cv2.LINE_AA)

    point_colors = ((0, 230, 255), (255, 60, 220), (255, 220, 0))
    for label, point, color in zip(("L", "R", "W"), integer_points, point_colors):
        if point is None or not (0 <= point[0] < width and 0 <= point[1] < height):
            continue
        cv2.circle(output, point, 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, point, 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            label,
            (min(point[0] + 6, width - 10), max(point[1] - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(output, (0, 0), (min(width - 1, 250), 19), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"t={int(frame_index):04d} gap={float(finger_distance_m) * 100:.1f}cm "
        f"area={float(triangle_area_m2) * 1e4:.1f}cm2",
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def render_triangle_video(
    source_path: str | Path,
    output_path: str | Path,
    *,
    uv: np.ndarray,
    valid: np.ndarray,
    geometry: dict[str, np.ndarray],
    suite: str,
    episode_id: int,
    snapshot_indices: Sequence[int],
) -> dict[str, Any]:
    """Overlay one complete trajectory while preserving source timing and length."""

    import av
    import cv2

    from examples.simBenchmarks.CoT.geometry_probe.episode_video import write_video_frames

    projected = np.asarray(uv, dtype=np.float64)
    projection_valid = np.asarray(valid, dtype=np.bool_)
    finger_distance = np.asarray(geometry["finger_distance_m"], dtype=np.float64)
    triangle_area = np.asarray(geometry["triangle_area_m2"], dtype=np.float64)
    frame_count = projected.shape[0]
    if projected.shape != (frame_count, 3, 2) or projection_valid.shape != (frame_count, 3):
        raise ValueError("uv/valid must have shapes [T,3,2] and [T,3]")
    if finger_distance.shape != (frame_count,) or triangle_area.shape != (frame_count,):
        raise ValueError("geometry arrays must have shape [T]")
    requested_snapshots = tuple(int(index) for index in snapshot_indices)
    if any(index < 0 or index >= frame_count for index in requested_snapshots):
        raise ValueError(f"snapshot indices must be within [0,{frame_count}), got {requested_snapshots}")

    container = av.open(str(source_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 20.0
    snapshots: dict[int, np.ndarray] = {}
    decoded_count = 0

    def overlay_frames():
        nonlocal decoded_count
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index >= frame_count:
                raise ValueError(f"source video has more than {frame_count} trajectory frames")
            rgb = frame.to_ndarray(format="rgb24")
            rendered = draw_triangle_overlay(
                rgb,
                projected[frame_index],
                projection_valid[frame_index],
                frame_index=frame_index,
                finger_distance_m=float(finger_distance[frame_index]),
                triangle_area_m2=float(triangle_area[frame_index]),
            )
            cv2.putText(
                rendered,
                f"{suite}  episode {int(episode_id):06d}",
                (4, rendered.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if frame_index in requested_snapshots:
                snapshots[frame_index] = rendered.copy()
            decoded_count += 1
            yield rendered
        if decoded_count != frame_count:
            raise ValueError(
                f"source video frame count {decoded_count} != trajectory frame count {frame_count}"
            )

    try:
        video_info = write_video_frames(output_path, overlay_frames(), fps=fps)
    finally:
        container.close()
    if len(snapshots) != len(set(requested_snapshots)):
        raise ValueError("not every requested snapshot was decoded")
    return {
        **video_info,
        "snapshots": [snapshots[index] for index in requested_snapshots],
    }


def write_suite_contact_sheet(
    path: str | Path,
    episode_snapshots: Sequence[tuple[int, Sequence[np.ndarray]]],
) -> Path:
    """Write one row of early/middle/late frames for every suite episode."""

    from PIL import Image, ImageDraw

    rows = list(episode_snapshots)
    if not rows:
        raise ValueError("at least one episode snapshot row is required")
    column_count = len(rows[0][1])
    if column_count < 1:
        raise ValueError("at least one snapshot per episode is required")
    first = np.asarray(rows[0][1][0])
    height, width = first.shape[:2]
    label_height = 22
    canvas = Image.new("RGB", (width * column_count, (height + label_height) * len(rows)), "black")
    draw = ImageDraw.Draw(canvas)
    for row_index, (episode_id, snapshots) in enumerate(rows):
        if len(snapshots) != column_count:
            raise ValueError("every episode must provide the same number of snapshots")
        row_y = row_index * (height + label_height)
        draw.text((5, row_y + 4), f"episode {int(episode_id):06d}  early / middle / late", fill="white")
        for column_index, snapshot in enumerate(snapshots):
            array = np.asarray(snapshot)
            if array.shape != first.shape or array.dtype != np.uint8:
                raise ValueError("contact-sheet snapshots must share uint8 RGB shape")
            canvas.paste(Image.fromarray(array, mode="RGB"), (column_index * width, row_y + label_height))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, subsampling=0)
    return output_path


SUITE_DIRECTORY_NAMES = (
    ("libero_10", "libero_10_no_noops_1.0.0_lerobot"),
    ("libero_goal", "libero_goal_no_noops_1.0.0_lerobot"),
    ("libero_object", "libero_object_no_noops_1.0.0_lerobot"),
    ("libero_spatial", "libero_spatial_no_noops_1.0.0_lerobot"),
)


@dataclass(frozen=True)
class SuiteRoot:
    name: str
    path: Path
    total_episodes: int


@dataclass(frozen=True)
class EpisodeRecord:
    suite: str
    dataset_root: Path
    episode_id: int
    frame_count: int
    hdf5_path: Path
    demo_id: str
    hdf5_indices: np.ndarray
    camera_path: Path
    video_path: Path


def _one_value(frame: Any, column: str) -> Any:
    if column not in frame.columns:
        raise ValueError(f"episode parquet is missing required column {column}")
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"{column} must contain exactly one value, got {values}")
    return values[0]


def episode_record_from_dataframe(
    dataset_root: str | Path,
    suite: str,
    episode_id: int,
    frame: Any,
) -> EpisodeRecord:
    """Validate one episode's replay mapping and resolve its artifact paths."""

    root = Path(dataset_root)
    episode_id = int(episode_id)
    if len(frame) < 1:
        raise ValueError(f"episode {episode_id} parquet is empty")
    hdf5_path = Path(str(_one_value(frame, "source.hdf5_path")))
    demo_id = str(_one_value(frame, "source.hdf5_demo_id"))
    camera_relative = Path(str(_one_value(frame, "observation.camera.params_path")))
    hdf5_indices = np.asarray(frame["source.hdf5_index"].to_numpy(), dtype=np.int64)
    chunk = episode_id // 1000
    return EpisodeRecord(
        suite=str(suite),
        dataset_root=root,
        episode_id=episode_id,
        frame_count=int(len(frame)),
        hdf5_path=hdf5_path,
        demo_id=demo_id,
        hdf5_indices=hdf5_indices,
        camera_path=root / camera_relative,
        video_path=(
            root
            / "videos"
            / f"chunk-{chunk:03d}"
            / "observation.images.image"
            / f"episode_{episode_id:06d}.mp4"
        ),
    )


def episode_parquet_path(dataset_root: str | Path, episode_id: int) -> Path:
    root = Path(dataset_root)
    episode_id = int(episode_id)
    expected = root / "data" / f"chunk-{episode_id // 1000:03d}" / f"episode_{episode_id:06d}.parquet"
    if expected.is_file():
        return expected
    matches = list((root / "data").glob(f"chunk-*/episode_{episode_id:06d}.parquet"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(expected)


def load_episode_record(suite: SuiteRoot, episode_id: int) -> EpisodeRecord:
    import pandas as pd

    frame = pd.read_parquet(episode_parquet_path(suite.path, episode_id))
    record = episode_record_from_dataframe(suite.path, suite.name, episode_id, frame)
    for required in (record.hdf5_path, record.camera_path, record.video_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    return record


def discover_suite_roots(dataset_root: str | Path) -> list[SuiteRoot]:
    root = Path(dataset_root)
    suites: list[SuiteRoot] = []
    for suite_name, directory_name in SUITE_DIRECTORY_NAMES:
        suite_path = root / directory_name
        info_path = suite_path / "meta" / "info.json"
        if not info_path.is_file():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        suites.append(
            SuiteRoot(
                name=suite_name,
                path=suite_path,
                total_episodes=int(info["total_episodes"]),
            )
        )
    if not suites:
        raise FileNotFoundError(f"no LIBERO rerender suites found below {root}")
    return suites


def load_agentview_camera(record: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
    with np.load(record.camera_path) as camera:
        intrinsics = np.asarray(camera["agentview_K"], dtype=np.float64)
        world_from_camera = np.asarray(camera["agentview_T_world_camera"], dtype=np.float64)
    camera_frames = int(intrinsics.shape[0])
    if camera_frames != record.frame_count:
        raise ValueError(
            f"camera frame count {camera_frames} != episode frame count {record.frame_count}"
        )
    if intrinsics.shape != (record.frame_count, 3, 3):
        raise ValueError(f"unexpected agentview_K shape {intrinsics.shape}")
    if world_from_camera.shape != (record.frame_count, 4, 4):
        raise ValueError(
            f"unexpected agentview_T_world_camera shape {world_from_camera.shape}"
        )
    return intrinsics, world_from_camera


def summarize_episode_metrics(
    points_world: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, float]:
    geometry = triangle_geometry(points_world)
    projected = np.asarray(uv, dtype=np.float64)
    projection_valid = np.asarray(valid, dtype=np.bool_)
    in_frame = (
        projection_valid
        & (projected[..., 0] >= 0.0)
        & (projected[..., 0] < int(width))
        & (projected[..., 1] >= 0.0)
        & (projected[..., 1] < int(height))
    )
    return {
        "min_finger_distance_m": float(np.min(geometry["finger_distance_m"])),
        "max_finger_distance_m": float(np.max(geometry["finger_distance_m"])),
        "min_triangle_area_m2": float(np.min(geometry["triangle_area_m2"])),
        "max_triangle_area_m2": float(np.max(geometry["triangle_area_m2"])),
        "projection_valid_point_ratio": float(np.mean(projection_valid)),
        "in_frame_point_ratio": float(np.mean(in_frame)),
        "all_three_in_frame_frame_ratio": float(np.mean(np.all(in_frame, axis=1))),
    }


def inspect_video(path: str | Path) -> dict[str, Any]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "fps": float(stream.average_rate) if stream.average_rate else 20.0,
            "codec": str(stream.codec_context.name),
            "pixel_format": str(stream.codec_context.pix_fmt),
        }


def extract_body_trajectory(env: Any, h5: Any, record: EpisodeRecord) -> np.ndarray:
    """Restore source states and read `[left tip, right tip, hand base]` positions."""

    state_dataset = h5[f"data/{record.demo_id}/states"]
    if np.any(record.hdf5_indices < 0) or np.any(record.hdf5_indices >= len(state_dataset)):
        raise IndexError(
            f"episode {record.episode_id} contains HDF5 indices outside [0,{len(state_dataset)})"
        )
    body_ids = []
    for body_name in POINT_BODY_NAMES:
        try:
            body_ids.append(int(env.sim.model.body_name2id(body_name)))
        except Exception as error:
            raise KeyError(f"MuJoCo model is missing required body {body_name}") from error
    trajectory = np.empty((record.frame_count, 3, 3), dtype=np.float64)
    for frame_index, source_index in enumerate(record.hdf5_indices):
        env.regenerate_obs_from_state(state_dataset[int(source_index)])
        trajectory[frame_index] = np.asarray(env.sim.data.body_xpos[body_ids], dtype=np.float64)
    return trajectory


def _episode_manifest_entry(
    record: EpisodeRecord,
    output_path: Path,
    video_info: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "suite": record.suite,
        "episode_id": record.episode_id,
        "frame_count": record.frame_count,
        "source": {
            "dataset_root": str(record.dataset_root),
            "video": str(record.video_path),
            "camera": str(record.camera_path),
            "hdf5": str(record.hdf5_path),
            "demo_id": record.demo_id,
            "hdf5_index_first": int(record.hdf5_indices[0]),
            "hdf5_index_last": int(record.hdf5_indices[-1]),
        },
        "output_video": str(output_path),
        "video": {
            key: value
            for key, value in video_info.items()
            if key != "snapshots"
        },
        "metrics": metrics,
    }


def _render_hdf5_group(
    hdf5_path: Path,
    records: Sequence[EpisodeRecord],
    suite_output: Path,
) -> tuple[list[dict[str, Any]], list[tuple[int, Sequence[np.ndarray]]]]:
    import h5py

    from examples.modelExtensions.CoT.scripts.export_libero_hdf5_state_depth import (
        env_kwargs_from_hdf5_attrs,
    )
    from libero.libero.envs import OffScreenRenderEnv

    entries: list[dict[str, Any]] = []
    snapshots: list[tuple[int, Sequence[np.ndarray]]] = []
    with h5py.File(hdf5_path, "r") as h5:
        env_kwargs = env_kwargs_from_hdf5_attrs(dict(h5["data"].attrs), ["agentview"], 256)
        env = OffScreenRenderEnv(**env_kwargs)
        try:
            for record in records:
                print(
                    f"  [episode] {record.suite}/{record.episode_id:06d} "
                    f"frames={record.frame_count} source={hdf5_path.name}",
                    flush=True,
                )
                points_world = extract_body_trajectory(env, h5, record)
                intrinsics, world_from_camera = load_agentview_camera(record)
                uv, _, valid = project_world_points(
                    points_world, intrinsics, world_from_camera
                )
                source_video = inspect_video(record.video_path)
                metrics = summarize_episode_metrics(
                    points_world,
                    uv,
                    valid,
                    width=int(source_video["width"]),
                    height=int(source_video["height"]),
                )
                if metrics["min_finger_distance_m"] <= 1e-4:
                    raise ValueError(
                        f"episode {record.episode_id} finger points collapse: {metrics}"
                    )
                if metrics["min_triangle_area_m2"] <= 1e-7:
                    raise ValueError(
                        f"episode {record.episode_id} triangle degenerates: {metrics}"
                    )
                if metrics["projection_valid_point_ratio"] < 1.0:
                    raise ValueError(
                        f"episode {record.episode_id} has invalid positive-depth projections: {metrics}"
                    )

                geometry = triangle_geometry(points_world)
                output_path = suite_output / "videos" / f"episode_{record.episode_id:06d}.mp4"
                snapshot_indices = tuple(
                    select_evenly_spaced_episode_ids(record.frame_count, min(3, record.frame_count))
                )
                video_info = render_triangle_video(
                    record.video_path,
                    output_path,
                    uv=uv,
                    valid=valid,
                    geometry=geometry,
                    suite=record.suite,
                    episode_id=record.episode_id,
                    snapshot_indices=snapshot_indices,
                )
                if int(video_info["frame_count"]) != record.frame_count:
                    raise ValueError(
                        f"output frame count {video_info['frame_count']} != {record.frame_count}"
                    )
                entries.append(
                    _episode_manifest_entry(record, output_path, video_info, metrics)
                )
                snapshots.append((record.episode_id, video_info["snapshots"]))
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    return entries, snapshots


def render_suite(
    suite: SuiteRoot,
    episode_ids: Sequence[int],
    output_root: str | Path,
) -> tuple[list[dict[str, Any]], Path]:
    suite_output = Path(output_root) / suite.name
    records = [load_episode_record(suite, episode_id) for episode_id in episode_ids]
    by_hdf5: dict[Path, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_hdf5[record.hdf5_path].append(record)
    entries: list[dict[str, Any]] = []
    snapshots_by_episode: dict[int, Sequence[np.ndarray]] = {}
    for group_index, (hdf5_path, group_records) in enumerate(by_hdf5.items(), start=1):
        print(
            f"[hdf5] {suite.name} {group_index}/{len(by_hdf5)} "
            f"{hdf5_path.name} episodes={len(group_records)}",
            flush=True,
        )
        group_entries, group_snapshots = _render_hdf5_group(
            hdf5_path, group_records, suite_output
        )
        entries.extend(group_entries)
        snapshots_by_episode.update(dict(group_snapshots))
    ordered_snapshots = [
        (int(episode_id), snapshots_by_episode[int(episode_id)])
        for episode_id in episode_ids
    ]
    contact_sheet = write_suite_contact_sheet(
        suite_output / "contact_sheet.jpg", ordered_snapshots
    )
    entries.sort(key=lambda entry: int(entry["episode_id"]))
    return entries, contact_sheet


def write_manifest(
    path: str | Path,
    *,
    dataset_root: Path,
    output_root: Path,
    entries: Sequence[dict[str, Any]],
    contact_sheets: dict[str, Path],
) -> Path:
    manifest = {
        "status": "success",
        "point_body_names": list(POINT_BODY_NAMES),
        "camera": "agentview",
        "occlusion_policy": "keep geometric projection without visibility masking",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "episode_count": len(entries),
        "contact_sheets": {key: str(value) for key, value in contact_sheets.items()},
        "episodes": list(entries),
    }
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay physical LIBERO finger-tip/hand-base triangles on agentview videos."
    )
    parser.add_argument(
        "--dataset-root",
        default="/root/data/yxz/datasets/libero_rerender",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/libero_gripper_triangle_preview",
    )
    parser.add_argument("--episodes-per-suite", type=int, default=10)
    parser.add_argument(
        "--suite",
        action="append",
        choices=[name for name, _ in SUITE_DIRECTORY_NAMES],
        help="Optional suite filter; may be repeated.",
    )
    parser.add_argument(
        "--episode-id",
        action="append",
        type=int,
        help="Optional explicit episode ID; may be repeated and applies to selected suites.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    suites = discover_suite_roots(dataset_root)
    if args.suite:
        selected_names = set(args.suite)
        suites = [suite for suite in suites if suite.name in selected_names]
        missing = selected_names - {suite.name for suite in suites}
        if missing:
            raise FileNotFoundError(f"requested suites not found: {sorted(missing)}")
    elif len(suites) != len(SUITE_DIRECTORY_NAMES):
        raise FileNotFoundError(
            f"expected four LIBERO suites, found {[suite.name for suite in suites]}"
        )

    entries: list[dict[str, Any]] = []
    contact_sheets: dict[str, Path] = {}
    for suite in suites:
        if args.episode_id:
            episode_ids = [int(index) for index in args.episode_id]
            if any(index < 0 or index >= suite.total_episodes for index in episode_ids):
                raise ValueError(
                    f"explicit episode IDs outside {suite.name} range [0,{suite.total_episodes})"
                )
        else:
            episode_ids = select_evenly_spaced_episode_ids(
                suite.total_episodes, args.episodes_per_suite
            )
        print(f"[suite] {suite.name} episodes={episode_ids}", flush=True)
        suite_entries, contact_sheet = render_suite(suite, episode_ids, output_root)
        entries.extend(suite_entries)
        contact_sheets[suite.name] = contact_sheet

    manifest_path = write_manifest(
        output_root / "manifest.json",
        dataset_root=dataset_root,
        output_root=output_root,
        entries=entries,
        contact_sheets=contact_sheets,
    )
    print(
        json.dumps(
            {
                "status": "success",
                "episodes": len(entries),
                "manifest": str(manifest_path),
                "contact_sheets": {key: str(value) for key, value in contact_sheets.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
