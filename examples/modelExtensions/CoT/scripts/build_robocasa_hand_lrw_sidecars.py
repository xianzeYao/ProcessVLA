#!/usr/bin/env python3
"""Generate and validate bilateral physical LRW sidecars for RoboCasa GR1."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from starVLA.robocasa_hand_lrw import (
    HAND_NAMES,
    LANDMARK_BODY_NAMES,
    LANDMARK_NAMES,
    HandLRWSidecar,
    hand_lrw_path,
    load_hand_lrw_sidecar,
    validate_hand_lrw_payload,
)


GEOMETRY_PATH_TEMPLATE = (
    "geometry/hand_lrw/chunk-{episode_chunk:03d}/"
    "episode_{episode_index:06d}.npz"
)


@dataclass(frozen=True)
class EpisodeSidecarSpec:
    episode_id: int
    frame_count: int
    width: int
    height: int


def extract_bilateral_lrw_world(
    env: Any,
    states: Any,
    *,
    reset_to: Any,
) -> np.ndarray:
    """Restore states and return physical `[left/right, thumb/index/wrist]` positions."""

    body_ids: list[list[int]] = []
    for hand_names in LANDMARK_BODY_NAMES:
        hand_ids: list[int] = []
        for name in hand_names:
            try:
                hand_ids.append(int(env.sim.model.body_name2id(name)))
            except Exception as error:
                raise KeyError(f"MuJoCo model is missing required body {name}") from error
        body_ids.append(hand_ids)
    body_index = np.asarray(body_ids, dtype=np.int64)

    trajectory = np.empty((len(states), 2, 3, 3), dtype=np.float32)
    for frame_index, state in enumerate(states):
        reset_to(env, {"states": state})
        env._get_observations(force_update=True)
        trajectory[frame_index] = np.asarray(
            env.sim.data.body_xpos[body_index], dtype=np.float32
        )
    if not np.isfinite(trajectory).all():
        raise ValueError("replayed LRW body positions must be finite")
    return trajectory


def write_sidecar_atomic(
    path: str | Path,
    payload: Mapping[str, np.ndarray],
    *,
    frame_count: int,
    width: int,
    height: int,
    overwrite: bool,
) -> str:
    """Validate and atomically write one sidecar, or validate and skip it."""

    target = Path(path)
    if target.exists() and not overwrite:
        load_hand_lrw_sidecar(
            target,
            frame_count=frame_count,
            width=width,
            height=height,
        )
        return "skipped"

    validated = validate_hand_lrw_payload(
        payload,
        frame_count=frame_count,
        width=width,
        height=height,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                world_xyz=validated.world_xyz,
                agentview_uvd_pixels=validated.agentview_uvd_pixels,
                agentview_projection_valid=validated.agentview_projection_valid,
                agentview_in_frame=validated.agentview_in_frame,
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        load_hand_lrw_sidecar(
            temporary_path,
            frame_count=frame_count,
            width=width,
            height=height,
        )
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return "written"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def finalize_geometry_metadata(
    dataset_root: str | Path,
    episode_specs: Sequence[EpisodeSidecarSpec],
) -> Path:
    """Advertise LRW geometry only after every requested episode validates."""

    root = Path(dataset_root)
    specs = list(episode_specs)
    if not specs:
        raise ValueError("at least one episode sidecar must be validated")
    for spec in specs:
        load_hand_lrw_sidecar(
            hand_lrw_path(root, spec.episode_id),
            frame_count=spec.frame_count,
            width=spec.width,
            height=spec.height,
        )

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    geometry_paths = dict(info.get("geometry_paths", {}))
    geometry_paths["hand_lrw"] = GEOMETRY_PATH_TEMPLATE
    info["geometry_paths"] = geometry_paths
    info["hand_lrw"] = {
        "hand_order": list(HAND_NAMES),
        "landmark_order": list(LANDMARK_NAMES),
        "frame": "world",
        "uvd_camera": "agentview",
    }
    _write_json_atomic(info_path, info)
    return info_path


def summarize_sidecar(sidecar: HandLRWSidecar) -> dict[str, float]:
    """Return physical-geometry and projection summaries for one episode."""

    points = np.asarray(sidecar.world_xyz, dtype=np.float64)
    metrics: dict[str, float] = {}
    for hand_index, hand_name in enumerate(HAND_NAMES):
        hand = points[:, hand_index]
        finger_axis = hand[:, 1] - hand[:, 0]
        wrist_axis = hand[:, 2] - hand[:, 0]
        finger_distance = np.linalg.norm(finger_axis, axis=-1)
        triangle_area = 0.5 * np.linalg.norm(
            np.cross(finger_axis, wrist_axis), axis=-1
        )
        metrics[f"{hand_name}_min_finger_distance_m"] = float(
            np.min(finger_distance)
        )
        metrics[f"{hand_name}_max_finger_distance_m"] = float(
            np.max(finger_distance)
        )
        metrics[f"{hand_name}_min_triangle_area_m2"] = float(
            np.min(triangle_area)
        )
        metrics[f"{hand_name}_max_triangle_area_m2"] = float(
            np.max(triangle_area)
        )
    metrics["projection_valid_point_ratio"] = float(
        np.mean(sidecar.agentview_projection_valid)
    )
    metrics["in_frame_point_ratio"] = float(
        np.mean(sidecar.agentview_in_frame)
    )
    metrics["all_six_in_frame_frame_ratio"] = float(
        np.mean(np.all(sidecar.agentview_in_frame, axis=(1, 2)))
    )
    return metrics
