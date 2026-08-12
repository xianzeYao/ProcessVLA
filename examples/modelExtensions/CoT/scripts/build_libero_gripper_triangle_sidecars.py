#!/usr/bin/env python3
"""Generate and validate LIBERO physical gripper-triangle sidecars."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from starVLA.gripper_triangle import (
    LANDMARK_BODY_NAMES,
    LANDMARK_NAMES,
    GripperTriangleSidecar,
    gripper_triangle_path,
    load_gripper_triangle_sidecar,
    validate_gripper_triangle_payload,
)


GEOMETRY_PATH_TEMPLATE = (
    "geometry/gripper_triangle/chunk-{episode_chunk:03d}/"
    "episode_{episode_index:06d}.npz"
)


@dataclass(frozen=True)
class EpisodeSidecarSpec:
    episode_id: int
    frame_count: int
    width: int
    height: int


def extract_gripper_triangle_world(
    env: Any,
    state_dataset: Any,
    source_indices: np.ndarray,
) -> np.ndarray:
    """Restore selected MuJoCo states and return physical `[L, R, W]` positions."""
    indices = np.asarray(source_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"source_indices must have shape [T], got {indices.shape}")
    if np.any(indices < 0) or np.any(indices >= len(state_dataset)):
        raise IndexError(f"source_indices must lie inside [0,{len(state_dataset)})")
    body_ids: list[int] = []
    for name in LANDMARK_BODY_NAMES:
        try:
            body_ids.append(int(env.sim.model.body_name2id(name)))
        except Exception as error:
            raise KeyError(f"MuJoCo model is missing required body {name}") from error

    trajectory = np.empty((len(indices), 3, 3), dtype=np.float32)
    for frame_index, source_index in enumerate(indices):
        env.regenerate_obs_from_state(state_dataset[int(source_index)])
        trajectory[frame_index] = np.asarray(
            env.sim.data.body_xpos[body_ids], dtype=np.float32
        )
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
        load_gripper_triangle_sidecar(
            target, frame_count=frame_count, width=width, height=height
        )
        return "skipped"

    validated = validate_gripper_triangle_payload(
        payload, frame_count=frame_count, width=width, height=height
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
        load_gripper_triangle_sidecar(
            temporary_path, frame_count=frame_count, width=width, height=height
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
    """Validate the complete requested selection, then atomically advertise it."""
    root = Path(dataset_root)
    specs = list(episode_specs)
    if not specs:
        raise ValueError("at least one episode sidecar must be validated")
    for spec in specs:
        load_gripper_triangle_sidecar(
            gripper_triangle_path(root, spec.episode_id),
            frame_count=spec.frame_count,
            width=spec.width,
            height=spec.height,
        )

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    geometry_paths = dict(info.get("geometry_paths", {}))
    geometry_paths["gripper_triangle"] = GEOMETRY_PATH_TEMPLATE
    info["geometry_paths"] = geometry_paths
    info["gripper_triangle"] = {
        "landmark_order": list(LANDMARK_NAMES),
        "frame": "world",
        "uvd_camera": "agentview",
    }
    _write_json_atomic(info_path, info)
    return info_path


def summarize_sidecar(sidecar: GripperTriangleSidecar) -> dict[str, float]:
    points = np.asarray(sidecar.world_xyz, dtype=np.float64)
    grasp_axis = points[:, 1] - points[:, 0]
    wrist_axis = points[:, 2] - points[:, 0]
    finger_distance = np.linalg.norm(grasp_axis, axis=-1)
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(grasp_axis, wrist_axis), axis=-1
    )
    return {
        "min_finger_distance_m": float(np.min(finger_distance)),
        "max_finger_distance_m": float(np.max(finger_distance)),
        "min_triangle_area_m2": float(np.min(triangle_area)),
        "max_triangle_area_m2": float(np.max(triangle_area)),
        "projection_valid_point_ratio": float(
            np.mean(sidecar.agentview_projection_valid)
        ),
        "in_frame_point_ratio": float(np.mean(sidecar.agentview_in_frame)),
        "all_three_in_frame_frame_ratio": float(
            np.mean(np.all(sidecar.agentview_in_frame, axis=1))
        ),
    }


from starVLA.gripper_triangle_cli import (  # noqa: E402
    main,
    parse_args,
    run,
    select_episode_ids,
)


if __name__ == "__main__":
    main()
