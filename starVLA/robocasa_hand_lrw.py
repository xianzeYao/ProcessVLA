"""Immutable bilateral LRW sidecar contract for RoboCasa GR1 hands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from starVLA.gripper_triangle import project_world_to_agentview_uvd


HAND_NAMES = ("left", "right")
LANDMARK_NAMES = ("thumb", "index", "wrist")
LANDMARK_BODY_NAMES = (
    (
        "gripper0_left_L_thumb_distal_link",
        "gripper0_left_L_index_intermediate_link",
        "gripper0_left_left_hand",
    ),
    (
        "gripper0_right_R_thumb_distal_link",
        "gripper0_right_R_index_intermediate_link",
        "gripper0_right_right_hand",
    ),
)
SIDECAR_KEYS = frozenset(
    {
        "world_xyz",
        "agentview_uvd_pixels",
        "agentview_projection_valid",
        "agentview_in_frame",
    }
)


@dataclass(frozen=True)
class HandLRWSidecar:
    """One episode of `[frame, hand, landmark, coordinate]` geometry."""

    world_xyz: np.ndarray
    agentview_uvd_pixels: np.ndarray
    agentview_projection_valid: np.ndarray
    agentview_in_frame: np.ndarray


def hand_lrw_path(dataset_root: str | Path, episode_id: int) -> Path:
    """Return the canonical episode sidecar path."""

    episode_id = int(episode_id)
    if episode_id < 0:
        raise ValueError(f"episode_id must be non-negative, got {episode_id}")
    return (
        Path(dataset_root)
        / "geometry"
        / "hand_lrw"
        / f"chunk-{episode_id // 1000:03d}"
        / f"episode_{episode_id:06d}.npz"
    )


def project_hand_lrw_world_to_agentview_uvd(
    world_xyz: np.ndarray,
    camera_k: np.ndarray,
    t_world_camera: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a strict `[T,2,3,3]` LRW trajectory into agentview pixels."""

    world = np.asarray(world_xyz, dtype=np.float32)
    if world.ndim != 4 or world.shape[1:] != (2, 3, 3):
        raise ValueError(
            "world_xyz must have shape [T,2 hands,3 landmarks,3], "
            f"got {world.shape}"
        )
    return project_world_to_agentview_uvd(
        world,
        camera_k,
        t_world_camera,
        width=int(width),
        height=int(height),
    )


def _require_array(
    payload: Mapping[str, np.ndarray],
    key: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    value = np.asarray(payload[key])
    if value.shape != shape:
        raise ValueError(f"{key} shape must be {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise ValueError(f"{key} dtype must be {dtype}, got {value.dtype}")
    return value


def validate_hand_lrw_payload(
    payload: Mapping[str, np.ndarray],
    *,
    frame_count: int,
    width: int,
    height: int,
) -> HandLRWSidecar:
    """Validate exact keys, axes, dtypes, finiteness, and image masks."""

    keys = set(payload.keys())
    if keys != SIDECAR_KEYS:
        raise ValueError(
            f"sidecar must have exact keys {sorted(SIDECAR_KEYS)}, got {sorted(keys)}"
        )
    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    if frame_count < 0:
        raise ValueError(f"frame count must be non-negative, got {frame_count}")
    if width < 1 or height < 1:
        raise ValueError(f"width and height must be positive, got {width}x{height}")

    expected_point_shape = (frame_count, len(HAND_NAMES), len(LANDMARK_NAMES))
    for key in SIDECAR_KEYS:
        value = np.asarray(payload[key])
        if value.ndim == 0 or value.shape[0] != frame_count:
            observed = 0 if value.ndim == 0 else int(value.shape[0])
            raise ValueError(
                f"{key} frame count must be {frame_count}, got {observed}"
            )

    world = _require_array(
        payload,
        "world_xyz",
        shape=(*expected_point_shape, 3),
        dtype=np.dtype(np.float32),
    )
    uvd = _require_array(
        payload,
        "agentview_uvd_pixels",
        shape=(*expected_point_shape, 3),
        dtype=np.dtype(np.float32),
    )
    projection_valid = _require_array(
        payload,
        "agentview_projection_valid",
        shape=expected_point_shape,
        dtype=np.dtype(np.bool_),
    )
    in_frame = _require_array(
        payload,
        "agentview_in_frame",
        shape=expected_point_shape,
        dtype=np.dtype(np.bool_),
    )

    if not np.isfinite(world).all():
        raise ValueError("world_xyz must be finite")
    if not np.isfinite(uvd[projection_valid]).all():
        raise ValueError("projection-valid UVD values must be finite")
    if np.any(uvd[..., 2][projection_valid] <= 0.0):
        raise ValueError("projection-valid UVD values must have positive depth")
    if np.any(in_frame & ~projection_valid):
        raise ValueError(
            "agentview_in_frame cannot exceed agentview_projection_valid"
        )
    in_frame_uv = uvd[..., :2][in_frame]
    if len(in_frame_uv) and np.any(
        (in_frame_uv[:, 0] < 0.0)
        | (in_frame_uv[:, 0] >= float(width))
        | (in_frame_uv[:, 1] < 0.0)
        | (in_frame_uv[:, 1] >= float(height))
    ):
        raise ValueError("in-frame UVD coordinates must lie inside the image")

    return HandLRWSidecar(
        world_xyz=world,
        agentview_uvd_pixels=uvd,
        agentview_projection_valid=projection_valid,
        agentview_in_frame=in_frame,
    )


def load_hand_lrw_sidecar(
    path: str | Path,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> HandLRWSidecar:
    """Load one NPZ without pickle and validate its complete contract."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    return validate_hand_lrw_payload(
        arrays,
        frame_count=frame_count,
        width=width,
        height=height,
    )


__all__ = [
    "HAND_NAMES",
    "LANDMARK_BODY_NAMES",
    "LANDMARK_NAMES",
    "HandLRWSidecar",
    "hand_lrw_path",
    "load_hand_lrw_sidecar",
    "project_hand_lrw_world_to_agentview_uvd",
    "validate_hand_lrw_payload",
]
