"""Immutable LIBERO gripper-triangle sidecar contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


LANDMARK_NAMES = ("left_finger_tip", "right_finger_tip", "wrist_hand_base")
LANDMARK_BODY_NAMES = (
    "gripper0_finger_joint1_tip",
    "gripper0_finger_joint2_tip",
    "gripper0_right_gripper",
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
class GripperTriangleSidecar:
    world_xyz: np.ndarray
    agentview_uvd_pixels: np.ndarray
    agentview_projection_valid: np.ndarray
    agentview_in_frame: np.ndarray


def gripper_triangle_path(dataset_root: str | Path, episode_id: int) -> Path:
    episode_id = int(episode_id)
    if episode_id < 0:
        raise ValueError(f"episode_id must be non-negative, got {episode_id}")
    return (
        Path(dataset_root)
        / "geometry"
        / "gripper_triangle"
        / f"chunk-{episode_id // 1000:03d}"
        / f"episode_{episode_id:06d}.npz"
    )


def project_world_to_agentview_uvd(
    world_xyz: np.ndarray,
    camera_k: np.ndarray,
    t_world_camera: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ``[..., 3]`` world points to pixel UVD and two validity masks."""
    xyz = np.asarray(world_xyz, dtype=np.float32)
    if xyz.ndim < 2 or xyz.shape[-1] != 3:
        raise ValueError(f"world_xyz must have shape [..., 3], got {xyz.shape}")
    if int(width) < 1 or int(height) < 1:
        raise ValueError(f"width and height must be positive, got {width}x{height}")

    frame_count = xyz.shape[0]
    points_per_frame = int(np.prod(xyz.shape[1:-1])) or 1
    flat_xyz = xyz.reshape(-1, 3)

    def per_point_matrix(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.shape == shape:
            return np.broadcast_to(matrix, (len(flat_xyz), *shape))
        if matrix.shape == (frame_count, *shape):
            return np.repeat(matrix, points_per_frame, axis=0)
        raise ValueError(
            f"{name} must have shape {shape} or {(frame_count, *shape)}, got {matrix.shape}"
        )

    k = per_point_matrix(camera_k, (3, 3), "camera_k")
    world_from_camera = per_point_matrix(
        t_world_camera, (4, 4), "t_world_camera"
    )
    homogeneous = np.concatenate(
        [flat_xyz, np.ones((len(flat_xyz), 1), dtype=np.float32)], axis=1
    )
    camera_from_world = np.linalg.inv(world_from_camera)
    camera_xyz = np.einsum("nij,nj->ni", camera_from_world, homogeneous)[:, :3]
    depth = camera_xyz[:, 2]
    projection_valid = np.isfinite(camera_xyz).all(axis=1) & (depth > 0.0)

    projected = np.einsum("nij,nj->ni", k, camera_xyz)
    uv = np.full((len(flat_xyz), 2), np.nan, dtype=np.float32)
    uv[projection_valid] = (
        projected[projection_valid, :2] / projected[projection_valid, 2:3]
    )
    uvd = np.concatenate([uv, depth[:, None].astype(np.float32)], axis=1)
    finite_uv = np.isfinite(uv).all(axis=1)
    in_frame = (
        projection_valid
        & finite_uv
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    output_shape = xyz.shape[:-1]
    return (
        uvd.reshape(*output_shape, 3),
        projection_valid.reshape(output_shape),
        in_frame.reshape(output_shape),
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


def validate_gripper_triangle_payload(
    payload: Mapping[str, np.ndarray],
    *,
    frame_count: int,
    width: int,
    height: int,
) -> GripperTriangleSidecar:
    keys = set(payload.keys())
    if keys != SIDECAR_KEYS:
        raise ValueError(
            f"sidecar must have exact keys {sorted(SIDECAR_KEYS)}, got {sorted(keys)}"
        )
    frame_count = int(frame_count)
    if frame_count < 0:
        raise ValueError(f"frame count must be non-negative, got {frame_count}")
    if int(width) < 1 or int(height) < 1:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    for key in SIDECAR_KEYS:
        value = np.asarray(payload[key])
        if value.ndim == 0 or value.shape[0] != frame_count:
            observed = 0 if value.ndim == 0 else value.shape[0]
            raise ValueError(
                f"{key} frame count must be {frame_count}, got {observed}"
            )

    world_xyz = _require_array(
        payload,
        "world_xyz",
        shape=(frame_count, 3, 3),
        dtype=np.dtype(np.float32),
    )
    uvd = _require_array(
        payload,
        "agentview_uvd_pixels",
        shape=(frame_count, 3, 3),
        dtype=np.dtype(np.float32),
    )
    projection_valid = _require_array(
        payload,
        "agentview_projection_valid",
        shape=(frame_count, 3),
        dtype=np.dtype(np.bool_),
    )
    in_frame = _require_array(
        payload,
        "agentview_in_frame",
        shape=(frame_count, 3),
        dtype=np.dtype(np.bool_),
    )

    if not np.isfinite(world_xyz).all():
        raise ValueError("world_xyz must be finite")
    if not np.isfinite(uvd[projection_valid]).all():
        raise ValueError("projection-valid UVD values must be finite")
    if np.any(uvd[..., 2][projection_valid] <= 0.0):
        raise ValueError("projection-valid UVD values must have positive depth")
    if np.any(in_frame & ~projection_valid):
        raise ValueError("agentview_in_frame cannot exceed agentview_projection_valid")
    in_frame_uv = uvd[..., :2][in_frame]
    if len(in_frame_uv) and np.any(
        (in_frame_uv[:, 0] < 0.0)
        | (in_frame_uv[:, 0] >= float(width))
        | (in_frame_uv[:, 1] < 0.0)
        | (in_frame_uv[:, 1] >= float(height))
    ):
        raise ValueError("in-frame UVD coordinates must lie inside the image")

    return GripperTriangleSidecar(
        world_xyz=world_xyz,
        agentview_uvd_pixels=uvd,
        agentview_projection_valid=projection_valid,
        agentview_in_frame=in_frame,
    )


def load_gripper_triangle_sidecar(
    path: str | Path,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> GripperTriangleSidecar:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    return validate_gripper_triangle_payload(
        arrays, frame_count=frame_count, width=width, height=height
    )
