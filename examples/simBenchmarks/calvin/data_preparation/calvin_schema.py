"""Pure helpers for the official per-timestep CALVIN dataset schema."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np


_SPLIT_RE = re.compile(r"^task_([A-D]+)_([A-D])$")


def parse_calvin_split(name: str) -> tuple[str, str]:
    """Return ``(training_environments, evaluation_environment)``."""
    match = _SPLIT_RE.fullmatch(str(name))
    if match is None:
        raise ValueError(f"expected task_<A-D+>_<A-D>, got {name!r}")
    return match.group(1), match.group(2)


def calvin_state(robot_obs: np.ndarray) -> np.ndarray:
    """Project CALVIN's 15-D proprioception into the 8-D Franka interface.

    The source layout is ``eef_xyz(3), eef_euler(3), gripper_width(1),
    arm_joints(7), gripper_action(1)``.  StarVLA's existing Franka schema
    expects the first six values, width, and the binary gripper action.
    """
    value = np.asarray(robot_obs)
    if value.shape != (15,):
        raise ValueError(f"robot_obs must have shape (15,), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("robot_obs contains non-finite values")
    return np.concatenate((value[:6], value[6:7], value[14:15])).astype(np.float32)


def _check_shape(frame: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...]) -> None:
    if key not in frame:
        raise ValueError(f"missing required CALVIN frame key: {key}")
    value = np.asarray(frame[key])
    if value.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {value.shape}")


def validate_frame(frame: Mapping[str, np.ndarray]) -> None:
    """Validate one official CALVIN NPZ frame without loading an episode."""
    for key in ("actions", "rel_actions"):
        _check_shape(frame, key, (7,))
        if not np.isfinite(np.asarray(frame[key])).all():
            raise ValueError(f"{key} contains non-finite values")
    for key, shape in (("robot_obs", (15,)), ("scene_obs", (24,))):
        _check_shape(frame, key, shape)
        if not np.isfinite(np.asarray(frame[key])).all():
            raise ValueError(f"{key} contains non-finite values")
    for key, shape in (
        ("rgb_static", (200, 200, 3)),
        ("rgb_gripper", (84, 84, 3)),
    ):
        _check_shape(frame, key, shape)
        if np.asarray(frame[key]).dtype != np.uint8:
            raise ValueError(f"{key} must have dtype uint8")
    for key, shape in (
        ("depth_static", (200, 200)),
        ("depth_gripper", (84, 84)),
    ):
        _check_shape(frame, key, shape)
        if not np.issubdtype(np.asarray(frame[key]).dtype, np.floating):
            raise ValueError(f"{key} must have a floating dtype")
        if not np.isfinite(np.asarray(frame[key])).all():
            raise ValueError(f"{key} contains non-finite values")


def assign_language_segments(
    frame_ids: np.ndarray, annotations: Mapping[str, Mapping[str, object]]
) -> list[dict[str, object]]:
    """Return annotation intervals intersecting ``frame_ids``.

    CALVIN's ``info.indx`` stores inclusive source frame boundaries.  The
    returned end index therefore remains inclusive and is kept as provenance;
    callers decide whether a frame belongs to an interval with ``start <= i <=
    end``.
    """
    ids = np.asarray(frame_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError(f"frame_ids must be one-dimensional, got {ids.shape}")
    language = annotations.get("language", {})
    info = annotations.get("info", {})
    ranges = np.asarray(info.get("indx", []), dtype=np.int64)
    instructions = list(language.get("ann", []))
    tasks = list(language.get("task", []))
    if ranges.size == 0:
        return []
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(f"info.indx must have shape (N, 2), got {ranges.shape}")
    if len(instructions) != len(ranges) or len(tasks) != len(ranges):
        raise ValueError("language annotations and info.indx have different lengths")
    if ids.size == 0:
        return []
    lo, hi = int(ids.min()), int(ids.max())
    result: list[dict[str, object]] = []
    for (start, end), task, instruction in zip(ranges, tasks, instructions):
        start, end = int(start), int(end)
        if end < lo or start > hi:
            continue
        result.append(
            {
                "source_start": start,
                "source_end": end,
                "task": str(task),
                "instruction": str(instruction),
            }
        )
    return result
