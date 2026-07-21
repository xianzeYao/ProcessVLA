"""Dependency-light geometry helpers for the simulator environment."""

from __future__ import annotations

import numpy as np


def sample_real_uvd_indices(start: int, end: int, k: int) -> np.ndarray:
    start, end, k = int(start), int(end), int(k)
    if end < start or k < 1:
        raise ValueError(f"invalid interval/count: start={start}, end={end}, k={k}")
    available = end - start + 1
    if available <= k:
        return np.arange(start, end + 1, dtype=np.int64)
    offsets = np.rint(np.linspace(0.0, float(end - start), k)).astype(np.int64)
    if len(np.unique(offsets)) != k:
        offsets = np.linspace(0, end - start, k, dtype=np.int64)
    return start + offsets


def transform_uvd_to_model_space(
    uvd_pixels: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    depth_scale: float,
) -> np.ndarray:
    out = np.asarray(uvd_pixels, dtype=np.float32).copy()
    out[..., 0] *= (target_width - 1) / float(source_width - 1)
    out[..., 1] *= (target_height - 1) / float(source_height - 1)
    out[..., 0] /= float(target_width - 1)
    out[..., 1] /= float(target_height - 1)
    out[..., 2] /= float(depth_scale)
    return out


def resize_depth(depth: np.ndarray, valid: np.ndarray, target_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    depth = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0)
    valid = np.asarray(valid, dtype=np.bool_)
    resized_depth = cv2.resize(depth, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    resized_valid = cv2.resize(valid.astype(np.uint8), (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
    return resized_depth, resized_valid
