"""Utilities for the standalone text/UVD conditioning experiments.

This module intentionally does not modify the existing geometric CoT adapter.
It reuses its projection convention while making the sampled UVD trace fixed
length and explicitly marking padded points as invalid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np

from starVLA.dataloader.gr00t_lerobot.cot_geometry import (
    project_eef_to_agentview_uvd,
    transform_uvd_to_model_space,
)


def sample_padded_uvd_indices(start: int, end: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample exactly ``k`` real-or-padded frame indices from an inclusive interval.

    When the interval has fewer than ``k`` real frames, the final real frame is
    repeated.  Repeated entries are marked invalid so callers can preserve a
    static tensor shape without treating synthetic tail points as supervision
    or action-relevant geometry.
    """

    start, end, k = int(start), int(end), int(k)
    if end < start:
        raise ValueError(f"end must be >= start, got start={start}, end={end}")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")

    available = end - start + 1
    if available >= k:
        offsets = np.rint(np.linspace(0.0, float(end - start), k)).astype(np.int64)
        if len(np.unique(offsets)) != k:
            offsets = np.linspace(0, end - start, k, dtype=np.int64)
        indices = start + offsets
        valid = np.ones(k, dtype=np.bool_)
    else:
        indices = np.concatenate(
            [np.arange(start, end + 1, dtype=np.int64), np.full(k - available, end, dtype=np.int64)]
        )
        valid = np.concatenate(
            [np.ones(available, dtype=np.bool_), np.zeros(k - available, dtype=np.bool_)]
        )

    denominator = float(max(end - start, 1))
    times = (indices - start).astype(np.float32) / denominator
    times[~valid] = 1.0
    return indices, valid, times


def build_padded_uvd_trace(
    eef_uvd: np.ndarray,
    eef_valid: np.ndarray,
    *,
    start: int,
    end: int,
    k: int,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    depth_scale: float,
) -> dict[str, np.ndarray]:
    """Return a fixed-K model-space UVD trace and validity metadata."""

    frame_indices, padding_valid, times = sample_padded_uvd_indices(start, end, k)
    raw = np.asarray(eef_uvd, dtype=np.float32)[frame_indices]
    geometric_valid = np.asarray(eef_valid, dtype=np.bool_)[frame_indices]
    valid = padding_valid & geometric_valid
    model_uvd = transform_uvd_to_model_space(
        raw,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        depth_scale=depth_scale,
    )
    return {
        "uvd": model_uvd.astype(np.float32),
        "uvd_valid_mask": valid.astype(np.bool_),
        "uvd_frame_indices": frame_indices.astype(np.int64),
        "uvd_time": times.astype(np.float32),
        "uvd_endpoint_indices": np.asarray([0, int(np.flatnonzero(valid).max()) if valid.any() else 0], dtype=np.int64),
    }


class EpisodeUVDCache:
    """Small LRU cache for per-episode projected EEF geometry."""

    def __init__(self, capacity: int = 1):
        self.capacity = int(capacity)
        if self.capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}")
        self._items: dict[int, Any] = {}
        self._order: list[int] = []

    def get(self, episode_index: int) -> Any | None:
        episode_index = int(episode_index)
        if episode_index not in self._items:
            return None
        self._order.remove(episode_index)
        self._order.append(episode_index)
        return self._items[episode_index]

    def put(self, episode_index: int, value: Any) -> None:
        if self.capacity == 0:
            return
        episode_index = int(episode_index)
        if episode_index in self._items:
            self._order.remove(episode_index)
        self._items[episode_index] = value
        self._order.append(episode_index)
        while len(self._order) > self.capacity:
            oldest = self._order.pop(0)
            self._items.pop(oldest, None)


def read_episode_camera_uvd(
    dataset_path: Path,
    trajectory_data,
    *,
    source_width: int,
    source_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project all EEF positions in one rerender episode to agentview UVD."""

    row0 = trajectory_data.iloc[0]
    camera_path = dataset_path / str(row0["observation.camera.params_path"])
    with np.load(camera_path, allow_pickle=False) as payload:
        camera_k = np.asarray(payload["agentview_K"], dtype=np.float32)
        t_world_camera = np.asarray(payload["agentview_T_world_camera"], dtype=np.float32)
    state = np.stack(trajectory_data["observation.state"].to_numpy()).astype(np.float32)
    if len(state) != len(camera_k) or len(state) != len(t_world_camera):
        raise ValueError(
            "episode geometry length mismatch: "
            f"state={len(state)}, K={len(camera_k)}, T={len(t_world_camera)}"
        )
    return project_eef_to_agentview_uvd(
        state[:, :3],
        camera_k,
        t_world_camera,
        width=int(source_width),
        height=int(source_height),
    )


class EmbodiedCotSidecar:
    """Lazy reader for per-episode decoded embodied-CoT JSON sidecars."""

    def __init__(self, dataset_path: Path, sidecar_dir: str = "meta/embodied_cot"):
        self.root = Path(dataset_path) / sidecar_dir
        self._cache: dict[int, list[str]] = {}

    def _load(self, episode_index: int) -> list[str]:
        episode_index = int(episode_index)
        if episode_index not in self._cache:
            path = self.root / f"episode_{episode_index:06d}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Embodied-CoT sidecar is missing: {path}. "
                    "Run prepare_libero_cot_sidecar.py before text-conditioned training."
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            texts = payload.get("cot_text") if isinstance(payload, dict) else payload
            if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                raise ValueError(f"Invalid embodied-CoT sidecar payload: {path}")
            declared_episode = payload.get("episode_index") if isinstance(payload, dict) else episode_index
            if int(declared_episode) != episode_index:
                raise ValueError(f"Sidecar episode mismatch in {path}: {declared_episode} != {episode_index}")
            self._cache[episode_index] = texts
        return self._cache[episode_index]

    def get(self, episode_index: int, frame_index: int) -> str:
        texts = self._load(episode_index)
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= len(texts):
            raise IndexError(
                f"Sidecar frame out of range: episode={episode_index}, frame={frame_index}, length={len(texts)}"
            )
        return texts[frame_index]
