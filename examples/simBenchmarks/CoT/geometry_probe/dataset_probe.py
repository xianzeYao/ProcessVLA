"""LIBERO rerender dataset reader for one-chunk geometry probing."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from examples.modelExtensions.CoT.scripts.robocasa_rerender_geometry import (
    dial_content_region_mask,
)
from examples.simBenchmarks.CoT.geometry_probe.probe_utils import EpisodeRef, SampleRef
from starVLA.dataloader.robocasa_eef_fields import select_robocasa_uvd_world_columns
from starVLA.dataloader.gr00t_lerobot.cot_geometry import (
    _read_npz_array,
    _resize_depth,
    project_eef_to_agentview_uvd,
    sample_real_uvd_indices,
    transform_uvd_to_model_space,
)
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps


DEFAULT_DATASET_SUFFIX = "_no_noops_1.0.0_lerobot"


@dataclass
class _LoadedEpisode:
    table: pd.DataFrame
    depth_m: np.ndarray
    eef_uvd: np.ndarray
    eef_valid: np.ndarray
    language: str
    primary_video: Path
    wrist_video: Path


@dataclass
class _LoadedRoboCasaEpisode:
    table: pd.DataFrame
    depth_m: np.ndarray
    hand_uvd: np.ndarray
    hand_valid: np.ndarray
    language: str
    ego_video: Path


class LiberoRerenderStore:
    """Lazily load RGB-D and EEF-UVD data using training conventions."""

    def __init__(
        self,
        root: str | Path,
        suites: Sequence[str],
        *,
        image_size: int = 224,
        horizon: int = 8,
        uvd_num_points: int | None = None,
        uvd_depth_scale: float = 1.0,
        video_backend: str = "torchvision_av",
        video_backend_kwargs: dict[str, Any] | None = None,
        episode_cache_size: int = 1,
    ) -> None:
        self.root = Path(root)
        self.suites = list(suites)
        self.image_size = int(image_size)
        self.horizon = int(horizon)
        self.uvd_num_points = (
            int(uvd_num_points)
            if uvd_num_points is not None
            else int(np.floor(0.3 * self.horizon)) + 2
        )
        self.uvd_depth_scale = float(uvd_depth_scale)
        self.video_backend = video_backend
        self.video_backend_kwargs = dict(video_backend_kwargs or {})
        self.episode_cache_size = max(int(episode_cache_size), 0)
        self._episode_cache: OrderedDict[tuple[str, int], _LoadedEpisode] = OrderedDict()
        self._dataset_paths = {suite: self._resolve_dataset_path(suite) for suite in self.suites}
        self._task_maps = {suite: self._read_tasks(self._dataset_paths[suite]) for suite in self.suites}

    def _resolve_dataset_path(self, suite: str) -> Path:
        candidates = [
            self.root / f"{suite}{DEFAULT_DATASET_SUFFIX}",
            self.root / f"{suite}_lerobot",
            self.root / suite,
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No LeRobot dataset found for suite={suite!r}; checked {[str(path) for path in candidates]}"
        )

    @staticmethod
    def _read_tasks(dataset_path: Path) -> dict[int, str]:
        path = dataset_path / "meta" / "tasks.jsonl"
        if not path.exists():
            return {}
        mapping: dict[int, str] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    mapping[int(row["task_index"])] = str(row["task"])
        return mapping

    @staticmethod
    def _read_episode_metadata(dataset_path: Path) -> list[EpisodeRef]:
        path = dataset_path / "meta" / "episodes.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing episode metadata: {path}")
        refs: list[EpisodeRef] = []
        # suite is filled by the caller; this method is only used internally.
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    refs.append(EpisodeRef("", int(row["episode_index"]), int(row["length"])))
        return refs

    def episode_refs(self) -> dict[str, list[EpisodeRef]]:
        return {
            suite: [
                EpisodeRef(suite, ref.episode_id, ref.episode_length)
                for ref in self._read_episode_metadata(self._dataset_paths[suite])
            ]
            for suite in self.suites
        }

    def _parquet_path(self, dataset_path: Path, episode_id: int) -> Path:
        chunk = int(episode_id) // 1000
        path = dataset_path / "data" / f"chunk-{chunk:03d}" / f"episode_{int(episode_id):06d}.parquet"
        if path.exists():
            return path
        matches = list((dataset_path / "data").glob(f"chunk-*/episode_{int(episode_id):06d}.parquet"))
        if matches:
            return matches[0]
        raise FileNotFoundError(path)

    @staticmethod
    def _video_path(dataset_path: Path, episode_id: int, key: str) -> Path:
        chunk = int(episode_id) // 1000
        return dataset_path / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{int(episode_id):06d}.mp4"

    def _load_episode(self, suite: str, episode_id: int) -> _LoadedEpisode:
        cache_key = (suite, int(episode_id))
        cached = self._episode_cache.get(cache_key)
        if cached is not None:
            self._episode_cache.move_to_end(cache_key)
            return cached

        dataset_path = self._dataset_paths[suite]
        table = pd.read_parquet(self._parquet_path(dataset_path, episode_id))
        row0 = table.iloc[0]
        depth_m = _read_npz_array(dataset_path / str(row0["observation.depth.image_m_path"]), "depth_m")
        camera_path = dataset_path / str(row0["observation.camera.params_path"])
        camera_k = _read_npz_array(camera_path, "agentview_K").astype(np.float32)
        t_world_camera = _read_npz_array(camera_path, "agentview_T_world_camera").astype(np.float32)
        state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
        if len(depth_m) != len(table) or len(camera_k) != len(table) or len(state) != len(table):
            raise ValueError(
                f"episode alignment mismatch suite={suite} episode={episode_id}: "
                f"table={len(table)}, depth={len(depth_m)}, K={len(camera_k)}, state={len(state)}"
            )
        eef_uvd, eef_valid = project_eef_to_agentview_uvd(
            state[:, :3],
            camera_k,
            t_world_camera,
            width=int(depth_m.shape[-1]),
            height=int(depth_m.shape[-2]),
        )
        task_index = int(np.asarray(row0["task_index"]).reshape(-1)[0])
        language = self._task_maps[suite].get(task_index, "")
        if not language:
            language = str(row0.get("task", ""))
        episode = _LoadedEpisode(
            table=table,
            depth_m=depth_m,
            eef_uvd=eef_uvd,
            eef_valid=eef_valid,
            language=language,
            primary_video=self._video_path(dataset_path, episode_id, "observation.images.image"),
            wrist_video=self._video_path(dataset_path, episode_id, "observation.images.wrist_image"),
        )
        if self.episode_cache_size > 0:
            self._episode_cache[cache_key] = episode
            self._episode_cache.move_to_end(cache_key)
            while len(self._episode_cache) > self.episode_cache_size:
                self._episode_cache.popitem(last=False)
        return episode

    def load_sample(self, sample: SampleRef) -> dict[str, Any]:
        episode = self._load_episode(sample.suite, sample.episode_id)
        t = int(sample.frame_index)
        future = t + self.horizon
        if future >= len(episode.table):
            raise ValueError(
                f"sample has no complete future: suite={sample.suite}, episode={sample.episode_id}, "
                f"t={t}, horizon={self.horizon}, length={len(episode.table)}"
            )

        timestamps = np.asarray([float(episode.table.iloc[t]["timestamp"])], dtype=np.float64)
        primary = get_frames_by_timestamps(
            str(episode.primary_video), timestamps,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )[0]
        wrist = get_frames_by_timestamps(
            str(episode.wrist_video), timestamps,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )[0]
        current_valid = np.isfinite(episode.depth_m[t]) & (episode.depth_m[t] > 0.0)
        future_valid = np.isfinite(episode.depth_m[future]) & (episode.depth_m[future] > 0.0)
        current_depth, current_valid = _resize_depth(
            episode.depth_m[t], current_valid, (self.image_size, self.image_size)
        )
        future_depth, future_valid = _resize_depth(
            episode.depth_m[future], future_valid, (self.image_size, self.image_size)
        )
        indices = sample_real_uvd_indices(t, future, self.uvd_num_points)
        uvd = transform_uvd_to_model_space(
            episode.eef_uvd[indices],
            source_width=int(episode.depth_m.shape[-1]),
            source_height=int(episode.depth_m.shape[-2]),
            target_width=self.image_size,
            target_height=self.image_size,
            depth_scale=self.uvd_depth_scale,
        )
        uvd_valid = episode.eef_valid[indices].astype(np.bool_)
        uvd_time = ((indices - t) / float(self.horizon)).astype(np.float32)
        state = np.asarray(episode.table.iloc[t]["observation.state"], dtype=np.float32)
        return {
            "example": {
                "image": [np.ascontiguousarray(primary), np.ascontiguousarray(wrist)],
                "lang": episode.language,
                "state": state,
            },
            "rgb": np.ascontiguousarray(primary),
            "wrist_rgb": np.ascontiguousarray(wrist),
            "depth_current": current_depth.astype(np.float32),
            "depth_future": future_depth.astype(np.float32),
            "depth_current_valid": current_valid.astype(np.bool_),
            "depth_future_valid": future_valid.astype(np.bool_),
            "uvd": uvd.astype(np.float32),
            "uvd_valid_mask": uvd_valid,
            "uvd_time": uvd_time,
            "uvd_frame_indices": indices.astype(np.int64),
            "metadata": {
                "suite": sample.suite,
                "episode_id": int(sample.episode_id),
                "frame_index": t,
                "timestamp": float(episode.table.iloc[t]["timestamp"]),
                "future_frame_index": future,
                "episode_length": int(sample.episode_length),
                "language": episode.language,
            },
        }


class RoboCasaRerenderStore:
    """Lazily load Fourier RoboCasa RGB-D and bilateral UVD targets."""

    def __init__(
        self,
        root: str | Path,
        task_names: Sequence[str],
        *,
        image_size: int = 224,
        horizon: int = 16,
        uvd_num_points: int = 6,
        uvd_depth_scale: float = 1.0,
        video_backend: str = "torchvision_av",
        video_backend_kwargs: dict[str, Any] | None = None,
        episode_cache_size: int = 1,
    ) -> None:
        self.root = Path(root)
        self.task_names = list(task_names)
        if not self.task_names:
            raise ValueError("at least one RoboCasa task is required")
        self.image_size = int(image_size)
        self.horizon = int(horizon)
        self.uvd_num_points = int(uvd_num_points)
        self.uvd_depth_scale = float(uvd_depth_scale)
        self.video_backend = str(video_backend)
        self.video_backend_kwargs = dict(video_backend_kwargs or {})
        self.episode_cache_size = max(int(episode_cache_size), 0)
        self._episode_cache: OrderedDict[tuple[str, int], _LoadedRoboCasaEpisode] = OrderedDict()
        self._dataset_paths = {task: self._resolve_dataset_path(task) for task in self.task_names}
        self._task_maps = {
            task: LiberoRerenderStore._read_tasks(path)
            for task, path in self._dataset_paths.items()
        }

    def _resolve_dataset_path(self, task_name: str) -> Path:
        path = self.root / task_name
        if not path.exists():
            raise FileNotFoundError(f"RoboCasa task dataset does not exist: {path}")
        return path

    def episode_refs(self) -> dict[str, list[EpisodeRef]]:
        output: dict[str, list[EpisodeRef]] = {}
        for task_name, path in self._dataset_paths.items():
            output[task_name] = [
                EpisodeRef(task_name, ref.episode_id, ref.episode_length)
                for ref in LiberoRerenderStore._read_episode_metadata(path)
            ]
        return output

    @staticmethod
    def _parquet_path(dataset_path: Path, episode_id: int) -> Path:
        chunk = int(episode_id) // 1000
        path = dataset_path / "data" / f"chunk-{chunk:03d}" / f"episode_{int(episode_id):06d}.parquet"
        if path.exists():
            return path
        matches = list((dataset_path / "data").glob(f"chunk-*/episode_{int(episode_id):06d}.parquet"))
        if matches:
            return matches[0]
        raise FileNotFoundError(path)

    @staticmethod
    def _video_path(dataset_path: Path, episode_id: int) -> Path:
        chunk = int(episode_id) // 1000
        return (
            dataset_path
            / "videos"
            / f"chunk-{chunk:03d}"
            / "observation.images.ego_view"
            / f"episode_{int(episode_id):06d}.mp4"
        )

    def _load_episode(self, task_name: str, episode_id: int) -> _LoadedRoboCasaEpisode:
        cache_key = (task_name, int(episode_id))
        cached = self._episode_cache.get(cache_key)
        if cached is not None:
            self._episode_cache.move_to_end(cache_key)
            return cached

        dataset_path = self._dataset_paths[task_name]
        table = pd.read_parquet(self._parquet_path(dataset_path, episode_id))
        if table.empty:
            raise ValueError(f"empty RoboCasa episode task={task_name} episode={episode_id}")
        row0 = table.iloc[0]
        depth_m = _read_npz_array(
            dataset_path / str(row0["observation.depth.image_m_path"]), "depth_m"
        )
        camera_path = dataset_path / str(row0["observation.camera.params_path"])
        camera_k = _read_npz_array(camera_path, "agentview_K").astype(np.float32)
        t_world_camera = _read_npz_array(
            camera_path, "agentview_T_world_camera"
        ).astype(np.float32)
        left_key, right_key = select_robocasa_uvd_world_columns(table.columns)
        left_world = np.stack(table[left_key].to_numpy()).astype(np.float32)
        right_world = np.stack(table[right_key].to_numpy()).astype(np.float32)
        hand_world = np.stack([left_world, right_world], axis=1)
        expected_length = len(table)
        if any(len(value) != expected_length for value in (depth_m, camera_k, t_world_camera, hand_world)):
            raise ValueError(
                f"RoboCasa alignment mismatch task={task_name} episode={episode_id}: "
                f"table={expected_length}, depth={len(depth_m)}, K={len(camera_k)}, "
                f"T={len(t_world_camera)}, hand={len(hand_world)}"
            )
        hand_uvd, hand_valid = project_eef_to_agentview_uvd(
            hand_world,
            camera_k,
            t_world_camera,
            width=int(depth_m.shape[-1]),
            height=int(depth_m.shape[-2]),
        )
        hand_valid = dial_content_region_mask(
            hand_uvd,
            hand_valid,
            output_size=int(depth_m.shape[-1]),
        )
        task_index = int(np.asarray(row0["task_index"]).reshape(-1)[0])
        language = self._task_maps[task_name].get(task_index, "")
        if not language:
            language = str(row0.get("task", ""))
        episode = _LoadedRoboCasaEpisode(
            table=table,
            depth_m=depth_m,
            hand_uvd=hand_uvd,
            hand_valid=hand_valid,
            language=language,
            ego_video=self._video_path(dataset_path, episode_id),
        )
        if self.episode_cache_size > 0:
            self._episode_cache[cache_key] = episode
            self._episode_cache.move_to_end(cache_key)
            while len(self._episode_cache) > self.episode_cache_size:
                self._episode_cache.popitem(last=False)
        return episode

    def load_sample(self, sample: SampleRef) -> dict[str, Any]:
        episode = self._load_episode(sample.suite, sample.episode_id)
        frame_index = int(sample.frame_index)
        future_index = frame_index + self.horizon
        if future_index >= len(episode.table):
            raise ValueError(
                f"sample has no complete future: task={sample.suite}, episode={sample.episode_id}, "
                f"t={frame_index}, horizon={self.horizon}, length={len(episode.table)}"
            )
        timestamps = np.asarray(
            [float(episode.table.iloc[frame_index]["timestamp"])], dtype=np.float64
        )
        ego_rgb = get_frames_by_timestamps(
            str(episode.ego_video),
            timestamps,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )[0]
        current_valid = np.isfinite(episode.depth_m[frame_index]) & (episode.depth_m[frame_index] > 0.0)
        future_valid = np.isfinite(episode.depth_m[future_index]) & (episode.depth_m[future_index] > 0.0)
        current_depth, current_valid = _resize_depth(
            episode.depth_m[frame_index], current_valid, (self.image_size, self.image_size)
        )
        future_depth, future_valid = _resize_depth(
            episode.depth_m[future_index], future_valid, (self.image_size, self.image_size)
        )
        indices = sample_real_uvd_indices(frame_index, future_index, self.uvd_num_points)
        sampled_pixels = episode.hand_uvd[indices]
        uvd = transform_uvd_to_model_space(
            sampled_pixels,
            source_width=int(episode.depth_m.shape[-1]),
            source_height=int(episode.depth_m.shape[-2]),
            target_width=self.image_size,
            target_height=self.image_size,
            depth_scale=self.uvd_depth_scale,
        )
        uvd_valid = episode.hand_valid[indices].astype(np.bool_)
        uvd_time = ((indices - frame_index) / float(self.horizon)).astype(np.float32)
        out_of_frame = (
            np.isfinite(sampled_pixels).all(axis=-1)
            & (sampled_pixels[..., 2] > 0.0)
            & ~(
                (sampled_pixels[..., 0] >= 0.0)
                & (sampled_pixels[..., 0] <= float(episode.depth_m.shape[-1] - 1))
                & (sampled_pixels[..., 1] >= 0.0)
                & (sampled_pixels[..., 1] <= float(episode.depth_m.shape[-2] - 1))
            )
        )
        return {
            "example": {
                "image": [np.ascontiguousarray(ego_rgb)],
                "lang": episode.language,
            },
            "rgb": np.ascontiguousarray(ego_rgb),
            "depth_current": current_depth.astype(np.float32),
            "depth_future": future_depth.astype(np.float32),
            "depth_current_valid": current_valid.astype(np.bool_),
            "depth_future_valid": future_valid.astype(np.bool_),
            "uvd": uvd.astype(np.float32),
            "uvd_valid_mask": uvd_valid,
            "uvd_out_of_frame_mask": out_of_frame.astype(np.bool_),
            "uvd_time": uvd_time,
            "uvd_frame_indices": indices.astype(np.int64),
            "metadata": {
                "suite": sample.suite,
                "episode_id": int(sample.episode_id),
                "frame_index": frame_index,
                "timestamp": float(episode.table.iloc[frame_index]["timestamp"]),
                "future_frame_index": future_index,
                "episode_length": int(sample.episode_length),
                "language": episode.language,
            },
        }
