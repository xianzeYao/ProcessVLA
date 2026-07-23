"""Convert CALVIN to the existing StarVLA/LIBERO LeRobot contract.

CALVIN semantics remain CALVIN-specific (robot_obs, rel_actions and language
annotations). Only the optional RGB-D/source metadata follows the LIBERO
rerender convention so existing geometry/data tools can consume it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import build_calvin_native_rgbd_lerobot as converter


CALVIN_DEPTH_PATH_KEYS = {
    "static": "observation.depth.image_m_path",
    "gripper": "observation.depth.wrist_m_path",
}
CALVIN_DEPTH_STORAGE_KEYS = {
    "static": "observation.depth.image_m",
    "gripper": "observation.depth.wrist_m",
}
CALVIN_MODALITY_FILE = "modality_calvin_lerobot.json"

_ORIGINAL_PREPARE_OUTPUT = converter.prepare_output
_ORIGINAL_CONVERT_SEGMENT = converter.convert_segment


def _source_episode_for_frame(frame_id: int, boundaries: np.ndarray) -> int:
    matches = np.flatnonzero(
        (boundaries[:, 0] <= int(frame_id)) & (int(frame_id) <= boundaries[:, 1])
    )
    if len(matches) != 1:
        raise ValueError(f"frame {frame_id} is covered by {len(matches)} source episodes")
    return int(matches[0])


def _mimsave(uri, ims, *args, **kwargs):
    kwargs.pop("macro_block_size", None)
    return converter.imageio.mimsave_original(uri, ims, *args, **kwargs)


def _canonicalize_info(output_root: Path) -> None:
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["depth_path"] = "depth/chunk-{episode_chunk:03d}/{depth_key}/episode_{episode_index:06d}.npz"
    features = info["features"]
    old_to_new = {
        "observation.depth.static_m_path": CALVIN_DEPTH_PATH_KEYS["static"],
        "observation.depth.gripper_m_path": CALVIN_DEPTH_PATH_KEYS["gripper"],
    }
    for old_key, new_key in old_to_new.items():
        if old_key in features:
            features[new_key] = features.pop(old_key)
    info["replay_rgbd"]["depth_path_convention"] = "LIBERO observation.depth.*_m_path"
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[4]
    modality_path = repo_root / "examples/simBenchmarks/calvin/train_files" / CALVIN_MODALITY_FILE
    (output_root / "meta" / "modality.json").write_bytes(modality_path.read_bytes())


def _prepare_output(output_root, dataset_root, split, tasks, total_segments, total_frames, fps, _modality_path):
    _ORIGINAL_PREPARE_OUTPUT(
        output_root,
        dataset_root,
        split,
        tasks,
        total_segments,
        total_frames,
        fps,
        _modality_path,
    )
    _canonicalize_info(Path(output_root))


def _canonicalize_episode(output_root: Path, episode_index: int) -> None:
    chunk = episode_index // 1000
    episode_name = f"episode_{episode_index:06d}.npz"
    for old_name, new_name, old_array in (
        ("observation.depth.static_m", CALVIN_DEPTH_STORAGE_KEYS["static"], "depth_static"),
        ("observation.depth.gripper_m", CALVIN_DEPTH_STORAGE_KEYS["gripper"], "depth_gripper"),
    ):
        old_path = output_root / "depth" / f"chunk-{chunk:03d}" / old_name / episode_name
        new_path = output_root / "depth" / f"chunk-{chunk:03d}" / new_name / episode_name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        with np.load(old_path) as payload:
            depth = np.asarray(payload[old_array], dtype=np.float32)
        np.savez_compressed(new_path, depth_m=depth)
        old_path.unlink()
        try:
            old_path.parent.rmdir()
        except OSError:
            pass

    parquet_path = output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    table = pd.read_parquet(parquet_path)
    table = table.rename(columns={
        "observation.depth.static_m_path": CALVIN_DEPTH_PATH_KEYS["static"],
        "observation.depth.gripper_m_path": CALVIN_DEPTH_PATH_KEYS["gripper"],
    })
    table.to_parquet(parquet_path, index=False)


def _convert_segment(segment, segment_index, frame_paths, boundaries, output_root, action_key, fps, task_index):
    result = _ORIGINAL_CONVERT_SEGMENT(
        segment, segment_index, frame_paths, boundaries, output_root, action_key, fps, task_index
    )
    _canonicalize_episode(Path(output_root), int(segment_index))
    return result


def main() -> None:
    converter.source_episode_for_frame = _source_episode_for_frame
    converter.prepare_output = _prepare_output
    converter.convert_segment = _convert_segment
    if not hasattr(converter.imageio, "mimsave_original"):
        converter.imageio.mimsave_original = converter.imageio.mimsave
        converter.imageio.mimsave = _mimsave
    converter.main()


if __name__ == "__main__":
    main()
