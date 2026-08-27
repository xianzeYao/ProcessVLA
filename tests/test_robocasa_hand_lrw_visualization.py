from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from starVLA.dataloader.robocasa_fourier_tasks import FOURIER_TASKS
from starVLA.robocasa_hand_lrw import hand_lrw_path


def make_visual_task(root: Path, task_index: int, *, frames: int = 40) -> Path:
    task = FOURIER_TASKS[task_index]
    task_root = root / task.official_dataset_name
    (task_root / "meta").mkdir(parents=True)
    (task_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "video_path": (
                    "videos/chunk-{episode_chunk:03d}/"
                    "observation.images.ego_view/episode_{episode_index:06d}.mp4"
                ),
                "features": {
                    "observation.images.ego_view": {
                        "dtype": "video",
                        "shape": [32, 64, 3],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (task_root / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "length": frames,
                "trajectory_id": f"{task.basename}-00007",
                "tasks": [f"do {task.basename}"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    video = (
        task_root
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / "episode_000000.mp4"
    )
    video.parent.mkdir(parents=True)
    video.touch()

    world = np.zeros((frames, 2, 3, 3), dtype=np.float32)
    world[..., 2] = 1.0
    uvd = np.zeros_like(world)
    for frame in range(frames):
        for hand in range(2):
            for landmark in range(3):
                uvd[frame, hand, landmark] = [
                    8.0 + frame + hand * 4,
                    14.0 + landmark * 5,
                    0.2 + 0.1 * landmark + 0.01 * frame,
                ]
    sidecar = hand_lrw_path(task_root, 0)
    sidecar.parent.mkdir(parents=True)
    np.savez_compressed(
        sidecar,
        world_xyz=world,
        agentview_uvd_pixels=uvd,
        agentview_projection_valid=np.ones((frames, 2, 3), dtype=np.bool_),
        agentview_in_frame=np.ones((frames, 2, 3), dtype=np.bool_),
    )
    return task_root


def test_fixed_seed_selects_ten_distinct_tasks_and_six_v2_frames(
    tmp_path: Path,
) -> None:
    from examples.modelExtensions.CoT.scripts.visualize_robocasa_hand_lrw_sidecars import (
        discover_visual_task_roots,
        select_audit_windows,
    )

    dataset_root = tmp_path / "rerender"
    for task_index in range(12):
        make_visual_task(dataset_root, task_index)
    roots = discover_visual_task_roots(
        dataset_root, allow_incomplete_preflight=True
    )

    first = select_audit_windows(
        roots,
        seed=42,
        task_count=10,
        action_horizon=16,
        uvd_num_points=6,
    )
    second = select_audit_windows(
        roots,
        seed=42,
        task_count=10,
        action_horizon=16,
        uvd_num_points=6,
    )

    assert first == second
    assert len(first) == 10
    assert len({item.task for item in first}) == 10
    assert all(len(item.frame_indices) == 6 for item in first)
    assert all(np.all(np.diff(item.frame_indices) > 0) for item in first)
    assert all(item.frame_indices[-1] - item.frame_indices[0] == 16 for item in first)


def test_overlay_encodes_bilateral_lrw_tracks_and_depth() -> None:
    from examples.modelExtensions.CoT.scripts.visualize_robocasa_hand_lrw_sidecars import (
        draw_lrw_trajectory_overlay,
    )

    image = np.zeros((64, 96, 3), dtype=np.uint8)
    uvd = np.zeros((6, 2, 3, 3), dtype=np.float32)
    for time in range(6):
        for hand in range(2):
            for landmark in range(3):
                uvd[time, hand, landmark] = [
                    12 + time * 8 + hand * 2,
                    16 + landmark * 14,
                    0.2 + landmark * 0.2,
                ]
    valid = np.ones((6, 2, 3), dtype=np.bool_)

    overlay = draw_lrw_trajectory_overlay(image, uvd, valid)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert np.any(overlay != image)
    # Thumb, index, and wrist use different RGB colors.
    nonzero_colors = np.unique(overlay.reshape(-1, 3), axis=0)
    assert len(nonzero_colors) >= 4
    # Near thumb points occupy more pixels than far wrist points.
    rgb = overlay.astype(np.int16)
    thumb_pixels = np.count_nonzero(
        (rgb[..., 0] > 1.5 * rgb[..., 1])
        & (rgb[..., 0] > 1.5 * rgb[..., 2])
    )
    wrist_pixels = np.count_nonzero(
        (rgb[..., 2] > 1.5 * rgb[..., 0])
        & (rgb[..., 2] > 1.5 * rgb[..., 1])
    )
    assert thumb_pixels > wrist_pixels


def test_run_writes_selection_before_render_and_preserves_failed_sample(
    tmp_path: Path,
) -> None:
    from examples.modelExtensions.CoT.scripts.visualize_robocasa_hand_lrw_sidecars import run

    dataset_root = tmp_path / "rerender"
    for task_index in range(12):
        make_visual_task(dataset_root, task_index)
    output_root = tmp_path / "audit"
    calls: list[str] = []

    def frame_loader(path: Path, frame_index: int) -> np.ndarray:
        manifest = json.loads(
            (output_root / "selection_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "selected"
        calls.append(path.parent.parent.parent.parent.parent.name)
        if len(calls) == 3:
            raise RuntimeError("synthetic frame failure")
        image = np.zeros((32, 64, 3), dtype=np.uint8)
        image[..., 1] = frame_index
        return image

    result = run(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--seed",
            "42",
            "--task-count",
            "10",
            "--allow-incomplete-preflight",
        ],
        frame_loader=frame_loader,
        raise_on_error=False,
    )

    manifest = json.loads(
        (output_root / "selection_manifest.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "failed"
    assert manifest["status"] == "failed"
    assert len(manifest["selections"]) == 10
    assert len({item["task"] for item in manifest["selections"]}) == 10
    assert len(manifest["errors"]) == 1
    assert manifest["errors"][0]["error"] == "synthetic frame failure"
    assert len(calls) == 10
    assert len(list(output_root.glob("*/episode_*/window.json"))) == 9
    assert (output_root / "contact_sheet.jpg").is_file()


def test_success_records_full_uvd_axes_and_invalid_counts(tmp_path: Path) -> None:
    from examples.modelExtensions.CoT.scripts.visualize_robocasa_hand_lrw_sidecars import run

    dataset_root = tmp_path / "rerender"
    for task_index in range(10):
        make_visual_task(dataset_root, task_index)
    output_root = tmp_path / "audit"

    result = run(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--seed",
            "9",
            "--task-count",
            "10",
            "--allow-incomplete-preflight",
        ],
        frame_loader=lambda path, frame_index: np.zeros(
            (32, 64, 3), dtype=np.uint8
        ),
    )

    records = list(output_root.glob("*/episode_*/window.json"))
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert len(records) == 10
    assert record["hand_order"] == ["left", "right"]
    assert record["landmark_order"] == ["thumb", "index", "wrist"]
    assert np.asarray(record["uvd_pixels"]).shape == (6, 2, 3, 3)
    assert np.asarray(record["projection_valid"]).shape == (6, 2, 3)
    assert record["invalid_projection_count"] == 0
    assert record["out_of_frame_count"] == 0
    assert (output_root / "contact_sheet.jpg").is_file()


def test_formal_audit_rejects_incomplete_task_roots(tmp_path: Path) -> None:
    from examples.modelExtensions.CoT.scripts.visualize_robocasa_hand_lrw_sidecars import (
        discover_visual_task_roots,
    )

    dataset_root = tmp_path / "rerender"
    make_visual_task(dataset_root, 0)

    with pytest.raises(FileNotFoundError, match="all 24 canonical tasks"):
        discover_visual_task_roots(dataset_root)
