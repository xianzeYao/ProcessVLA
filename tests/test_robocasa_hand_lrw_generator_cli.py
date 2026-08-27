from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from starVLA.dataloader.robocasa_fourier_tasks import FOURIER_TASKS
from starVLA.robocasa_hand_lrw import hand_lrw_path


def write_info(root: Path, total_episodes: int) -> None:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": total_episodes,
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


def write_episode(root: Path, episode_id: int, demo_id: str, frames: int = 2) -> None:
    chunk = episode_id // 1000
    camera_relative = Path("camera") / f"chunk-{chunk:03d}" / f"episode_{episode_id:06d}.npz"
    camera_path = root / camera_relative
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        camera_path,
        agentview_K=np.broadcast_to(
            np.eye(3, dtype=np.float32), (frames, 3, 3)
        ).copy(),
        agentview_T_world_camera=np.broadcast_to(
            np.eye(4, dtype=np.float32), (frames, 4, 4)
        ).copy(),
    )
    parquet_path = (
        root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_id:06d}.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "episode_index": [episode_id] * frames,
            "index": np.arange(100, 100 + frames),
            "timestamp": np.arange(frames, dtype=np.float64) / 20.0,
            "observation.camera.params_path": [camera_relative.as_posix()] * frames,
            "source.hdf5_demo_id": [demo_id] * frames,
        }
    ).to_parquet(parquet_path, index=False)


def valid_payload(frames: int = 2) -> dict[str, np.ndarray]:
    world = np.zeros((frames, 2, 3, 3), dtype=np.float32)
    world[..., 2] = 1.0
    uvd = np.zeros_like(world)
    uvd[..., 0] = 20.0
    uvd[..., 1] = 10.0
    uvd[..., 2] = 1.0
    return {
        "world_xyz": world,
        "agentview_uvd_pixels": uvd,
        "agentview_projection_valid": np.ones((frames, 2, 3), dtype=np.bool_),
        "agentview_in_frame": np.ones((frames, 2, 3), dtype=np.bool_),
    }


def create_task_fixture(
    base: Path,
    *,
    task_index: int = 0,
    total_episodes: int = 2,
) -> tuple[Path, Path, Path]:
    task = FOURIER_TASKS[task_index]
    dataset_root = base / "rerender"
    hdf5_root = base / "hdf5"
    task_root = dataset_root / task.official_dataset_name
    write_info(task_root, total_episodes)
    hdf5_root.mkdir(parents=True, exist_ok=True)
    (hdf5_root / task.hdf5_filename).touch()
    for episode_id in range(total_episodes):
        write_episode(task_root, episode_id, f"demo_{episode_id}")
    return dataset_root, hdf5_root, task_root


def test_cli_import_does_not_require_training_accelerate_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    code = """
import builtins
real_import = builtins.__import__
def block_accelerate(name, *args, **kwargs):
    if name == 'accelerate' or name.startswith('accelerate.'):
        raise ModuleNotFoundError('blocked accelerate for replay-boundary test')
    return real_import(name, *args, **kwargs)
builtins.__import__ = block_accelerate
import starVLA.robocasa_hand_lrw_cli as module
print(len(module.FOURIER_TASKS))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "24"


def test_discovers_all_canonical_tasks_in_fourier_order(tmp_path: Path) -> None:
    from starVLA.robocasa_hand_lrw_cli import discover_task_roots

    dataset_root = tmp_path / "rerender"
    hdf5_root = tmp_path / "hdf5"
    hdf5_root.mkdir()
    for task in FOURIER_TASKS:
        write_info(dataset_root / task.official_dataset_name, 1)
        (hdf5_root / task.hdf5_filename).touch()

    roots = discover_task_roots(dataset_root, hdf5_root)

    assert [item.task.basename for item in roots] == [
        task.basename for task in FOURIER_TASKS
    ]
    assert all(item.total_episodes == 1 for item in roots)


def test_resolves_episode_source_and_camera_alignment(tmp_path: Path) -> None:
    from starVLA.robocasa_hand_lrw_cli import (
        discover_task_roots,
        load_episode_camera,
        resolve_episode_source,
    )

    dataset_root, hdf5_root, _ = create_task_fixture(tmp_path)
    task_root = discover_task_roots(
        dataset_root,
        hdf5_root,
        task_names=[FOURIER_TASKS[0].basename],
    )[0]

    source = resolve_episode_source(task_root, 1)
    camera_k, camera_pose = load_episode_camera(source)

    assert source.episode_id == 1
    assert source.demo_id == "demo_1"
    assert source.frame_count == 2
    assert source.width == 64 and source.height == 32
    assert camera_k.shape == (2, 3, 3)
    assert camera_pose.shape == (2, 4, 4)


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("source.hdf5_demo_id", ["demo_0", "demo_1"], "exactly one value"),
        ("episode_index", [0, 1], "episode_index"),
    ],
)
def test_rejects_ambiguous_episode_mapping(
    tmp_path: Path, column: str, values: list[object], message: str
) -> None:
    from starVLA.robocasa_hand_lrw_cli import (
        discover_task_roots,
        episode_parquet_path,
        resolve_episode_source,
    )

    dataset_root, hdf5_root, _ = create_task_fixture(tmp_path)
    task_root = discover_task_roots(
        dataset_root,
        hdf5_root,
        task_names=[FOURIER_TASKS[0].basename],
    )[0]
    path = episode_parquet_path(task_root.dataset_path, 0)
    frame = pd.read_parquet(path)
    frame[column] = values
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match=message):
        resolve_episode_source(task_root, 0)


def test_validate_only_reports_missing_then_finalizes_complete_task(
    tmp_path: Path,
) -> None:
    from examples.modelExtensions.CoT.scripts.build_robocasa_hand_lrw_sidecars import (
        write_sidecar_atomic,
    )
    from starVLA.robocasa_hand_lrw_cli import run

    dataset_root, hdf5_root, task_root = create_task_fixture(tmp_path)
    write_sidecar_atomic(
        hand_lrw_path(task_root, 0),
        valid_payload(),
        frame_count=2,
        width=64,
        height=32,
        overwrite=False,
    )
    failed_report = tmp_path / "failed.json"
    common = [
        "--dataset-root",
        str(dataset_root),
        "--hdf5-root",
        str(hdf5_root),
        "--task",
        FOURIER_TASKS[0].basename,
        "--validate-only",
    ]

    with pytest.raises(RuntimeError, match="1 sidecar.*failed validation"):
        run([*common, "--report-path", str(failed_report)])

    failed = json.loads(failed_report.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["totals"]["selected_episodes"] == 2
    assert failed["missing"] == [
        f"{FOURIER_TASKS[0].basename}/episode_000001"
    ]
    info = json.loads((task_root / "meta" / "info.json").read_text())
    assert "hand_lrw" not in info.get("geometry_paths", {})

    write_sidecar_atomic(
        hand_lrw_path(task_root, 1),
        valid_payload(),
        frame_count=2,
        width=64,
        height=32,
        overwrite=False,
    )
    success_report = tmp_path / "success.json"
    result = run([*common, "--report-path", str(success_report)])

    assert result["status"] == "success"
    assert result["totals"]["validated_episodes"] == 2
    assert result["missing"] == []
    info = json.loads((task_root / "meta" / "info.json").read_text())
    assert "hand_lrw" in info["geometry_paths"]


def test_partial_validate_never_advertises_task_metadata(tmp_path: Path) -> None:
    from examples.modelExtensions.CoT.scripts.build_robocasa_hand_lrw_sidecars import (
        write_sidecar_atomic,
    )
    from starVLA.robocasa_hand_lrw_cli import run

    dataset_root, hdf5_root, task_root = create_task_fixture(tmp_path)
    write_sidecar_atomic(
        hand_lrw_path(task_root, 0),
        valid_payload(),
        frame_count=2,
        width=64,
        height=32,
        overwrite=False,
    )

    result = run(
        [
            "--dataset-root",
            str(dataset_root),
            "--hdf5-root",
            str(hdf5_root),
            "--task",
            FOURIER_TASKS[0].basename,
            "--episode-end",
            "1",
            "--validate-only",
            "--report-path",
            str(tmp_path / "partial.json"),
        ]
    )

    assert result["status"] == "success"
    assert result["tasks"][FOURIER_TASKS[0].basename]["metadata_finalized"] is False
    info = json.loads((task_root / "meta" / "info.json").read_text())
    assert "hand_lrw" not in info.get("geometry_paths", {})
