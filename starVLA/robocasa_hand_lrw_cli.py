"""CLI orchestration for RoboCasa GR1 bilateral LRW sidecars."""

from __future__ import annotations

import argparse
import json
import os
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from examples.modelExtensions.CoT.scripts.robocasa_rerender_geometry import (
    dial_content_region_mask,
)
from starVLA.robocasa_hand_lrw import (
    hand_lrw_path,
    load_hand_lrw_sidecar,
    project_hand_lrw_world_to_agentview_uvd,
)


def _load_fourier_task_metadata() -> Any:
    """Load the pure task table without importing the training dataloader package."""

    module_path = Path(__file__).resolve().parent / "dataloader" / "robocasa_fourier_tasks.py"
    module_name = "robocasa_fourier_tasks_for_hand_lrw"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Fourier task metadata helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_FOURIER_TASK_METADATA = _load_fourier_task_metadata()
FOURIER_TASKS = _FOURIER_TASK_METADATA.FOURIER_TASKS
FourierTask = _FOURIER_TASK_METADATA.FourierTask


@dataclass(frozen=True)
class TaskRoot:
    task: FourierTask
    dataset_path: Path
    hdf5_path: Path
    total_episodes: int


@dataclass(frozen=True)
class EpisodeSource:
    task_root: TaskRoot
    episode_id: int
    frame_count: int
    width: int
    height: int
    demo_id: str
    camera_path: Path


def discover_task_roots(
    dataset_root: str | Path,
    hdf5_root: str | Path,
    *,
    task_names: Sequence[str] | None = None,
) -> list[TaskRoot]:
    """Resolve requested canonical tasks in the declared 24-task order."""

    dataset_root = Path(dataset_root)
    hdf5_root = Path(hdf5_root)
    requested = None if task_names is None else {str(name) for name in task_names}
    canonical_names = {task.basename for task in FOURIER_TASKS}
    if requested is not None:
        unknown = requested - canonical_names
        if unknown:
            raise ValueError(f"unknown RoboCasa Fourier tasks: {sorted(unknown)}")
        if not requested:
            raise ValueError("task selection must not be empty")

    roots: list[TaskRoot] = []
    for task in FOURIER_TASKS:
        if requested is not None and task.basename not in requested:
            continue
        dataset_path = dataset_root / task.official_dataset_name
        info_path = dataset_path / "meta" / "info.json"
        hdf5_path = hdf5_root / task.hdf5_filename
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        if not hdf5_path.is_file():
            raise FileNotFoundError(hdf5_path)
        info = json.loads(info_path.read_text(encoding="utf-8"))
        total_episodes = int(info["total_episodes"])
        if total_episodes < 1:
            raise ValueError(f"{info_path} total_episodes must be positive")
        roots.append(
            TaskRoot(
                task=task,
                dataset_path=dataset_path,
                hdf5_path=hdf5_path,
                total_episodes=total_episodes,
            )
        )
    if not roots:
        raise FileNotFoundError(f"no RoboCasa Fourier task roots found below {dataset_root}")
    return roots


def select_episode_ids(
    total_episodes: int,
    *,
    episode_start: int,
    episode_end: int | None,
    max_episodes: int | None,
) -> list[int]:
    total_episodes = int(total_episodes)
    start = int(episode_start)
    end = total_episodes if episode_end is None else int(episode_end)
    if start < 0 or start > total_episodes:
        raise ValueError(
            f"episode_start must lie inside [0,{total_episodes}], got {start}"
        )
    if end < start or end > total_episodes:
        raise ValueError(
            f"episode_end must lie inside [{start},{total_episodes}], got {end}"
        )
    episode_ids = list(range(start, end))
    if max_episodes is not None:
        limit = int(max_episodes)
        if limit < 1:
            raise ValueError(f"max_episodes must be positive, got {limit}")
        episode_ids = episode_ids[:limit]
    if not episode_ids:
        raise ValueError("episode selection is empty")
    return episode_ids


def episode_parquet_path(dataset_root: str | Path, episode_id: int) -> Path:
    root = Path(dataset_root)
    episode_id = int(episode_id)
    expected = (
        root
        / "data"
        / f"chunk-{episode_id // 1000:03d}"
        / f"episode_{episode_id:06d}.parquet"
    )
    if expected.is_file():
        return expected
    matches = list(
        (root / "data").glob(f"chunk-*/episode_{episode_id:06d}.parquet")
    )
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(expected)


def _stored_image_size(dataset_root: Path) -> tuple[int, int]:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    try:
        shape = info["features"]["observation.images.ego_view"]["shape"]
        height, width = int(shape[0]), int(shape[1])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(
            f"{info_path} lacks observation.images.ego_view [height,width,channels]"
        ) from error
    if height < 1 or width < 1:
        raise ValueError(f"stored image dimensions must be positive, got {width}x{height}")
    return width, height


def _one_value(frame: Any, column: str) -> Any:
    if column not in frame.columns:
        raise ValueError(f"episode parquet is missing required column {column}")
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"{column} must contain exactly one value, got {values}")
    return values[0]


def _validate_episode_frame_order(frame: Any) -> None:
    """Require Parquet rows to preserve the source state's temporal order."""

    if "index" not in frame.columns:
        raise ValueError("episode parquet is missing required column index")
    observed = np.asarray(frame["index"])
    if not np.issubdtype(observed.dtype, np.number):
        raise ValueError("index must be an ordered contiguous integer sequence")
    if not np.isfinite(observed).all():
        raise ValueError("index must be an ordered contiguous integer sequence")
    integral = observed.astype(np.int64)
    if not np.array_equal(observed, integral):
        raise ValueError("index must be an ordered contiguous integer sequence")
    if len(integral) > 1 and not np.all(np.diff(integral) == 1):
        raise ValueError(
            "index must be an ordered contiguous integer sequence with step 1"
        )


def resolve_episode_source(task_root: TaskRoot, episode_id: int) -> EpisodeSource:
    """Validate one rerender episode's source-demo and camera mapping."""

    import pandas as pd

    episode_id = int(episode_id)
    frame = pd.read_parquet(episode_parquet_path(task_root.dataset_path, episode_id))
    if frame.empty:
        raise ValueError(f"episode {episode_id} parquet is empty")
    _validate_episode_frame_order(frame)
    observed_episode = int(_one_value(frame, "episode_index"))
    if observed_episode != episode_id:
        raise ValueError(
            f"episode_index must be {episode_id}, got {observed_episode}"
        )
    demo_id = str(_one_value(frame, "source.hdf5_demo_id"))
    if not demo_id:
        raise ValueError("source.hdf5_demo_id must be non-empty")
    camera_relative = Path(
        str(_one_value(frame, "observation.camera.params_path"))
    )
    camera_path = task_root.dataset_path / camera_relative
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    width, height = _stored_image_size(task_root.dataset_path)
    return EpisodeSource(
        task_root=task_root,
        episode_id=episode_id,
        frame_count=int(len(frame)),
        width=width,
        height=height,
        demo_id=demo_id,
        camera_path=camera_path,
    )


def load_episode_camera(
    source: EpisodeSource,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(source.camera_path, allow_pickle=False) as camera:
        camera_k = np.asarray(camera["agentview_K"], dtype=np.float32)
        camera_pose = np.asarray(
            camera["agentview_T_world_camera"], dtype=np.float32
        )
    expected_k = (source.frame_count, 3, 3)
    expected_pose = (source.frame_count, 4, 4)
    if camera_k.shape != expected_k:
        raise ValueError(
            f"agentview_K shape must be {expected_k}, got {camera_k.shape}"
        )
    if camera_pose.shape != expected_pose:
        raise ValueError(
            "agentview_T_world_camera shape must be "
            f"{expected_pose}, got {camera_pose.shape}"
        )
    return camera_k, camera_pose


def _default_h5_open(path: str | Path, mode: str) -> Any:
    import h5py

    return h5py.File(path, mode)


def _load_robocasa_runtime(
    robocasa_repo: Path,
) -> tuple[Callable[[Path], Any], Callable[[Any, dict[str, Any]], Any]]:
    repo = str(robocasa_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from argparse import Namespace

    from examples.modelExtensions.CoT.scripts.robocasa_book_filter import (
        make_book_safe_reset_to,
    )
    from robocasa.scripts.playback_dataset import (
        make_env_from_args,
        reset_to as robocasa_reset_to,
    )

    def env_factory(hdf5_path: Path) -> Any:
        return make_env_from_args(
            Namespace(
                dataset=str(hdf5_path),
                use_abs_actions=False,
                render=False,
                verbose=False,
            )
        )

    return env_factory, make_book_safe_reset_to(robocasa_reset_to)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate RoboCasa bilateral LRW sidecars."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--hdf5-root", required=True)
    parser.add_argument(
        "--robocasa-repo",
        default="/root/data/yxz/benchmarks/robocasa-gr1-tabletop-tasks",
    )
    parser.add_argument("--task", action="append")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--report-path",
        default="artifacts/robocasa_hand_lrw_v5/sidecar_generation_report.json",
    )
    return parser.parse_args(argv)


def run(
    argv: Sequence[str] | None = None,
    *,
    env_factory: Callable[[Path], Any] | None = None,
    reset_to: Callable[[Any, dict[str, Any]], Any] | None = None,
    h5_open: Callable[[str | Path, str], Any] | None = None,
) -> dict[str, Any]:
    from examples.modelExtensions.CoT.scripts.build_robocasa_hand_lrw_sidecars import (
        EpisodeSidecarSpec,
        _write_json_atomic,
        extract_bilateral_lrw_world,
        finalize_geometry_metadata,
        revoke_geometry_metadata,
        summarize_sidecar,
        write_sidecar_atomic,
    )

    args = parse_args(argv)
    task_roots = discover_task_roots(
        args.dataset_root,
        args.hdf5_root,
        task_names=args.task,
    )
    h5_open = h5_open or _default_h5_open

    selected: dict[str, list[EpisodeSource]] = {}
    requested: dict[str, list[int]] = {}
    selection_errors: list[dict[str, str]] = []
    for task_root in task_roots:
        episode_ids = select_episode_ids(
            task_root.total_episodes,
            episode_start=args.episode_start,
            episode_end=args.episode_end,
            max_episodes=args.max_episodes,
        )
        requested[task_root.task.basename] = episode_ids
        sources: list[EpisodeSource] = []
        for episode_id in episode_ids:
            try:
                sources.append(resolve_episode_source(task_root, episode_id))
            except Exception as error:
                selection_errors.append(
                    {
                        "episode": f"{task_root.task.basename}/episode_{episode_id:06d}",
                        "error": str(error),
                    }
                )
        first_episode_by_demo: dict[str, int] = {}
        for source in sources:
            first_episode = first_episode_by_demo.setdefault(
                source.demo_id, source.episode_id
            )
            if first_episode != source.episode_id:
                selection_errors.append(
                    {
                        "episode": (
                            f"{task_root.task.basename}/"
                            f"episode_{source.episode_id:06d}"
                        ),
                        "error": (
                            f"duplicate source.hdf5_demo_id {source.demo_id}; "
                            f"already mapped by episode_{first_episode:06d}"
                        ),
                    }
                )
        selected[task_root.task.basename] = sources

    written_count = 0
    skipped_count = 0
    generation_errors: list[dict[str, str]] = list(selection_errors)
    if not args.validate_only and not selection_errors:
        if env_factory is None or reset_to is None:
            default_env_factory, default_reset_to = _load_robocasa_runtime(
                Path(args.robocasa_repo)
            )
            env_factory = env_factory or default_env_factory
            reset_to = reset_to or default_reset_to
        for task_root in task_roots:
            sources_to_generate: list[EpisodeSource] = []
            for source in selected[task_root.task.basename]:
                path = hand_lrw_path(task_root.dataset_path, source.episode_id)
                if path.exists() and not args.overwrite:
                    try:
                        load_hand_lrw_sidecar(
                            path,
                            frame_count=source.frame_count,
                            width=source.width,
                            height=source.height,
                        )
                    except Exception:
                        # The validation pass below assigns the sole terminal
                        # status, so this is reported as corrupt only once.
                        continue
                    else:
                        skipped_count += 1
                    continue
                sources_to_generate.append(source)
            if not sources_to_generate:
                continue

            try:
                with h5_open(task_root.hdf5_path, "r") as h5:
                    env = env_factory(task_root.hdf5_path)
                    try:
                        for source in sources_to_generate:
                            label = (
                                f"{task_root.task.basename}/"
                                f"episode_{source.episode_id:06d}"
                            )
                            try:
                                demo = h5[f"data/{source.demo_id}"]
                                states = demo["states"]
                                if len(states) != source.frame_count:
                                    raise ValueError(
                                        f"HDF5 states={len(states)} but rerender frames="
                                        f"{source.frame_count}"
                                    )
                                if source.frame_count < 1:
                                    raise ValueError("episode must contain at least one state")
                                reset_to(
                                    env,
                                    {
                                        "model": demo.attrs["model_file"],
                                        "ep_meta": demo.attrs.get("ep_meta"),
                                        "states": states[0],
                                    },
                                )
                                env._get_observations(force_update=True)
                                world = extract_bilateral_lrw_world(
                                    env,
                                    states,
                                    reset_to=reset_to,
                                )
                                camera_k, camera_pose = load_episode_camera(source)
                                uvd, projection_valid, _ = (
                                    project_hand_lrw_world_to_agentview_uvd(
                                        world,
                                        camera_k,
                                        camera_pose,
                                        width=source.width,
                                        height=source.height,
                                    )
                                )
                                if source.width != source.height:
                                    raise ValueError(
                                        "DIAL content mask requires square stored images, "
                                        f"got {source.width}x{source.height}"
                                    )
                                in_frame = dial_content_region_mask(
                                    uvd,
                                    projection_valid,
                                    output_size=source.width,
                                )
                                result = write_sidecar_atomic(
                                    hand_lrw_path(
                                        task_root.dataset_path, source.episode_id
                                    ),
                                    {
                                        "world_xyz": world.astype(np.float32),
                                        "agentview_uvd_pixels": uvd.astype(np.float32),
                                        "agentview_projection_valid": projection_valid.astype(
                                            np.bool_
                                        ),
                                        "agentview_in_frame": in_frame.astype(np.bool_),
                                    },
                                    frame_count=source.frame_count,
                                    width=source.width,
                                    height=source.height,
                                    overwrite=args.overwrite,
                                )
                                written_count += int(result == "written")
                                skipped_count += int(result == "skipped")
                            except Exception as error:
                                generation_errors.append(
                                    {"episode": label, "error": str(error)}
                                )
                    finally:
                        close = getattr(env, "close", None)
                        if callable(close):
                            close()
            except Exception as error:
                for source in sources_to_generate:
                    label = (
                        f"{task_root.task.basename}/episode_{source.episode_id:06d}"
                    )
                    if not any(item["episode"] == label for item in generation_errors):
                        generation_errors.append(
                            {"episode": label, "error": str(error)}
                        )

    missing: list[str] = []
    corrupt: list[dict[str, str]] = []
    validated_count = 0
    validated_frames = 0
    task_reports: dict[str, dict[str, Any]] = {}
    for task_root in task_roots:
        sources = selected[task_root.task.basename]
        episode_metrics: list[dict[str, float]] = []
        for source in sources:
            label = (
                f"{task_root.task.basename}/episode_{source.episode_id:06d}"
            )
            path = hand_lrw_path(task_root.dataset_path, source.episode_id)
            if not path.is_file():
                missing.append(label)
                continue
            try:
                sidecar = load_hand_lrw_sidecar(
                    path,
                    frame_count=source.frame_count,
                    width=source.width,
                    height=source.height,
                )
                episode_metrics.append(summarize_sidecar(sidecar))
            except Exception as error:
                corrupt.append({"episode": label, "error": str(error)})
                continue
            validated_count += 1
            validated_frames += source.frame_count

        requested_ids = requested[task_root.task.basename]
        complete_selection = requested_ids == list(range(task_root.total_episodes))
        task_failed = any(
            label.startswith(f"{task_root.task.basename}/")
            for label in missing
        ) or any(
            item["episode"].startswith(f"{task_root.task.basename}/")
            for item in (*corrupt, *generation_errors)
        )
        metadata_finalized = False
        if complete_selection and not task_failed:
            specs = [
                EpisodeSidecarSpec(
                    source.episode_id,
                    source.frame_count,
                    source.width,
                    source.height,
                )
                for source in sources
            ]
            finalize_geometry_metadata(task_root.dataset_path, specs)
            metadata_finalized = True
        elif complete_selection and task_failed:
            revoke_geometry_metadata(task_root.dataset_path)
        task_reports[task_root.task.basename] = {
            "selected_episodes": len(sources),
            "selected_frames": int(sum(source.frame_count for source in sources)),
            "validated_episodes": len(episode_metrics),
            "metadata_finalized": metadata_finalized,
        }

    failure_labels = set(missing)
    failure_labels.update(item["episode"] for item in corrupt)
    failure_labels.update(item["episode"] for item in generation_errors)
    failure_count = len(failure_labels)
    report: dict[str, Any] = {
        "status": "success" if failure_count == 0 else "failed",
        "dataset_root": str(Path(args.dataset_root)),
        "hdf5_root": str(Path(args.hdf5_root)),
        "validate_only": bool(args.validate_only),
        "totals": {
            "selected_tasks": len(task_roots),
            "selected_episodes": int(sum(len(value) for value in selected.values())),
            "selected_frames": int(
                sum(source.frame_count for value in selected.values() for source in value)
            ),
            "written_episodes": written_count,
            "skipped_episodes": skipped_count,
            "validated_episodes": validated_count,
            "validated_frames": validated_frames,
            "missing_episodes": len(missing),
            "corrupt_episodes": len(corrupt),
            "generation_error_episodes": len(generation_errors),
            "failed_episodes": failure_count,
        },
        "tasks": task_reports,
        "missing": missing,
        "corrupt": corrupt,
        "generation_errors": generation_errors,
    }
    _write_json_atomic(Path(args.report_path), report)
    if failure_count:
        raise RuntimeError(
            f"{failure_count} sidecar episode(s) failed validation; "
            f"see {args.report_path}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    report = run(argv)
    print(
        json.dumps(
            {
                "status": report["status"],
                "tasks": report["totals"]["selected_tasks"],
                "episodes": report["totals"]["validated_episodes"],
                "report": report["dataset_root"],
            },
            indent=2,
        ),
        flush=True,
    )


__all__ = [
    "EpisodeSource",
    "TaskRoot",
    "discover_task_roots",
    "episode_parquet_path",
    "load_episode_camera",
    "main",
    "parse_args",
    "resolve_episode_source",
    "run",
    "select_episode_ids",
]
