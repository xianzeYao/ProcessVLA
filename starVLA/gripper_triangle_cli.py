"""CLI orchestration for LIBERO gripper-triangle sidecars."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from starVLA.gripper_triangle import (
    gripper_triangle_path,
    load_gripper_triangle_sidecar,
    project_world_to_agentview_uvd,
)


SUITE_DIRECTORY_NAMES = (
    ("libero_10", "libero_10_no_noops_1.0.0_lerobot"),
    ("libero_goal", "libero_goal_no_noops_1.0.0_lerobot"),
    ("libero_object", "libero_object_no_noops_1.0.0_lerobot"),
    ("libero_spatial", "libero_spatial_no_noops_1.0.0_lerobot"),
)


@dataclass(frozen=True)
class SuiteRoot:
    name: str
    path: Path
    total_episodes: int


@dataclass(frozen=True)
class EpisodeSource:
    suite: SuiteRoot
    episode_id: int
    frame_count: int
    width: int
    height: int
    hdf5_path: Path
    demo_id: str
    hdf5_indices: np.ndarray
    camera_path: Path


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
    if total_episodes < 0:
        raise ValueError(f"total_episodes must be non-negative, got {total_episodes}")
    if start < 0 or start > total_episodes:
        raise ValueError(f"episode_start must lie inside [0,{total_episodes}], got {start}")
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


def discover_suite_roots(dataset_root: str | Path) -> list[SuiteRoot]:
    root = Path(dataset_root)
    suites: list[SuiteRoot] = []
    for suite_name, directory_name in SUITE_DIRECTORY_NAMES:
        suite_path = root / directory_name
        info_path = suite_path / "meta" / "info.json"
        if not info_path.is_file():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        suites.append(
            SuiteRoot(
                name=suite_name,
                path=suite_path,
                total_episodes=int(info["total_episodes"]),
            )
        )
    if not suites:
        raise FileNotFoundError(f"no LIBERO rerender suites found below {root}")
    return suites


def _episode_parquet_path(dataset_root: Path, episode_id: int) -> Path:
    expected = (
        dataset_root
        / "data"
        / f"chunk-{episode_id // 1000:03d}"
        / f"episode_{episode_id:06d}.parquet"
    )
    if expected.is_file():
        return expected
    matches = list(
        (dataset_root / "data").glob(f"chunk-*/episode_{episode_id:06d}.parquet")
    )
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(expected)


def _stored_image_size(dataset_root: Path) -> tuple[int, int]:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    try:
        shape = info["features"]["observation.images.image"]["shape"]
        height, width = int(shape[0]), int(shape[1])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(
            f"{info_path} lacks observation.images.image [height,width,channels]"
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


def _read_episode_frame(suite: SuiteRoot, episode_id: int) -> Any:
    import pandas as pd

    return pd.read_parquet(_episode_parquet_path(suite.path, episode_id))


def _episode_spec(suite: SuiteRoot, episode_id: int, frame: Any) -> Any:
    from examples.modelExtensions.CoT.scripts.build_libero_gripper_triangle_sidecars import (
        EpisodeSidecarSpec,
    )

    width, height = _stored_image_size(suite.path)
    return EpisodeSidecarSpec(
        episode_id=episode_id,
        frame_count=int(len(frame)),
        width=width,
        height=height,
    )


def _episode_source(suite: SuiteRoot, spec: Any, frame: Any) -> EpisodeSource:
    hdf5_path = Path(str(_one_value(frame, "source.hdf5_path")))
    demo_id = str(_one_value(frame, "source.hdf5_demo_id"))
    camera_relative = Path(str(_one_value(frame, "observation.camera.params_path")))
    if "source.hdf5_index" not in frame.columns:
        raise ValueError("episode parquet is missing required column source.hdf5_index")
    indices = np.asarray(frame["source.hdf5_index"].to_numpy(), dtype=np.int64)
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    camera_path = suite.path / camera_relative
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    return EpisodeSource(
        suite=suite,
        episode_id=spec.episode_id,
        frame_count=spec.frame_count,
        width=spec.width,
        height=spec.height,
        hdf5_path=hdf5_path,
        demo_id=demo_id,
        hdf5_indices=indices,
        camera_path=camera_path,
    )


def _default_h5_open(path: str | Path, mode: str) -> Any:
    import h5py

    return h5py.File(path, mode)


def _default_env_factory(attrs: dict[str, Any], width: int, height: int) -> Any:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from examples.modelExtensions.CoT.scripts.export_libero_hdf5_state_depth import (
        env_kwargs_from_hdf5_attrs,
    )
    from libero.libero.envs import OffScreenRenderEnv

    if width != height:
        raise ValueError(f"LIBERO replay expects square images, got {width}x{height}")
    env_kwargs = env_kwargs_from_hdf5_attrs(attrs, ["agentview"], height)
    return OffScreenRenderEnv(**env_kwargs)


def _load_camera(source: EpisodeSource) -> tuple[np.ndarray, np.ndarray]:
    with np.load(source.camera_path, allow_pickle=False) as camera:
        k = np.asarray(camera["agentview_K"], dtype=np.float32)
        t_world_camera = np.asarray(
            camera["agentview_T_world_camera"], dtype=np.float32
        )
    if k.shape != (source.frame_count, 3, 3):
        raise ValueError(
            f"agentview_K shape must be {(source.frame_count, 3, 3)}, got {k.shape}"
        )
    if t_world_camera.shape != (source.frame_count, 4, 4):
        raise ValueError(
            "agentview_T_world_camera shape must be "
            f"{(source.frame_count, 4, 4)}, got {t_world_camera.shape}"
        )
    return k, t_world_camera


def _aggregate_suite_report(
    specs: Sequence[Any], metrics: Sequence[dict[str, float]]
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "episode_count": len(specs),
        "frame_count": int(sum(spec.frame_count for spec in specs)),
    }
    if metrics:
        weights = np.asarray(
            [max(spec.frame_count, 1) for spec in specs[: len(metrics)]], dtype=np.float64
        )
        report.update(
            {
                "min_finger_distance_m": float(
                    min(item["min_finger_distance_m"] for item in metrics)
                ),
                "min_triangle_area_m2": float(
                    min(item["min_triangle_area_m2"] for item in metrics)
                ),
                "projection_valid_point_ratio": float(
                    np.average(
                        [item["projection_valid_point_ratio"] for item in metrics],
                        weights=weights,
                    )
                ),
                "in_frame_point_ratio": float(
                    np.average(
                        [item["in_frame_point_ratio"] for item in metrics],
                        weights=weights,
                    )
                ),
            }
        )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate physical LIBERO gripper-triangle sidecars."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--suite",
        action="append",
        choices=[name for name, _ in SUITE_DIRECTORY_NAMES],
    )
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--report-path",
        default="artifacts/libero_gripper_triangle_v3/sidecar_generation_report.json",
    )
    return parser.parse_args(argv)


def run(
    argv: Sequence[str] | None = None,
    *,
    env_factory: Callable[[dict[str, Any], int, int], Any] | None = None,
    h5_open: Callable[[str | Path, str], Any] | None = None,
) -> dict[str, Any]:
    from examples.modelExtensions.CoT.scripts.build_libero_gripper_triangle_sidecars import (
        _write_json_atomic,
        extract_gripper_triangle_world,
        finalize_geometry_metadata,
        summarize_sidecar,
        write_sidecar_atomic,
    )

    args = parse_args(argv)
    env_factory = env_factory or _default_env_factory
    h5_open = h5_open or _default_h5_open
    suites = discover_suite_roots(Path(args.dataset_root))
    if args.suite:
        requested = set(args.suite)
        suites = [suite for suite in suites if suite.name in requested]
        missing_suites = requested - {suite.name for suite in suites}
        if missing_suites:
            raise FileNotFoundError(f"requested suites not found: {sorted(missing_suites)}")

    selected: dict[str, list[Any]] = {}
    frames: dict[tuple[str, int], Any] = {}
    for suite in suites:
        episode_ids = select_episode_ids(
            suite.total_episodes,
            episode_start=args.episode_start,
            episode_end=args.episode_end,
            max_episodes=args.max_episodes,
        )
        specs: list[Any] = []
        for episode_id in episode_ids:
            frame = _read_episode_frame(suite, episode_id)
            frames[(suite.name, episode_id)] = frame
            specs.append(_episode_spec(suite, episode_id, frame))
        selected[suite.name] = specs

    written_count = 0
    skipped_count = 0
    generation_errors: list[dict[str, str]] = []
    if not args.validate_only:
        by_hdf5: dict[Path, list[EpisodeSource]] = defaultdict(list)
        for suite in suites:
            for spec in selected[suite.name]:
                path = gripper_triangle_path(suite.path, spec.episode_id)
                if path.exists() and not args.overwrite:
                    try:
                        load_gripper_triangle_sidecar(
                            path,
                            frame_count=spec.frame_count,
                            width=spec.width,
                            height=spec.height,
                        )
                    except Exception as error:
                        generation_errors.append(
                            {
                                "episode": f"{suite.name}/episode_{spec.episode_id:06d}",
                                "error": str(error),
                            }
                        )
                    else:
                        skipped_count += 1
                    continue
                try:
                    source = _episode_source(
                        suite, spec, frames[(suite.name, spec.episode_id)]
                    )
                except Exception as error:
                    generation_errors.append(
                        {
                            "episode": f"{suite.name}/episode_{spec.episode_id:06d}",
                            "error": str(error),
                        }
                    )
                else:
                    by_hdf5[source.hdf5_path].append(source)

        for hdf5_path, sources in by_hdf5.items():
            try:
                with h5_open(hdf5_path, "r") as h5:
                    env = env_factory(
                        dict(h5["data"].attrs), sources[0].width, sources[0].height
                    )
                    try:
                        for source in sources:
                            try:
                                states = h5[f"data/{source.demo_id}/states"]
                                world_xyz = extract_gripper_triangle_world(
                                    env, states, source.hdf5_indices
                                )
                                k, t_world_camera = _load_camera(source)
                                uvd, projection_valid, in_frame = (
                                    project_world_to_agentview_uvd(
                                        world_xyz,
                                        k,
                                        t_world_camera,
                                        width=source.width,
                                        height=source.height,
                                    )
                                )
                                write_sidecar_atomic(
                                    gripper_triangle_path(
                                        source.suite.path, source.episode_id
                                    ),
                                    {
                                        "world_xyz": world_xyz,
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
                                written_count += 1
                            except Exception as error:
                                generation_errors.append(
                                    {
                                        "episode": (
                                            f"{source.suite.name}/"
                                            f"episode_{source.episode_id:06d}"
                                        ),
                                        "error": str(error),
                                    }
                                )
                    finally:
                        close = getattr(env, "close", None)
                        if callable(close):
                            close()
            except Exception as error:
                labels = [
                    f"{source.suite.name}/episode_{source.episode_id:06d}"
                    for source in sources
                ]
                generation_errors.extend(
                    {"episode": label, "error": str(error)} for label in labels
                )

    missing: list[str] = []
    corrupt: list[dict[str, str]] = list(generation_errors)
    corrupt_labels = {item["episode"] for item in corrupt}
    suite_metrics: dict[str, list[dict[str, float]]] = {
        suite.name: [] for suite in suites
    }
    validated_count = 0
    for suite in suites:
        for spec in selected[suite.name]:
            label = f"{suite.name}/episode_{spec.episode_id:06d}"
            try:
                sidecar = load_gripper_triangle_sidecar(
                    gripper_triangle_path(suite.path, spec.episode_id),
                    frame_count=spec.frame_count,
                    width=spec.width,
                    height=spec.height,
                )
            except FileNotFoundError:
                missing.append(label)
            except Exception as error:
                if label not in corrupt_labels:
                    corrupt.append({"episode": label, "error": str(error)})
                    corrupt_labels.add(label)
            else:
                validated_count += 1
                suite_metrics[suite.name].append(summarize_sidecar(sidecar))

    failure_count = len(missing) + len(corrupt)
    report = {
        "status": "failed" if failure_count else "success",
        "mode": "validate_only" if args.validate_only else "generate",
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "totals": {
            "selected_episodes": int(sum(len(specs) for specs in selected.values())),
            "selected_frames": int(
                sum(spec.frame_count for specs in selected.values() for spec in specs)
            ),
            "validated_episodes": validated_count,
            "written_episodes": written_count,
            "skipped_episodes": skipped_count,
        },
        "missing": missing,
        "corrupt": corrupt,
        "suites": {
            suite.name: _aggregate_suite_report(
                selected[suite.name], suite_metrics[suite.name]
            )
            for suite in suites
        },
    }
    _write_json_atomic(Path(args.report_path), report)
    if failure_count:
        raise RuntimeError(f"{failure_count} sidecar(s) failed validation")
    for suite in suites:
        finalize_geometry_metadata(suite.path, selected[suite.name])
    return report


def main(argv: Sequence[str] | None = None) -> None:
    report = run(argv)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
