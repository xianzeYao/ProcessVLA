#!/usr/bin/env python3
"""Render a fixed-seed audit of bilateral RoboCasa LRW UVD trajectories."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from examples.modelExtensions.CoT.scripts.build_robocasa_hand_lrw_sidecars import (
    _write_json_atomic,
)
from starVLA.robocasa_hand_lrw import (
    HAND_NAMES,
    LANDMARK_NAMES,
    hand_lrw_path,
    load_hand_lrw_sidecar,
    validate_hand_lrw_dataset_metadata,
)
from starVLA.robocasa_hand_lrw_cli import FOURIER_TASKS


LANDMARK_COLORS_RGB = (
    (255, 64, 64),
    (64, 230, 96),
    (64, 128, 255),
)


@dataclass(frozen=True)
class VisualTaskRoot:
    task: str
    dataset_path: Path
    total_episodes: int
    width: int
    height: int
    video_path_template: str


@dataclass(frozen=True)
class AuditWindow:
    task: str
    dataset_path: Path
    episode_id: int
    demo_id: str
    frame_count: int
    base_frame: int
    frame_indices: tuple[int, ...]
    width: int
    height: int
    video_path: Path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def discover_visual_task_roots(
    dataset_root: str | Path,
    *,
    allow_incomplete_preflight: bool = False,
) -> list[VisualTaskRoot]:
    """Discover available canonical task roots without training dependencies."""

    root = Path(dataset_root)
    tasks: list[VisualTaskRoot] = []
    missing: list[str] = []
    for task in FOURIER_TASKS:
        dataset_path = root / task.official_dataset_name
        info_path = dataset_path / "meta" / "info.json"
        if not info_path.is_file():
            missing.append(task.basename)
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        try:
            shape = info["features"]["observation.images.ego_view"]["shape"]
            height, width = int(shape[0]), int(shape[1])
            total_episodes = int(info["total_episodes"])
            video_template = str(info["video_path"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(f"invalid visualization metadata in {info_path}") from error
        tasks.append(
            VisualTaskRoot(
                task=task.basename,
                dataset_path=dataset_path,
                total_episodes=total_episodes,
                width=width,
                height=height,
                video_path_template=video_template,
            )
        )
    if missing and not allow_incomplete_preflight:
        raise FileNotFoundError(
            "formal LRW audit requires all 24 canonical tasks; "
            f"missing {missing} below {root}"
        )
    if not allow_incomplete_preflight:
        for task_root in tasks:
            validate_hand_lrw_dataset_metadata(task_root.dataset_path)
    if not tasks:
        raise FileNotFoundError(f"no canonical RoboCasa task roots below {root}")
    return tasks


def _sample_real_uvd_indices(start: int, end: int, count: int) -> np.ndarray:
    start = int(start)
    end = int(end)
    count = int(count)
    if end < start:
        raise ValueError(f"end must be >= start, got {start}/{end}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    available = end - start + 1
    if available <= count:
        return np.arange(start, end + 1, dtype=np.int64)
    offsets = np.rint(np.linspace(0.0, float(end - start), count)).astype(
        np.int64
    )
    offsets = np.unique(offsets)
    if len(offsets) != count:
        offsets = np.linspace(0, end - start, count, dtype=np.int64)
    return start + offsets


def _episode_video_path(task_root: VisualTaskRoot, episode_id: int) -> Path:
    relative = task_root.video_path_template.format(
        episode_chunk=int(episode_id) // 1000,
        episode_index=int(episode_id),
    )
    return task_root.dataset_path / relative


def _demo_id_from_metadata(row: dict[str, Any]) -> str:
    trajectory_id = str(row.get("trajectory_id", ""))
    try:
        return f"demo_{int(trajectory_id.rsplit('-', 1)[1])}"
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"cannot parse demo ID from trajectory_id={trajectory_id!r}"
        ) from error


def select_audit_windows(
    task_roots: Sequence[VisualTaskRoot],
    *,
    seed: int,
    task_count: int,
    action_horizon: int,
    uvd_num_points: int,
) -> list[AuditWindow]:
    """Select distinct tasks and complete windows reproducibly."""

    roots = list(task_roots)
    task_count = int(task_count)
    action_horizon = int(action_horizon)
    uvd_num_points = int(uvd_num_points)
    if task_count < 1 or task_count > len(roots):
        raise ValueError(
            f"task_count must lie inside [1,{len(roots)}], got {task_count}"
        )
    if action_horizon < 1 or uvd_num_points < 2:
        raise ValueError("action_horizon must be positive and uvd_num_points >= 2")
    rng = np.random.default_rng(int(seed))
    selected_root_indices = rng.choice(
        len(roots), size=task_count, replace=False
    ).tolist()
    selections: list[AuditWindow] = []
    for root_index in selected_root_indices:
        task_root = roots[int(root_index)]
        rows = _read_jsonl(task_root.dataset_path / "meta" / "episodes.jsonl")
        candidates = [
            row
            for row in rows
            if int(row["length"]) >= action_horizon + 1
            and hand_lrw_path(
                task_root.dataset_path, int(row["episode_index"])
            ).is_file()
            and _episode_video_path(
                task_root, int(row["episode_index"])
            ).is_file()
        ]
        if not candidates:
            raise RuntimeError(
                f"{task_root.task} has no complete episode with a valid LRW sidecar"
            )
        row = candidates[int(rng.integers(0, len(candidates)))]
        episode_id = int(row["episode_index"])
        frame_count = int(row["length"])
        max_base = frame_count - 1 - action_horizon
        base_frame = int(rng.integers(0, max_base + 1))
        frame_indices = _sample_real_uvd_indices(
            base_frame,
            base_frame + action_horizon,
            uvd_num_points,
        )
        if len(frame_indices) != uvd_num_points:
            raise RuntimeError(
                f"{task_root.task}/{episode_id} did not yield {uvd_num_points} frames"
            )
        selections.append(
            AuditWindow(
                task=task_root.task,
                dataset_path=task_root.dataset_path,
                episode_id=episode_id,
                demo_id=_demo_id_from_metadata(row),
                frame_count=frame_count,
                base_frame=base_frame,
                frame_indices=tuple(int(value) for value in frame_indices),
                width=task_root.width,
                height=task_root.height,
                video_path=_episode_video_path(task_root, episode_id),
            )
        )
    return selections


def _point_radius(depth: float, min_depth: float, max_depth: float) -> int:
    if not np.isfinite(depth):
        return 2
    span = max(max_depth - min_depth, 1e-6)
    closeness = np.clip((max_depth - depth) / span, 0.0, 1.0)
    return 3 + int(np.rint(4.0 * closeness))


def _draw_right_hand_segment(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    import cv2

    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return
    segment_count = max(int(math.ceil(length / 6.0)), 1)
    for index in range(segment_count):
        if index % 2:
            continue
        alpha0 = index / segment_count
        alpha1 = min((index + 1) / segment_count, 1.0)
        point0 = tuple(
            int(np.rint(value))
            for value in (np.asarray(start) + alpha0 * delta)
        )
        point1 = tuple(
            int(np.rint(value))
            for value in (np.asarray(start) + alpha1 * delta)
        )
        cv2.line(image, point0, point1, color, 2, cv2.LINE_AA)


def draw_lrw_trajectory_overlay(
    image: np.ndarray,
    uvd_pixels: np.ndarray,
    in_frame: np.ndarray,
) -> np.ndarray:
    """Draw six-step bilateral LRW tracks with color, style, time, and depth."""

    import cv2

    source = np.asarray(image)
    uvd = np.asarray(uvd_pixels, dtype=np.float32)
    valid = np.asarray(in_frame, dtype=np.bool_)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError(
            f"image must be uint8 RGB [H,W,3], got {source.shape}/{source.dtype}"
        )
    if uvd.ndim != 4 or uvd.shape[1:] != (2, 3, 3):
        raise ValueError(f"uvd_pixels must have shape [T,2,3,3], got {uvd.shape}")
    if valid.shape != uvd.shape[:-1]:
        raise ValueError(f"in_frame must have shape {uvd.shape[:-1]}, got {valid.shape}")
    output = source.copy()
    finite_depth = uvd[..., 2][valid & np.isfinite(uvd[..., 2])]
    min_depth = float(np.min(finite_depth)) if len(finite_depth) else 0.0
    max_depth = float(np.max(finite_depth)) if len(finite_depth) else 1.0

    def point(time: int, hand: int, landmark: int) -> tuple[int, int] | None:
        value = uvd[time, hand, landmark, :2]
        if not valid[time, hand, landmark] or not np.isfinite(value).all():
            return None
        return int(np.rint(value[0])), int(np.rint(value[1]))

    for hand in range(2):
        for landmark, color in enumerate(LANDMARK_COLORS_RGB):
            for time in range(len(uvd) - 1):
                start = point(time, hand, landmark)
                end = point(time + 1, hand, landmark)
                if start is None or end is None:
                    continue
                if hand == 0:
                    cv2.line(output, start, end, color, 3, cv2.LINE_AA)
                else:
                    _draw_right_hand_segment(output, start, end, color)

    for time in range(len(uvd)):
        for hand in range(2):
            triangle = [point(time, hand, landmark) for landmark in range(3)]
            if all(value is not None for value in triangle):
                points = np.asarray(triangle, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    output,
                    [points],
                    isClosed=True,
                    color=(210, 210, 210),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            for landmark, color in enumerate(LANDMARK_COLORS_RGB):
                location = point(time, hand, landmark)
                if location is None:
                    continue
                radius = _point_radius(
                    float(uvd[time, hand, landmark, 2]),
                    min_depth,
                    max_depth,
                )
                if hand == 1:
                    cv2.circle(output, location, radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(output, location, radius, color, -1, cv2.LINE_AA)
                if landmark == 0:
                    cv2.putText(
                        output,
                        f"{time}{'L' if hand == 0 else 'R'}",
                        (location[0] + radius + 1, location[1] - radius),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
    legend = "LRW: thumb/red index/green wrist/blue | solid=L dashed=R | size=near"
    cv2.rectangle(output, (0, 0), (min(output.shape[1] - 1, 510), 14), (0, 0, 0), -1)
    cv2.putText(
        output,
        legend,
        (3, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    """Decode one zero-based RGB frame through PyAV."""

    import av

    frame_index = int(frame_index)
    if frame_index < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame_index}")
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
    raise IndexError(f"video {path} has no frame {frame_index}")


def _selection_json(selection: AuditWindow) -> dict[str, Any]:
    return {
        "task": selection.task,
        "dataset_path": str(selection.dataset_path),
        "episode_id": selection.episode_id,
        "demo_id": selection.demo_id,
        "frame_count": selection.frame_count,
        "base_frame": selection.base_frame,
        "frame_indices": list(selection.frame_indices),
        "video_path": str(selection.video_path),
    }


def write_contact_sheet(
    path: str | Path,
    overlays: Sequence[tuple[str, np.ndarray]],
) -> Path:
    from PIL import Image, ImageDraw

    rows = list(overlays)
    if not rows:
        raise ValueError("at least one overlay is required for a contact sheet")
    first = np.asarray(rows[0][1])
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ValueError("contact sheet overlays must be uint8 RGB")
    height, width = first.shape[:2]
    columns = min(5, len(rows))
    row_count = int(math.ceil(len(rows) / columns))
    label_height = 20
    canvas = Image.new(
        "RGB", (columns * width, row_count * (height + label_height)), "black"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, overlay) in enumerate(rows):
        array = np.asarray(overlay)
        if array.shape != first.shape or array.dtype != np.uint8:
            raise ValueError("all contact sheet overlays must share shape and dtype")
        column = index % columns
        row = index // columns
        x = column * width
        y = row * (height + label_height)
        canvas.paste(Image.fromarray(array, mode="RGB"), (x, y + label_height))
        draw.text((x + 3, y + 3), str(label), fill="white")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--output-root",
        default="artifacts/robocasa_hand_lrw_v5/audit_seed42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--uvd-num-points", type=int, default=6)
    parser.add_argument(
        "--allow-incomplete-preflight",
        action="store_true",
        help="allow geometry spot-checks before all task sidecars are finalized",
    )
    return parser.parse_args(argv)


def run(
    argv: Sequence[str] | None = None,
    *,
    frame_loader: Callable[[Path, int], np.ndarray] = read_video_frame,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    task_roots = discover_visual_task_roots(
        args.dataset_root,
        allow_incomplete_preflight=bool(args.allow_incomplete_preflight),
    )
    selections = select_audit_windows(
        task_roots,
        seed=args.seed,
        task_count=args.task_count,
        action_horizon=args.action_horizon,
        uvd_num_points=args.uvd_num_points,
    )
    manifest_path = output_root / "selection_manifest.json"
    manifest: dict[str, Any] = {
        "status": "selected",
        "seed": int(args.seed),
        "task_count": int(args.task_count),
        "action_horizon": int(args.action_horizon),
        "uvd_num_points": int(args.uvd_num_points),
        "allow_incomplete_preflight": bool(args.allow_incomplete_preflight),
        "hand_order": list(HAND_NAMES),
        "landmark_order": list(LANDMARK_NAMES),
        "selections": [_selection_json(selection) for selection in selections],
        "errors": [],
    }
    _write_json_atomic(manifest_path, manifest)

    overlays: list[tuple[str, np.ndarray]] = []
    for selection in selections:
        try:
            sidecar = load_hand_lrw_sidecar(
                hand_lrw_path(selection.dataset_path, selection.episode_id),
                frame_count=selection.frame_count,
                width=selection.width,
                height=selection.height,
            )
            indices = np.asarray(selection.frame_indices, dtype=np.int64)
            uvd = sidecar.agentview_uvd_pixels[indices]
            projection_valid = sidecar.agentview_projection_valid[indices]
            in_frame = sidecar.agentview_in_frame[indices]
            image = frame_loader(selection.video_path, selection.base_frame)
            overlay = draw_lrw_trajectory_overlay(image, uvd, in_frame)

            episode_output = (
                output_root
                / selection.task
                / f"episode_{selection.episode_id:06d}"
            )
            episode_output.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            Image.fromarray(overlay, mode="RGB").save(
                episode_output / "window.jpg", quality=95
            )
            record = {
                **_selection_json(selection),
                "hand_order": list(HAND_NAMES),
                "landmark_order": list(LANDMARK_NAMES),
                "uvd_pixels": uvd.tolist(),
                "projection_valid": projection_valid.tolist(),
                "in_frame": in_frame.tolist(),
                "invalid_projection_count": int(np.count_nonzero(~projection_valid)),
                "out_of_frame_count": int(
                    np.count_nonzero(projection_valid & ~in_frame)
                ),
                "overlay_path": str(episode_output / "window.jpg"),
            }
            _write_json_atomic(episode_output / "window.json", record)
            overlays.append(
                (
                    f"{selection.task} ep{selection.episode_id} f{selection.base_frame}",
                    overlay,
                )
            )
        except Exception as error:
            manifest["errors"].append(
                {
                    "task": selection.task,
                    "episode_id": selection.episode_id,
                    "error": str(error),
                }
            )

    if overlays:
        write_contact_sheet(output_root / "contact_sheet.jpg", overlays)
    manifest["status"] = "success" if not manifest["errors"] else "failed"
    manifest["rendered_count"] = len(overlays)
    manifest["invalid_or_failed_count"] = len(manifest["errors"])
    manifest["contact_sheet"] = (
        str(output_root / "contact_sheet.jpg") if overlays else None
    )
    _write_json_atomic(manifest_path, manifest)
    if manifest["errors"] and raise_on_error:
        raise RuntimeError(
            f"{len(manifest['errors'])} selected LRW audit window(s) failed; "
            f"see {manifest_path}"
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    result = run(argv)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
