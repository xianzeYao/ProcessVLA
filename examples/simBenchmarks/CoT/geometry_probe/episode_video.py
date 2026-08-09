"""Episode planning and video helpers for paired geometry comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from examples.simBenchmarks.CoT.geometry_probe.probe_utils import EpisodeRef, SampleRef


def select_manifest_episodes(
    manifest: Sequence[SampleRef],
    *,
    groups: Sequence[str],
) -> list[EpisodeRef]:
    """Select the first manifest episode for every requested group."""
    ordered_groups = [str(group) for group in groups]
    if not ordered_groups:
        raise ValueError("at least one group is required")
    if len(set(ordered_groups)) != len(ordered_groups):
        raise ValueError(f"groups must be unique, got {ordered_groups}")
    first_by_group: dict[str, EpisodeRef] = {}
    requested = set(ordered_groups)
    for sample in manifest:
        group = str(sample.suite)
        if group in requested and group not in first_by_group:
            first_by_group[group] = EpisodeRef(
                suite=group,
                episode_id=int(sample.episode_id),
                episode_length=int(sample.episode_length),
            )
    missing = [group for group in ordered_groups if group not in first_by_group]
    if missing:
        raise ValueError(f"manifest is missing requested groups: {missing}")
    return [first_by_group[group] for group in ordered_groups]


def build_episode_anchor_plan(
    episodes: Sequence[EpisodeRef],
    *,
    horizon: int,
    stride: int = 8,
) -> list[SampleRef]:
    """Expand episodes into regularly spaced anchors with complete futures."""
    horizon = int(horizon)
    stride = int(stride)
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")
    plan: list[SampleRef] = []
    for episode in episodes:
        length = int(episode.episode_length)
        anchors = list(range(0, max(length - horizon, 0), stride))
        if not anchors:
            raise ValueError(
                f"episode {episode.suite}/{episode.episode_id} has no complete "
                f"horizon={horizon} anchor at length={length}"
            )
        plan.extend(
            SampleRef(
                suite=str(episode.suite),
                episode_id=int(episode.episode_id),
                episode_length=length,
                frame_index=anchor,
            )
            for anchor in anchors
        )
    return plan


def infer_video_fps(
    timestamps: Sequence[float],
    *,
    fallback_fps: float = 2.5,
) -> float:
    """Infer playback FPS from sampled-frame timestamps."""
    fallback_fps = float(fallback_fps)
    if not np.isfinite(fallback_fps) or fallback_fps <= 0.0:
        raise ValueError(f"fallback_fps must be finite and positive, got {fallback_fps}")
    finite = np.asarray(timestamps, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return fallback_fps
    positive_steps = np.diff(finite)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0.0)]
    if positive_steps.size == 0:
        return fallback_fps
    fps = 1.0 / float(np.median(positive_steps))
    return fps if np.isfinite(fps) and fps > 0.0 else fallback_fps


def write_video_frames(
    path: str | Path,
    frames: Iterable[np.ndarray],
    *,
    fps: float,
    codec: str = "libx264",
) -> dict[str, float | int | str]:
    """Write constant-sized RGB frames to a validated MP4 file."""

    import cv2

    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    iterator = iter(frames)
    try:
        first = np.asarray(next(iterator))
    except StopIteration as error:
        raise ValueError("at least one video frame is required") from error

    if (
        first.ndim != 3
        or first.shape[2] != 3
        or first.dtype != np.uint8
        or first.shape[0] % 2
        or first.shape[1] % 2
    ):
        raise ValueError(f"RGB frames must be even-sized uint8 [H,W,3], got {first.shape}/{first.dtype}")
    height, width = first.shape[:2]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(codec) in {"libx264", "h264"}:
        import av
        from fractions import Fraction
        from itertools import chain

        rate = Fraction(str(fps)).limit_denominator(100000)
        output = av.open(str(path), "w", options={"movflags": "+faststart"})
        stream = output.add_stream("libx264", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        count = 0
        try:
            for array in chain((first,), iterator):
                array = np.asarray(array)
                if array.shape != first.shape or array.dtype != np.uint8:
                    raise ValueError(
                        f"video frame shape/dtype changed: {array.shape}/{array.dtype} "
                        f"vs {first.shape}/{first.dtype}"
                    )
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                frame.pts = count
                frame.time_base = Fraction(rate.denominator, rate.numerator)
                for packet in stream.encode(frame):
                    output.mux(packet)
                count += 1
            for packet in stream.encode():
                output.mux(packet)
        finally:
            output.close()
        return {
            "path": str(path),
            "fps": fps,
            "frame_count": count,
            "width": width,
            "height": height,
            "codec": "h264",
        }
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*str(codec)), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    count = 0
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        count += 1
        for frame in iterator:
            array = np.asarray(frame)
            if array.shape != first.shape or array.dtype != np.uint8:
                raise ValueError(
                    f"video frame shape/dtype changed: {array.shape}/{array.dtype} "
                    f"vs {first.shape}/{first.dtype}"
                )
            writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            count += 1
    finally:
        writer.release()
    return {
        "path": str(path),
        "fps": fps,
        "frame_count": count,
        "width": width,
        "height": height,
    }
