#!/usr/bin/env python3
"""Render stride-sampled full-episode GT/v1/v2 geometry comparison videos."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from examples.simBenchmarks.CoT.geometry_probe.episode_video import (
    build_episode_anchor_plan,
    infer_video_fps,
    select_manifest_episodes,
    write_video_frames,
)
from examples.simBenchmarks.CoT.geometry_probe.paired_probe import (
    load_materialized_sample,
    load_prediction,
    run_checkpoint,
    write_paired_results,
)
from examples.simBenchmarks.CoT.geometry_probe.run_paired_geometry_probe import (
    _build_store,
    _load_manifest,
    _materialize_or_reuse,
    _prediction_paths,
    benchmark_spec,
)
from examples.simBenchmarks.CoT.geometry_probe.visualization import (
    render_paired_sample_frame,
)


def default_source_dir(benchmark: str) -> Path:
    """Return the completed fixed-240 paired probe used for episode selection."""

    if benchmark not in ("libero", "robocasa"):
        raise ValueError(f"unknown benchmark: {benchmark!r}")
    return Path(f"/root/data/yxz/outputs/paired_geometry_probe_{benchmark}_240")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, choices=("libero", "robocasa"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint-a", type=Path)
    parser.add_argument("--checkpoint-b", type=Path)
    parser.add_argument("--label-a")
    parser.add_argument("--label-b")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--fallback-fps", type=float, default=2.5)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=("pyav", "decord", "torchvision_av"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe or "group"


def _video_info(path: Path) -> dict[str, float | int | str]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open cached video: {path}")
        return {
            "path": str(path),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def _render_episode_videos(
    *,
    episodes: Sequence[Any],
    plan: Sequence[Any],
    sample_paths: Sequence[Path],
    prediction_paths: dict[str, Sequence[Path]],
    labels: Sequence[str],
    summary: dict[str, Any],
    output_dir: Path,
    fallback_fps: float,
    codec: str,
) -> list[dict[str, Any]]:
    row_by_index = {int(row["sample_index"]): row for row in summary["samples"]}
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes):
        indices = [
            index
            for index, ref in enumerate(plan)
            if ref.suite == episode.suite and ref.episode_id == episode.episode_id
        ]
        samples = [load_materialized_sample(sample_paths[index]) for index in indices]
        timestamps = [sample["metadata"].get("timestamp", float("nan")) for sample in samples]
        fps = infer_video_fps(timestamps, fallback_fps=fallback_fps)
        path = video_dir / (
            f"{episode_index:02d}_{_safe_component(episode.suite)}_"
            f"episode_{int(episode.episode_id):06d}.mp4"
        )
        if path.is_file():
            info = _video_info(path)
            if int(info["frame_count"]) != len(indices):
                raise ValueError(
                    f"cached video frame mismatch for {path}: "
                    f"{info['frame_count']} vs {len(indices)}"
                )
        else:
            def frames():
                for index, sample in zip(indices, samples):
                    predictions = {
                        label: load_prediction(prediction_paths[label][index])
                        for label in labels
                    }
                    yield render_paired_sample_frame(
                        sample=sample,
                        predictions=predictions,
                        labels=labels,
                        metrics=row_by_index[index]["metrics"],
                    )

            info = write_video_frames(path, frames(), fps=fps, codec=codec)
        outputs.append(
            {
                **info,
                "suite": episode.suite,
                "episode_id": int(episode.episode_id),
                "episode_length": int(episode.episode_length),
                "sample_indices": indices,
                "source_frames": [int(plan[index].frame_index) for index in indices],
                "timestamps": timestamps,
            }
        )
        print(
            f"video {episode_index + 1}/{len(episodes)}: {path} "
            f"frames={len(indices)} fps={fps:.3f}",
            flush=True,
        )
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    spec = benchmark_spec(args.bench)
    source_dir = args.source_dir or default_source_dir(spec.name)
    source_manifest = args.manifest or source_dir / "manifest.json"
    output_dir = args.output_dir or source_dir / f"episode_stride{int(args.stride)}"
    dataset_root = args.dataset_root or spec.dataset_root
    checkpoint_a = args.checkpoint_a or spec.checkpoint_a
    checkpoint_b = args.checkpoint_b or spec.checkpoint_b
    label_a = args.label_a or spec.label_a
    label_b = args.label_b or spec.label_b
    if label_a == label_b:
        raise ValueError("paired labels must be distinct")

    fixed_plan = _load_manifest(source_manifest, spec)
    episodes = select_manifest_episodes(fixed_plan, groups=spec.groups)
    plan = build_episode_anchor_plan(
        episodes,
        horizon=spec.horizon,
        stride=args.stride,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "benchmark": spec.name,
        "source_manifest": str(source_manifest),
        "stride": int(args.stride),
        "horizon": int(spec.horizon),
        "episodes": [asdict(episode) for episode in episodes],
        "samples": [asdict(sample) for sample in plan],
    }
    (output_dir / "episode_plan.json").write_text(
        json.dumps(plan_payload, indent=2), encoding="utf-8"
    )
    counts = {group: sum(ref.suite == group for ref in plan) for group in spec.groups}
    print(json.dumps({"benchmark": spec.name, "anchors": len(plan), "counts": counts}, indent=2))
    if args.dry_run:
        print(f"validated episode plan: {output_dir / 'episode_plan.json'}")
        return

    for checkpoint in (checkpoint_a, checkpoint_b):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    store = _build_store(spec, dataset_root, args.video_backend)
    sample_paths = _materialize_or_reuse(store, plan, output_dir / "samples")
    predictions_dir = output_dir / "predictions"
    paths_a = _prediction_paths(predictions_dir, label_a, len(plan))
    if not all(path.exists() for path in paths_a):
        paths_a = run_checkpoint(
            checkpoint_a,
            label=label_a,
            sample_paths=sample_paths,
            output_dir=predictions_dir,
            device=args.device,
        )
    paths_b = _prediction_paths(predictions_dir, label_b, len(plan))
    if not all(path.exists() for path in paths_b):
        paths_b = run_checkpoint(
            checkpoint_b,
            label=label_b,
            sample_paths=sample_paths,
            output_dir=predictions_dir,
            device=args.device,
        )
    prediction_paths = {label_a: paths_a, label_b: paths_b}
    summary = write_paired_results(
        sample_paths=sample_paths,
        prediction_paths=prediction_paths,
        labels=(label_a, label_b),
        output_dir=output_dir / "results",
        image_size=224,
    )
    videos = _render_episode_videos(
        episodes=episodes,
        plan=plan,
        sample_paths=sample_paths,
        prediction_paths=prediction_paths,
        labels=(label_a, label_b),
        summary=summary,
        output_dir=output_dir,
        fallback_fps=args.fallback_fps,
        codec=args.codec,
    )
    manifest = {**plan_payload, "videos": videos}
    (output_dir / "episode_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    run_config = {
        "spec": asdict(spec),
        "dataset_root": str(dataset_root),
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "label_a": label_a,
        "label_b": label_b,
        "device": args.device,
        "video_backend": args.video_backend,
        "fallback_fps": args.fallback_fps,
        "codec": args.codec,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str), encoding="utf-8"
    )
    print(f"artifacts: {output_dir}")


if __name__ == "__main__":
    main()
