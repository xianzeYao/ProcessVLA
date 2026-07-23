#!/usr/bin/env python3
"""Rerender every language segment fully covered by a CALVIN frame shard.

Structured public mirrors split frame files at arbitrary boundaries. This
entry point is useful for a verified shard: it rerenders complete language
segments without pretending that the shard contains complete source episodes.
For the full dataset, the regular episode rerender entry point remains valid.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

from .build_calvin_native_rgbd_lerobot import build_segments, load_annotations, load_source_episodes
FPS = 30.0

from .rerender_calvin_episodes import (
    make_env,
    render_episode,
    resolve_config_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-split", choices=("training", "validation"), default="training")
    parser.add_argument("--config-split", choices=("auto", "training", "validation"), default="auto")
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--start-segment", type=int, default=0)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.dataset_root / args.source_split
    boundaries = load_source_episodes(source_dir)
    annotations = load_annotations(source_dir)
    frame_ids = {
        int(path.stem.split("_", 1)[1])
        for path in source_dir.glob("episode_*.npz")
    }
    segments = build_segments(annotations, frame_ids, boundaries)
    segments = segments[args.start_segment :]
    if args.max_segments > 0:
        segments = segments[: args.max_segments]
    if not segments:
        raise RuntimeError("no complete language segments are covered by the raw shard")

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_root}; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    config_dir = resolve_config_dir(args.dataset_root, args.config_split)
    env = make_env(config_dir)
    summaries = []
    try:
        for output_index, segment in enumerate(segments):
            summary = render_episode(
                env,
                source_dir,
                args.output_root,
                output_index,
                int(segment["source_start"]),
                int(segment["source_end"]),
                args.fps,
            )
            summary.update(
                {
                    "source_segment_index": int(segment["source_segment_index"]),
                    "instruction": segment["instruction"],
                    "task": segment["task"],
                }
            )
            summaries.append(summary)
            print(
                f"[segment {output_index + 1}/{len(segments)}] "
                f"{segment['source_start']}:{segment['source_end']}",
                flush=True,
            )
    finally:
        env.close()

    payload = {
        "status": "success",
        "source_dataset": str(args.dataset_root),
        "source_split": args.source_split,
        "config_dir": str(config_dir),
        "num_segments": len(summaries),
        "segments": summaries,
        "output_root": str(args.output_root),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
