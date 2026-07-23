#!/usr/bin/env python3
"""Merge frame-sharded CALVIN training directories into one raw dataset.

The public CALVIN shard mirrors split the per-frame episode_*.npz files at
arbitrary frame boundaries. Each shard carries the same global metadata, so
rerendering must happen after the frame files are merged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def find_training_dirs(input_root: Path) -> list[Path]:
    candidates = []
    for path in sorted(input_root.rglob("training")):
        if path.is_dir() and (path / "ep_start_end_ids.npy").exists():
            candidates.append(path)
    if (input_root / "training" / "ep_start_end_ids.npy").exists():
        candidates.insert(0, input_root / "training")
    unique = list(dict.fromkeys(path.resolve() for path in candidates))
    if not unique:
        raise FileNotFoundError(f"no CALVIN training directories under {input_root}")
    return unique


def _copy_metadata(source: Path, destination: Path) -> None:
    for name in ("scene_info.npy", "ep_lens.npy", "ep_start_end_ids.npy"):
        source_path = source / name
        if not source_path.exists():
            raise FileNotFoundError(f"missing {source_path}")
        shutil.copy2(source_path, destination / name)
    source_lang = source / "lang_annotations"
    destination_lang = destination / "lang_annotations"
    if not (source_lang / "auto_lang_ann.npy").exists():
        raise FileNotFoundError(f"missing {source_lang / 'auto_lang_ann.npy'}")
    shutil.copytree(source_lang, destination_lang, dirs_exist_ok=True)
    source_hydra = source / ".hydra"
    if source_hydra.exists():
        shutil.copytree(source_hydra, destination / ".hydra", dirs_exist_ok=True)


def merge_training_shards(input_root: Path, output_root: Path, overwrite: bool = False, mode: str = "hardlink") -> dict:
    sources = find_training_dirs(input_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)
    destination = output_root / "training"
    destination.mkdir(parents=True, exist_ok=True)
    _copy_metadata(sources[0], destination)

    frame_count = 0
    frame_ids: set[int] = set()
    for source in sources:
        for source_frame in sorted(source.glob("episode_*.npz")):
            try:
                frame_id = int(source_frame.stem.split("_", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid CALVIN frame filename: {source_frame}") from exc
            if frame_id in frame_ids:
                raise ValueError(f"duplicate frame id {frame_id} across shards")
            frame_ids.add(frame_id)
            destination_frame = destination / source_frame.name
            if mode == "hardlink":
                os.link(source_frame, destination_frame)
            else:
                shutil.copy2(source_frame, destination_frame)
            frame_count += 1

    if not frame_ids:
        raise FileNotFoundError(f"no episode_*.npz files under {input_root}")
    manifest = {
        "status": "success",
        "source_root": str(input_root),
        "source_training_dirs": [str(path) for path in sources],
        "output_root": str(output_root),
        "frame_count": frame_count,
        "min_frame_id": min(frame_ids),
        "max_frame_id": max(frame_ids),
        "link_mode": mode,
        "metadata_source": str(sources[0]),
    }
    (output_root / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    print(json.dumps(merge_training_shards(args.input_root, args.output_root, args.overwrite, args.mode), indent=2))


if __name__ == "__main__":
    main()
