"""Decode DeepThinkVLA LIBERO reasoning into local per-episode sidecars.

The source dataset stores Gemma/PaliGemma token ids.  This script performs
that one-time source-tokenizer conversion; training later reads UTF-8 text and
lets the Qwen3.5 processor tokenize it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


SUITES = (
    ("libero_10_no_noops_1.0.0_lerobot", 0),
    ("libero_goal_no_noops_1.0.0_lerobot", 379),
    ("libero_object_no_noops_1.0.0_lerobot", 807),
    ("libero_spatial_no_noops_1.0.0_lerobot", 1261),
)


def load_lengths(path: Path) -> dict[int, int]:
    return {
        int(row["episode_index"]): int(row["length"])
        for row in (json.loads(line) for line in (path / "meta/episodes.jsonl").read_text().splitlines())
    }


def trim_padding(ids: list[int], pad_token_id: int) -> list[int]:
    end = len(ids)
    while end > 0 and ids[end - 1] == pad_token_id:
        end -= 1
    return ids[:end]


def decode_episode(path: Path, tokenizer) -> list[str]:
    table = pq.read_table(path, columns=["reasoning", "episode_index", "frame_index"])
    episode_ids = {int(value) for value in table["episode_index"].to_pylist()}
    if len(episode_ids) != 1:
        raise ValueError(f"Expected one episode per parquet, got {episode_ids} in {path}")
    frames = table["frame_index"].to_pylist()
    expected_frames = list(range(len(frames)))
    if frames != expected_frames:
        raise ValueError(f"Non-contiguous frame indices in {path}")
    ids = [trim_padding(list(row), int(tokenizer.pad_token_id)) for row in table["reasoning"].to_pylist()]
    return tokenizer.batch_decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot-root", type=Path, required=True)
    parser.add_argument("--rerender-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    source_files = {
        int(path.stem.split("_")[-1]): path
        for path in sorted((args.cot_root / "data").glob("chunk-*/*.parquet"))
    }
    if len(source_files) != 1693:
        raise ValueError(f"Expected 1693 source episodes, found {len(source_files)}")

    manifest = {
        "source_dataset": str(args.cot_root),
        "source_tokenizer": str(args.tokenizer),
        "alignment": "suite + local episode_index + frame_index",
        "source_pad_token_id": int(tokenizer.pad_token_id),
    }
    for dataset_name, offset in SUITES:
        dataset_path = args.rerender_root / dataset_name
        lengths = load_lengths(dataset_path)
        output_root = dataset_path / "meta/embodied_cot"
        output_root.mkdir(parents=True, exist_ok=True)
        for local_episode, length in sorted(lengths.items()):
            output_path = output_root / f"episode_{local_episode:06d}.json"
            if output_path.exists() and not args.overwrite:
                continue
            source_episode = offset + local_episode
            texts = decode_episode(source_files[source_episode], tokenizer)
            if len(texts) != length:
                raise ValueError(
                    f"Length mismatch for {dataset_name} local episode {local_episode}: "
                    f"local={length}, cot={len(texts)}, source={source_episode}"
                )
            output_path.write_text(
                json.dumps(
                    {
                        "episode_index": local_episode,
                        "source_episode_index": source_episode,
                        "cot_text": texts,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        (dataset_path / "meta/embodied_cot_manifest.json").write_text(
            json.dumps({**manifest, "dataset_name": dataset_name, "episode_offset": offset}, indent=2),
            encoding="utf-8",
        )
        print(f"prepared {dataset_name}: {len(lengths)} episodes")


if __name__ == "__main__":
    main()
