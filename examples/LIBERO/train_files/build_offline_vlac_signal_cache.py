from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.signal_utils import (
    compute_vlac_curve_signal,
    normalize_curve,
    read_video_fps,
    save_signal_curve_npz,
    select_primary_video_key,
    signal_cache_curve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build offline VLAC signal caches aligned to the training dataset defined by a YAML config."
    )
    parser.add_argument("--config-yaml", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--data-mix", type=str, default=None)
    parser.add_argument("--signal-name", type=str, default="vlac")
    parser.add_argument(
        "--signal-kind",
        type=str,
        default="value",
        choices=["value", "critic", "done"],
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reference-video-path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_output_root(args, vla_cfg) -> Path:
    configured = args.output_root or vla_cfg.get("signal_cache_root", None)
    if configured:
        return Path(configured).expanduser().resolve()
    return (WORKSPACE_ROOT / "results" / "signal_cache" / str(vla_cfg.data_mix) / args.signal_name).resolve()


def resolve_instruction(dataset, trajectory_id: int) -> str:
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    instructions = dataset.get_language(trajectory_id, dataset.modality_keys["language"][0], 0)
    instruction = str(instructions[0]).strip() if instructions else ""
    if not instruction:
        raise ValueError(f"Empty instruction for trajectory {trajectory_id} in dataset {dataset.dataset_name}")
    return instruction


def iter_datasets(mixture_dataset):
    for dataset in mixture_dataset.datasets:
        yield dataset


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config_yaml)
    if args.data_mix is not None:
        cfg.datasets.vla_data.data_mix = args.data_mix

    vla_cfg = cfg.datasets.vla_data
    output_root = resolve_output_root(args, vla_cfg)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset = get_vla_dataset(data_cfg=vla_cfg, mode="train")

    total_saved = 0
    total_skipped = 0
    total_failed = 0

    print(f"[config] data_mix={vla_cfg.data_mix}")
    print(f"[config] output_root={output_root}")
    print(f"[config] signal_name={args.signal_name}")
    print(f"[config] signal_kind={args.signal_kind}")
    print(f"[config] device={device}")

    for single_dataset in iter_datasets(dataset):
        video_key = select_primary_video_key(single_dataset.modality_keys["video"])
        video_subkey = video_key.replace("video.", "")
        iterable = zip(single_dataset.trajectory_ids.tolist(), single_dataset.trajectory_lengths.tolist())
        desc = f"{single_dataset.dataset_name}"

        for idx, (trajectory_id, trajectory_length) in enumerate(
            tqdm(iterable, total=len(single_dataset.trajectory_ids), desc=desc)
        ):
            if args.max_trajectories is not None and idx >= args.max_trajectories:
                break

            trajectory_id = int(trajectory_id)
            trajectory_length = int(trajectory_length)
            curve_path = signal_cache_curve_path(
                cache_root=output_root,
                dataset_name=single_dataset.dataset_name,
                trajectory_id=trajectory_id,
                signal_name=args.signal_name,
            )
            if curve_path.exists() and not args.overwrite:
                total_skipped += 1
                continue

            try:
                instruction = resolve_instruction(single_dataset, trajectory_id)
                video_path = single_dataset.get_video_path(trajectory_id, video_subkey)
                fps = read_video_fps(video_path)
                curve = compute_vlac_curve_signal(
                    video_path=video_path,
                    instruction=instruction,
                    fps=fps,
                    frame_count=trajectory_length,
                    reference_video_path=args.reference_video_path,
                    signal_kind=args.signal_kind,
                    device=device,
                )
                curve = normalize_curve(curve, trajectory_length)
                curve_path.parent.mkdir(parents=True, exist_ok=True)
                save_signal_curve_npz(
                    output_path=curve_path,
                    curve=curve,
                    signal_name=args.signal_name,
                    instruction=instruction,
                    video_path=video_path,
                    fps=fps,
                    status=f"offline_vlac_{args.signal_kind}",
                )
                total_saved += 1
            except Exception as exc:
                total_failed += 1
                print(
                    f"[failed] dataset={single_dataset.dataset_name} "
                    f"trajectory_id={trajectory_id} error={exc}"
                )

    print(
        f"[done] saved={total_saved} skipped={total_skipped} failed={total_failed} "
        f"output_root={output_root}"
    )


if __name__ == "__main__":
    main()
