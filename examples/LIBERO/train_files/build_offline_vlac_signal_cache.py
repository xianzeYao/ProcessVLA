from __future__ import annotations
from starVLA.model.framework.signal_utils import (
    compute_vlac_curve_signal,
    normalize_curve,
    read_video_fps,
    save_signal_curve_npz,
    select_primary_video_key,
    signal_cache_curve_path,
)
from starVLA.dataloader.lerobot_datasets import get_vla_dataset

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_output_root(
    *,
    output_root: str | Path | None,
    vla_cfg,
    signal_name: str,
) -> Path:
    configured = output_root or vla_cfg.get("signal_cache_root", None)
    if configured:
        return Path(configured).expanduser().resolve()
    return (WORKSPACE_ROOT / "results" / "signal_cache" / str(vla_cfg.data_mix) / signal_name).resolve()


def resolve_instruction(dataset, trajectory_id: int) -> str:
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    instructions = dataset.get_language(
        trajectory_id, dataset.modality_keys["language"][0], 0
    )
    instruction = str(instructions[0]).strip() if instructions else ""
    if not instruction:
        raise ValueError(
            f"Empty instruction for trajectory {trajectory_id} in dataset {dataset.dataset_name}"
        )
    return instruction


def iter_datasets(mixture_dataset):
    for dataset in mixture_dataset.datasets:
        yield dataset


def build_dataset_records(dataset, video_subkey: str):
    records = []
    task_to_records = defaultdict(list)
    iterable = zip(dataset.trajectory_ids.tolist(),
                   dataset.trajectory_lengths.tolist())
    for trajectory_id, trajectory_length in tqdm(
        iterable,
        total=len(dataset.trajectory_ids),
        desc=f"{dataset.dataset_name} metadata",
    ):
        trajectory_id = int(trajectory_id)
        trajectory_length = int(trajectory_length)
        instruction = resolve_instruction(dataset, trajectory_id)
        video_path = Path(dataset.get_video_path(
            trajectory_id, video_subkey)).expanduser().resolve()
        record = {
            "trajectory_id": trajectory_id,
            "trajectory_length": trajectory_length,
            "instruction": instruction,
            "video_path": video_path,
        }
        records.append(record)
        task_to_records[instruction].append(record)
    return records, task_to_records


def resolve_reference_video_path(
    *,
    reference_mode: str,
    reference_seed: int,
    dataset_name: str,
    record: dict,
    task_to_records,
) -> Path | None:
    if reference_mode == "none":
        return None

    if reference_mode != "same_task_random":
        raise ValueError(f"Unsupported reference_mode={reference_mode!r}")

    candidates = [
        candidate
        for candidate in task_to_records.get(record["instruction"], [])
        if int(candidate["trajectory_id"]) != int(record["trajectory_id"])
    ]
    if not candidates:
        return None

    selector = random.Random(
        f"{reference_seed}|{dataset_name}|{record['instruction']}|{record['trajectory_id']}"
    )
    chosen = selector.choice(candidates)
    return Path(chosen["video_path"]).expanduser().resolve()


def build_offline_vlac_signal_cache(
    *,
    config_yaml: str | Path,
    output_root: str | Path | None = None,
    data_mix: str | None = None,
    signal_name: str = "vlac",
    signal_kind: str = "value",
    device: str | None = None,
    reference_mode: str = "same_task_random",
    reference_seed: int = 42,
    vlac_ref_num: int = 6,
    vlac_batch_num: int = 5,
    vlac_skip: int = 5,
    vlac_rich: bool = False,
    vlac_frame_skip: bool = False,
    vlac_think: bool = False,
    vlac_python: str | Path | None = None,
    overwrite: bool = False,
    max_trajectories: int | None = None,
) -> dict:
    cfg = OmegaConf.load(config_yaml)
    if data_mix is not None:
        cfg.datasets.vla_data.data_mix = data_mix

    vla_cfg = cfg.datasets.vla_data
    resolved_output_root = resolve_output_root(
        output_root=output_root,
        vla_cfg=vla_cfg,
        signal_name=signal_name,
    )
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)

    dataset = get_vla_dataset(data_cfg=vla_cfg, mode="train")

    total_saved = 0
    total_skipped = 0
    total_failed = 0
    raw_curve_lengths = []
    target_curve_lengths = []
    density_ratios = []

    print(f"[config] data_mix={vla_cfg.data_mix}")
    print(f"[config] output_root={resolved_output_root}")
    print(f"[config] signal_name={signal_name}")
    print(f"[config] signal_kind={signal_kind}")
    print(f"[config] device={resolved_device}")
    print(f"[config] reference_mode={reference_mode}")
    print(f"[config] reference_seed={reference_seed}")
    print(f"[config] vlac_ref_num={vlac_ref_num}")
    print(f"[config] vlac_batch_num={vlac_batch_num}")
    print(f"[config] vlac_skip={vlac_skip}")
    print(f"[config] vlac_rich={bool(vlac_rich)}")
    print(f"[config] vlac_frame_skip={bool(vlac_frame_skip)}")
    print(f"[config] vlac_think={bool(vlac_think)}")
    print(f"[config] vlac_python={vlac_python}")

    for single_dataset in iter_datasets(dataset):
        video_key = select_primary_video_key(
            single_dataset.modality_keys["video"])
        video_subkey = video_key.replace("video.", "")
        desc = f"{single_dataset.dataset_name}"
        records, task_to_records = build_dataset_records(
            single_dataset, video_subkey)

        for idx, record in enumerate(tqdm(records, total=len(records), desc=desc)):
            if max_trajectories is not None and idx >= max_trajectories:
                break

            trajectory_id = int(record["trajectory_id"])
            trajectory_length = int(record["trajectory_length"])
            curve_path = signal_cache_curve_path(
                cache_root=resolved_output_root,
                dataset_name=single_dataset.dataset_name,
                trajectory_id=trajectory_id,
                signal_name=signal_name,
            )
            if curve_path.exists() and not overwrite:
                total_skipped += 1
                continue

            try:
                instruction = str(record["instruction"])
                video_path = Path(record["video_path"]).expanduser().resolve()
                fps = read_video_fps(video_path)
                reference_video_path = resolve_reference_video_path(
                    reference_mode=reference_mode,
                    reference_seed=reference_seed,
                    dataset_name=single_dataset.dataset_name,
                    record=record,
                    task_to_records=task_to_records,
                )
                curve, raw_curve = compute_vlac_curve_signal(
                    video_path=video_path,
                    instruction=instruction,
                    fps=fps,
                    frame_count=trajectory_length,
                    reference_video_path=reference_video_path,
                    signal_kind=signal_kind,
                    ref_num=int(vlac_ref_num),
                    batch_num=int(vlac_batch_num),
                    skip=int(vlac_skip),
                    rich=bool(vlac_rich),
                    frame_skip=bool(vlac_frame_skip),
                    think=bool(vlac_think),
                    device=resolved_device,
                    return_raw_curve=True,
                    vlac_python=vlac_python,
                )
                curve = normalize_curve(curve, trajectory_length)
                raw_curve_length = int(np.asarray(
                    raw_curve).reshape(-1).shape[0])
                dense_ratio = (
                    float(raw_curve_length) / float(trajectory_length)
                    if trajectory_length > 0
                    else float("nan")
                )
                curve_path.parent.mkdir(parents=True, exist_ok=True)
                save_signal_curve_npz(
                    output_path=curve_path,
                    curve=curve,
                    signal_name=signal_name,
                    instruction=instruction,
                    video_path=video_path,
                    fps=fps,
                    status=f"offline_vlac_{signal_kind}_{reference_mode}",
                    extra_arrays={
                        "raw_curve_length": np.asarray([raw_curve_length], dtype=np.int32),
                        "target_curve_length": np.asarray([trajectory_length], dtype=np.int32),
                        "density_ratio_raw_over_target": np.asarray([dense_ratio], dtype=np.float32),
                    },
                )
                raw_curve_lengths.append(raw_curve_length)
                target_curve_lengths.append(trajectory_length)
                density_ratios.append(dense_ratio)
                print(
                    f"[dense] dataset={single_dataset.dataset_name} "
                    f"trajectory_id={trajectory_id} raw_curve_length={raw_curve_length} "
                    f"target_length={trajectory_length} raw_over_target={dense_ratio:.4f}"
                )
                total_saved += 1
            except Exception as exc:
                total_failed += 1
                print(
                    f"[failed] dataset={single_dataset.dataset_name} "
                    f"trajectory_id={trajectory_id} error={exc}"
                )

    summary = {
        "saved": total_saved,
        "skipped": total_skipped,
        "failed": total_failed,
        "output_root": str(resolved_output_root),
        "raw_curve_length_mean": float(np.mean(raw_curve_lengths)) if raw_curve_lengths else None,
        "target_length_mean": float(np.mean(target_curve_lengths)) if target_curve_lengths else None,
        "raw_over_target_mean": float(np.mean(density_ratios)) if density_ratios else None,
        "raw_over_target_min": float(np.min(density_ratios)) if density_ratios else None,
        "raw_over_target_max": float(np.max(density_ratios)) if density_ratios else None,
    }
    if raw_curve_lengths:
        print(
            "[dense-summary] "
            f"raw_curve_length_mean={summary['raw_curve_length_mean']:.2f} "
            f"target_length_mean={summary['target_length_mean']:.2f} "
            f"raw_over_target_mean={summary['raw_over_target_mean']:.4f} "
            f"raw_over_target_min={summary['raw_over_target_min']:.4f} "
            f"raw_over_target_max={summary['raw_over_target_max']:.4f}"
        )
    print(
        f"[done] saved={total_saved} skipped={total_skipped} failed={total_failed} "
        f"output_root={resolved_output_root}"
    )
    return summary


def main():
    config_yaml = WORKSPACE_ROOT / "starVLA" / "config" / \
        "training" / "starvla_cotrain_libero_signal.yaml"
    output_root = None
    data_mix = "libero_goal"
    signal_name = "vlac"
    signal_kind = "value"
    device = "cuda:0"
    reference_mode = "same_task_random"
    reference_seed = 42
    vlac_ref_num = 6
    vlac_batch_num = 5
    vlac_skip = 5
    vlac_rich = False
    vlac_frame_skip = False
    vlac_think = False
    vlac_python = None
    overwrite = False
    max_trajectories = None

    build_offline_vlac_signal_cache(
        config_yaml=config_yaml,
        output_root=output_root,
        data_mix=data_mix,
        signal_name=signal_name,
        signal_kind=signal_kind,
        device=device,
        reference_mode=reference_mode,
        reference_seed=reference_seed,
        vlac_ref_num=vlac_ref_num,
        vlac_batch_num=vlac_batch_num,
        vlac_skip=vlac_skip,
        vlac_rich=vlac_rich,
        vlac_frame_skip=vlac_frame_skip,
        vlac_think=vlac_think,
        vlac_python=vlac_python,
        overwrite=overwrite,
        max_trajectories=max_trajectories,
    )


if __name__ == "__main__":
    main()
