from __future__ import annotations

import json
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

from starVLA.model.framework.signal_utils import read_video_fps, run_vlac_web_trajectory_critic, select_primary_video_key
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_output_dir(
    *,
    output_dir: str | Path | None,
    vla_cfg,
    trajectory_seed: int,
    reference_seed: int,
) -> Path:
    if output_dir:
        base_dir = Path(output_dir).expanduser().resolve()
    else:
        base_dir = (WORKSPACE_ROOT / "results" / "test_offline_vlac").resolve()
    run_dir = (
        f"{vla_cfg.data_mix}"
        f"_trajectory_seed_{int(trajectory_seed)}"
        f"_reference_seed_{int(reference_seed)}"
    )
    return base_dir / run_dir


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


def build_global_records(mixture_dataset):
    global_records = []
    task_to_records_by_dataset = {}
    for single_dataset in iter_datasets(mixture_dataset):
        video_key = select_primary_video_key(
            single_dataset.modality_keys["video"])
        video_subkey = video_key.replace("video.", "")
        task_to_records = defaultdict(list)
        iterable = zip(single_dataset.trajectory_ids.tolist(),
                       single_dataset.trajectory_lengths.tolist())
        for trajectory_id, trajectory_length in tqdm(
            iterable,
            total=len(single_dataset.trajectory_ids),
            desc=f"{single_dataset.dataset_name} metadata",
        ):
            trajectory_id = int(trajectory_id)
            trajectory_length = int(trajectory_length)
            instruction = resolve_instruction(single_dataset, trajectory_id)
            video_path = Path(single_dataset.get_video_path(
                trajectory_id, video_subkey)).expanduser().resolve()
            record = {
                "dataset_name": single_dataset.dataset_name,
                "trajectory_id": trajectory_id,
                "trajectory_length": trajectory_length,
                "instruction": instruction,
                "video_path": video_path,
            }
            global_records.append(record)
            task_to_records[instruction].append(record)
        task_to_records_by_dataset[single_dataset.dataset_name] = task_to_records
    if not global_records:
        raise ValueError(
            "No trajectories found in the resolved mixture dataset.")
    return global_records, task_to_records_by_dataset


def select_target_record(records, trajectory_seed: int) -> dict:
    selector = random.Random(str(trajectory_seed))
    return dict(selector.choice(records))


def select_reference_record(target_record: dict, task_to_records_by_dataset, reference_seed: int) -> dict:
    candidates = [
        candidate
        for candidate in task_to_records_by_dataset[target_record["dataset_name"]].get(target_record["instruction"], [])
        if int(candidate["trajectory_id"]) != int(target_record["trajectory_id"])
    ]
    if not candidates:
        raise ValueError(
            "No same-dataset same-task reference candidates found after excluding the selected trajectory."
        )
    selector = random.Random(
        f"{reference_seed}|{target_record['dataset_name']}|{target_record['instruction']}|{target_record['trajectory_id']}"
    )
    return dict(selector.choice(candidates))


def save_result_npz(
    output_path: Path,
    *,
    target_record: dict,
    reference_record: dict,
    result_path: str | None,
    fps: float,
    value_list,
    critic_list,
    done_list,
    trajectory_seed: int,
    reference_seed: int,
    signal_kind: str,
) -> None:
    np.savez(
        output_path,
        dataset_name=np.asarray([target_record["dataset_name"]], dtype=object),
        instruction=np.asarray([target_record["instruction"]], dtype=object),
        trajectory_id=np.asarray(
            [int(target_record["trajectory_id"])], dtype=np.int32),
        reference_trajectory_id=np.asarray(
            [int(reference_record["trajectory_id"])], dtype=np.int32),
        video_path=np.asarray(
            [str(target_record["video_path"])], dtype=object),
        reference_video_path=np.asarray(
            [str(reference_record["video_path"])], dtype=object),
        result_video_path=np.asarray(
            [str(result_path) if result_path is not None else ""], dtype=object),
        fps=np.asarray([fps], dtype=np.float32),
        value_curve=np.asarray(value_list, dtype=np.float32),
        critic_curve=np.asarray(critic_list, dtype=np.float32),
        done_curve=np.asarray(
            [] if done_list is None else done_list, dtype=np.float32),
        trajectory_seed=np.asarray([trajectory_seed], dtype=np.int32),
        reference_seed=np.asarray([reference_seed], dtype=np.int32),
        signal_kind=np.asarray([signal_kind], dtype=object),
    )


def save_metadata_json(
    output_path: Path,
    *,
    target_record: dict,
    reference_record: dict,
    result_path: str | None,
    fps: float,
    data_mix: str | None,
    trajectory_seed: int,
    reference_seed: int,
    signal_kind: str,
    vlac_ref_num: int,
    vlac_batch_num: int,
    vlac_skip: int,
    vlac_rich: bool,
    vlac_frame_skip: bool,
    vlac_think: bool,
    done_flag: bool,
    done_threshold: float,
) -> None:
    payload = {
        "data_mix": data_mix,
        "trajectory_seed": trajectory_seed,
        "reference_seed": reference_seed,
        "dataset_name": target_record["dataset_name"],
        "instruction": target_record["instruction"],
        "trajectory_id": int(target_record["trajectory_id"]),
        "reference_trajectory_id": int(reference_record["trajectory_id"]),
        "video_path": str(target_record["video_path"]),
        "reference_video_path": str(reference_record["video_path"]),
        "result_video_path": result_path,
        "fps": fps,
        "vlac": {
            "signal_kind": signal_kind,
            "ref_num": vlac_ref_num,
            "batch_num": vlac_batch_num,
            "skip": vlac_skip,
            "rich": bool(vlac_rich),
            "frame_skip": bool(vlac_frame_skip),
            "think": bool(vlac_think),
            "done_flag": bool(done_flag),
            "done_threshold": float(done_threshold),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def build_vla_cfg(
    *,
    data_root_dir: str | Path,
    data_mix: str,
    video_backend: str = "torchvision_av",
):
    return OmegaConf.create(
        {
            "data_root_dir": str(Path(data_root_dir).expanduser().resolve()),
            "data_mix": str(data_mix),
            "video_backend": str(video_backend),
        }
    )


def test_offline_vlac(
    *,
    data_root_dir: str | Path,
    data_mix: str,
    output_dir: str | Path | None = None,
    trajectory_seed: int = 42,
    reference_seed: int = 42,
    device: str | None = None,
    signal_kind: str = "value",
    vlac_ref_num: int = 6,
    vlac_batch_num: int = 5,
    vlac_skip: int = 5,
    vlac_rich: bool = False,
    vlac_frame_skip: bool = False,
    vlac_think: bool = False,
    vlac_python: str | Path | None = None,
    done_flag: bool = False,
    done_threshold: float = 0.9,
) -> dict:
    vla_cfg = build_vla_cfg(
        data_root_dir=data_root_dir,
        data_mix=data_mix,
    )
    resolved_output_dir = resolve_output_dir(
        output_dir=output_dir,
        vla_cfg=vla_cfg,
        trajectory_seed=trajectory_seed,
        reference_seed=reference_seed,
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)

    dataset = get_vla_dataset(data_cfg=vla_cfg, mode="train")
    records, task_to_records_by_dataset = build_global_records(dataset)

    target_record = select_target_record(records, trajectory_seed)
    reference_record = select_reference_record(
        target_record,
        task_to_records_by_dataset,
        reference_seed,
    )

    print("[selection] target")
    print(f"  dataset_name: {target_record['dataset_name']}")
    print(f"  episode_id: {int(target_record['trajectory_id'])}")
    print(f"  instruction: {target_record['instruction']}")
    print("[selection] reference")
    print(f"  dataset_name: {reference_record['dataset_name']}")
    print(f"  episode_id: {int(reference_record['trajectory_id'])}")
    print(f"  instruction: {reference_record['instruction']}")

    fps = read_video_fps(target_record["video_path"])
    need_done = bool(done_flag or signal_kind == "done")
    result_path, value_list, critic_list, done_list = run_vlac_web_trajectory_critic(
        video_path=target_record["video_path"],
        instruction=target_record["instruction"],
        fps=float(fps),
        reference_video_path=reference_record["video_path"],
        ref_num=int(vlac_ref_num),
        batch_num=int(vlac_batch_num),
        skip=int(vlac_skip),
        rich=bool(vlac_rich),
        frame_skip=bool(vlac_frame_skip),
        think=bool(vlac_think),
        device=resolved_device,
        done_flag=need_done,
        in_context_done=True,
        done_threshold=float(done_threshold),
        output_path=str(resolved_output_dir),
        video_output=True,
        vlac_python=vlac_python,
    )

    stem = (
        f"{target_record['dataset_name']}"
        f"_traj{int(target_record['trajectory_id'])}"
        f"_ref{int(reference_record['trajectory_id'])}"
        f"_ts{int(trajectory_seed)}"
        f"_rs{int(reference_seed)}"
    )
    curves_npz = resolved_output_dir / f"{stem}_vlac_curves.npz"
    metadata_json = resolved_output_dir / f"{stem}_selection.json"
    save_result_npz(
        curves_npz,
        target_record=target_record,
        reference_record=reference_record,
        result_path=result_path,
        fps=fps,
        value_list=value_list,
        critic_list=critic_list,
        done_list=done_list,
        trajectory_seed=trajectory_seed,
        reference_seed=reference_seed,
        signal_kind=signal_kind,
    )
    save_metadata_json(
        metadata_json,
        target_record=target_record,
        reference_record=reference_record,
        result_path=result_path,
        fps=fps,
        data_mix=data_mix,
        trajectory_seed=trajectory_seed,
        reference_seed=reference_seed,
        signal_kind=signal_kind,
        vlac_ref_num=vlac_ref_num,
        vlac_batch_num=vlac_batch_num,
        vlac_skip=vlac_skip,
        vlac_rich=vlac_rich,
        vlac_frame_skip=vlac_frame_skip,
        vlac_think=vlac_think,
        done_flag=done_flag,
        done_threshold=done_threshold,
    )

    summary = {
        "data_mix": str(vla_cfg.data_mix),
        "dataset_name": target_record["dataset_name"],
        "instruction": target_record["instruction"],
        "trajectory_seed": trajectory_seed,
        "reference_seed": reference_seed,
        "trajectory_id": int(target_record["trajectory_id"]),
        "reference_trajectory_id": int(reference_record["trajectory_id"]),
        "main_video": str(target_record["video_path"]),
        "reference_video": str(reference_record["video_path"]),
        "result_video": None if result_path is None else str(result_path),
        "curves_npz": str(curves_npz),
        "metadata_json": str(metadata_json),
        "value_curve_length": int(len(value_list)),
        "critic_curve_length": int(len(critic_list)),
        "done_curve_length": int(0 if done_list is None else len(done_list)),
    }

    print("[OK] Offline VLAC test complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


def main():
    data_root_dir = "/data/yxz/dataset/libero_lerobot"
    output_dir = "test_vlac4train"
    data_mix = "libero_goal"
    trajectory_seed = 42
    reference_seed = 10
    device = "cuda:0"
    signal_kind = "value"
    vlac_ref_num = 6
    vlac_batch_num = 5
    vlac_skip = 5
    vlac_rich = False
    vlac_frame_skip = False
    vlac_think = False
    vlac_python = "/data/yxz/conda/envs/VLAC/bin/python"
    done_flag = False
    done_threshold = 0.9

    test_offline_vlac(
        data_root_dir=data_root_dir,
        output_dir=output_dir,
        data_mix=data_mix,
        trajectory_seed=trajectory_seed,
        reference_seed=reference_seed,
        device=device,
        signal_kind=signal_kind,
        vlac_ref_num=vlac_ref_num,
        vlac_batch_num=vlac_batch_num,
        vlac_skip=vlac_skip,
        vlac_rich=vlac_rich,
        vlac_frame_skip=vlac_frame_skip,
        vlac_think=vlac_think,
        vlac_python=vlac_python,
        done_flag=done_flag,
        done_threshold=done_threshold,
    )


if __name__ == "__main__":
    main()
