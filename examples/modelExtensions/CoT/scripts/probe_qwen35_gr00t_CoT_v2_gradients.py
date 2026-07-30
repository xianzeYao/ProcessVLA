#!/usr/bin/env python3
"""Offline per-objective gradient probe for QwenGR00TCoTV2 checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.training.cot_gradient_probe import (
    extract_v2_objectives,
    measure_objective_gradients,
    select_probe_sample_indices,
    select_shared_parameter_groups,
    sha256_file,
    summarize_probe_batches,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen_tail_layers", type=int, default=2)
    parser.add_argument("--sample_indices", default=None)
    parser.add_argument("--device", default="cuda")
    return parser


def _set_random_state(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    raise ValueError(f"checkpoint must end in .pt or .safetensors, got {path}")


def _build_training_dataset(cfg):
    dataset_py = str(cfg.datasets.vla_data.dataset_py)
    module = importlib.import_module(f"starVLA.dataloader.{dataset_py}")
    if not hasattr(module, "get_vla_dataset"):
        raise ValueError(f"dataset module {dataset_py!r} has no get_vla_dataset")
    data_cfg = cfg.datasets.vla_data
    return module.get_vla_dataset(
        data_cfg=data_cfg,
        mode="train",
        balance_dataset_weights=bool(data_cfg.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(data_cfg.get("balance_trajectory_weights", False)),
    )


def _git_command(repo_root: Path, *args: str, text: bool = True):
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=text,
        capture_output=True,
        check=False,
    )


def _git_identity(repo_root: Path) -> dict[str, Any]:
    revision = _git_command(repo_root, "rev-parse", "HEAD")
    status = _git_command(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = _git_command(repo_root, "diff", "--binary", "HEAD", text=False)
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "status_porcelain": status.stdout if status.returncode == 0 else None,
        "tracked_diff_sha256": (
            hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
        ),
    }


def _source_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in paths}


def _provenance(
    *,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    model: torch.nn.Module,
    parameter_metadata: dict[str, dict[str, Any]],
    sample_indices: list[int],
    args: argparse.Namespace,
    config: Any,
) -> dict[str, Any]:
    checkpoint_stat = checkpoint_path.stat()
    config_bytes = config_path.read_bytes()
    first_parameter = next(model.parameters())
    repo_root = Path(__file__).resolve().parents[4]
    probe_module_path = Path(__file__).resolve()
    gradient_module_path = Path(sys.modules[extract_v2_objectives.__module__].__file__).resolve()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "git": _git_identity(repo_root),
        "source_sha256": _source_hashes([probe_module_path, gradient_module_path]),
        "effective_config": OmegaConf.to_container(config, resolve=True),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "model_parameter_dtype": str(first_parameter.dtype),
        "seed": int(args.seed),
        "num_batches": int(args.num_batches),
        "batch_size": int(args.batch_size),
        "qwen_tail_layers": int(args.qwen_tail_layers),
        "sample_indices": sample_indices,
        "parameter_groups": parameter_metadata,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("group                  aux/action     aux cosine")
    for group_name, group in summary["groups"].items():
        combined = group["combined_aux"]
        ratio = combined["aux_to_action_grad_norm"]["mean"]
        cosine = combined["cosine_with_action"]["mean"]
        print(f"{group_name:<22} {ratio:>10.4f} {cosine:>14.4f}")


def main() -> None:
    args = build_parser().parse_args()
    if args.num_batches < 1 or args.batch_size < 1:
        raise ValueError("num_batches and batch_size must be positive")
    config_path = Path(args.config_yaml).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    cfg = OmegaConf.load(config_path)
    if str(cfg.framework.name) != "QwenGR00TCoTV2":
        raise ValueError(f"gradient probe requires QwenGR00TCoTV2, got {cfg.framework.name}")
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.persistent_workers = False
    cfg.datasets.vla_data.per_device_batch_size = int(args.batch_size)
    cfg.trainer.pretrained_checkpoint = None

    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

    model = build_framework(cfg)
    model.load_state_dict(_load_state_dict(checkpoint_path), strict=True)
    model = TrainerUtils.freeze_backbones(
        model,
        freeze_modules=str(cfg.trainer.get("freeze_modules", "")),
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device)
    model.train()

    parameter_groups, parameter_metadata = select_shared_parameter_groups(
        model,
        qwen_tail_layers=int(args.qwen_tail_layers),
    )
    dataset = _build_training_dataset(cfg)
    sample_count = int(args.num_batches) * int(args.batch_size)
    selected_indices = select_probe_sample_indices(
        dataset_length=len(dataset),
        sample_count=sample_count,
        seed=int(args.seed),
        sample_indices=args.sample_indices,
    )

    records = []
    action_nonzero = False
    for batch_index in range(int(args.num_batches)):
        start = batch_index * int(args.batch_size)
        batch_indices = selected_indices[start : start + int(args.batch_size)]
        _set_random_state(int(args.seed) + batch_index)
        examples = [dataset[index] for index in batch_indices]
        _set_random_state(int(args.seed) + 100_000 + batch_index)
        autocast_enabled = device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = model.forward(examples)
        objectives = extract_v2_objectives(output, model)
        measurement = measure_objective_gradients(objectives, parameter_groups)
        action_nonzero = action_nonzero or (
            measurement["groups"]["shared_total"]["objectives"]["action"]["raw_grad_norm"] > 0.0
        )
        records.append(
            {
                "batch_index": batch_index,
                "sample_indices": batch_indices,
                **measurement,
            }
        )
        del objectives, output, measurement, examples
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not action_nonzero:
        raise RuntimeError("action gradient is zero across every measured batch and shared parameter")
    summary = summarize_probe_batches(records)
    result = {
        "provenance": _provenance(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            model=model,
            parameter_metadata=parameter_metadata,
            sample_indices=selected_indices,
            args=args,
            config=cfg,
        ),
        "batches": records,
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _print_summary(summary)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
