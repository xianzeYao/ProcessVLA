#!/usr/bin/env python3
"""Measure raw CoT losses on real LIBERO samples and recommend loss weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.cot_lerobot_datasets import LiberoCoTDataConfig
from starVLA.dataloader.gr00t_lerobot.cot_geometry import CoTLeRobotSingleDataset
from starVLA.model.framework.base_framework import build_framework


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--dataset", default="libero_goal_no_noops_1.0.0_lerobot")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.datasets.vla_data.num_workers = 0
    video_backend = str(cfg.datasets.vla_data.get("video_backend", "torchvision_av"))
    model = build_framework(cfg).cuda().eval()
    model.qwen_vl_interface.requires_grad_(False)
    data_cfg = cfg.datasets.vla_data
    data_config = LiberoCoTDataConfig()
    dataset = CoTLeRobotSingleDataset(
        dataset_path=Path(str(data_cfg.data_root_dir)) / args.dataset,
        modality_configs=data_config.modality_config(),
        transforms=data_config.transform(),
        embodiment_tag=data_config.embodiment_tag,
        video_backend=video_backend,
        data_cfg=data_cfg,
    )

    values: dict[str, list[float]] = {"action_loss": [], "depth_current_loss": [], "depth_future_loss": [], "uvd_loss": [], "geometry_loss": []}
    with torch.inference_mode():
        for sample_index in range(args.num_samples):
            torch.manual_seed(sample_index)
            sample = dataset[sample_index]
            output = model([sample])
            for key in values:
                values[key].append(float(output[key].detach().cpu()))

    means = {key: float(np.mean(items)) for key, items in values.items()}
    target_fraction = {
        "depth_current_loss": 0.04,
        "depth_future_loss": 0.04,
        "uvd_loss": 0.01,
        "geometry_loss": 0.01,
    }
    action_mean = max(means["action_loss"], 1e-8)
    recommendations = {
        key: target_fraction[key] * action_mean / max(means[key], 1e-8)
        for key in target_fraction
    }
    weighted = {key: recommendations[key] * means[key] for key in recommendations}
    result = {
        "num_samples": args.num_samples,
        "raw_losses": values,
        "mean_raw_losses": means,
        "target_aux_fraction_of_action": target_fraction,
        "recommended_weights": recommendations,
        "expected_weighted_aux_losses": weighted,
        "expected_total_aux_fraction": float(sum(weighted.values()) / action_mean),
        "notes": "Raw initialization-scale probe; confirm with a short pilot before treating weights as final scientific evidence.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
