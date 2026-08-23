"""VLA trainer entry point for forward coarse-to-local QwenGR00TCoTV4."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.cot_test_diagnostics import (
    compute_token_utilization_metrics,
    write_run_manifest,
)
from starVLA.training.cot_v4_diagnostics import (
    compute_v4_geometry_metrics,
)
from starVLA.training.train_starvla import (
    accelerator,
    logger,
    normalize_dotlist_args,
    prepare_data,
    setup_directories,
    setup_optimizer_and_scheduler,
)
from starVLA.training.train_starvla_cot_v1 import CotV1Trainer
from starVLA.training.trainer_utils.config_tracker import wrap_config


class CotV4Trainer(CotV1Trainer):
    """Add V4 coarse objectives and diagnostics through generic base hooks."""

    def _extra_objective_metrics(
        self,
        output_dict,
    ) -> tuple[dict[str, float], float]:
        weight = float(self.model.lambda_uvd_coarse)
        coarse_loss = float(output_dict["uvd_coarse_loss"].item())
        metrics = {
            "uvd_coarse_loss": coarse_loss,
            "weighted_uvd_coarse_loss": weight * coarse_loss,
        }
        if "uvd_coarse_absolute_loss" in output_dict:
            absolute = float(
                output_dict["uvd_coarse_absolute_loss"].item()
            )
            metrics["uvd_coarse_absolute_loss"] = absolute
            metrics["weighted_uvd_coarse_absolute_loss"] = (
                weight * absolute
            )
        if "uvd_coarse_relative_loss" in output_dict:
            relative = float(
                output_dict["uvd_coarse_relative_loss"].item()
            )
            metrics["uvd_coarse_relative_loss"] = relative
            metrics["weighted_uvd_coarse_relative_loss"] = (
                weight
                * float(self.model.lambda_uvd_coarse_relative)
                * relative
            )
        return metrics, weight * coarse_loss

    def _extra_geometry_metrics(
        self,
        *,
        model,
        predictions,
        examples,
        diagnostic_config,
    ) -> dict[str, float]:
        metrics = compute_v4_geometry_metrics(
            predictions,
            examples,
            depth_scale=float(model.uvd_depth_scale),
            image_size=int(model.depth_output_size),
            hand_count=int(model.uvd_hand_count),
        )
        if (
            bool(
                diagnostic_config.get(
                    "log_token_utilization",
                    False,
                )
            )
            and "uvd_coarse_tokens" in predictions
        ):
            metrics.update(
                compute_token_utilization_metrics(
                    predictions["uvd_coarse_tokens"],
                    prefix="uvd_coarse",
                )
            )
        return metrics


def main(cfg) -> None:
    logger.info("QwenGR00TCoT V4 training :: Warming Up")
    cfg = wrap_config(cfg)
    output_dir = setup_directories(cfg=cfg)
    diagnostic_config = cfg.trainer.get("test_diagnostics", {})
    if (
        bool(diagnostic_config.get("enabled", False))
        and accelerator.is_main_process
    ):
        write_run_manifest(
            output_dir,
            cfg,
            Path(__file__).resolve().parents[2],
        )
    model = build_framework(cfg)
    dataloader = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(
        model=model,
        cfg=cfg,
    )
    trainer = CotV4Trainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    trainer.train()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.from_dotlist(
            normalize_dotlist_args(clipargs)
        ),
    )
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    main(cfg)
