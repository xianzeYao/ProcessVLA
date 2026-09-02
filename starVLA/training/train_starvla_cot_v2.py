"""VLA trainer entry point for QwenGR00TCoTV2."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.cot_test_diagnostics import write_run_manifest
from starVLA.training.cot_gradient_probe import (
    extract_v2_objectives,
    measure_paired_depth_token_gradients,
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


class CotV2Trainer(CotV1Trainer):
    """Reuse the CoT lifecycle and log optional V2 wrist-depth objectives."""

    def _forward_diagnostic_kwargs(self) -> dict:
        diagnostic_config = self._diagnostic_config()
        interval = max(
            int(diagnostic_config.get("objective_gradient_interval", 50)),
            1,
        )
        should_collect = (
            bool(diagnostic_config.get("enabled", False))
            and bool(diagnostic_config.get("log_objective_gradients", False))
            and self.accelerator.sync_gradients
            and (self.completed_steps + 1) % interval == 0
        )
        self._collect_online_depth_objective_gradients = should_collect
        return {"capture_depth_token_gradients": True} if should_collect else {}

    def _pre_backward_diagnostic_metrics(
        self,
        output_dict: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        if not bool(
            getattr(self, "_collect_online_depth_objective_gradients", False)
        ):
            return {"diagnostic/objective_gradients_collected": 0.0}

        start = time.perf_counter()
        depth_tokens = {
            "current": output_dict.pop("_probe_depth_current_tokens"),
            "future": output_dict.pop("_probe_depth_future_tokens"),
        }
        model = self.accelerator.unwrap_model(self.model)
        objectives = extract_v2_objectives(output_dict, model)
        metrics = measure_paired_depth_token_gradients(
            objectives,
            depth_tokens,
            distributed=True,
        )
        metrics["diagnostic/objective_gradients_collected"] = 1.0
        metrics["timing/objective_gradient_probe"] = time.perf_counter() - start
        return metrics

    def _extra_objective_metrics(
        self,
        output_dict,
    ) -> tuple[dict[str, float], float]:
        keys = (
            "wrist_depth_current_loss",
            "wrist_depth_future_loss",
        )
        present = [key in output_dict for key in keys]
        if not any(present):
            return {}, 0.0
        if not all(present):
            missing = [key for key in keys if key not in output_dict]
            raise KeyError(f"incomplete wrist-depth objective: missing {missing}")

        current = output_dict["wrist_depth_current_loss"].item()
        future = output_dict["wrist_depth_future_loss"].item()
        weighted_current = self.model.lambda_wrist_depth_current * current
        weighted_future = self.model.lambda_wrist_depth_future * future
        return (
            {
                "wrist_depth_current_loss": current,
                "wrist_depth_future_loss": future,
                "weighted_wrist_depth_current_loss": weighted_current,
                "weighted_wrist_depth_future_loss": weighted_future,
            },
            weighted_current + weighted_future,
        )


def main(cfg) -> None:
    logger.info("QwenGR00TCoT V2 training :: Warming Up")
    cfg = wrap_config(cfg)
    output_dir = setup_directories(cfg=cfg)
    diagnostic_config = cfg.trainer.get("test_diagnostics", {})
    if bool(diagnostic_config.get("enabled", False)) and accelerator.is_main_process:
        write_run_manifest(output_dir, cfg, Path(__file__).resolve().parents[2])
    model = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)
    trainer = CotV2Trainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=vla_train_dataloader,
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
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(clipargs)))
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    main(cfg)
