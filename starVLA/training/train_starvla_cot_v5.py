"""VLA trainer entry point for RoboCasa QwenGR00TCoTV5."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.cot_test_diagnostics import write_run_manifest
from starVLA.training.train_starvla import (
    accelerator,
    logger,
    normalize_dotlist_args,
    prepare_data,
    setup_directories,
    setup_optimizer_and_scheduler,
)
from starVLA.training.train_starvla_cot_v2 import CotV2Trainer
from starVLA.training.trainer_utils.config_tracker import wrap_config


class CotV5Trainer(CotV2Trainer):
    """Reuse the V2 lifecycle with the independent V5 framework."""


def main(cfg) -> None:
    logger.info("QwenGR00TCoT V5 training :: Warming Up")
    cfg = wrap_config(cfg)
    output_dir = setup_directories(cfg=cfg)
    diagnostic_config = cfg.trainer.get("test_diagnostics", {})
    if bool(diagnostic_config.get("enabled", False)) and accelerator.is_main_process:
        write_run_manifest(output_dir, cfg, Path(__file__).resolve().parents[2])
    model = build_framework(cfg)
    vla_train_dataloader = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)
    trainer = CotV5Trainer(
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
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.from_dotlist(normalize_dotlist_args(clipargs)),
    )
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    main(cfg)
