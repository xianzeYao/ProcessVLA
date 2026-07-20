"""Training entry point for standalone text/UVD condition ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.dataloader.text_uvd_lerobot_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.train_starvla import (
    VLATrainer,
    accelerator,
    logger,
    normalize_dotlist_args,
    setup_directories,
    setup_optimizer_and_scheduler,
)
from starVLA.training.trainer_utils.config_tracker import wrap_config


def load_config(config_yaml: str, overrides: list[str]):
    config_path = Path(config_yaml).resolve()
    cfg = OmegaConf.load(config_path)
    base_config = cfg.get("_base_config")
    if base_config:
        base_path = (config_path.parent / str(base_config)).resolve()
        cfg = OmegaConf.merge(OmegaConf.load(base_path), cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(overrides)))
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = str(config_path)
    return cfg


def prepare_text_uvd_data(cfg, output_dir):
    data_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(
        data_cfg=data_cfg,
        balance_dataset_weights=data_cfg.get("balance_dataset_weights", False),
        balance_trajectory_weights=data_cfg.get("balance_trajectory_weights", False),
    )
    if accelerator.is_main_process:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from torch.utils.data import DataLoader

    num_workers = int(data_cfg.get("num_workers", 4))
    loader_kwargs = {
        "batch_size": data_cfg.per_device_batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=bool(data_cfg.get("persistent_workers", True)),
            prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        )
    return DataLoader(dataset, **loader_kwargs)


def main(cfg) -> None:
    logger.info("Standalone QwenGR00T text/UVD ablation training")
    cfg = wrap_config(cfg)
    output_dir = setup_directories(cfg=cfg)
    model = build_framework(cfg)
    dataloader = prepare_text_uvd_data(cfg=cfg, output_dir=output_dir)
    # Each batch is a Python list of heterogeneous example dictionaries.
    # Do not let Accelerate split it as if it were a tensor batch.
    accelerator.dataloader_config.dispatch_batches = False
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)
    trainer = VLATrainer(
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
    main(load_config(args.config_yaml, clipargs))
