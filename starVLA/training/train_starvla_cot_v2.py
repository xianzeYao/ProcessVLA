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
        depth_tokens = {}
        wrist_depth_tokens = {}
        depth_queries = {}
        wrist_depth_queries = {}
        for branch in ("current", "future"):
            token_key = f"_probe_depth_{branch}_tokens"
            wrist_token_key = f"_probe_wrist_depth_{branch}_tokens"
            query_key = f"_probe_depth_{branch}_query"
            wrist_query_key = f"_probe_wrist_depth_{branch}_query"
            if token_key in output_dict:
                depth_tokens[branch] = output_dict.pop(token_key)
            if wrist_token_key in output_dict:
                wrist_depth_tokens[branch] = output_dict.pop(wrist_token_key)
            if query_key in output_dict:
                depth_queries[branch] = output_dict.pop(query_key)
            if wrist_query_key in output_dict:
                wrist_depth_queries[branch] = output_dict.pop(wrist_query_key)
        model = self.accelerator.unwrap_model(self.model)
        objectives = extract_v2_objectives(output_dict, model)
        metrics = measure_paired_depth_token_gradients(
            objectives,
            depth_tokens,
            wrist_depth_tokens=wrist_depth_tokens,
            distributed=True,
        )
        if depth_queries:
            metrics.update(
                measure_paired_depth_token_gradients(
                    objectives,
                    depth_queries,
                    wrist_depth_tokens=wrist_depth_queries,
                    distributed=True,
                    metric_root="depth_query_gradient",
                )
            )
        metrics["diagnostic/objective_gradients_collected"] = 1.0
        metrics["timing/objective_gradient_probe"] = time.perf_counter() - start
        return metrics

    def _extra_objective_metrics(
        self,
        output_dict,
    ) -> tuple[dict[str, float], float]:
        metrics: dict[str, float] = {}
        weighted_total = 0.0
        for branch in ("current", "future"):
            key = f"wrist_depth_{branch}_loss"
            if key not in output_dict:
                continue
            value = output_dict[key].item()
            weighted = float(getattr(self.model, f"lambda_wrist_depth_{branch}")) * value
            metrics[key] = value
            metrics[f"weighted_{key}"] = weighted
            weighted_total += weighted
        return metrics, weighted_total


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
