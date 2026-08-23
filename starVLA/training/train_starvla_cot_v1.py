"""VLA trainer for QwenGR00TCoT with auxiliary geometry losses."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.train_starvla import (
    VLATrainer,
    accelerator,
    logger,
    normalize_dotlist_args,
    prepare_data,
    setup_directories,
    setup_optimizer_and_scheduler,
)
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.cot_test_diagnostics import (
    collect_batch_valid_ratios,
    collect_module_grad_norms,
    compute_decoder_reliance_metrics,
    compute_gradient_clipping_metrics,
    install_module_grad_norm_hooks,
    compute_geometry_metrics,
    compute_token_utilization_metrics,
    save_prediction_bundle,
    resolve_post_step_gradient_norm,
    write_run_manifest,
)


class CotV1Trainer(VLATrainer):
    """Reuse baseline lifecycle and replace only the loss aggregation step."""

    def _diagnostic_config(self):
        return self.config.trainer.get("test_diagnostics", {})

    @staticmethod
    def _uvd_track_count(model) -> int:
        """Resolve V3 landmarks before the V1/V2 hand-count compatibility field."""

        return int(getattr(model, "landmark_count", getattr(model, "uvd_hand_count", 1)))

    def _extra_objective_metrics(
        self,
        output_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, float], float]:
        return {}, 0.0

    def _extra_geometry_metrics(
        self,
        *,
        model,
        predictions: dict,
        examples: list[dict],
        diagnostic_config,
    ) -> dict[str, float]:
        return {}


    def _log_metrics(self, metrics):
        super()._log_metrics(metrics)
        should_write = self.completed_steps % self.config.trainer.logging_frequency == 0
        if not self.accelerator.is_main_process or not should_write:
            return
        record = {"step": int(self.completed_steps)}
        for key, value in metrics.items():
            if isinstance(value, (bool, int, float)):
                record[key] = value
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        except Exception:
            logger.exception("Failed to append train_metrics.jsonl; training will continue")
    def _train_step(self, batch_vla, batch_vlm=None):
        diagnostic_config = self._diagnostic_config()
        if bool(diagnostic_config.get("enabled", False)) and not hasattr(self, "_diagnostic_examples"):
            sample_count = int(diagnostic_config.get("num_samples", 4))
            self._diagnostic_examples = [dict(example) for example in batch_vla[:sample_count]]
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                total_loss = output_dict["total_loss"]
            module_grad_metrics = {}
            grad_diagnostics_time = 0.0
            hook_state = None
            if (
                bool(diagnostic_config.get("enabled", False))
                and bool(diagnostic_config.get("log_module_gradients", False))
            ):
                gradient_interval = max(
                    int(diagnostic_config.get("module_gradient_interval", 100)), 1
                )
                should_collect_gradients = self.accelerator.sync_gradients and (
                    (self.completed_steps + 1) % gradient_interval == 0
                )
                if should_collect_gradients:
                    hook_state = install_module_grad_norm_hooks(self.model)
            self.accelerator.backward(total_loss)
            if hook_state is not None:
                grad_diagnostics_start = time.perf_counter()
                module_grad_metrics = collect_module_grad_norms(self.model, hook_state=hook_state)
                grad_diagnostics_time = time.perf_counter() - grad_diagnostics_start
            grad_norm = None
            if self.config.trainer.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
            self.optimizer.step()
            if (
                grad_norm is None
                and self.config.trainer.gradient_clipping is not None
                and self.accelerator.sync_gradients
            ):
                grad_norm = resolve_post_step_gradient_norm(None, self.model)
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        metrics = {
            "total_loss": total_loss.item(),
            "action_dit_loss": output_dict["action_loss"].item(),
            "weighted_action_loss": self.model.lambda_action * output_dict["action_loss"].item(),
            "depth_current_loss": output_dict["depth_current_loss"].item(),
            "depth_future_loss": output_dict["depth_future_loss"].item(),
            "uvd_loss": output_dict["uvd_loss"].item(),
            "weighted_depth_current_loss": self.model.lambda_depth_current * output_dict["depth_current_loss"].item(),
            "weighted_depth_future_loss": self.model.lambda_depth_future * output_dict["depth_future_loss"].item(),
            "weighted_uvd_loss": self.model.lambda_uvd * output_dict["uvd_loss"].item(),
        }
        if "uvd_absolute_loss" in output_dict:
            metrics["uvd_absolute_loss"] = output_dict["uvd_absolute_loss"].item()
            metrics["weighted_uvd_absolute_loss"] = (
                self.model.lambda_uvd * metrics["uvd_absolute_loss"]
            )
        if "uvd_relative_loss" in output_dict:
            metrics["uvd_relative_loss"] = output_dict["uvd_relative_loss"].item()
            metrics["weighted_uvd_relative_loss"] = (
                self.model.lambda_uvd
                * float(getattr(self.model, "lambda_uvd_relative", 0.0))
                * metrics["uvd_relative_loss"]
            )
        if "uvd_temporal_loss" in output_dict:
            metrics["uvd_temporal_loss"] = output_dict["uvd_temporal_loss"].item()
            metrics["weighted_uvd_temporal_loss"] = (
                self.model.lambda_uvd
                * float(getattr(self.model, "lambda_uvd_temporal", 0.0))
                * metrics["uvd_temporal_loss"]
            )
        if "uvd_shape_loss" in output_dict:
            metrics["uvd_shape_loss"] = output_dict["uvd_shape_loss"].item()
            metrics["weighted_uvd_shape_loss"] = (
                self.model.lambda_uvd
                * float(getattr(self.model, "lambda_uvd_shape", 0.0))
                * metrics["uvd_shape_loss"]
            )
        extra_metrics, weighted_extra_loss = self._extra_objective_metrics(
            output_dict
        )
        metrics.update(extra_metrics)
        weighted_aux_loss = (
            metrics["weighted_depth_current_loss"]
            + metrics["weighted_depth_future_loss"]
            + metrics["weighted_uvd_loss"]
            + weighted_extra_loss
        )
        action_scale = max(abs(metrics["weighted_action_loss"]), 1.0e-12)
        total_weighted_scale = action_scale + weighted_aux_loss
        metrics["weighted_aux_loss"] = weighted_aux_loss
        metrics["loss_balance/aux_to_action"] = weighted_aux_loss / action_scale
        metrics["loss_balance/action_fraction"] = action_scale / max(total_weighted_scale, 1.0e-12)
        metrics["timing/grad_diagnostics"] = grad_diagnostics_time
        metrics["diagnostic/module_gradients_collected"] = float(hook_state is not None)
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
            metrics.update(
                compute_gradient_clipping_metrics(
                    pre_clip_norm=grad_norm,
                    threshold=float(self.config.trainer.gradient_clipping),
                )
            )
        metrics.update(module_grad_metrics)
        if bool(diagnostic_config.get("enabled", False)):
            metrics.update(collect_batch_valid_ratios(batch_vla))
        return metrics

    def eval_action_model(self, step_metrics=None):
        eval_start = time.perf_counter()
        step_metrics = super().eval_action_model(step_metrics)
        action_eval_time = time.perf_counter() - eval_start
        step_metrics["timing/action_eval"] = action_eval_time
        diagnostic_config = self._diagnostic_config()
        if not bool(diagnostic_config.get("enabled", False)) or not hasattr(self, "_diagnostic_examples"):
            step_metrics["timing/eval_total"] = time.perf_counter() - eval_start
            return step_metrics
        interval = int(diagnostic_config.get("interval", self.config.trainer.eval_interval))
        if self.completed_steps % interval != 0:
            step_metrics["timing/eval_total"] = time.perf_counter() - eval_start
            return step_metrics
        geometry_diagnostics_start = time.perf_counter()
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()
        log_token_utilization = bool(
            diagnostic_config.get("log_token_utilization", False)
        )
        log_decoder_reliance = bool(
            diagnostic_config.get("log_decoder_reliance", False)
        )
        if hasattr(model, "predict_geometry_diagnostics") and (
            log_token_utilization or log_decoder_reliance
        ):
            predictions = model.predict_geometry_diagnostics(
                self._diagnostic_examples,
                include_decoder_interventions=log_decoder_reliance,
            )
        else:
            predictions = model.predict_geometry(self._diagnostic_examples)
        if was_training:
            model.train()
        geometry_metrics = compute_geometry_metrics(
            predictions,
            self._diagnostic_examples,
            depth_scale=float(model.uvd_depth_scale),
            image_size=int(model.depth_output_size),
            uvd_hand_count=self._uvd_track_count(model),
            uvd_order=str(getattr(model, "uvd_token_order", "hand_major")),
            include_uvd_time_metrics=bool(
                diagnostic_config.get("log_uvd_time_metrics", False)
            ),
        )
        if log_token_utilization:
            geometry_metrics.update(
                compute_token_utilization_metrics(
                    predictions["depth_current_tokens"],
                    prefix="depth_current",
                    attention_weights=predictions["depth_current_pool_weights"],
                )
            )
            geometry_metrics.update(
                compute_token_utilization_metrics(
                    predictions["depth_future_tokens"],
                    prefix="depth_future",
                    attention_weights=predictions["depth_future_pool_weights"],
                )
            )
            geometry_metrics.update(
                compute_token_utilization_metrics(
                    predictions["uvd_tokens"],
                    prefix="uvd",
                )
            )
        if log_decoder_reliance:
            geometry_metrics.update(
                compute_decoder_reliance_metrics(
                    predictions,
                    self._diagnostic_examples,
                    depth_scale=float(model.uvd_depth_scale),
                )
            )
        geometry_metrics.update(
            self._extra_geometry_metrics(
                model=model,
                predictions=predictions,
                examples=self._diagnostic_examples,
                diagnostic_config=diagnostic_config,
            )
        )
        step_metrics.update({f"diagnostic/{key}": value for key, value in geometry_metrics.items()})
        if self.accelerator.is_main_process:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            record = {"step": int(self.completed_steps), **geometry_metrics}
            with (output_dir / "test_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            save_interval = int(diagnostic_config.get("prediction_save_interval", 1000))
            if bool(diagnostic_config.get("save_predictions", True)) and self.completed_steps % save_interval == 0:
                try:
                    save_prediction_bundle(
                        output_dir,
                        self.completed_steps,
                        predictions,
                        self._diagnostic_examples,
                        uvd_hand_count=self._uvd_track_count(model),
                        uvd_order=str(getattr(model, "uvd_token_order", "hand_major")),
                    )
                    step_metrics["diagnostic/prediction_save_failed"] = 0.0
                except Exception:
                    # Prediction bundles are observability artifacts, not training state.
                    step_metrics["diagnostic/prediction_save_failed"] = 1.0
                    logger.exception(
                        f"Prediction diagnostics save failed at step {self.completed_steps}; training will continue"
                    )
        step_metrics["timing/geometry_diagnostics"] = time.perf_counter() - geometry_diagnostics_start
        step_metrics["timing/eval_total"] = time.perf_counter() - eval_start
        return step_metrics


def main(cfg) -> None:
    logger.info("QwenGR00TCoT V1 training :: Warming Up")
    cfg = wrap_config(cfg)
    output_dir = setup_directories(cfg=cfg)
    diagnostic_config = cfg.trainer.get("test_diagnostics", {})
    if bool(diagnostic_config.get("enabled", False)) and accelerator.is_main_process:
        write_run_manifest(output_dir, cfg, Path(__file__).resolve().parents[2])
    model = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)
    trainer = CotV1Trainer(
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
