from contextlib import contextmanager

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import starVLA.training.train_starvla_cot_v1 as cot_trainer_module
from starVLA.training.train_starvla_cot_v1 import CotV1Trainer


class _FourLossModel(nn.Module):
    lambda_action = 1.0
    lambda_depth_current = 0.14
    lambda_depth_future = 0.15
    lambda_uvd = 0.62

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, examples):
        action = self.scale.square()
        depth_current = 0.5 * self.scale.square()
        depth_future = 0.25 * self.scale.square()
        uvd = 0.125 * self.scale.square()
        total = (
            self.lambda_action * action
            + self.lambda_depth_current * depth_current
            + self.lambda_depth_future * depth_future
            + self.lambda_uvd * uvd
        )
        return {
            "action_loss": action,
            "depth_current_loss": depth_current,
            "depth_future_loss": depth_future,
            "uvd_loss": uvd,
            "total_loss": total,
        }


class _V2LossModel(_FourLossModel):
    lambda_uvd_relative = 0.1

    def forward(self, examples):
        action = self.scale.square()
        depth_current = 0.5 * self.scale.square()
        depth_future = 0.25 * self.scale.square()
        uvd_absolute = 0.125 * self.scale.square()
        uvd_relative = 0.25 * self.scale.square()
        uvd = uvd_absolute + self.lambda_uvd_relative * uvd_relative
        total = (
            self.lambda_action * action
            + self.lambda_depth_current * depth_current
            + self.lambda_depth_future * depth_future
            + self.lambda_uvd * uvd
        )
        return {
            "action_loss": action,
            "depth_current_loss": depth_current,
            "depth_future_loss": depth_future,
            "uvd_loss": uvd,
            "uvd_absolute_loss": uvd_absolute,
            "uvd_relative_loss": uvd_relative,
            "total_loss": total,
        }


class _Accelerator:
    sync_gradients = True
    num_processes = 1
    gradient_accumulation_steps = 1

    @contextmanager
    def accumulate(self, model):
        yield

    @staticmethod
    def backward(loss):
        loss.backward()

    @staticmethod
    def clip_grad_norm_(parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class _Scheduler:
    def step(self):
        return None


def test_cot_trainer_accepts_four_loss_objective_without_geometry_consistency():
    model = _FourLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = OmegaConf.create(
        {
            "datasets": {"vla_data": {"per_device_batch_size": 1}},
            "trainer": {
                "gradient_clipping": 1.0,
                "test_diagnostics": {"enabled": False, "log_module_gradients": False},
            }
        }
    )
    trainer = CotV1Trainer(config, model, [], optimizer, _Scheduler(), _Accelerator())

    metrics = trainer._train_step([])

    assert metrics["action_dit_loss"] == 1.0
    assert metrics["depth_current_loss"] == 0.5
    assert metrics["depth_future_loss"] == 0.25
    assert metrics["uvd_loss"] == 0.125
    assert all("geometry" not in key for key in metrics)
    assert metrics["train/grad_norm_pre_clip"] == pytest.approx(2.37)
    assert metrics["train/grad_clip_threshold"] == 1.0
    assert metrics["train/grad_clip_triggered"] == 1.0
    assert metrics["train/grad_clip_scale"] == pytest.approx(1.0 / 2.37)


def test_cot_trainer_logs_v2_uvd_absolute_and_relative_components():
    model = _V2LossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = OmegaConf.create(
        {
            "datasets": {"vla_data": {"per_device_batch_size": 1}},
            "trainer": {
                "gradient_clipping": 1.0,
                "test_diagnostics": {"enabled": False, "log_module_gradients": False},
            },
        }
    )
    trainer = CotV1Trainer(config, model, [], optimizer, _Scheduler(), _Accelerator())

    metrics = trainer._train_step([])

    assert metrics["uvd_absolute_loss"] == 0.125
    assert metrics["uvd_relative_loss"] == 0.25
    assert metrics["uvd_loss"] == pytest.approx(0.15)
    assert metrics["weighted_uvd_absolute_loss"] == pytest.approx(0.62 * 0.125)
    assert metrics["weighted_uvd_relative_loss"] == pytest.approx(0.62 * 0.1 * 0.25)


def test_disabled_diagnostics_never_install_module_gradient_hooks(monkeypatch):
    model = _FourLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = OmegaConf.create(
        {
            "datasets": {"vla_data": {"per_device_batch_size": 1}},
            "trainer": {
                "gradient_clipping": 1.0,
                "test_diagnostics": {
                    "enabled": False,
                    "log_module_gradients": True,
                    "module_gradient_interval": 1,
                },
            },
        }
    )
    trainer = CotV1Trainer(config, model, [], optimizer, _Scheduler(), _Accelerator())

    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled diagnostics must not install gradient hooks")

    monkeypatch.setattr(cot_trainer_module, "install_module_grad_norm_hooks", fail_if_called)

    metrics = trainer._train_step([])

    assert metrics["diagnostic/module_gradients_collected"] == 0.0
