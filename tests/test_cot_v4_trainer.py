from contextlib import contextmanager

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from starVLA.training.train_starvla_cot_v4 import CotV4Trainer


class _V4LossModel(nn.Module):
    lambda_action = 1.0
    lambda_depth_current = 0.14
    lambda_depth_future = 0.15
    lambda_uvd = 0.62
    lambda_uvd_relative = 0.1
    lambda_uvd_coarse = 0.2
    lambda_uvd_coarse_relative = 0.1

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, examples):
        square = self.scale.square()
        action = square
        current = 0.5 * square
        future = 0.25 * square
        local_absolute = 0.125 * square
        local_relative = 0.25 * square
        local = local_absolute + self.lambda_uvd_relative * local_relative
        coarse_absolute = 0.375 * square
        coarse_relative = 0.5 * square
        coarse = (
            coarse_absolute
            + self.lambda_uvd_coarse_relative * coarse_relative
        )
        total = (
            action
            + self.lambda_depth_current * current
            + self.lambda_depth_future * future
            + self.lambda_uvd * local
            + self.lambda_uvd_coarse * coarse
        )
        return {
            "action_loss": action,
            "depth_current_loss": current,
            "depth_future_loss": future,
            "uvd_loss": local,
            "uvd_absolute_loss": local_absolute,
            "uvd_relative_loss": local_relative,
            "uvd_coarse_loss": coarse,
            "uvd_coarse_absolute_loss": coarse_absolute,
            "uvd_coarse_relative_loss": coarse_relative,
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
    @staticmethod
    def step():
        return None


def test_v4_trainer_adds_coarse_loss_to_weighted_aux_balance():
    model = _V4LossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = OmegaConf.create(
        {
            "datasets": {"vla_data": {"per_device_batch_size": 1}},
            "trainer": {
                "gradient_clipping": 1.0,
                "test_diagnostics": {
                    "enabled": False,
                    "log_module_gradients": False,
                },
            },
        }
    )
    trainer = CotV4Trainer(
        config,
        model,
        [],
        optimizer,
        _Scheduler(),
        _Accelerator(),
    )

    metrics = trainer._train_step([])

    assert metrics["uvd_coarse_loss"] == pytest.approx(0.425)
    assert metrics["weighted_uvd_coarse_loss"] == pytest.approx(0.2 * 0.425)
    assert metrics["uvd_coarse_absolute_loss"] == pytest.approx(0.375)
    assert metrics["uvd_coarse_relative_loss"] == pytest.approx(0.5)
    assert metrics["weighted_aux_loss"] == pytest.approx(
        0.14 * 0.5 + 0.15 * 0.25 + 0.62 * 0.15 + 0.2 * 0.425
    )
