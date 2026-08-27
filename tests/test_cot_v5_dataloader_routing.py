from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from starVLA.dataloader import build_dataloader


class _ToyDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {"index": index}

    def save_dataset_statistics(self, path):
        self.statistics_path = path


def _fake_dataset_module():
    dataset = _ToyDataset()
    return SimpleNamespace(
        get_vla_dataset=lambda **kwargs: dataset,
        collate_fn=lambda batch: batch,
    )


def test_build_dataloader_routes_only_the_explicit_robocasa_v5_selector(
    monkeypatch, tmp_path
):
    fake_v5 = _fake_dataset_module()
    monkeypatch.setitem(
        sys.modules,
        "starVLA.dataloader.robocasa_v5_lerobot_datasets",
        fake_v5,
    )
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "datasets": {
                "vla_data": {
                    "per_device_batch_size": 3,
                    "num_workers": 0,
                }
            },
        }
    )

    dataloader = build_dataloader(
        cfg,
        dataset_py="robocasa_v5_lerobot_datasets",
    )

    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.batch_size == 3
    assert dataloader.dataset is fake_v5.get_vla_dataset()
