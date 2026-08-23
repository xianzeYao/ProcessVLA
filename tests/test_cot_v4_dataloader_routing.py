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


def test_build_dataloader_routes_libero_v4_module(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "starVLA.dataloader.cot_v4_lerobot_datasets",
        _fake_dataset_module(),
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
        dataset_py="cot_v4_lerobot_datasets",
    )

    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.batch_size == 3


def test_build_dataloader_routes_robocasa_v4_module(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "starVLA.dataloader.robocasa_v4_lerobot_datasets",
        _fake_dataset_module(),
    )
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "datasets": {
                "vla_data": {
                    "per_device_batch_size": 2,
                    "num_workers": 0,
                }
            },
        }
    )

    dataloader = build_dataloader(
        cfg,
        dataset_py="robocasa_v4_lerobot_datasets",
    )

    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.batch_size == 2
