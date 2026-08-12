from __future__ import annotations

import unittest
from unittest import mock


class CotV3DataloaderDispatchTest(unittest.TestCase):
    def test_dispatches_v3_without_changing_v2_branch(self) -> None:
        import starVLA.dataloader as dataloader_module
        import starVLA.dataloader.cot_lerobot_datasets as v2_module
        import starVLA.dataloader.cot_v3_lerobot_datasets as v3_module

        class Config(dict):
            def __getattr__(self, key):
                return self[key]

        data_cfg = Config(
            per_device_batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        cfg = Config(
            output_dir="/tmp/output",
            datasets=Config(vla_data=data_cfg),
        )
        v2_dataset = type("V2Dataset", (), {"save_dataset_statistics": lambda *_: None})()
        v3_dataset = type("V3Dataset", (), {"save_dataset_statistics": lambda *_: None})()

        with mock.patch.object(v2_module, "get_vla_dataset", return_value=v2_dataset), mock.patch.object(
            v3_module, "get_vla_dataset", return_value=v3_dataset
        ), mock.patch.object(
            dataloader_module,
            "DataLoader",
            side_effect=lambda dataset, **kwargs: (dataset, kwargs),
        ), mock.patch.object(
            dataloader_module.dist, "is_initialized", return_value=True
        ), mock.patch.object(
            dataloader_module.dist, "get_rank", return_value=1
        ):
            v3_loader = dataloader_module.build_dataloader(
                cfg, dataset_py="cot_v3_lerobot_datasets"
            )
            v2_loader = dataloader_module.build_dataloader(
                cfg, dataset_py="cot_lerobot_datasets"
            )

        self.assertIs(v3_loader[0], v3_dataset)
        self.assertIs(v2_loader[0], v2_dataset)
        self.assertEqual(v3_loader[1]["batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
