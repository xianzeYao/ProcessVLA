import subprocess
import sys
import textwrap

from starVLA.dataloader.gr00t_lerobot.registry import _find_registry_dirs


def test_nested_benchmark_registry_directories_are_discovered():
    registry_paths = {path.as_posix() for path in _find_registry_dirs()}

    assert any(
        path.endswith("examples/simBenchmarks/LIBERO/train_files/data_registry")
        for path in registry_paths
    )


def test_calvin_mixtures_use_calvin_training_config():
    code = textwrap.dedent(
        """
        from starVLA.dataloader.calvin_lerobot_datasets import CalvinDataConfig
        from starVLA.dataloader.gr00t_lerobot.registry import (
            DATASET_NAMED_MIXTURES,
            ROBOT_TYPE_CONFIG_MAP,
        )

        for name in ("calvin_task_ABC_D", "calvin_task_ABCD_D"):
            assert DATASET_NAMED_MIXTURES[name] == [
                (name, 1.0, "calvin_franka")
            ]
        assert isinstance(ROBOT_TYPE_CONFIG_MAP["calvin_franka"], CalvinDataConfig)
        """
    )

    subprocess.run([sys.executable, "-c", code], check=True)

