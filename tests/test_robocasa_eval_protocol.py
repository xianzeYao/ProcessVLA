from pathlib import Path

import pytest

from examples.simBenchmarks.Robocasa_tabletop.eval_files.robocasa_eval_protocol import (
    build_manifest,
)


def _manifest(gpus: list[str]):
    return build_manifest(
        checkpoint="checkpoint.pt",
        gpus=gpus,
        num_episodes=50,
        base_port=18000,
        run_dir=Path("/tmp/robocasa-eval"),
        env_names=[f"suite/task_{index:02d}" for index in range(24)],
        save_video=False,
    )


def test_build_manifest_supports_eight_unique_gpus():
    manifest = _manifest([str(index) for index in range(8)])

    assert manifest["gpus"] == list(range(8))
    assert len(manifest["tasks"]) == 24
    assert [task["worker_id"] for task in manifest["tasks"]] == [
        index % 8 for index in range(24)
    ]
    worker_counts = {
        worker: sum(task["worker_id"] == worker for task in manifest["tasks"])
        for worker in range(8)
    }
    assert worker_counts == {worker: 3 for worker in range(8)}


@pytest.mark.parametrize(
    ("gpus", "message"),
    [
        ([], "between 1 and 24"),
        ([str(index) for index in range(25)], "between 1 and 24"),
        (["0", "1", "1"], "unique"),
    ],
)
def test_build_manifest_rejects_invalid_gpu_lists(gpus, message):
    with pytest.raises(ValueError, match=message):
        _manifest(gpus)
