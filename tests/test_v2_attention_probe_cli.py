from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from examples.simBenchmarks.CoT.attention_probe.run_v2_attention_probe import (
    _canonical_uvd,
    _optional_state,
    _render_report,
    build_parser,
)


def test_parser_exposes_offline_checkpoint_sample_and_output_contract():
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "/tmp/model.pt",
            "--sample-npz",
            "/tmp/sample.npz",
            "--output-dir",
            "/tmp/report",
        ]
    )

    assert args.checkpoint == Path("/tmp/model.pt")
    assert args.sample_npz == Path("/tmp/sample.npz")
    assert args.output_dir == Path("/tmp/report")
    assert args.device == "cuda:0"
    assert args.seed == 42


def test_optional_state_matches_existing_inference_contract_when_sample_has_none():
    framework = SimpleNamespace(
        config=SimpleNamespace(
            framework=SimpleNamespace(action_model={"state_dim": 8})
        )
    )
    condition = torch.zeros(1, 4, 16)

    assert _optional_state(framework, {}, condition) is None


def test_canonical_uvd_restores_time_major_two_hand_layout():
    flat = np.arange(36, dtype=np.float32).reshape(12, 3)

    canonical = _canonical_uvd(flat, points_per_hand=6, hand_count=2)

    assert canonical.shape == (6, 2, 3)
    np.testing.assert_array_equal(canonical[0, 0], [0, 1, 2])
    np.testing.assert_array_equal(canonical[0, 1], [3, 4, 5])
    np.testing.assert_array_equal(canonical[5, 1], [33, 34, 35])


def test_render_report_accepts_single_view_and_two_hand_uvd(tmp_path):
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    depth = np.arange(64, dtype=np.float32).reshape(8, 8)
    uvd = np.asarray(
        [
            [[0.2, 0.3, 0.5], [0.8, 0.3, 0.6]],
            [[0.3, 0.5, 0.7], [0.7, 0.5, 0.8]],
        ],
        dtype=np.float32,
    )
    sample = {
        "images": [image],
        "language": "move both hands",
        "depth_current": depth,
        "depth_future": depth + 1,
        "uvd": uvd,
    }
    predicted = {
        "depth_current": depth + 0.1,
        "depth_future": depth + 1.1,
        "uvd": uvd,
    }
    qwen_maps = {
        name: {"agentview": np.ones((4, 4), dtype=np.float32)}
        for name in ("depth_current", "depth_future", "uvd")
    }
    output = tmp_path / "report.png"

    _render_report(
        output,
        sample=sample,
        predicted=predicted,
        qwen_maps=qwen_maps,
        action_maps={"agentview": np.ones((4, 4), dtype=np.float32)},
        action_mass={"agentview": 0.5, "language": 0.5},
    )

    assert output.is_file()
    assert Image.open(output).size == (2880, 2560)
