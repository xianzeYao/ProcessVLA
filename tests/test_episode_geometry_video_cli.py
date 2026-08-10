import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from examples.simBenchmarks.CoT.geometry_probe import run_episode_geometry_videos as runner

from examples.simBenchmarks.CoT.geometry_probe.run_episode_geometry_videos import (
    _parse_args,
    default_source_dir,
)


class EpisodeVideoCliTest(unittest.TestCase):
    def test_defaults_to_stride_eight_and_previous_fixed_probe(self):
        libero = _parse_args(["--bench", "libero"])
        robocasa = _parse_args(["--bench", "robocasa"])

        self.assertEqual(libero.stride, 8)
        self.assertEqual(robocasa.stride, 8)
        self.assertEqual(
            default_source_dir("libero"),
            Path("/root/data/yxz/outputs/paired_geometry_probe_libero_240"),
        )
        self.assertEqual(
            default_source_dir("robocasa"),
            Path("/root/data/yxz/outputs/paired_geometry_probe_robocasa_240"),
        )

    def test_expected_video_group_counts_are_four_and_twenty_four(self):
        from examples.simBenchmarks.CoT.geometry_probe.run_paired_geometry_probe import (
            benchmark_spec,
        )

        self.assertEqual(len(benchmark_spec("libero").groups), 4)
        self.assertEqual(len(benchmark_spec("robocasa").groups), 24)

    def test_runner_builds_one_context_and_reuses_it_for_every_episode_frame(self):
        episodes = [SimpleNamespace(suite="suite", episode_id=7, episode_length=10)]
        plan = [
            SimpleNamespace(suite="suite", episode_id=7, frame_index=0),
            SimpleNamespace(suite="suite", episode_id=7, frame_index=1),
        ]
        samples = [
            {"metadata": {"timestamp": 0.0}},
            {"metadata": {"timestamp": 0.05}},
        ]
        prediction_values = {
            "v1_0.npz": {"label": "v1", "frame": 0},
            "v1_1.npz": {"label": "v1", "frame": 1},
            "v2_0.npz": {"label": "v2", "frame": 0},
            "v2_1.npz": {"label": "v2", "frame": 1},
        }
        context = object()
        rendered = []

        def load_prediction(path):
            return prediction_values[Path(path).name]

        def render_frame(**kwargs):
            rendered.append(kwargs)
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def write_frames(path, frames, *, fps, codec):
            arrays = list(frames)
            return {
                "path": str(path),
                "fps": fps,
                "frame_count": len(arrays),
                "width": 8,
                "height": 8,
                "codec": codec,
            }

        summary = {
            "samples": [
                {"sample_index": 0, "metrics": {"frame": 0}},
                {"sample_index": 1, "metrics": {"frame": 1}},
            ]
        }
        prediction_paths = {
            "v1": [Path("v1_0.npz"), Path("v1_1.npz")],
            "v2": [Path("v2_0.npz"), Path("v2_1.npz")],
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runner, "load_materialized_sample", side_effect=samples
        ), mock.patch.object(
            runner, "load_prediction", side_effect=load_prediction
        ), mock.patch.object(
            runner, "build_episode_curve_context", return_value=context
        ) as build_context, mock.patch.object(
            runner, "render_paired_episode_frame", side_effect=render_frame
        ), mock.patch.object(
            runner, "write_video_frames", side_effect=write_frames
        ):
            outputs = runner._render_episode_videos(
                episodes=episodes,
                plan=plan,
                sample_paths=[Path("sample_0.npz"), Path("sample_1.npz")],
                prediction_paths=prediction_paths,
                labels=("v1", "v2"),
                summary=summary,
                output_dir=Path(temporary),
                fallback_fps=20.0,
                codec="libx264",
            )

        build_context.assert_called_once_with(
            samples,
            {
                "v1": [prediction_values["v1_0.npz"], prediction_values["v1_1.npz"]],
                "v2": [prediction_values["v2_0.npz"], prediction_values["v2_1.npz"]],
            },
            labels=("v1", "v2"),
        )
        self.assertEqual([call["episode_index"] for call in rendered], [0, 1])
        self.assertTrue(all(call["episode_context"] is context for call in rendered))
        self.assertEqual(outputs[0]["frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
