import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.simBenchmarks.CoT.geometry_probe.episode_video import (
    build_episode_anchor_plan,
    infer_video_fps,
    select_manifest_episodes,
)
from examples.simBenchmarks.CoT.geometry_probe.probe_utils import SampleRef


class EpisodePlanTest(unittest.TestCase):
    def test_selects_first_manifest_episode_per_group_in_requested_order(self):
        manifest = [
            SampleRef("b", 21, 41, 7),
            SampleRef("a", 11, 25, 3),
            SampleRef("a", 12, 29, 4),
            SampleRef("b", 22, 49, 5),
        ]

        selected = select_manifest_episodes(manifest, groups=("a", "b"))

        self.assertEqual(
            [(ref.suite, ref.episode_id, ref.episode_length) for ref in selected],
            [("a", 11, 25), ("b", 21, 41)],
        )

    def test_rejects_missing_group(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            select_manifest_episodes(
                [SampleRef("a", 11, 25, 3)],
                groups=("a", "b"),
            )

    def test_expands_complete_stride_eight_anchors(self):
        selected = select_manifest_episodes(
            [SampleRef("a", 11, 25, 3)],
            groups=("a",),
        )

        plan = build_episode_anchor_plan(selected, horizon=8, stride=8)

        self.assertEqual([ref.frame_index for ref in plan], [0, 8, 16])
        self.assertTrue(all(ref.frame_index + 8 < ref.episode_length for ref in plan))

    def test_rejects_episode_without_complete_anchor(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            build_episode_anchor_plan(
                select_manifest_episodes(
                    [SampleRef("a", 1, 8, 0)], groups=("a",)
                ),
                horizon=8,
                stride=8,
            )


class VideoTimingTest(unittest.TestCase):
    def test_infers_fps_from_anchor_timestamps(self):
        self.assertAlmostEqual(infer_video_fps([0.0, 0.4, 0.8]), 2.5)

    def test_ignores_invalid_and_nonpositive_timestamp_steps(self):
        self.assertAlmostEqual(
            infer_video_fps([0.0, 0.4, 0.4, float("nan"), 0.8]),
            2.5,
        )

    def test_uses_fallback_without_enough_timestamps(self):
        self.assertEqual(infer_video_fps([1.0], fallback_fps=3.0), 3.0)


if __name__ == "__main__":
    unittest.main()
