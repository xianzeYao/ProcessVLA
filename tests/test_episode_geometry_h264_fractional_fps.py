import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from examples.simBenchmarks.CoT.geometry_probe.episode_video import write_video_frames


class H264FractionalFpsTest(unittest.TestCase):
    def test_accepts_floating_point_noise_around_integer_fps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.mp4"
            frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(2)]

            write_video_frames(
                path,
                frames,
                fps=20.000000000000004,
                codec="libx264",
            )

            with av.open(str(path)) as container:
                self.assertEqual(float(container.streams.video[0].average_rate), 20.0)


if __name__ == "__main__":
    unittest.main()
