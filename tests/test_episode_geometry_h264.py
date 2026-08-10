import tempfile
import unittest
from pathlib import Path

import av
import cv2
import numpy as np

from examples.simBenchmarks.CoT.geometry_probe.episode_video import write_video_frames


class H264VideoWriterTest(unittest.TestCase):
    def test_libx264_writes_yuv420p_with_exact_frames_and_fps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.mp4"
            frames = [
                np.full((48, 64, 3), value, dtype=np.uint8)
                for value in (20, 100, 220)
            ]

            info = write_video_frames(path, frames, fps=20.0, codec="libx264")

            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                self.assertEqual(stream.codec_context.name, "h264")
                self.assertEqual(stream.codec_context.format.name, "yuv420p")
                self.assertEqual(float(stream.average_rate), 20.0)
            capture = cv2.VideoCapture(str(path))
            decoded = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded += 1
            capture.release()
            self.assertEqual(decoded, 3)
            self.assertEqual(info["frame_count"], 3)


if __name__ == "__main__":
    unittest.main()
