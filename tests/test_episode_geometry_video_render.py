import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.simBenchmarks.CoT.geometry_probe.episode_video import write_video_frames
from examples.simBenchmarks.CoT.geometry_probe.visualization import render_paired_sample_frame


def _synthetic_render_inputs():
    depth = np.linspace(0.5, 1.5, 64, dtype=np.float32).reshape(8, 8)
    uvd = np.asarray([[0.2, 0.3, 0.8], [0.7, 0.6, 1.2]], dtype=np.float32)
    sample = {
        "images": [np.full((8, 8, 3), 100, dtype=np.uint8)],
        "language": "move the object",
        "depth_current": depth,
        "depth_future": depth + 0.1,
        "uvd": uvd,
        "uvd_valid_mask": np.ones(2, dtype=np.bool_),
        "uvd_time": np.asarray([0.0, 1.0], dtype=np.float32),
        "metadata": {"suite": "suite", "episode_id": 3, "frame_index": 8},
    }
    predictions = {
        "v1": {"depth_current": depth + 0.05, "depth_future": depth + 0.15, "uvd": uvd + 0.01},
        "v2": {"depth_current": depth + 0.02, "depth_future": depth + 0.12, "uvd": uvd + 0.02},
    }
    metrics = {
        label: {
            "uvd_uv_ade_px": 1.0,
            "uvd_z_mae_m": 0.02,
            "depth_current_mae": 0.03,
            "depth_future_mae": 0.04,
        }
        for label in predictions
    }
    return sample, predictions, metrics


class VideoRenderingTest(unittest.TestCase):
    def test_renders_shared_paired_layout_to_rgb_frame(self):
        sample, predictions, metrics = _synthetic_render_inputs()
        frame = render_paired_sample_frame(
            sample=sample,
            predictions=predictions,
            labels=("v1", "v2"),
            metrics=metrics,
        )
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(frame.shape[0] % 2, 0)
        self.assertEqual(frame.shape[1] % 2, 0)

    def test_writes_readable_two_frame_mp4(self):
        import cv2

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.mp4"
            frames = [
                np.full((48, 64, 3), 20, dtype=np.uint8),
                np.full((48, 64, 3), 220, dtype=np.uint8),
            ]
            info = write_video_frames(path, frames, fps=2.5)
            capture = cv2.VideoCapture(str(path))
            decoded = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded += 1
            capture.release()
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(info["frame_count"], 2)
            self.assertEqual(decoded, 2)


if __name__ == "__main__":
    unittest.main()
