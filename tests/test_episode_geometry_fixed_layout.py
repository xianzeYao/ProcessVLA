import unittest

import matplotlib.pyplot as plt
import numpy as np

from examples.simBenchmarks.CoT.geometry_probe.episode_curves import (
    build_episode_curve_context,
)
from examples.simBenchmarks.CoT.geometry_probe.visualization import (
    _build_paired_episode_figure,
    _build_paired_sample_figure,
    render_paired_episode_frame,
)
from tests.test_episode_geometry_video_render import _synthetic_render_inputs


def _two_hand(sample, predictions):
    sample = dict(sample)
    sample["uvd"] = np.stack((sample["uvd"], sample["uvd"] + 0.05), axis=1)
    sample["uvd_valid_mask"] = np.ones(sample["uvd"].shape[:2], dtype=np.bool_)
    predictions = {
        label: {
            **prediction,
            "uvd": np.stack((prediction["uvd"], prediction["uvd"] + 0.05), axis=1),
        }
        for label, prediction in predictions.items()
    }
    return sample, predictions


def _episode_fixture():
    sample0, predictions0, metrics = _synthetic_render_inputs()
    sample0 = dict(sample0)
    sample0["metadata"] = {**sample0["metadata"], "frame_index": 0, "timestamp": 0.0}
    sample1 = {
        **sample0,
        "uvd": sample0["uvd"] + np.asarray([0.1, 0.05, 0.2], dtype=np.float32),
        "depth_current": sample0["depth_current"] + 0.2,
        "depth_future": sample0["depth_future"] + 0.2,
        "metadata": {**sample0["metadata"], "frame_index": 1, "timestamp": 0.05},
    }
    predictions1 = {
        label: {
            "uvd": prediction["uvd"] + np.asarray([0.1, 0.05, 0.2], dtype=np.float32),
            "depth_current": prediction["depth_current"] + 0.2,
            "depth_future": prediction["depth_future"] + 0.2,
        }
        for label, prediction in predictions0.items()
    }
    samples = [sample0, sample1]
    predictions = {
        label: [predictions0[label], predictions1[label]] for label in ("v1", "v2")
    }
    context = build_episode_curve_context(samples, predictions, labels=("v1", "v2"))
    return samples, predictions, metrics, context


class FixedLayoutTest(unittest.TestCase):
    def test_static_panel_bounds_do_not_depend_on_hand_count(self):
        one_sample, one_predictions, metrics = _synthetic_render_inputs()
        two_sample, two_predictions = _two_hand(one_sample, one_predictions)
        figures = [
            _build_paired_sample_figure(
                sample=sample,
                predictions=predictions,
                labels=("v1", "v2"),
                metrics=metrics,
                figsize=(16.5, 9.75),
                dpi=80,
            )
            for sample, predictions in (
                (one_sample, one_predictions),
                (two_sample, two_predictions),
            )
        ]
        try:
            for figure in figures:
                figure.canvas.draw()
            bounds = [
                np.asarray([axis.get_position().bounds for axis in figure.axes[:15]])
                for figure in figures
            ]
            np.testing.assert_allclose(bounds[0], bounds[1], rtol=0.0, atol=1e-10)
        finally:
            for figure in figures:
                plt.close(figure)

    def test_episode_curves_and_limits_stay_fixed_while_marker_moves(self):
        samples, predictions, metrics, context = _episode_fixture()
        figures = [
            _build_paired_episode_figure(
                sample=samples[index],
                predictions={label: predictions[label][index] for label in predictions},
                labels=("v1", "v2"),
                metrics=metrics,
                episode_context=context,
                episode_index=index,
                figsize=(16.5, 9.75),
                dpi=80,
            )
            for index in (0, 1)
        ]
        try:
            for figure in figures:
                figure.canvas.draw()
            depth_axes = [figure.axes[11] for figure in figures]
            uv_axes = [figure.axes[13] for figure in figures]
            for axes in (depth_axes, uv_axes):
                self.assertEqual(axes[0].get_xlim(), axes[1].get_xlim())
                self.assertEqual(axes[0].get_ylim(), axes[1].get_ylim())
                for label in ("GT episode", "v1 episode", "v2 episode"):
                    lines = [
                        next(line for line in axis.lines if line.get_label() == label)
                        for axis in axes
                    ]
                    np.testing.assert_allclose(lines[0].get_xdata(), lines[1].get_xdata())
                    np.testing.assert_allclose(lines[0].get_ydata(), lines[1].get_ydata())
                markers = [
                    next(line for line in axis.lines if line.get_label() == "current GT")
                    for axis in axes
                ]
                self.assertNotEqual(tuple(markers[0].get_xdata()), tuple(markers[1].get_xdata()))
            np.testing.assert_allclose(
                next(
                    line for line in depth_axes[0].lines if line.get_label() == "GT episode"
                ).get_xdata(),
                [0.0, 0.05],
            )
        finally:
            for figure in figures:
                plt.close(figure)

    def test_video_entry_point_returns_fixed_size_rgb(self):
        samples, predictions, metrics, context = _episode_fixture()
        frame = render_paired_episode_frame(
            sample=samples[0],
            predictions={label: predictions[label][0] for label in predictions},
            labels=("v1", "v2"),
            metrics=metrics,
            episode_context=context,
            episode_index=0,
        )
        self.assertEqual(frame.shape, (780, 1320, 3))
        self.assertEqual(frame.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
