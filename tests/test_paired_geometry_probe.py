import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from examples.simBenchmarks.CoT.geometry_probe.dataset_probe import RoboCasaRerenderStore
from examples.simBenchmarks.CoT.geometry_probe.run_paired_geometry_probe import benchmark_spec
from examples.simBenchmarks.CoT.geometry_probe.paired_probe import (
    build_geometry_example,
    load_materialized_sample,
    materialize_samples,
    run_checkpoint,
    run_framework_predictions,
    write_paired_results,
)
from examples.simBenchmarks.CoT.geometry_probe.probe_utils import (
    EpisodeRef,
    SampleRef,
    build_episode_balanced_sample_plan,
    canonicalize_uvd_prediction,
    masked_depth_metrics,
    uvd_trajectory_metrics,
)
from examples.simBenchmarks.CoT.geometry_probe.visualization import save_paired_sample_figure


class BalancedSamplingTest(unittest.TestCase):
    def test_selects_exact_count_per_group_from_distinct_episodes(self):
        groups = {
            "a": [EpisodeRef("a", index, 40) for index in range(20)],
            "b": [EpisodeRef("b", index, 40) for index in range(20)],
        }

        plan = build_episode_balanced_sample_plan(
            groups,
            samples_per_group=10,
            horizon=8,
            seed=42,
        )

        self.assertEqual({group: sum(ref.suite == group for ref in plan) for group in groups}, {"a": 10, "b": 10})
        self.assertEqual(len({(ref.suite, ref.episode_id) for ref in plan}), 20)
        self.assertTrue(all(ref.frame_index + 8 < ref.episode_length for ref in plan))

    def test_is_deterministic_for_same_seed(self):
        groups = {"a": [EpisodeRef("a", index, 40 + index) for index in range(20)]}

        first = build_episode_balanced_sample_plan(groups, samples_per_group=10, horizon=8, seed=7)
        second = build_episode_balanced_sample_plan(groups, samples_per_group=10, horizon=8, seed=7)

        self.assertEqual(first, second)

    def test_rejects_group_without_enough_complete_episodes(self):
        groups = {"a": [EpisodeRef("a", 0, 8), EpisodeRef("a", 1, 40)]}

        with self.assertRaisesRegex(ValueError, "distinct episodes"):
            build_episode_balanced_sample_plan(groups, samples_per_group=2, horizon=8, seed=42)


class GeometryMetricTest(unittest.TestCase):
    def test_canonicalizes_hand_major_and_time_major_to_time_hand_layout(self):
        hand_major = np.asarray([[0, 0, 1], [1, 0, 1], [10, 0, 1], [11, 0, 1]], dtype=np.float32)
        time_major = np.asarray([[0, 0, 1], [10, 0, 1], [1, 0, 1], [11, 0, 1]], dtype=np.float32)
        expected = np.asarray([[[0, 0, 1], [10, 0, 1]], [[1, 0, 1], [11, 0, 1]]], dtype=np.float32)

        np.testing.assert_array_equal(
            canonicalize_uvd_prediction(hand_major, points_per_hand=2, hand_count=2, token_order="hand_major"),
            expected,
        )
        np.testing.assert_array_equal(
            canonicalize_uvd_prediction(time_major, points_per_hand=2, hand_count=2, token_order="time_major"),
            expected,
        )

    def test_depth_metrics_include_relative_threshold_and_training_loss(self):
        target = np.asarray([[1.0, 2.0], [1.0, 4.0]], dtype=np.float32)
        prediction = np.asarray([[2.0, 4.0], [1.0, 8.0]], dtype=np.float32)

        metrics = masked_depth_metrics(prediction, target, np.ones((2, 2), dtype=np.bool_))

        self.assertEqual(metrics["valid_count"], 4)
        self.assertAlmostEqual(metrics["mae"], 1.75)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(21.0 / 4.0), places=6)
        self.assertAlmostEqual(metrics["abs_rel"], 0.75)
        self.assertAlmostEqual(metrics["delta1"], 0.25)
        self.assertAlmostEqual(metrics["smooth_l1"], 1.375)

    def test_uvd_metrics_report_pixel_ade_fde_and_per_hand_depth(self):
        target = np.asarray(
            [
                [[0.1, 0.2, 1.0], [0.3, 0.4, 2.0]],
                [[0.2, 0.2, 1.5], [0.4, 0.4, 2.5]],
            ],
            dtype=np.float32,
        )
        prediction = target.copy()
        prediction[..., 0] += 0.1
        prediction[..., 2] += 0.2

        metrics = uvd_trajectory_metrics(
            prediction,
            target,
            np.ones((2, 2), dtype=np.bool_),
            image_size=11,
        )

        self.assertEqual(metrics["valid_count"], 4)
        self.assertAlmostEqual(metrics["uv_ade_px"], 1.0, places=6)
        self.assertAlmostEqual(metrics["uv_fde_px"], 1.0, places=6)
        self.assertAlmostEqual(metrics["z_mae_m"], 0.2, places=6)
        self.assertEqual(set(metrics["per_hand"]), {"hand_0", "hand_1"})
        self.assertAlmostEqual(metrics["per_hand"]["hand_0"]["z_mae_m"], 0.2, places=6)


class RoboCasaRerenderStoreTest(unittest.TestCase):
    def test_loads_training_aligned_dual_hand_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_name = "task_a"
            task = root / task_name
            (task / "meta").mkdir(parents=True)
            (task / "data" / "chunk-000").mkdir(parents=True)
            (task / "depth" / "chunk-000").mkdir(parents=True)
            (task / "camera" / "chunk-000").mkdir(parents=True)
            (task / "videos" / "chunk-000" / "observation.images.ego_view").mkdir(parents=True)
            (task / "meta" / "episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "length": 20}) + "\n",
                encoding="utf-8",
            )
            (task / "meta" / "tasks.jsonl").write_text(
                json.dumps({"task_index": 1, "task": "unlocked_waist: move object"}) + "\n",
                encoding="utf-8",
            )

            depth_relative = "depth/chunk-000/episode_000000.npz"
            camera_relative = "camera/chunk-000/episode_000000.npz"
            rows = []
            for frame in range(20):
                rows.append(
                    {
                        "timestamp": frame / 20.0,
                        "task_index": 1,
                        "observation.state": np.zeros(44, dtype=np.float32),
                        "observation.depth.image_m_path": depth_relative,
                        "observation.camera.params_path": camera_relative,
                        "observation.left_thumb_index_pinch_pos": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                        "observation.right_thumb_index_pinch_pos": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
                    }
                )
            pd.DataFrame(rows).to_parquet(task / "data" / "chunk-000" / "episode_000000.parquet")
            np.savez_compressed(task / depth_relative, depth_m=np.ones((20, 8, 8), dtype=np.float32))
            intrinsic = np.asarray([[1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            np.savez_compressed(
                task / camera_relative,
                agentview_K=np.repeat(intrinsic[None], 20, axis=0),
                agentview_T_world_camera=np.repeat(np.eye(4, dtype=np.float32)[None], 20, axis=0),
            )

            store = RoboCasaRerenderStore(
                root,
                [task_name],
                image_size=8,
                horizon=4,
                uvd_num_points=3,
            )
            frame = np.full((1, 8, 8, 3), 127, dtype=np.uint8)
            with mock.patch(
                "examples.simBenchmarks.CoT.geometry_probe.dataset_probe.get_frames_by_timestamps",
                return_value=frame,
            ):
                sample = store.load_sample(SampleRef(task_name, 0, 20, 0))

            self.assertEqual(sample["uvd"].shape, (3, 2, 3))
            self.assertEqual(sample["uvd_valid_mask"].shape, (3, 2))
            self.assertTrue(sample["uvd_valid_mask"].all())
            self.assertEqual(len(sample["example"]["image"]), 1)
            self.assertEqual(sample["example"]["lang"], "unlocked_waist: move object")
            np.testing.assert_allclose(sample["uvd"][:, 0, :2], [[4 / 7, 4 / 7]] * 3)
            np.testing.assert_allclose(sample["uvd"][:, 1, :2], [[5 / 7, 4 / 7]] * 3)


class _SyntheticStore:
    def load_sample(self, ref):
        return {
            "example": {
                "image": [
                    np.full((4, 4, 3), 10, dtype=np.uint8),
                    np.full((4, 4, 3), 20, dtype=np.uint8),
                ],
                "lang": "move object",
            },
            "rgb": np.full((4, 4, 3), 10, dtype=np.uint8),
            "wrist_rgb": np.full((4, 4, 3), 20, dtype=np.uint8),
            "depth_current": np.ones((4, 4), dtype=np.float32),
            "depth_future": np.full((4, 4), 2.0, dtype=np.float32),
            "depth_current_valid": np.ones((4, 4), dtype=np.bool_),
            "depth_future_valid": np.ones((4, 4), dtype=np.bool_),
            "uvd": np.asarray(
                [
                    [[0.1, 0.2, 1.0], [0.3, 0.4, 2.0]],
                    [[0.2, 0.2, 1.5], [0.4, 0.4, 2.5]],
                ],
                dtype=np.float32,
            ),
            "uvd_valid_mask": np.ones((2, 2), dtype=np.bool_),
            "uvd_out_of_frame_mask": np.zeros((2, 2), dtype=np.bool_),
            "uvd_time": np.asarray([0.0, 1.0], dtype=np.float32),
            "uvd_frame_indices": np.asarray([0, 8], dtype=np.int64),
            "metadata": {
                "suite": ref.suite,
                "episode_id": ref.episode_id,
                "frame_index": ref.frame_index,
                "language": "move object",
            },
        }


class _FakeGeometryFramework:
    uvd_hand_count = 2
    uvd_token_order = "hand_major"

    def _trajectory_point_count(self):
        return 2

    def to(self, device):
        self.device = str(device)
        return self

    def eval(self):
        self.is_eval = True
        return self

    def predict_geometry(self, examples):
        np.testing.assert_array_equal(examples[0]["uvd"], np.zeros((2, 2, 3), dtype=np.float32))
        np.testing.assert_array_equal(examples[0]["uvd_valid_mask"], np.ones((2, 2), dtype=np.bool_))
        return {
            "depth_current": np.ones((1, 1, 4, 4), dtype=np.float32),
            "depth_future": np.full((1, 1, 4, 4), 2.0, dtype=np.float32),
            "uvd": np.asarray(
                [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [10.0, 0.0, 1.0], [11.0, 0.0, 1.0]]],
                dtype=np.float32,
            ),
        }


class _SingleHandSyntheticStore(_SyntheticStore):
    def load_sample(self, ref):
        sample = super().load_sample(ref)
        sample["uvd"] = sample["uvd"][:, 0, :]
        sample["uvd_valid_mask"] = sample["uvd_valid_mask"][:, 0]
        sample["uvd_out_of_frame_mask"] = sample["uvd_out_of_frame_mask"][:, 0]
        return sample


class _FakeSingleHandFramework(_FakeGeometryFramework):
    uvd_hand_count = 1

    def predict_geometry(self, examples):
        return {
            "depth_current": np.ones((1, 1, 4, 4), dtype=np.float32),
            "depth_future": np.full((1, 1, 4, 4), 2.0, dtype=np.float32),
            "uvd": np.asarray([[[0.1, 0.2, 1.0], [0.2, 0.2, 1.5]]], dtype=np.float32),
        }


class MaterializedProbeTest(unittest.TestCase):
    def test_materializes_once_and_builds_gt_safe_geometry_example(self):
        with tempfile.TemporaryDirectory() as temporary:
            ref = SampleRef("suite", 3, 20, 4)
            paths = materialize_samples(_SyntheticStore(), [ref], Path(temporary))
            sample = load_materialized_sample(paths[0])
            example = build_geometry_example(sample)

            self.assertEqual(len(paths), 1)
            self.assertEqual(len(example["image"]), 2)
            self.assertEqual(example["lang"], "move object")
            self.assertNotIn("depth_current", example)
            np.testing.assert_array_equal(example["uvd"], np.zeros((2, 2, 3), dtype=np.float32))
            np.testing.assert_array_equal(example["uvd_valid_mask"], np.ones((2, 2), dtype=np.bool_))
            np.testing.assert_array_equal(example["uvd_time"], [0.0, 1.0])

    def test_framework_predictions_are_saved_in_canonical_time_hand_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_paths = materialize_samples(
                _SyntheticStore(),
                [SampleRef("suite", 3, 20, 4)],
                root / "samples",
            )

            prediction_paths = run_framework_predictions(
                _FakeGeometryFramework(),
                label="v1",
                sample_paths=sample_paths,
                output_dir=root / "predictions",
            )

            with np.load(prediction_paths[0]) as payload:
                self.assertEqual(payload["depth_current"].shape, (4, 4))
                self.assertEqual(payload["depth_future"].shape, (4, 4))
                np.testing.assert_array_equal(
                    payload["uvd"],
                    np.asarray(
                        [
                            [[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]],
                            [[1.0, 0.0, 1.0], [11.0, 0.0, 1.0]],
                        ],
                        dtype=np.float32,
                    ),
                )

    def test_single_hand_predictions_match_libero_target_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_paths = materialize_samples(
                _SingleHandSyntheticStore(),
                [SampleRef("suite", 3, 20, 4)],
                root / "samples",
            )

            prediction_paths = run_framework_predictions(
                _FakeSingleHandFramework(),
                label="v1_5",
                sample_paths=sample_paths,
                output_dir=root / "predictions",
            )

            with np.load(prediction_paths[0]) as payload:
                self.assertEqual(payload["uvd"].shape, (2, 3))

    def test_checkpoint_loader_runs_and_writes_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_paths = materialize_samples(
                _SyntheticStore(),
                [SampleRef("suite", 3, 20, 4)],
                root / "samples",
            )
            loaded_paths = []

            def loader(path):
                loaded_paths.append(path)
                return _FakeGeometryFramework()

            prediction_paths = run_checkpoint(
                "checkpoint.pt",
                label="v1",
                sample_paths=sample_paths,
                output_dir=root / "predictions",
                device="cpu",
                framework_loader=loader,
            )

            self.assertEqual(loaded_paths, ["checkpoint.pt"])
            self.assertEqual(len(prediction_paths), 1)
            self.assertTrue(prediction_paths[0].exists())


class PairedArtifactTest(unittest.TestCase):
    def test_benchmark_specs_are_exactly_balanced_to_240(self):
        libero = benchmark_spec("libero")
        robocasa = benchmark_spec("robocasa")

        self.assertEqual(len(libero.groups), 4)
        self.assertEqual(libero.samples_per_group, 60)
        self.assertEqual(len(libero.groups) * libero.samples_per_group, 240)
        self.assertEqual(len(robocasa.groups), 24)
        self.assertEqual(robocasa.samples_per_group, 10)
        self.assertEqual(len(robocasa.groups) * robocasa.samples_per_group, 240)

    def test_writes_paired_metrics_deltas_and_figure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_paths = materialize_samples(
                _SyntheticStore(),
                [SampleRef("suite", 3, 20, 4)],
                root / "samples",
            )
            sample = load_materialized_sample(sample_paths[0])
            predictions = {}
            prediction_paths = {}
            for label, uv_offset in (("v1", 0.0), ("v2", 0.1)):
                directory = root / "predictions" / label
                directory.mkdir(parents=True)
                path = directory / "sample_0000.npz"
                uvd = sample["uvd"].copy()
                uvd[..., 0] += uv_offset
                np.savez_compressed(
                    path,
                    depth_current=sample["depth_current"],
                    depth_future=sample["depth_future"],
                    uvd=uvd,
                )
                prediction_paths[label] = [path]
                predictions[label] = {
                    "depth_current": sample["depth_current"],
                    "depth_future": sample["depth_future"],
                    "uvd": uvd,
                }

            summary = write_paired_results(
                sample_paths=sample_paths,
                prediction_paths=prediction_paths,
                labels=["v1", "v2"],
                output_dir=root / "results",
                image_size=4,
            )
            figure_path = save_paired_sample_figure(
                root / "results" / "paired.png",
                sample=sample,
                predictions=predictions,
                labels=["v1", "v2"],
                metrics=summary["samples"][0]["metrics"],
            )

            self.assertAlmostEqual(summary["overall"]["v1"]["uvd_uv_ade_px"], 0.0)
            self.assertAlmostEqual(summary["overall"]["v2"]["uvd_uv_ade_px"], 0.3, places=6)
            self.assertAlmostEqual(
                summary["overall"]["delta_v2_minus_v1"]["uvd_uv_ade_px"],
                0.3,
                places=6,
            )
            self.assertTrue((root / "results" / "samples.jsonl").exists())
            self.assertTrue((root / "results" / "summary.json").exists())
            self.assertTrue((root / "results" / "summary.csv").exists())
            self.assertTrue(figure_path.exists())

    def test_no_empty_legend_warning_for_sample_without_valid_uvd(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample = _SyntheticStore().load_sample(SampleRef("suite", 3, 20, 4))
            sample = {
                **sample,
                "images": sample["example"]["image"],
                "language": sample["example"]["lang"],
                "uvd_valid_mask": np.zeros_like(sample["uvd_valid_mask"]),
            }
            prediction = {
                "depth_current": sample["depth_current"],
                "depth_future": sample["depth_future"],
                "uvd": sample["uvd"],
            }
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                save_paired_sample_figure(
                    Path(temporary) / "no_valid.png",
                    sample=sample,
                    predictions={"v1": prediction, "v2": prediction},
                    labels=["v1", "v2"],
                    metrics={"v1": {}, "v2": {}},
                )
            self.assertFalse(any("No artists with labels" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
