from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


class EpisodeSelectionTest(unittest.TestCase):
    def test_selects_ten_evenly_spaced_episodes_including_both_endpoints(self) -> None:
        from examples.modelExtensions.CoT.scripts.visualize_libero_gripper_triangle import (
            select_evenly_spaced_episode_ids,
        )

        self.assertEqual(
            select_evenly_spaced_episode_ids(total_episodes=20, count=10),
            [0, 2, 4, 6, 8, 11, 13, 15, 17, 19],
        )

    def test_rejects_more_samples_than_episodes(self) -> None:
        from examples.modelExtensions.CoT.scripts.visualize_libero_gripper_triangle import (
            select_evenly_spaced_episode_ids,
        )

        with self.assertRaisesRegex(ValueError, "count.*total_episodes"):
            select_evenly_spaced_episode_ids(total_episodes=3, count=4)


class ProjectionTest(unittest.TestCase):
    def test_projects_world_points_and_marks_nonpositive_depth_invalid(self) -> None:
        from examples.modelExtensions.CoT.scripts.visualize_libero_gripper_triangle import (
            project_world_points,
        )

        intrinsics = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        world_from_camera = np.eye(4, dtype=np.float64)
        points_world = np.asarray(
            [[[1.0, 2.0, 10.0], [-1.0, 1.0, 5.0], [0.0, 0.0, -1.0]]],
            dtype=np.float64,
        )

        uv, depth, valid = project_world_points(
            points_world,
            intrinsics[None],
            world_from_camera[None],
        )

        np.testing.assert_allclose(uv[0, :2], [[60.0, 60.0], [30.0, 60.0]])
        np.testing.assert_allclose(depth, [[10.0, 5.0, -1.0]])
        np.testing.assert_array_equal(valid, [[True, True, False]])
        self.assertTrue(np.isnan(uv[0, 2]).all())


class TriangleGeometryTest(unittest.TestCase):
    def test_measures_finger_separation_and_triangle_area(self) -> None:
        from examples.modelExtensions.CoT.scripts.visualize_libero_gripper_triangle import (
            triangle_geometry,
        )

        triangles = np.asarray(
            [
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=np.float64,
        )

        metrics = triangle_geometry(triangles)

        np.testing.assert_allclose(metrics["finger_distance_m"], [2.0, 1.0])
        np.testing.assert_allclose(metrics["triangle_area_m2"], [2.0, 0.5])


class TriangleOverlayTest(unittest.TestCase):
    def test_draws_vertices_and_edges_without_changing_rgb_contract(self) -> None:
        from examples.modelExtensions.CoT.scripts.visualize_libero_gripper_triangle import (
            draw_triangle_overlay,
        )

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        uv = np.asarray([[12.0, 48.0], [52.0, 48.0], [32.0, 12.0]])
        rendered = draw_triangle_overlay(
            image,
            uv,
            np.asarray([True, True, True]),
            frame_index=3,
            finger_distance_m=0.04,
            triangle_area_m2=0.002,
        )

        self.assertEqual(rendered.shape, image.shape)
        self.assertEqual(rendered.dtype, np.uint8)
        self.assertFalse(np.shares_memory(rendered, image))
        for x, y in ((12, 48), (52, 48), (32, 12), (32, 48), (22, 30)):
            self.assertGreater(int(rendered[y, x].max()), 0)

    def test_round_trips_a_real_h264_episode_and_writes_contact_sheet(self) -> None:
        import av

        from examples.modelExtensions.CoT.scripts.visualize_libero_gripper_triangle import (
            render_triangle_video,
            write_suite_contact_sheet,
        )
        from examples.simBenchmarks.CoT.geometry_probe.episode_video import write_video_frames

        frames = [np.full((64, 64, 3), i * 20, dtype=np.uint8) for i in range(4)]
        uv = np.asarray(
            [
                [[10 + i, 50], [50 - i, 50], [30, 10 + i]]
                for i in range(len(frames))
            ],
            dtype=np.float64,
        )
        valid = np.ones((len(frames), 3), dtype=np.bool_)
        geometry = {
            "finger_distance_m": np.linspace(0.05, 0.04, len(frames)),
            "triangle_area_m2": np.linspace(0.002, 0.001, len(frames)),
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            output = root / "overlay.mp4"
            sheet = root / "contact_sheet.jpg"
            write_video_frames(source, frames, fps=20.0)

            result = render_triangle_video(
                source,
                output,
                uv=uv,
                valid=valid,
                geometry=geometry,
                suite="libero_test",
                episode_id=7,
                snapshot_indices=(0, 2, 3),
            )

            self.assertEqual(result["frame_count"], 4)
            self.assertEqual(len(result["snapshots"]), 3)
            with av.open(str(output)) as container:
                stream = container.streams.video[0]
                decoded = list(container.decode(stream))
                self.assertEqual(stream.codec_context.name, "h264")
                self.assertEqual(stream.codec_context.pix_fmt, "yuv420p")
                self.assertEqual(len(decoded), 4)

            write_suite_contact_sheet(
                sheet,
                [(7, result["snapshots"]), (8, result["snapshots"])],
            )
            self.assertTrue(sheet.is_file())
            from PIL import Image

            with Image.open(sheet) as contact:
                self.assertEqual(contact.size, (64 * 3, (64 + 22) * 2))


class EpisodeRecordTest(unittest.TestCase):
    def test_resolves_source_camera_and_agentview_paths(self) -> None:
        from examples.modelExtensions.CoT.scripts.visualize_libero_gripper_triangle import (
            episode_record_from_dataframe,
        )

        frame = pd.DataFrame(
            {
                "source.hdf5_path": ["/source/task.hdf5"] * 2,
                "source.hdf5_demo_id": ["demo_4"] * 2,
                "source.hdf5_index": [3, 7],
                "observation.camera.params_path": [
                    "camera/chunk-000/episode_000007.npz"
                ]
                * 2,
            }
        )

        record = episode_record_from_dataframe(
            Path("/dataset/libero_goal_no_noops_1.0.0_lerobot"),
            "libero_goal",
            7,
            frame,
        )

        self.assertEqual(record.frame_count, 2)
        self.assertEqual(record.hdf5_path, Path("/source/task.hdf5"))
        self.assertEqual(record.demo_id, "demo_4")
        np.testing.assert_array_equal(record.hdf5_indices, [3, 7])
        self.assertEqual(
            record.camera_path,
            Path("/dataset/libero_goal_no_noops_1.0.0_lerobot/camera/chunk-000/episode_000007.npz"),
        )
        self.assertEqual(
            record.video_path,
            Path(
                "/dataset/libero_goal_no_noops_1.0.0_lerobot/videos/chunk-000/"
                "observation.images.image/episode_000007.mp4"
            ),
        )

    def test_rejects_an_episode_mapped_to_multiple_hdf5_files(self) -> None:
        from examples.modelExtensions.CoT.scripts.visualize_libero_gripper_triangle import (
            episode_record_from_dataframe,
        )

        frame = pd.DataFrame(
            {
                "source.hdf5_path": ["/source/a.hdf5", "/source/b.hdf5"],
                "source.hdf5_demo_id": ["demo_0", "demo_0"],
                "source.hdf5_index": [0, 1],
                "observation.camera.params_path": ["camera/episode.npz"] * 2,
            }
        )
        with self.assertRaisesRegex(ValueError, "source.hdf5_path.*one value"):
            episode_record_from_dataframe(Path("/dataset"), "libero_goal", 0, frame)


class SuiteDiscoveryTest(unittest.TestCase):
    def test_discovers_only_rerender_suites_in_canonical_order(self) -> None:
        from examples.modelExtensions.CoT.scripts.visualize_libero_gripper_triangle import (
            discover_suite_roots,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, total in (
                ("libero_spatial_no_noops_1.0.0_lerobot", 12),
                ("libero_10_no_noops_1.0.0_lerobot", 11),
                ("unrelated", 99),
            ):
                (root / name / "meta").mkdir(parents=True)
                (root / name / "meta" / "info.json").write_text(
                    json.dumps({"total_episodes": total}), encoding="utf-8"
                )

            suites = discover_suite_roots(root)

        self.assertEqual(
            [(suite.name, suite.total_episodes) for suite in suites],
            [("libero_10", 11), ("libero_spatial", 12)],
        )


class EpisodeMetricsTest(unittest.TestCase):
    def test_summarizes_geometry_projection_and_in_frame_rates(self) -> None:
        from examples.modelExtensions.CoT.scripts.visualize_libero_gripper_triangle import (
            summarize_episode_metrics,
        )

        points = np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.02, 0.10, 0.0]],
                [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.01, 0.05, 0.0]],
            ]
        )
        uv = np.asarray(
            [
                [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
                [[10.0, 10.0], [70.0, 20.0], [30.0, 30.0]],
            ]
        )
        valid = np.asarray([[True, True, True], [True, True, False]])

        metrics = summarize_episode_metrics(points, uv, valid, width=64, height=64)

        self.assertAlmostEqual(metrics["min_finger_distance_m"], 0.02)
        self.assertAlmostEqual(metrics["min_triangle_area_m2"], 0.0005)
        self.assertAlmostEqual(metrics["projection_valid_point_ratio"], 5 / 6)
        self.assertAlmostEqual(metrics["in_frame_point_ratio"], 4 / 6)
        self.assertAlmostEqual(metrics["all_three_in_frame_frame_ratio"], 0.5)

    def test_rejects_camera_length_that_differs_from_episode(self) -> None:
        from examples.modelExtensions.CoT.scripts.visualize_libero_gripper_triangle import (
            EpisodeRecord,
            load_agentview_camera,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera = root / "camera.npz"
            np.savez_compressed(
                camera,
                agentview_K=np.repeat(np.eye(3)[None], 2, axis=0),
                agentview_T_world_camera=np.repeat(np.eye(4)[None], 2, axis=0),
            )
            record = EpisodeRecord(
                suite="libero_goal",
                dataset_root=root,
                episode_id=0,
                frame_count=3,
                hdf5_path=root / "source.hdf5",
                demo_id="demo_0",
                hdf5_indices=np.arange(3),
                camera_path=camera,
                video_path=root / "source.mp4",
            )

            with self.assertRaisesRegex(ValueError, "camera frame count 2.*episode frame count 3"):
                load_agentview_camera(record)


if __name__ == "__main__":
    unittest.main()
