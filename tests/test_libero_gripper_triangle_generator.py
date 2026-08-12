from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from starVLA.dataloader.gr00t_lerobot.gripper_triangle import (
    gripper_triangle_path,
    load_gripper_triangle_sidecar,
    validate_gripper_triangle_payload,
)


def valid_payload() -> dict[str, np.ndarray]:
    return {
        "world_xyz": np.asarray(
            [
                [[-0.02, 0.00, 1.0], [0.02, 0.00, 1.0], [0.00, 0.05, 1.0]],
                [[-0.01, 0.00, 2.0], [0.01, 0.00, 2.0], [0.00, 0.05, 2.0]],
            ],
            dtype=np.float32,
        ),
        "agentview_uvd_pixels": np.asarray(
            [
                [[30.0, 16.0, 1.0], [34.0, 16.0, 1.0], [32.0, 21.0, 1.0]],
                [[31.5, 16.0, 2.0], [32.5, 16.0, 2.0], [32.0, 18.5, 2.0]],
            ],
            dtype=np.float32,
        ),
        "agentview_projection_valid": np.ones((2, 3), dtype=np.bool_),
        "agentview_in_frame": np.ones((2, 3), dtype=np.bool_),
    }


class GripperTriangleGenerationTest(unittest.TestCase):
    def test_extracts_mujoco_bodies_in_landmark_order_for_each_restored_state(self) -> None:
        from examples.modelExtensions.CoT.scripts.build_libero_gripper_triangle_sidecars import (
            extract_gripper_triangle_world,
        )

        class FakeModel:
            body_ids = {
                "gripper0_finger_joint1_tip": 2,
                "gripper0_finger_joint2_tip": 0,
                "gripper0_right_gripper": 1,
            }

            def body_name2id(self, name: str) -> int:
                return self.body_ids[name]

        class FakeData:
            body_xpos = np.zeros((3, 3), dtype=np.float64)

        class FakeSim:
            model = FakeModel()
            data = FakeData()

        class FakeEnv:
            sim = FakeSim()

            def regenerate_obs_from_state(self, state: np.ndarray) -> None:
                offset = float(state[0])
                self.sim.data.body_xpos = np.asarray(
                    [
                        [20.0 + offset, 0.0, 1.0],
                        [30.0 + offset, 0.0, 1.0],
                        [10.0 + offset, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )

        states = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
        trajectory = extract_gripper_triangle_world(
            FakeEnv(), states, np.asarray([2, 0], dtype=np.int64)
        )

        np.testing.assert_allclose(
            trajectory,
            [
                [[12.0, 0.0, 1.0], [22.0, 0.0, 1.0], [32.0, 0.0, 1.0]],
                [[10.0, 0.0, 1.0], [20.0, 0.0, 1.0], [30.0, 0.0, 1.0]],
            ],
        )
        self.assertEqual(trajectory.dtype, np.float32)

    def test_atomic_write_is_resumable_and_leaves_no_temporary_sibling(self) -> None:
        from examples.modelExtensions.CoT.scripts.build_libero_gripper_triangle_sidecars import (
            write_sidecar_atomic,
        )

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "geometry" / "episode_000000.npz"
            result = write_sidecar_atomic(
                path,
                valid_payload(),
                frame_count=2,
                width=64,
                height=32,
                overwrite=False,
            )
            original_bytes = path.read_bytes()
            skipped = write_sidecar_atomic(
                path,
                valid_payload(),
                frame_count=2,
                width=64,
                height=32,
                overwrite=False,
            )

            self.assertEqual(result, "written")
            self.assertEqual(skipped, "skipped")
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])
            load_gripper_triangle_sidecar(path, frame_count=2, width=64, height=32)

    def test_metadata_is_updated_only_after_every_requested_sidecar_validates(self) -> None:
        from examples.modelExtensions.CoT.scripts.build_libero_gripper_triangle_sidecars import (
            EpisodeSidecarSpec,
            finalize_geometry_metadata,
            write_sidecar_atomic,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            info_path = root / "meta" / "info.json"
            original = {"total_episodes": 1, "features": {"kept": True}}
            info_path.write_text(json.dumps(original), encoding="utf-8")
            sidecar_path = gripper_triangle_path(root, 0)
            sidecar_path.parent.mkdir(parents=True)
            sidecar_path.write_bytes(b"corrupt")
            spec = EpisodeSidecarSpec(episode_id=0, frame_count=2, width=64, height=32)

            with self.assertRaises(Exception):
                finalize_geometry_metadata(root, [spec])
            self.assertEqual(json.loads(info_path.read_text(encoding="utf-8")), original)

            write_sidecar_atomic(
                sidecar_path,
                valid_payload(),
                frame_count=2,
                width=64,
                height=32,
                overwrite=True,
            )
            finalize_geometry_metadata(root, [spec])
            updated = json.loads(info_path.read_text(encoding="utf-8"))

        self.assertEqual(updated["features"], {"kept": True})
        self.assertEqual(
            updated["geometry_paths"]["gripper_triangle"],
            "geometry/gripper_triangle/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.npz",
        )
        self.assertEqual(
            updated["gripper_triangle"],
            {
                "landmark_order": [
                    "left_finger_tip",
                    "right_finger_tip",
                    "wrist_hand_base",
                ],
                "frame": "world",
                "uvd_camera": "agentview",
            },
        )

    def test_report_metrics_keep_projection_and_in_frame_rates_separate(self) -> None:
        from examples.modelExtensions.CoT.scripts.build_libero_gripper_triangle_sidecars import (
            summarize_sidecar,
        )

        payload = valid_payload()
        payload["agentview_projection_valid"][1, 2] = False
        payload["agentview_in_frame"][1, 1:] = False
        sidecar = validate_gripper_triangle_payload(
            payload, frame_count=2, width=64, height=32
        )

        metrics = summarize_sidecar(sidecar)

        self.assertAlmostEqual(metrics["min_finger_distance_m"], 0.02)
        self.assertAlmostEqual(metrics["min_triangle_area_m2"], 0.0005)
        self.assertAlmostEqual(metrics["projection_valid_point_ratio"], 5 / 6)
        self.assertAlmostEqual(metrics["in_frame_point_ratio"], 4 / 6)


if __name__ == "__main__":
    unittest.main()
