from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from starVLA.dataloader.gr00t_lerobot.gripper_triangle import (
    LANDMARK_BODY_NAMES,
    LANDMARK_NAMES,
    gripper_triangle_path,
    load_gripper_triangle_sidecar,
    project_world_to_agentview_uvd,
    validate_gripper_triangle_payload,
)


def valid_payload() -> dict[str, np.ndarray]:
    return {
        "world_xyz": np.asarray(
            [
                [[-0.02, 0.00, 1.0], [0.02, 0.00, 1.0], [0.00, 0.10, 1.0]],
                [[-0.01, 0.00, 2.0], [0.01, 0.00, 2.0], [0.00, 0.10, 2.0]],
            ],
            dtype=np.float32,
        ),
        "agentview_uvd_pixels": np.asarray(
            [
                [[30.0, 16.0, 1.0], [34.0, 16.0, 1.0], [32.0, 26.0, 1.0]],
                [[31.5, 16.0, 2.0], [32.5, 16.0, 2.0], [32.0, 21.0, 2.0]],
            ],
            dtype=np.float32,
        ),
        "agentview_projection_valid": np.ones((2, 3), dtype=np.bool_),
        "agentview_in_frame": np.ones((2, 3), dtype=np.bool_),
    }


class GripperTriangleSidecarContractTest(unittest.TestCase):
    def test_path_and_landmark_order_are_stable(self) -> None:
        self.assertEqual(
            gripper_triangle_path(Path("/dataset/libero_goal"), 1007),
            Path(
                "/dataset/libero_goal/geometry/gripper_triangle/"
                "chunk-001/episode_001007.npz"
            ),
        )
        self.assertEqual(
            LANDMARK_NAMES,
            ("left_finger_tip", "right_finger_tip", "wrist_hand_base"),
        )
        self.assertEqual(
            LANDMARK_BODY_NAMES,
            (
                "gripper0_finger_joint1_tip",
                "gripper0_finger_joint2_tip",
                "gripper0_right_gripper",
            ),
        )

    def test_validates_and_loads_exact_schema_without_dtype_coercion(self) -> None:
        payload = valid_payload()
        validated = validate_gripper_triangle_payload(
            payload, frame_count=2, width=64, height=32
        )
        self.assertEqual(validated.world_xyz.dtype, np.float32)
        self.assertEqual(validated.agentview_uvd_pixels.shape, (2, 3, 3))
        self.assertEqual(validated.agentview_projection_valid.dtype, np.bool_)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.npz"
            np.savez_compressed(path, **payload)
            loaded = load_gripper_triangle_sidecar(
                path, frame_count=2, width=64, height=32
            )

        np.testing.assert_array_equal(loaded.world_xyz, payload["world_xyz"])
        np.testing.assert_array_equal(
            loaded.agentview_in_frame, payload["agentview_in_frame"]
        )

    def test_rejects_wrong_keys_shapes_dtypes_and_frame_count(self) -> None:
        cases: list[tuple[str, dict[str, np.ndarray], int]] = []

        extra = valid_payload()
        extra["unexpected"] = np.zeros(1, dtype=np.float32)
        cases.append(("exact keys", extra, 2))

        wrong_shape = valid_payload()
        wrong_shape["world_xyz"] = wrong_shape["world_xyz"][:, :2]
        cases.append(("world_xyz.*shape", wrong_shape, 2))

        wrong_dtype = valid_payload()
        wrong_dtype["agentview_projection_valid"] = np.ones((2, 3), dtype=np.uint8)
        cases.append(("projection_valid.*dtype", wrong_dtype, 2))

        cases.append(("frame count", valid_payload(), 3))

        for message, payload, frame_count in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_gripper_triangle_payload(
                        payload, frame_count=frame_count, width=64, height=32
                    )

    def test_rejects_invalid_depth_masks_and_in_frame_coordinates(self) -> None:
        nonpositive_depth = valid_payload()
        nonpositive_depth["agentview_uvd_pixels"][0, 0, 2] = 0.0
        with self.assertRaisesRegex(ValueError, "positive depth"):
            validate_gripper_triangle_payload(
                nonpositive_depth, frame_count=2, width=64, height=32
            )

        impossible_mask = valid_payload()
        impossible_mask["agentview_projection_valid"][0, 0] = False
        with self.assertRaisesRegex(ValueError, "in_frame.*projection_valid"):
            validate_gripper_triangle_payload(
                impossible_mask, frame_count=2, width=64, height=32
            )

        outside = valid_payload()
        outside["agentview_uvd_pixels"][0, 0, 0] = 64.0
        with self.assertRaisesRegex(ValueError, "in-frame.*coordinates"):
            validate_gripper_triangle_payload(
                outside, frame_count=2, width=64, height=32
            )


class GripperTriangleProjectionTest(unittest.TestCase):
    def test_projection_separates_positive_depth_from_image_bounds(self) -> None:
        k = np.asarray(
            [[100.0, 0.0, 32.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        points = np.asarray(
            [
                [[0.0, 0.0, 2.0], [1.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            ],
            dtype=np.float32,
        )

        uvd, projection_valid, in_frame = project_world_to_agentview_uvd(
            points,
            k[None],
            np.eye(4, dtype=np.float32)[None],
            width=64,
            height=32,
        )

        np.testing.assert_allclose(uvd[0, 0], [32.0, 16.0, 2.0], atol=1e-6)
        np.testing.assert_array_equal(projection_valid, [[True, True, False]])
        np.testing.assert_array_equal(in_frame, [[True, False, False]])
        self.assertTrue(np.isnan(uvd[0, 2, :2]).all())
        self.assertEqual(float(uvd[0, 2, 2]), -1.0)


if __name__ == "__main__":
    unittest.main()
