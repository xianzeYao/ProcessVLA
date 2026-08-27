from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from starVLA.robocasa_hand_lrw import (
    HAND_NAMES,
    LANDMARK_BODY_NAMES,
    LANDMARK_NAMES,
    hand_lrw_path,
    load_hand_lrw_sidecar,
    project_hand_lrw_world_to_agentview_uvd,
    validate_hand_lrw_payload,
)


def valid_payload(frame_count: int = 2) -> dict[str, np.ndarray]:
    world = np.zeros((frame_count, 2, 3, 3), dtype=np.float32)
    uvd = np.zeros((frame_count, 2, 3, 3), dtype=np.float32)
    for frame_index in range(frame_count):
        for hand_index in range(2):
            for landmark_index in range(3):
                world[frame_index, hand_index, landmark_index] = (
                    0.01 * (landmark_index + 1),
                    0.02 * (hand_index + 1),
                    1.0 + frame_index,
                )
                uvd[frame_index, hand_index, landmark_index] = (
                    20.0 + 10.0 * hand_index + landmark_index,
                    12.0 + frame_index + landmark_index,
                    1.0 + frame_index,
                )
    return {
        "world_xyz": world,
        "agentview_uvd_pixels": uvd,
        "agentview_projection_valid": np.ones(
            (frame_count, 2, 3), dtype=np.bool_
        ),
        "agentview_in_frame": np.ones((frame_count, 2, 3), dtype=np.bool_),
    }


def test_contract_has_explicit_hand_landmark_axes_and_body_order(tmp_path: Path) -> None:
    assert HAND_NAMES == ("left", "right")
    assert LANDMARK_NAMES == ("thumb", "index", "wrist")
    assert LANDMARK_BODY_NAMES == (
        (
            "gripper0_left_L_thumb_distal_link",
            "gripper0_left_L_index_intermediate_link",
            "gripper0_left_left_hand",
        ),
        (
            "gripper0_right_R_thumb_distal_link",
            "gripper0_right_R_index_intermediate_link",
            "gripper0_right_right_hand",
        ),
    )
    assert hand_lrw_path(tmp_path, 1001).relative_to(tmp_path).as_posix() == (
        "geometry/hand_lrw/chunk-001/episode_001001.npz"
    )


def test_validates_and_loads_exact_bilateral_lrw_payload(tmp_path: Path) -> None:
    payload = valid_payload()
    sidecar = validate_hand_lrw_payload(
        payload, frame_count=2, width=64, height=32
    )

    assert sidecar.world_xyz.shape == (2, 2, 3, 3)
    assert sidecar.world_xyz.dtype == np.float32
    assert sidecar.agentview_uvd_pixels.shape == (2, 2, 3, 3)
    assert sidecar.agentview_projection_valid.shape == (2, 2, 3)
    assert sidecar.agentview_in_frame.dtype == np.bool_

    path = hand_lrw_path(tmp_path, 7)
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, **payload)
    loaded = load_hand_lrw_sidecar(
        path, frame_count=2, width=64, height=32
    )
    np.testing.assert_array_equal(loaded.world_xyz, payload["world_xyz"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.pop("world_xyz"), "exact keys"),
        (
            lambda p: p.__setitem__(
                "world_xyz", p["world_xyz"].astype(np.float64)
            ),
            "world_xyz dtype",
        ),
        (
            lambda p: p.__setitem__(
                "agentview_uvd_pixels",
                np.zeros((2, 3, 2, 3), dtype=np.float32),
            ),
            "agentview_uvd_pixels shape",
        ),
        (
            lambda p: p["agentview_projection_valid"].__setitem__(
                (0, 0, 0), False
            )
            or p["agentview_in_frame"].__setitem__((0, 0, 0), True),
            "cannot exceed",
        ),
        (
            lambda p: p["agentview_uvd_pixels"].__setitem__(
                (0, 0, 0, 2), 0.0
            ),
            "positive depth",
        ),
    ],
)
def test_rejects_invalid_payloads(mutation, message: str) -> None:
    payload = valid_payload()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        validate_hand_lrw_payload(payload, frame_count=2, width=64, height=32)


def test_rejects_negative_episode_and_frame_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        hand_lrw_path(tmp_path, -1)
    with pytest.raises(ValueError, match="frame count"):
        validate_hand_lrw_payload(
            valid_payload(), frame_count=3, width=64, height=32
        )


def test_projects_all_hand_landmark_axes_with_per_frame_cameras() -> None:
    world = np.asarray(
        [
            [
                [[1.0, 2.0, 2.0], [2.0, 2.0, 2.0], [3.0, 2.0, 2.0]],
                [[1.0, 4.0, 2.0], [2.0, 4.0, 2.0], [3.0, 4.0, 2.0]],
            ],
            [
                [[2.0, 2.0, 4.0], [4.0, 2.0, 4.0], [6.0, 2.0, 4.0]],
                [[2.0, 4.0, 4.0], [4.0, 4.0, 4.0], [6.0, 4.0, 4.0]],
            ],
        ],
        dtype=np.float32,
    )
    camera_k = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy()
    camera_pose = np.broadcast_to(
        np.eye(4, dtype=np.float32), (2, 4, 4)
    ).copy()

    uvd, projection_valid, in_frame = project_hand_lrw_world_to_agentview_uvd(
        world, camera_k, camera_pose, width=8, height=8
    )

    assert uvd.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(uvd[0, 0, 0], [0.5, 1.0, 2.0])
    np.testing.assert_allclose(uvd[1, 1, 2], [1.5, 1.0, 4.0])
    assert projection_valid.all()
    assert in_frame.all()
