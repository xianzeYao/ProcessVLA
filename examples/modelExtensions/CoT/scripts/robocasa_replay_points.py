"""MuJoCo point extraction helpers for RoboCasa replay rendering."""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_bilateral_positions(positions: np.ndarray, label: str) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.shape != (2, 3) or not np.isfinite(positions).all():
        raise ValueError(f"GR1 bilateral {label} positions must be finite [2, 3], got {positions}")
    return positions.copy()


def bilateral_eef_world_positions(env: Any) -> np.ndarray:
    """Return Fourier gripper left/right ``grip_site`` positions.

    RoboCasa's GR1 observation dictionary exposes joint/controller arrays under
    ``robot0_left`` and ``robot0_right``; it does not expose bilateral EEF xyz
    fields.  In robosuite, the robot's ``eef_site_id`` is populated from
    ``gripper[arm].important_sites["grip_site"]``.  For this task that is the
    Fourier hand's ``gripper0_{left,right}_grip_site``, not the raw arm site
    named ``robot0_{left,right}_eef_site``.
    """

    robot = env.robots[0]
    site_xpos = np.asarray(env.sim.data.site_xpos, dtype=np.float32)
    positions = np.stack(
        [
            site_xpos[int(robot.eef_site_id["left"])],
            site_xpos[int(robot.eef_site_id["right"])],
        ],
        axis=0,
    )
    return _validate_bilateral_positions(positions, "EEF")


def bilateral_robot_eef_site_world_positions(env: Any) -> np.ndarray:
    """Return raw GR1 arm-end positions in world coordinates.

    The released Teleop XML is asymmetric: the left arm has
    ``robot0_left_eef_site``, while the right arm has no
    ``robot0_right_eef_site`` and exposes the coincident arm-end marker as
    ``robot0_r_wrist_site``.  Prefer the explicit EEF names and use the
    corresponding wrist marker only when the XML lacks that EEF site.
    """

    site_xpos = np.asarray(env.sim.data.site_xpos, dtype=np.float32)
    site_ids = []
    for candidates in (
        ("robot0_left_eef_site", "robot0_l_wrist_site"),
        ("robot0_right_eef_site", "robot0_r_wrist_site"),
    ):
        for name in candidates:
            try:
                site_ids.append(int(env.sim.model.site_name2id(name)))
                break
            except (KeyError, ValueError):
                continue
        else:
            raise ValueError(f"None of the GR1 arm-end sites exist: {candidates}")
    positions = np.stack(
        [site_xpos[site_ids[0]], site_xpos[site_ids[1]]],
        axis=0,
    )
    return _validate_bilateral_positions(positions, "robot EEF site")


def bilateral_pinch_center_world_positions(env: Any) -> np.ndarray:
    """Return visible left/right hand-center points from the four pinch sites.

    RoboCasa's GR1 pinch sites are IK/debug markers, so the renderer hides them
    from RGB. Their midpoint is nevertheless the hand-center point that remains
    inside the DIAL egoview crop and is suitable for a visible UVD trajectory.
    """

    site_xpos = np.asarray(env.sim.data.site_xpos, dtype=np.float32)
    centers = []
    for side in ("left", "right"):
        try:
            site_ids = [
                int(env.sim.model.site_name2id(f"robot0_{side}_pinch_spheres_{index}"))
                for index in range(4)
            ]
        except (KeyError, ValueError):
            return bilateral_eef_world_positions(env)
        centers.append(site_xpos[site_ids].mean(axis=0))
    return _validate_bilateral_positions(np.stack(centers, axis=0), "pinch-center")


def bilateral_thumb_index_pinch_world_positions(env: Any) -> np.ndarray:
    """Return true thumb-index pinch proxies from DIAL's finger bodies.

    DIAL tracks the thumb distal body and index intermediate body for each
    hand.  Their midpoint is a dynamic pinch proxy derived by forward
    kinematics, unlike the fixed ``robot0_*_pinch_spheres`` markers.
    """

    body_xpos = np.asarray(env.sim.data.body_xpos, dtype=np.float32)
    body_pairs = (
        (
            "gripper0_left_L_thumb_distal_link",
            "gripper0_left_L_index_intermediate_link",
        ),
        (
            "gripper0_right_R_thumb_distal_link",
            "gripper0_right_R_index_intermediate_link",
        ),
    )
    positions = []
    for thumb_name, index_name in body_pairs:
        thumb_id = int(env.sim.model.body_name2id(thumb_name))
        index_id = int(env.sim.model.body_name2id(index_name))
        positions.append(0.5 * (body_xpos[thumb_id] + body_xpos[index_id]))
    return _validate_bilateral_positions(np.stack(positions, axis=0), "thumb-index-pinch")
