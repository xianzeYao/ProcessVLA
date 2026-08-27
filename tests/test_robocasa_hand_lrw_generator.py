from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from starVLA.robocasa_hand_lrw import (
    LANDMARK_BODY_NAMES,
    hand_lrw_path,
    load_hand_lrw_sidecar,
    validate_hand_lrw_payload,
)


def valid_payload(frame_count: int = 2) -> dict[str, np.ndarray]:
    world = np.zeros((frame_count, 2, 3, 3), dtype=np.float32)
    uvd = np.zeros_like(world)
    for frame in range(frame_count):
        for hand in range(2):
            for landmark in range(3):
                world[frame, hand, landmark] = [
                    0.01 * landmark,
                    0.02 * hand,
                    1.0 + frame,
                ]
                uvd[frame, hand, landmark] = [
                    10.0 + landmark,
                    10.0 + hand,
                    1.0 + frame,
                ]
    return {
        "world_xyz": world,
        "agentview_uvd_pixels": uvd,
        "agentview_projection_valid": np.ones(
            (frame_count, 2, 3), dtype=np.bool_
        ),
        "agentview_in_frame": np.ones((frame_count, 2, 3), dtype=np.bool_),
    }


class FakeModel:
    def __init__(self, *, missing: str | None = None) -> None:
        names = [name for hand in LANDMARK_BODY_NAMES for name in hand]
        self.body_ids = {
            name: index for index, name in enumerate(names) if name != missing
        }

    def body_name2id(self, name: str) -> int:
        return self.body_ids[name]


class FakeData:
    body_xpos = np.zeros((6, 3), dtype=np.float64)


class FakeSim:
    def __init__(self, *, missing: str | None = None) -> None:
        self.model = FakeModel(missing=missing)
        self.data = FakeData()


class FakeEnv:
    def __init__(self, *, missing: str | None = None) -> None:
        self.sim = FakeSim(missing=missing)
        self.observation_calls = 0

    def _get_observations(self, *, force_update: bool) -> None:
        assert force_update is True
        self.observation_calls += 1


def make_reset(states_seen: list[float]):
    def reset_to(env: FakeEnv, state: dict) -> None:
        value = float(np.asarray(state["states"])[0])
        states_seen.append(value)
        env.sim.data.body_xpos = np.asarray(
            [[10.0 * body + value, body, 1.0] for body in range(6)],
            dtype=np.float64,
        )

    return reset_to


def test_extracts_left_right_then_thumb_index_wrist_for_each_state() -> None:
    from examples.modelExtensions.CoT.scripts.build_robocasa_hand_lrw_sidecars import (
        extract_bilateral_lrw_world,
    )

    env = FakeEnv()
    states_seen: list[float] = []
    states = np.asarray([[2.0], [5.0]], dtype=np.float64)

    trajectory = extract_bilateral_lrw_world(
        env, states, reset_to=make_reset(states_seen)
    )

    assert trajectory.shape == (2, 2, 3, 3)
    assert trajectory.dtype == np.float32
    np.testing.assert_allclose(
        trajectory[0],
        [
            [[2.0, 0.0, 1.0], [12.0, 1.0, 1.0], [22.0, 2.0, 1.0]],
            [[32.0, 3.0, 1.0], [42.0, 4.0, 1.0], [52.0, 5.0, 1.0]],
        ],
    )
    assert states_seen == [2.0, 5.0]
    assert env.observation_calls == 2


def test_extraction_fails_loudly_for_missing_body() -> None:
    from examples.modelExtensions.CoT.scripts.build_robocasa_hand_lrw_sidecars import (
        extract_bilateral_lrw_world,
    )

    missing = LANDMARK_BODY_NAMES[1][2]
    with pytest.raises(KeyError, match=missing):
        extract_bilateral_lrw_world(
            FakeEnv(missing=missing),
            np.asarray([[0.0]], dtype=np.float64),
            reset_to=make_reset([]),
        )


def test_atomic_write_is_resumable_and_cleans_temporary_files(tmp_path: Path) -> None:
    from examples.modelExtensions.CoT.scripts.build_robocasa_hand_lrw_sidecars import (
        write_sidecar_atomic,
    )

    path = hand_lrw_path(tmp_path, 0)
    written = write_sidecar_atomic(
        path,
        valid_payload(),
        frame_count=2,
        width=64,
        height=32,
        overwrite=False,
    )
    original = path.read_bytes()
    skipped = write_sidecar_atomic(
        path,
        valid_payload(),
        frame_count=2,
        width=64,
        height=32,
        overwrite=False,
    )

    assert written == "written"
    assert skipped == "skipped"
    assert path.read_bytes() == original
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    load_hand_lrw_sidecar(path, frame_count=2, width=64, height=32)


def test_metadata_waits_for_every_episode_and_records_axis_order(tmp_path: Path) -> None:
    from examples.modelExtensions.CoT.scripts.build_robocasa_hand_lrw_sidecars import (
        EpisodeSidecarSpec,
        finalize_geometry_metadata,
        write_sidecar_atomic,
    )

    info_path = tmp_path / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    original = {"total_episodes": 2, "features": {"kept": True}}
    info_path.write_text(json.dumps(original), encoding="utf-8")
    specs = [
        EpisodeSidecarSpec(0, 2, 64, 32),
        EpisodeSidecarSpec(1, 2, 64, 32),
    ]
    write_sidecar_atomic(
        hand_lrw_path(tmp_path, 0),
        valid_payload(),
        frame_count=2,
        width=64,
        height=32,
        overwrite=False,
    )

    with pytest.raises(FileNotFoundError):
        finalize_geometry_metadata(tmp_path, specs)
    assert json.loads(info_path.read_text(encoding="utf-8")) == original

    write_sidecar_atomic(
        hand_lrw_path(tmp_path, 1),
        valid_payload(),
        frame_count=2,
        width=64,
        height=32,
        overwrite=False,
    )
    finalize_geometry_metadata(tmp_path, specs)
    updated = json.loads(info_path.read_text(encoding="utf-8"))

    assert updated["geometry_paths"]["hand_lrw"] == (
        "geometry/hand_lrw/chunk-{episode_chunk:03d}/"
        "episode_{episode_index:06d}.npz"
    )
    assert updated["hand_lrw"] == {
        "hand_order": ["left", "right"],
        "landmark_order": ["thumb", "index", "wrist"],
        "frame": "world",
        "uvd_camera": "agentview",
    }


def test_summary_keeps_each_hands_geometry_and_masks_separate() -> None:
    from examples.modelExtensions.CoT.scripts.build_robocasa_hand_lrw_sidecars import (
        summarize_sidecar,
    )

    payload = valid_payload()
    payload["world_xyz"][:, 0, 1, 0] = 0.04
    payload["world_xyz"][:, 1, 1, 0] = 0.08
    payload["agentview_projection_valid"][1, 1, 2] = False
    payload["agentview_in_frame"][1, 1, 1:] = False
    sidecar = validate_hand_lrw_payload(
        payload, frame_count=2, width=64, height=32
    )

    metrics = summarize_sidecar(sidecar)

    assert metrics["left_min_finger_distance_m"] == pytest.approx(0.04)
    assert metrics["right_min_finger_distance_m"] == pytest.approx(0.08)
    assert metrics["projection_valid_point_ratio"] == pytest.approx(11 / 12)
    assert metrics["in_frame_point_ratio"] == pytest.approx(10 / 12)
