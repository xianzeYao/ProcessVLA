import json

import numpy as np
import pytest

from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_metrics import (
    align_realized_trace,
    backproject_uvd,
    canonicalize_v3_uvd,
    compute_anchor_metrics,
    metrics_to_jsonable,
)


def test_canonicalize_v3_uvd_preserves_time_major_landmark_order():
    flat = np.asarray([[token, token + 100, token + 200] for token in range(12)], dtype=np.float32)

    canonical = canonicalize_v3_uvd(flat)

    assert canonical.shape == (4, 3, 3)
    np.testing.assert_array_equal(canonical[:, :, 0], [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])


def test_align_realized_trace_marks_early_termination_and_landmark_gaps():
    steps = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    steps[2, 1] = np.nan

    target, valid = align_realized_trace(steps, anchor=1, offsets=[0, 1, 2])

    np.testing.assert_array_equal(target[0], steps[1])
    np.testing.assert_array_equal(target[1, 0], steps[2, 0])
    assert not valid[1, 1]
    assert not valid[2].any()
    assert np.isnan(target[2]).all()


def test_anchor_metrics_report_exact_per_landmark_errors_and_persistence_baseline():
    target = np.asarray(
        [
            [[0.50, 0.50, 1.000], [0.20, 0.10, 1.000], [0.80, 0.90, 1.000]],
            [[0.50, 0.50, 1.010], [0.20, 0.10, 1.010], [0.80, 0.90, 1.000]],
        ],
        dtype=np.float32,
    )
    prediction = np.asarray(
        [
            [[0.50, 0.50, 1.002], [0.20, 0.10, 1.000], [0.80, 0.90, 1.000]],
            [[0.51, 0.50, 1.014], [0.20, 0.10, 1.000], [0.80, 0.90, 1.500]],
        ],
        dtype=np.float32,
    )
    valid = np.asarray([[True, True, True], [True, True, False]])

    metrics = compute_anchor_metrics(
        prediction,
        target,
        valid,
        image_size=(101, 201),
        depth_dead_zone_m=0.002,
    )

    assert metrics["left"]["valid_count"] == 2
    assert np.isclose(metrics["left"]["uv_ade_px"], 1.0, atol=1e-3)
    assert np.isclose(metrics["left"]["uv_fde_px"], 2.0, atol=1e-3)
    assert np.isclose(metrics["left"]["d_mae_mm"], 3.0, atol=1e-3)
    assert np.isclose(metrics["left"]["delta_d_mae_mm"], 2.0, atol=1e-3)
    assert np.isclose(metrics["left"]["delta_d_direction_accuracy"], 1.0, atol=1e-3)
    assert np.isclose(metrics["right"]["d_mae_mm"], 5.0, atol=1e-3)
    assert np.isclose(metrics["right"]["delta_d_mae_mm"], 10.0, atol=1e-3)
    assert np.isclose(metrics["right"]["delta_d_direction_accuracy"], 0.0, atol=1e-3)
    assert metrics["wrist"]["valid_count"] == 1
    assert np.isnan(metrics["wrist"]["delta_d_mae_mm"])
    assert np.isnan(metrics["wrist"]["delta_d_direction_accuracy"])
    assert metrics["aggregate"]["valid_count"] == 5
    assert np.isclose(metrics["aggregate"]["uv_ade_px"], 0.4, atol=1e-3)
    assert np.isclose(metrics["aggregate"]["d_mae_mm"], 3.2, atol=1e-3)
    assert np.isclose(metrics["aggregate"]["delta_d_mae_mm"], 6.0, atol=1e-3)
    assert np.isclose(metrics["aggregate"]["delta_d_direction_accuracy"], 0.5, atol=1e-3)
    assert np.isclose(metrics["persistence"]["left"]["d_mae_mm"], 5.0, atol=1e-3)
    assert np.isclose(metrics["persistence"]["aggregate"]["d_mae_mm"], 4.0, atol=1e-3)
    assert np.isclose(metrics["persistence"]["aggregate"]["delta_d_mae_mm"], 10.0, atol=1e-3)


def test_metrics_separate_projection_validity_from_in_frame_and_backproject_invalid_depths():
    uvd = np.asarray(
        [[[0.5, 0.5, 2.0], [1.2, 0.5, 1.0], [0.5, 0.5, -1.0]]],
        dtype=np.float32,
    )
    camera_k = np.asarray([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    xyz = backproject_uvd(uvd, camera_k, image_size=11)

    np.testing.assert_allclose(xyz[0, 0], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(xyz[0, 1], [0.7, 0.0, 1.0])
    assert np.isnan(xyz[0, 2]).all()

    target = np.asarray([[[0.5, 0.5, 2.0], [0.5, 0.5, 1.0], [0.5, 0.5, 1.0]]], dtype=np.float32)
    metrics = compute_anchor_metrics(uvd, target, np.ones((1, 3), dtype=np.bool_), image_size=11)
    assert metrics["aggregate"]["projection_valid_count"] == 2
    assert metrics["aggregate"]["in_frame_count"] == 1
    assert metrics["aggregate"]["valid_count"] == 2


def test_v3_inputs_reject_fractional_shapes_and_malformed_time_major_metadata():
    flat = np.zeros((12, 3), dtype=np.float32)
    uvd_time = np.repeat(np.arange(4, dtype=np.float32), 3)
    landmark_ids = np.tile(np.arange(3, dtype=np.int64), 4)

    canonical = canonicalize_v3_uvd(flat, uvd_time=uvd_time, uvd_landmark_ids=landmark_ids)
    assert canonical.shape == (4, 3, 3)

    with pytest.raises(ValueError, match="positive integer"):
        canonicalize_v3_uvd(flat, time_points=1.5)
    with pytest.raises(ValueError, match="exactly 3"):
        canonicalize_v3_uvd(flat, landmarks=2)
    with pytest.raises(ValueError, match="same.*time"):
        canonicalize_v3_uvd(flat, uvd_time=np.arange(12), uvd_landmark_ids=landmark_ids)
    with pytest.raises(ValueError, match="time-major"):
        canonicalize_v3_uvd(flat, uvd_time=uvd_time, uvd_landmark_ids=landmark_ids[::-1])

    steps = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\[step, 3, 3\]"):
        align_realized_trace(np.zeros((2, 2, 3), dtype=np.float32), anchor=0, offsets=[0])
    with pytest.raises(ValueError, match="integer"):
        align_realized_trace(steps, anchor=0.5, offsets=[0])
    with pytest.raises(ValueError, match="integer"):
        align_realized_trace(steps, anchor=0, offsets=[0.5])
    with pytest.raises(ValueError, match="non-negative"):
        align_realized_trace(steps, anchor=0, offsets=[-1])
    with pytest.raises(ValueError, match=r"step_valid.*\[step, 3\]"):
        align_realized_trace(steps, anchor=0, offsets=[0], step_valid=np.ones((2, 2), dtype=np.bool_))


def test_fde_uses_only_planned_final_horizon_and_persistence_matches_coverage():
    target = np.zeros((3, 3, 3), dtype=np.float32)
    target[..., 2] = 1.0
    prediction = target.copy()
    prediction[2, 0, 0] = 0.1
    prediction[2, 2, 0] = 0.2
    valid = np.asarray([[True, True, True], [True, True, True], [True, False, True]])

    metrics = compute_anchor_metrics(prediction, target, valid, image_size=11)

    assert np.isclose(metrics["left"]["uv_fde_px"], 1.0)
    assert np.isnan(metrics["right"]["uv_fde_px"])
    assert np.isclose(metrics["wrist"]["uv_fde_px"], 2.0)
    assert metrics["aggregate"]["fde_valid_count"] == 2
    assert np.isclose(metrics["aggregate"]["uv_fde_px"], 1.5)
    assert metrics["persistence"]["aggregate"]["fde_valid_count"] == 2
    assert np.isclose(metrics["persistence"]["aggregate"]["uv_fde_px"], 0.0)


def test_metrics_to_jsonable_replaces_nonfinite_values_and_numpy_scalars():
    target = np.zeros((2, 3, 3), dtype=np.float32)
    target[..., 2] = 1.0
    valid = np.asarray([[True, True, True], [True, False, True]])
    metrics = compute_anchor_metrics(target, target, valid, image_size=11)
    jsonable = metrics_to_jsonable({"metrics": metrics, "extra": np.asarray([np.inf, np.float32(2.0)])})

    assert jsonable["metrics"]["right"]["delta_d_mae_mm"] is None
    assert jsonable["extra"] == [None, 2.0]
    json.dumps(jsonable, allow_nan=False)


def test_backproject_uses_height_width_pixel_scales_for_rectangular_images():
    camera_k = np.asarray([[10.0, 0.0, 10.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    xyz = backproject_uvd(np.asarray([[[0.5, 1.0, 2.0]]], dtype=np.float32), camera_k, image_size=(11, 21))

    np.testing.assert_allclose(xyz[0, 0], [0.0, 1.0, 2.0])
