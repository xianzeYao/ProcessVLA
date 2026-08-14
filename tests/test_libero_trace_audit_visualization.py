from dataclasses import replace
from pathlib import Path

import av
import cv2
import numpy as np

from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_rollout import (
    RolloutRecord,
    validate_rollout_record,
)
from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_visualization import (
    prepare_dashboard_view_model,
    render_dashboard_frame,
    render_rollout_summary,
    render_rollout_video,
    render_suite_contact_sheet,
    render_task_seed_summary,
)


def _record(seed: int = 7) -> RolloutRecord:
    states, height, width, anchors, points = 5, 12, 16, 2, 3
    anchor_steps = np.asarray([0, 2], np.int32)
    times = np.repeat(np.linspace(0.0, 1.0, points, dtype=np.float32), 3)
    landmark_ids = np.tile(np.arange(3, dtype=np.int64), points)
    realized = np.zeros((states, 3, 3), np.float32)
    realized[..., 0] = np.asarray([0.25, 0.55, 0.75], np.float32)
    realized[..., 1] = np.asarray([0.30, 0.45, 0.65], np.float32)
    realized[..., 2] = 0.7 + np.arange(states, dtype=np.float32)[:, None] * 0.02
    predicted = np.empty((anchors, points, 3, 3), np.float32)
    targets = np.empty_like(predicted)
    target_valid = np.ones((anchors, points, 3), np.bool_)
    for anchor_index, anchor in enumerate(anchor_steps):
        for point, offset in enumerate((0, 1, 2)):
            targets[anchor_index, point] = realized[anchor + offset]
            predicted[anchor_index, point] = targets[anchor_index, point]
    predicted[0, 1, 0, 0] += 0.03
    target_valid[1, 1, 1] = False
    targets[1, 1, 1] = np.nan
    raw = np.zeros((states - 1, 7), np.float32)
    raw[:, :6] = np.linspace(-0.1, 0.2, states - 1, dtype=np.float32)[:, None]
    executed = raw.copy()
    executed[:, 6] = 1.0
    record = RolloutRecord(
        agent_rgb=np.full((states, height, width, 3), 80, np.uint8),
        wrist_rgb=np.full((states, height, width, 3), 160, np.uint8),
        agent_depth=np.full((states, height, width), 0.8, np.float32),
        policy_actions_raw=raw,
        executed_actions=executed,
        anchor_steps=anchor_steps,
        predicted_uvd=predicted,
        predicted_uvd_time=np.tile(times, (anchors, 1)),
        predicted_uvd_landmark_ids=np.tile(landmark_ids, (anchors, 1)),
        predicted_depth_current=np.full((anchors, 6, 8), 0.8, np.float32),
        predicted_depth_future=np.full((anchors, 6, 8), 0.9, np.float32),
        realized_uvd=realized,
        realized_xyz=np.ones_like(realized),
        realized_valid=np.ones((states, 3), np.bool_),
        realized_in_frame=np.ones((states, 3), np.bool_),
        anchor_target_uvd=targets,
        anchor_target_valid=target_valid,
        dense_depth_current_target=np.full((anchors, 6, 8), 0.8, np.float32),
        dense_depth_future_target=np.full((anchors, 6, 8), 0.9, np.float32),
        latency_ms=np.asarray([12.0, 18.0], np.float64),
        camera_k_agentview_flipped=np.asarray(
            [[-10.0, 0.0, 7.0], [0.0, -10.0, 5.0], [0.0, 0.0, 1.0]], np.float32
        ),
        metadata={
            "case": {
                "suite": "libero_goal",
                "task_id": 3,
                "language": "place the bowl",
                "rank_group": "best",
                "initial_state_index": 0,
                "seed": seed,
                "original_success": True,
            },
            "outcome": {"success": False, "end_reason": "max_steps"},
            "action_horizon": 2,
            "image_size": [height, width],
            "metrics": {},
            "schema": "state_timeline_v2",
        },
    )
    validate_rollout_record(record)
    return record


def test_view_model_has_exact_lrw_error_curves_units_anchor_x_and_nan_gaps() -> None:
    view = prepare_dashboard_view_model(_record(), frame_index=2)

    assert view["landmarks"] == ("left", "right", "wrist")
    for metric, unit in (("uv_ade", "px"), ("d_mae", "mm"), ("delta_d_mae", "mm")):
        panel = view["error_panels"][metric]
        assert panel["unit"] == unit
        np.testing.assert_array_equal(panel["x"], [0, 2])
        assert tuple(panel["curves"]) == ("left", "right", "wrist")
    assert np.isnan(view["error_panels"]["uv_ade"]["curves"]["right"][1])


def test_early_late_dashboard_frames_keep_fixed_axes_and_move_cursor() -> None:
    record = _record()
    early, late = (
        prepare_dashboard_view_model(record, frame_index=index) for index in (0, 3)
    )
    assert early["axis_limits"] == late["axis_limits"]
    assert early["cursor"]["action_step"] != late["cursor"]["action_step"]
    assert early["cursor"]["anchor_step"] != late["cursor"]["anchor_step"]
    for index in (0, 3):
        frame = render_dashboard_frame(record, index)
        assert frame.shape == (1440, 2560, 3)
        assert frame.dtype == np.uint8


def test_rollout_video_is_h264_yuv420p_and_decodes_requested_frames(tmp_path: Path) -> None:
    path = render_rollout_video(_record(), tmp_path / "rollout.mp4", fps=10, summary_seconds=0, frame_indices=(0, 3))

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.format.name == "yuv420p"
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == 2


def test_rollout_task_and_suite_pngs_are_nonempty(tmp_path: Path) -> None:
    record = _record()
    rollout = render_rollout_summary(record, tmp_path / "rollout.png")
    records = [replace(record, metadata={**record.metadata, "case": {**record.metadata["case"], "seed": seed}}) for seed in range(7, 12)]
    task = render_task_seed_summary(records, tmp_path / "task.png")
    sheets = [render_task_seed_summary(records, tmp_path / f"task_{index}.png") for index in range(4)]
    suite = render_suite_contact_sheet(sheets, tmp_path / "suite.png")
    for path in (rollout, task, suite):
        image = cv2.imread(str(path))
        assert image is not None and image.size > 0
