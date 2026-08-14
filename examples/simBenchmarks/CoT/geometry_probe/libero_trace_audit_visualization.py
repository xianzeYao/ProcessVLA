"""Fixed-layout visual audits for validated LIBERO V3 rollout records.

The module deliberately renders simulator observations and simulator-realized
landmarks only.  It never accepts, draws, or labels an expert demonstration as
ground truth: dashed triangles are explicitly self-consistency observations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from .libero_trace_audit_metrics import compute_anchor_metrics
from .libero_trace_audit_rollout import RolloutRecord, validate_rollout_record


LANDMARKS = ("left", "right", "wrist")
LANDMARK_COLORS = {"left": "#e45756", "right": "#4c9f70", "wrist": "#4c78a8"}
FRAME_SIZE = (1440, 2560)


def _finite_limit(values: Sequence[np.ndarray], *, floor: float = 1.0) -> tuple[float, float]:
    """Return padded limits that preserve both signs across the whole episode."""

    joined = np.concatenate([np.asarray(value, float).reshape(-1) for value in values])
    finite = joined[np.isfinite(joined)]
    if not len(finite):
        return (-floor, floor)
    magnitude = max(float(np.max(np.abs(finite))), floor)
    padding = max(magnitude * 0.15, floor * 0.05)
    return (min(float(np.min(finite)) - padding, -floor), max(float(np.max(finite)) + padding, floor))


def _overlay_pixels(uvd: np.ndarray, valid: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Map normalized UVD to image pixels, leaving non-drawable vertices NaN."""

    uvd = np.asarray(uvd, dtype=np.float32)
    drawable = np.asarray(valid, dtype=np.bool_) & np.isfinite(uvd).all(axis=1)
    drawable &= uvd[:, 2] > 0.0
    drawable &= (uvd[:, 0] >= 0.0) & (uvd[:, 0] <= 1.0)
    drawable &= (uvd[:, 1] >= 0.0) & (uvd[:, 1] <= 1.0)
    pixels = np.full((len(uvd), 2), np.nan, dtype=np.float32)
    pixels[drawable, 0] = uvd[drawable, 0] * (width - 1)
    pixels[drawable, 1] = uvd[drawable, 1] * (height - 1)
    return pixels


def _validate_frame_index(record: RolloutRecord, frame_index: int) -> int:
    if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, (int, np.integer)):
        raise ValueError("frame_index must be an integer state index")
    index = int(frame_index)
    if not 0 <= index < len(record.agent_rgb):
        raise ValueError(f"frame_index {index} outside state timeline")
    return index


def _metric_curves(record: RolloutRecord) -> dict[str, dict[str, np.ndarray]]:
    result = {
        "uv_ade": {name: np.full(len(record.anchor_steps), np.nan, np.float64) for name in LANDMARKS},
        "d_mae": {name: np.full(len(record.anchor_steps), np.nan, np.float64) for name in LANDMARKS},
        "delta_d_mae": {name: np.full(len(record.anchor_steps), np.nan, np.float64) for name in LANDMARKS},
    }
    height, width = record.agent_rgb.shape[1:3]
    for anchor_index in range(len(record.anchor_steps)):
        metrics = compute_anchor_metrics(
            record.predicted_uvd[anchor_index],
            record.anchor_target_uvd[anchor_index],
            record.anchor_target_valid[anchor_index],
            image_size=(height, width),
            camera_k=record.camera_k_agentview_flipped,
            uvd_time=record.predicted_uvd_time[anchor_index],
            uvd_landmark_ids=record.predicted_uvd_landmark_ids[anchor_index],
        )
        for name in LANDMARKS:
            if not record.anchor_target_valid[anchor_index, :, LANDMARKS.index(name)].all():
                continue
            result["uv_ade"][name][anchor_index] = float(metrics[name]["uv_ade_px"])
            result["d_mae"][name][anchor_index] = float(metrics[name]["d_mae_mm"])
            result["delta_d_mae"][name][anchor_index] = float(metrics[name]["delta_d_mae_mm"])
    return result


def prepare_dashboard_view_model(record: RolloutRecord, frame_index: int) -> dict[str, object]:
    """Build a pure, inspectable description of one fixed dashboard frame."""

    validate_rollout_record(record)
    frame_index = _validate_frame_index(record, frame_index)
    state_step = frame_index
    action_step = min(state_step, len(record.executed_actions) - 1)
    anchor_index = max(0, int(np.searchsorted(record.anchor_steps, state_step, side="right") - 1))
    anchor_step = int(record.anchor_steps[anchor_index])
    horizon = int(record.metadata["action_horizon"])
    offsets = np.rint(
        record.predicted_uvd_time[anchor_index].reshape(-1, 3)[:, 0] * horizon
    ).astype(np.int64)
    relative_step = int(np.clip(state_step - anchor_step, 0, horizon))
    point_index = int(np.argmin(np.abs(offsets - relative_step)))
    height, width = record.agent_rgb.shape[1:3]
    predicted_overlay = _overlay_pixels(
        record.predicted_uvd[anchor_index, point_index],
        np.ones(3, dtype=np.bool_), height=height, width=width,
    )
    realized_overlay = _overlay_pixels(
        record.anchor_target_uvd[anchor_index, point_index],
        record.anchor_target_valid[anchor_index, point_index], height=height, width=width,
    )
    curves = _metric_curves(record)
    error_panels = {
        "uv_ade": {"unit": "px", "x": record.anchor_steps.copy(), "curves": curves["uv_ade"]},
        "d_mae": {"unit": "mm", "x": record.anchor_steps.copy(), "curves": curves["d_mae"]},
        "delta_d_mae": {"unit": "mm", "x": record.anchor_steps.copy(), "curves": curves["delta_d_mae"]},
    }
    chunk_target = record.anchor_target_uvd[anchor_index, :, :, 2]
    view: dict[str, object] = {
        "landmarks": LANDMARKS,
        "colors": LANDMARK_COLORS.copy(),
        "frame_index": frame_index,
        "anchor_index": anchor_index,
        "anchor_step": anchor_step,
        "offsets": offsets,
        "predicted_uvd": record.predicted_uvd[anchor_index],
        "realized_uvd": record.anchor_target_uvd[anchor_index],
        "target_valid": record.anchor_target_valid[anchor_index],
        "chunk_depth": {
            "predicted_m": record.predicted_uvd[anchor_index, :, :, 2],
            "realized_m": chunk_target,
            "persistence_m": np.broadcast_to(chunk_target[:1], chunk_target.shape).copy(),
        },
        "error_panels": error_panels,
        "overlay": {"predicted_xy_px": predicted_overlay, "realized_xy_px": realized_overlay},
        "cursor": {"state_step": state_step, "action_step": action_step, "anchor_step": anchor_step, "relative_step": relative_step, "chunk_point": point_index},
        "axis_limits": {
            "actions_x": (-0.25, len(record.executed_actions) - 0.75),
            "actions_y": _finite_limit([record.executed_actions[:, :6]], floor=0.25),
            "gripper_y": (-1.15, 1.15),
            "chunk_x": (float(offsets.min()) - 0.15, float(offsets.max()) + 0.15),
            "chunk_y": _finite_limit([record.predicted_uvd[..., 2], record.anchor_target_uvd[..., 2]], floor=0.1),
            "uv_ade_y": _finite_limit(list(curves["uv_ade"].values()), floor=1.0),
            "d_mae_y": _finite_limit(list(curves["d_mae"].values()), floor=1.0),
            "delta_d_mae_y": _finite_limit(list(curves["delta_d_mae"].values()), floor=1.0),
        },
    }
    return view


def _style_axis(axis: plt.Axes, title: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=9, loc="left", fontweight="bold")
    axis.set_ylabel(ylabel, fontsize=8)
    axis.grid(True, color="#d9d9d9", linewidth=0.5)
    axis.tick_params(labelsize=7)


def _draw_triangle(axis: plt.Axes, pixels: np.ndarray, *, linestyle: str, label: str) -> None:
    """Draw only finite image-pixel vertices; NaNs produce visible line breaks."""

    points = np.asarray(pixels, dtype=float)
    closed = np.r_[np.arange(3), 0]
    axis.plot(points[closed, 0], points[closed, 1], color="#f4f4f4", linewidth=2.0, linestyle=linestyle, label=label)
    for index, name in enumerate(LANDMARKS):
        if np.isfinite(points[index]).all():
            axis.scatter(points[index, 0], points[index, 1], s=36, color=LANDMARK_COLORS[name], edgecolor="black", linewidth=0.4, zorder=4)


def _figure_to_rgb(figure: plt.Figure) -> np.ndarray:
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[..., :3])
    if rgb.shape != (*FRAME_SIZE, 3) or rgb.dtype != np.uint8:
        raise RuntimeError(f"unexpected dashboard raster {rgb.shape}/{rgb.dtype}")
    return rgb


def _build_dashboard_figure(record: RolloutRecord, view: Mapping[str, object]) -> plt.Figure:
    figure = plt.figure(figsize=(12.8, 7.2), dpi=200, facecolor="#fbfbfc")
    agent = figure.add_axes((0.02, 0.48, 0.42, 0.47))
    wrist = figure.add_axes((0.305, 0.705, 0.125, 0.215))
    translation = figure.add_axes((0.48, 0.755, 0.15, 0.19))
    rotation = figure.add_axes((0.655, 0.755, 0.15, 0.19))
    gripper = figure.add_axes((0.83, 0.755, 0.15, 0.19))
    chunk = figure.add_axes((0.48, 0.49, 0.50, 0.18))
    error_axes = [figure.add_axes((x, 0.105, 0.27, 0.265)) for x in (0.04, 0.365, 0.69)]
    state = int(view["frame_index"])
    cursor = view["cursor"]
    limits = view["axis_limits"]
    agent.imshow(record.agent_rgb[state], origin="upper")
    agent.set_title("agentview: predicted vs simulator-realized landmarks", fontsize=10, loc="left", fontweight="bold")
    agent.set_axis_off()
    overlay = view["overlay"]
    _draw_triangle(agent, np.asarray(overlay["predicted_xy_px"]), linestyle="-", label="predicted")
    _draw_triangle(agent, np.asarray(overlay["realized_xy_px"]), linestyle="--", label="simulator-realized")
    agent.text(0.01, 0.01, "solid: predicted • dashed: simulator-realized (self-consistency), not expert GT", transform=agent.transAxes, color="white", fontsize=6.8, bbox={"facecolor": "black", "alpha": 0.72, "pad": 2})
    case = record.metadata["case"]
    outcome = record.metadata["outcome"]
    agent.text(0.01, 0.985, f"{case['suite']} / task {case['task_id']}  |  {case['language']}\nrank={case['rank_group']} seed={case['seed']} outcome={outcome['success']} ({outcome['end_reason']})\nstate={state} anchor={view['anchor_step']} latency={record.latency_ms[int(view['anchor_index'])]:.1f} ms", transform=agent.transAxes, va="top", color="white", fontsize=7, bbox={"facecolor": "black", "alpha": 0.72, "pad": 2})
    wrist.imshow(record.wrist_rgb[state], origin="upper")
    wrist.set_title("wrist RGB", fontsize=7)
    wrist.set_axis_off()

    action_x = np.arange(len(record.executed_actions))
    for axis, start, title in ((translation, 0, "translation Δ (xyz)"), (rotation, 3, "rotation-vector Δ (xyz)")):
        for offset, color in enumerate(("#e45756", "#4c78a8", "#72b7b2")):
            axis.plot(action_x, record.executed_actions[:, start + offset], color=color, linewidth=1.2, label="xyz"[offset])
        axis.axvline(cursor["action_step"], color="#222", linewidth=1)
        axis.set_xlim(limits["actions_x"]); axis.set_ylim(limits["actions_y"])
        _style_axis(axis, title, "action")
    gripper.plot(action_x, record.executed_actions[:, 6], color="#7f7f7f", linewidth=1.3)
    gripper.axvline(cursor["action_step"], color="#222", linewidth=1)
    gripper.set_xlim(limits["actions_x"]); gripper.set_ylim(limits["gripper_y"])
    _style_axis(gripper, "gripper", "open/close")

    depths = view["chunk_depth"]
    offsets = np.asarray(view["offsets"])
    for name_index, name in enumerate(LANDMARKS):
        color = LANDMARK_COLORS[name]
        chunk.plot(offsets, np.asarray(depths["predicted_m"])[:, name_index], color=color, linewidth=1.5, label=f"{name} predicted")
        chunk.plot(offsets, np.asarray(depths["realized_m"])[:, name_index], color=color, linestyle="--", linewidth=1.2, label=f"{name} realized")
        chunk.plot(offsets, np.asarray(depths["persistence_m"])[:, name_index], color=color, linestyle=":", linewidth=1.0, label=f"{name} persistence")
    chunk.axvline(offsets[int(cursor["chunk_point"])], color="#222", linewidth=1)
    chunk.set_xlim(limits["chunk_x"]); chunk.set_ylim(limits["chunk_y"])
    _style_axis(chunk, "active chunk depth: predicted / simulator-realized / persistence", "depth (m)")
    chunk.set_xlabel("chunk offset (action steps)", fontsize=8)
    chunk.legend(ncol=3, fontsize=5.5, frameon=False, loc="upper center")

    metric_specs = (("uv_ade", "UV ADE", "px"), ("d_mae", "d MAE", "mm"), ("delta_d_mae", "Δd MAE", "mm"))
    panels = view["error_panels"]
    for axis, (metric, title, unit) in zip(error_axes, metric_specs):
        panel = panels[metric]
        for name in LANDMARKS:
            axis.plot(panel["x"], panel["curves"][name], color=LANDMARK_COLORS[name], linewidth=1.6, label=name)
        axis.axvspan(float(cursor["anchor_step"]) - 0.035, float(cursor["anchor_step"]) + 0.035, color="#222", alpha=0.35)
        axis.set_xlim(float(record.anchor_steps.min()) - 0.25, float(record.anchor_steps.max()) + 0.25)
        axis.set_ylim(limits[f"{metric}_y"])
        _style_axis(axis, f"full episode {title}", unit)
        axis.set_xlabel("anchor step", fontsize=8)
        axis.legend(fontsize=6, frameon=False, loc="upper right")
    figure.text(0.02, 0.02, "LRW colors: left / right / wrist. Invalid or out-of-frame landmarks are omitted; no coordinates are clamped to image borders.", fontsize=7, color="#333")
    return figure


def render_dashboard_frame(record: RolloutRecord, frame_index: int) -> np.ndarray:
    """Rasterize one 2560×1440 RGB dashboard state and close its figure."""

    view = prepare_dashboard_view_model(record, frame_index)
    figure = _build_dashboard_figure(record, view)
    try:
        return _figure_to_rgb(figure)
    finally:
        plt.close(figure)


def _summary_card(record: RolloutRecord, frame_index: int) -> np.ndarray:
    figure = plt.figure(figsize=(12.8, 7.2), dpi=200, facecolor="#16202a")
    axis = figure.add_axes((0.08, 0.12, 0.84, 0.76)); axis.set_axis_off()
    case, outcome = record.metadata["case"], record.metadata["outcome"]
    view = prepare_dashboard_view_model(record, frame_index)
    aggregate = {key: float(np.nanmean(np.concatenate(list(view["error_panels"][key]["curves"].values())))) for key in ("uv_ade", "d_mae", "delta_d_mae")}
    axis.text(0.0, 0.85, "LIBERO landmark trace audit", fontsize=34, color="white", fontweight="bold")
    axis.text(0.0, 0.62, f"{case['suite']} • task {case['task_id']} • seed {case['seed']} • {case['language']}", fontsize=20, color="#d8e2ea")
    axis.text(0.0, 0.42, f"Outcome: {outcome['success']} ({outcome['end_reason']})", fontsize=25, color="#f4b183" if not outcome["success"] else "#8fd19e")
    axis.text(0.0, 0.21, f"mean anchor metrics — UV ADE {aggregate['uv_ade']:.2f} px | d MAE {aggregate['d_mae']:.2f} mm | Δd MAE {aggregate['delta_d_mae']:.2f} mm", fontsize=17, color="white")
    axis.text(0.0, 0.04, "Dashed traces are simulator-realized self-consistency observations, not expert ground truth.", fontsize=13, color="#b8c5cf")
    try:
        return _figure_to_rgb(figure)
    finally:
        plt.close(figure)


def render_rollout_video(record: RolloutRecord, path: str | Path, fps: float = 10, *, summary_seconds: float = 2.0, frame_indices: Sequence[int] | None = None) -> Path:
    """Stream rollout frames and one cached summary card to H.264/yuv420p."""

    validate_rollout_record(record)
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(float(summary_seconds)) or summary_seconds < 0:
        raise ValueError("summary_seconds must be finite and non-negative")
    indices = range(len(record.agent_rgb)) if frame_indices is None else tuple(
        _validate_frame_index(record, index) for index in frame_indices
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    card_count = int(round(fps * float(summary_seconds)))
    with imageio.get_writer(str(output), format="pyav", fps=fps, codec="libx264", out_pixel_format="yuv420p", is_batch=False) as writer:
        seen = False
        for index in indices:
            writer.append_data(render_dashboard_frame(record, index))
            seen = True
        if not seen:
            raise ValueError("frame_indices must not be empty")
        if card_count:
            card = _summary_card(record, len(record.agent_rgb) - 1)
            for _ in range(card_count):
                writer.append_data(card)
    return output


def _metric_table(record: RolloutRecord) -> dict[str, dict[str, float]]:
    curves = _metric_curves(record)
    table: dict[str, dict[str, float]] = {}
    for name in (*LANDMARKS, "aggregate"):
        per_name = []
        if name == "aggregate":
            for metric in curves:
                values = np.stack([curves[metric][landmark] for landmark in LANDMARKS])
                per_name.append(float(np.nanmean(values)))
        else:
            per_name = [float(np.nanmean(curves[metric][name])) for metric in curves]
        table[name] = dict(zip(("uv_ade_px", "d_mae_mm", "delta_d_mae_mm"), per_name))
    return table


def _save_figure(figure: plt.Figure, path: str | Path) -> Path:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output, dpi=160, facecolor=figure.get_facecolor())
    finally:
        plt.close(figure)
    return output


def render_rollout_summary(record: RolloutRecord, path: str | Path) -> Path:
    """Render outcome, coverage, aggregate/per-LRW and persistence audit PNG."""

    validate_rollout_record(record)
    table = _metric_table(record)
    view = prepare_dashboard_view_model(record, 0)
    figure, axis = plt.subplots(figsize=(14, 8), facecolor="#fbfbfc")
    axis.set_axis_off()
    case, outcome = record.metadata["case"], record.metadata["outcome"]
    coverage = int(record.anchor_target_valid.sum())
    total = int(record.anchor_target_valid.size)
    lines = [f"{case['suite']} task {case['task_id']} | rank={case['rank_group']} seed={case['seed']}", f"{case['language']}", f"outcome={outcome['success']} ({outcome['end_reason']}); original_success={case['original_success']}  |  coverage={coverage}/{total}", "", "landmark       UV ADE (px)    d MAE (mm)    Δd MAE (mm)"]
    for name in (*LANDMARKS, "aggregate"):
        values = table[name]
        lines.append(f"{name:<14} {values['uv_ade_px']:>10.3f}    {values['d_mae_mm']:>10.3f}    {values['delta_d_mae_mm']:>11.3f}")
    persistence = np.asarray(view["chunk_depth"]["persistence_m"])
    lines.extend(("", f"persistence baseline: first simulator-realized depth repeated across chunk; active depth range {np.nanmin(persistence):.3f}–{np.nanmax(persistence):.3f} m", "simulator-realized is a self-consistency target, not expert ground truth."))
    axis.text(0.04, 0.94, "LIBERO rollout trace audit", fontsize=24, fontweight="bold", va="top")
    axis.text(0.04, 0.82, "\n".join(lines), family="monospace", fontsize=13, va="top")
    return _save_figure(figure, path)


def render_task_seed_summary(records: Sequence[RolloutRecord], path: str | Path) -> Path:
    """Render an unscreened fixed-seed (7…11) task comparison PNG."""

    if len(records) != 5:
        raise ValueError("task seed summary requires exactly five records")
    for record in records:
        validate_rollout_record(record)
    ordered = sorted(records, key=lambda item: int(item.metadata["case"]["seed"]))
    seeds = [int(item.metadata["case"]["seed"]) for item in ordered]
    if seeds != [7, 8, 9, 10, 11]:
        raise ValueError("task seed summary requires fixed seeds 7..11")
    values = np.asarray([[ _metric_table(record)["aggregate"][metric] for metric in ("uv_ade_px", "d_mae_mm", "delta_d_mae_mm") ] for record in ordered], float)
    figure, axis = plt.subplots(figsize=(15, 8), facecolor="#fbfbfc"); axis.set_axis_off()
    case = ordered[0].metadata["case"]
    lines = [f"{case['suite']} task {case['task_id']} — five-seed audit (no outcome filtering)", "seed   outcome / original     UV ADE px     d MAE mm     Δd MAE mm"]
    for record, row in zip(ordered, values):
        info, outcome = record.metadata["case"], record.metadata["outcome"]
        mismatch = " MISMATCH" if bool(outcome["success"]) != bool(info["original_success"]) else ""
        lines.append(f"{info['seed']:>4}   {str(outcome['success']):<5}/{str(info['original_success']):<5}{mismatch:<10} {row[0]:>10.3f}   {row[1]:>10.3f}   {row[2]:>10.3f}")
    means, stds = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    lines.extend(("", f"mean ± std                  {means[0]:.3f} ± {stds[0]:.3f}   {means[1]:.3f} ± {stds[1]:.3f}   {means[2]:.3f} ± {stds[2]:.3f}", "All five selected seeds are retained, including original-vs-rollout outcome mismatches."))
    axis.text(0.04, 0.93, "Task seed summary", fontsize=25, fontweight="bold", va="top")
    axis.text(0.04, 0.80, "\n".join(lines), family="monospace", fontsize=13, va="top")
    return _save_figure(figure, path)


def render_suite_contact_sheet(task_summaries: Sequence[tuple[str | Path, str]], path: str | Path) -> Path:
    """Render exactly two best and two worst task summary cards from explicit rank data."""

    if len(task_summaries) != 4:
        raise ValueError("suite contact sheet requires exactly four task summaries")
    ranks = [str(rank) for _, rank in task_summaries]
    if ranks.count("best") != 2 or ranks.count("worst") != 2 or set(ranks) != {"best", "worst"}:
        raise ValueError("suite contact sheet requires exactly two best and two worst summaries")
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor="#fbfbfc")
    for index, (axis, (summary, rank)) in enumerate(zip(axes.flat, task_summaries), start=1):
        image = plt.imread(Path(summary))
        axis.imshow(image)
        axis.set_axis_off()
        axis.set_title(f"Task {index} • {str(rank).upper()}", loc="left", fontsize=12, fontweight="bold")
    figure.suptitle("LIBERO suite contact sheet — four task seed summaries", fontsize=18, fontweight="bold")
    figure.tight_layout()
    return _save_figure(figure, path)
