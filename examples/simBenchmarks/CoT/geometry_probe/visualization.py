"""Visualization and artifact writers for geometry probe samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from examples.simBenchmarks.CoT.geometry_probe.episode_curves import EpisodeCurveContext


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def save_sample_bundle(
    output_dir: str | Path,
    sample_index: int,
    *,
    sample: dict[str, Any],
    prediction: dict[str, np.ndarray],
    metrics: dict[str, Any],
    latency_ms: float,
    latency_fields: dict[str, float] | None = None,
) -> Path:
    """Write one self-contained NPZ bundle and return its path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sample_{int(sample_index):04d}.npz"
    metadata = dict(sample.get("metadata", {}))
    metadata["latency_ms"] = float(latency_ms)
    if latency_fields is not None:
        metadata["latency"] = {key: float(value) for key, value in latency_fields.items()}
    np.savez_compressed(
        path,
        rgb=np.asarray(sample["rgb"]),
        wrist_rgb=np.asarray(sample["wrist_rgb"]),
        depth_current_gt=np.asarray(sample["depth_current"], dtype=np.float32),
        depth_future_gt=np.asarray(sample["depth_future"], dtype=np.float32),
        depth_current_pred=np.asarray(prediction["depth_current"], dtype=np.float32),
        depth_future_pred=np.asarray(prediction["depth_future"], dtype=np.float32),
        uvd_gt=np.asarray(sample["uvd"], dtype=np.float32),
        uvd_pred=np.asarray(prediction["uvd"], dtype=np.float32),
        uvd_valid_mask=np.asarray(sample["uvd_valid_mask"], dtype=np.bool_),
        uvd_time=np.asarray(sample["uvd_time"], dtype=np.float32),
        metrics_json=np.asarray(json.dumps(metrics, default=_json_default)),
        metadata_json=np.asarray(json.dumps(metadata, default=_json_default)),
    )
    return path


def save_sample_figure(
    path: str | Path,
    *,
    sample: dict[str, Any],
    prediction: dict[str, np.ndarray],
    metrics: dict[str, Any],
    latency_ms: float,
) -> Path:
    """Save current/future depth pairs and a 2D/3D UVD comparison."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current_gt = np.asarray(sample["depth_current"], dtype=np.float32)
    future_gt = np.asarray(sample["depth_future"], dtype=np.float32)
    current_pred = np.asarray(prediction["depth_current"], dtype=np.float32)
    future_pred = np.asarray(prediction["depth_future"], dtype=np.float32)
    finite_values = [x[np.isfinite(x)].reshape(-1) for x in (current_gt, future_gt, current_pred, future_pred)]
    finite_values = [x for x in finite_values if len(x)]
    depth_values = np.concatenate(finite_values) if finite_values else np.asarray([], dtype=np.float32)
    if len(depth_values):
        vmin, vmax = np.percentile(depth_values, [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(depth_values.min()), float(depth_values.max() + 1e-3)
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    panels = [
        (current_gt, "Current GT depth"),
        (current_pred, "Current predicted depth"),
        (future_gt, "Future GT depth"),
        (future_pred, "Future predicted depth"),
    ]
    for axis, (depth, title) in zip(axes[0], panels):
        image = axis.imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)

    rgb = np.asarray(sample["rgb"])
    height, width = rgb.shape[:2]
    axis = axes[1, 0]
    axis.imshow(rgb)
    for points, label, color in (
        (np.asarray(sample["uvd"]), "GT UVD", "lime"),
        (np.asarray(prediction["uvd"]), "Predicted UVD", "red"),
    ):
        valid = np.ones(len(points), dtype=np.bool_)
        if label == "GT UVD":
            valid = np.asarray(sample["uvd_valid_mask"], dtype=np.bool_)
        xy = points[valid, :2].copy()
        xy[:, 0] *= width - 1
        xy[:, 1] *= height - 1
        axis.plot(xy[:, 0], xy[:, 1], "o-", color=color, label=label, linewidth=2)
    axis.set_title("UVD projected on current RGB")
    axis.legend(loc="best", fontsize=8)
    axis.set_xlim(0, width - 1)
    axis.set_ylim(height - 1, 0)

    axis = axes[1, 1]
    gt = np.asarray(sample["uvd"])
    pred = np.asarray(prediction["uvd"])
    valid = np.asarray(sample["uvd_valid_mask"], dtype=np.bool_)
    if valid.any():
        tau = np.asarray(sample["uvd_time"])[valid]
        axis.plot(tau, gt[valid, 2], "o-", color="lime", label="GT depth")
        axis.plot(tau, pred[valid, 2], "o-", color="red", label="Predicted depth")
    axis.set_title("UVD depth over normalized time")
    axis.set_xlabel("tau")
    axis.set_ylabel("camera depth (m)")
    axis.legend(fontsize=8)

    axis = axes[1, 2]
    if valid.any():
        axis.plot(gt[valid, 0], gt[valid, 1], "o-", color="lime", label="GT")
        axis.plot(pred[valid, 0], pred[valid, 1], "o-", color="red", label="Predicted")
    axis.set_title("UVD normalized image path")
    axis.set_xlabel("u")
    axis.set_ylabel("v")
    axis.invert_yaxis()
    axis.legend(fontsize=8)

    fig.delaxes(axes[1, 3])
    axis = fig.add_subplot(2, 4, 8, projection="3d")
    if valid.any():
        axis.plot(gt[valid, 0], gt[valid, 1], gt[valid, 2], "o-", color="lime", label="GT")
        axis.plot(pred[valid, 0], pred[valid, 1], pred[valid, 2], "o-", color="red", label="Predicted")
    axis.set_title("UVD 3D path")
    axis.set_xlabel("u")
    axis.set_ylabel("v")
    axis.set_zlabel("depth (m)")
    axis.legend(fontsize=8)

    metadata = sample.get("metadata", {})
    fig.suptitle(
        f"sample={metadata.get('suite', '?')}/{metadata.get('episode_id', '?')}"
        f" frame={metadata.get('frame_index', '?')} latency={latency_ms:.2f} ms\n"
        f"metrics={json.dumps(metrics, default=_json_default, sort_keys=True)}",
        fontsize=9,
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _create_paired_figure(
    *,
    figsize: tuple[float, float],
    dpi: float | None,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Create deterministic paired panel and colorbar slots."""

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=dpi)
    grid = fig.add_gridspec(
        3,
        10,
        width_ratios=(1.0, 0.045) * 5,
        height_ratios=(1.0, 1.0, 1.08),
        left=0.035,
        right=0.985,
        bottom=0.065,
        top=0.875,
        wspace=0.28,
        hspace=0.38,
    )
    axes = np.empty((3, 5), dtype=object)
    for row in range(2):
        for column in range(5):
            axes[row, column] = fig.add_subplot(grid[row, 2 * column])
    for column in range(5):
        axes[2, column] = fig.add_subplot(grid[2, 2 * column : 2 * column + 2])
    colorbar_axes = np.empty((2, 5), dtype=object)
    for row in range(2):
        for column in range(5):
            colorbar_axes[row, column] = fig.add_subplot(grid[row, 2 * column + 1])
    return fig, axes, colorbar_axes


def _build_paired_figure(
    *,
    sample: dict[str, Any],
    predictions: dict[str, dict[str, np.ndarray]],
    labels: Sequence[str],
    metrics: dict[str, Any],
    figsize: tuple[float, float] = (22.0, 13.0),
    dpi: float | None = None,
    episode_context: EpisodeCurveContext | None = None,
    episode_index: int | None = None,
) -> Any:
    """Build fixed paired panels for either one sample or an episode frame."""

    labels = [str(label) for label in labels]
    if len(labels) != 2 or any(label not in predictions for label in labels):
        raise ValueError(f"two available prediction labels are required, got {labels}")

    rgb = np.asarray(sample["images"][0])
    gt_depths = [
        np.asarray(sample["depth_current"], dtype=np.float32),
        np.asarray(sample["depth_future"], dtype=np.float32),
    ]
    prediction_depths = {
        label: [
            np.asarray(predictions[label]["depth_current"], dtype=np.float32),
            np.asarray(predictions[label]["depth_future"], dtype=np.float32),
        ]
        for label in labels
    }
    finite = [
        depth[np.isfinite(depth)]
        for depth in gt_depths + [depth for label in labels for depth in prediction_depths[label]]
    ]
    finite = [values for values in finite if values.size]
    values = np.concatenate(finite) if finite else np.asarray([], dtype=np.float32)
    if episode_context is not None:
        vmin, vmax = episode_context.depth_image_limits
    elif values.size:
        vmin, vmax = np.percentile(values, [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(values.min()), float(values.max() + 1e-3)
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes, colorbar_axes = _create_paired_figure(figsize=figsize, dpi=dpi)
    depth_names = ("Current", "Future")
    for row, (gt_depth, depth_name) in enumerate(zip(gt_depths, depth_names)):
        panels = [(gt_depth, f"{depth_name} GT")]
        panels.extend(
            (prediction_depths[label][row], f"{depth_name} {label}") for label in labels
        )
        panels.extend(
            (np.abs(prediction_depths[label][row] - gt_depth), f"|{label} - GT|")
            for label in labels
        )
        error_arrays = [panel[0][np.isfinite(panel[0])] for panel in panels[3:]]
        error_arrays = [values for values in error_arrays if values.size]
        error_values = np.concatenate(error_arrays) if error_arrays else np.asarray([])
        if episode_context is not None:
            error_max = episode_context.depth_error_limit
        else:
            error_max = float(np.percentile(error_values, 98)) if error_values.size else 1.0
            error_max = max(error_max, 1e-6)
        for column, (depth, title) in enumerate(panels):
            is_error = column >= 3
            image = axes[row, column].imshow(
                depth,
                cmap="magma" if is_error else "turbo",
                vmin=0.0 if is_error else vmin,
                vmax=error_max if is_error else vmax,
            )
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
            fig.colorbar(image, cax=colorbar_axes[row, column])

    gt_uvd = np.asarray(sample["uvd"], dtype=np.float32)
    valid = np.asarray(sample["uvd_valid_mask"], dtype=np.bool_)
    if gt_uvd.ndim == 2:
        gt_uvd = gt_uvd[:, None, :]
        valid = valid[:, None]
    time = np.asarray(sample["uvd_time"], dtype=np.float32)
    height, width = rgb.shape[:2]
    colors = {"gt": "lime", labels[0]: "cyan", labels[1]: "red"}
    line_styles = ("-", "--", ":", "-.")

    axis = axes[2, 0]
    axis.imshow(rgb)
    trajectories = {"GT": gt_uvd}
    trajectories.update(
        {label: np.asarray(predictions[label]["uvd"], dtype=np.float32) for label in labels}
    )
    for name, trajectory in trajectories.items():
        if trajectory.ndim == 2:
            trajectory = trajectory[:, None, :]
        color = colors["gt"] if name == "GT" else colors[name]
        for hand in range(trajectory.shape[1]):
            hand_valid = valid[:, hand]
            if hand_valid.any():
                xy = trajectory[hand_valid, hand, :2].copy()
                xy[:, 0] *= width - 1
                xy[:, 1] *= height - 1
                axis.plot(
                    xy[:, 0], xy[:, 1], marker="o", color=color,
                    linestyle=line_styles[hand % len(line_styles)], linewidth=2,
                    label=f"{name} hand {hand}",
                )
    axis.set_title("UVD over current RGB")
    axis.set_xlim(0, width - 1)
    axis.set_ylim(height - 1, 0)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=7)

    if episode_context is None:
        for hand in range(min(gt_uvd.shape[1], 2)):
            axis = axes[2, 1 + hand]
            hand_valid = valid[:, hand]
            if not hand_valid.any():
                continue
            axis.plot(
                time[hand_valid], gt_uvd[hand_valid, hand, 2],
                "o-", color=colors["gt"], label="GT",
            )
            for label in labels:
                trajectory = np.asarray(predictions[label]["uvd"], dtype=np.float32)
                if trajectory.ndim == 2:
                    trajectory = trajectory[:, None, :]
                axis.plot(
                    time[hand_valid], trajectory[hand_valid, hand, 2],
                    "o-", color=colors[label], label=label,
                )
            axis.set_title(f"Hand {hand} depth over time")
            axis.set_xlabel("normalized time")
            axis.set_ylabel("camera depth (m)")
            axis.legend(loc="upper right", fontsize=7)
        if gt_uvd.shape[1] == 1:
            axes[2, 2].axis("off")

        axis = axes[2, 3]
        for name, trajectory in trajectories.items():
            if trajectory.ndim == 2:
                trajectory = trajectory[:, None, :]
            color = colors["gt"] if name == "GT" else colors[name]
            for hand in range(trajectory.shape[1]):
                hand_valid = valid[:, hand]
                if hand_valid.any():
                    axis.plot(
                        trajectory[hand_valid, hand, 0], trajectory[hand_valid, hand, 1],
                        marker="o", color=color,
                        linestyle=line_styles[hand % len(line_styles)],
                        label=f"{name} h{hand}",
                    )
        axis.set_title("Normalized UV paths")
        axis.set_xlabel("u")
        axis.set_ylabel("v")
        axis.invert_yaxis()
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="upper right", fontsize=7)
    else:
        if episode_index is None or not 0 <= int(episode_index) < len(episode_context.timestamps):
            raise ValueError(f"invalid episode index: {episode_index}")
        episode_index = int(episode_index)
        episode_trajectories = {"GT": episode_context.gt, **episode_context.predictions}
        episode_time = episode_context.timestamps
        time_span = max(float(episode_time.max() - episode_time.min()), 1e-3)
        time_limits = (
            float(episode_time.min() - 0.02 * time_span),
            float(episode_time.max() + 0.02 * time_span),
        )
        current_time = float(episode_time[episode_index])
        for hand in range(min(episode_context.gt.shape[1], 2)):
            axis = axes[2, 1 + hand]
            for name, trajectory in episode_trajectories.items():
                color = colors["gt"] if name == "GT" else colors[name]
                axis.plot(
                    episode_time,
                    trajectory[:, hand, 2],
                    color=color,
                    linewidth=1.8,
                    label=f"{name} episode",
                )
                axis.plot(
                    [current_time],
                    [trajectory[episode_index, hand, 2]],
                    marker="o",
                    markersize=7,
                    color=color,
                    linestyle="None",
                    label="current GT" if name == "GT" else "_nolegend_",
                )
            axis.axvline(current_time, color="white", alpha=0.65, linewidth=1.0)
            axis.set_xlim(*time_limits)
            axis.set_ylim(*episode_context.depth_limits[hand])
            axis.set_title(f"Hand {hand} depth over episode")
            axis.set_xlabel("episode time (s)")
            axis.set_ylabel("camera depth (m)")
            axis.legend(loc="upper right", fontsize=7)
        if episode_context.gt.shape[1] == 1:
            axes[2, 2].axis("off")

        axis = axes[2, 3]
        for name, trajectory in episode_trajectories.items():
            color = colors["gt"] if name == "GT" else colors[name]
            for hand in range(trajectory.shape[1]):
                line_label = f"{name} episode" if hand == 0 else f"{name} episode h{hand}"
                axis.plot(
                    trajectory[:, hand, 0],
                    trajectory[:, hand, 1],
                    color=color,
                    linewidth=1.8,
                    linestyle=line_styles[hand % len(line_styles)],
                    label=line_label,
                )
                axis.plot(
                    [trajectory[episode_index, hand, 0]],
                    [trajectory[episode_index, hand, 1]],
                    marker="o",
                    markersize=7,
                    color=color,
                    linestyle="None",
                    label="current GT" if name == "GT" and hand == 0 else "_nolegend_",
                )
        axis.set_title("Normalized UV path over episode")
        axis.set_xlabel("u")
        axis.set_ylabel("v")
        axis.set_xlim(*episode_context.uv_limits[0])
        axis.set_ylim(
            episode_context.uv_limits[1][1],
            episode_context.uv_limits[1][0],
        )
        axis.legend(loc="upper right", fontsize=7)

    axes[2, 4].axis("off")
    metric_lines = []
    for label in labels:
        label_metrics = metrics.get(label, {})
        metric_lines.append(
            f"{label}: UV ADE={label_metrics.get('uvd_uv_ade_px', float('nan')):.3f}px, "
            f"Z MAE={label_metrics.get('uvd_z_mae_m', float('nan')):.4f}m\n"
            f"  depth now MAE={label_metrics.get('depth_current_mae', float('nan')):.4f}m, "
            f"future MAE={label_metrics.get('depth_future_mae', float('nan')):.4f}m"
        )
    axes[2, 4].text(
        0.0, 1.0, "\n\n".join(metric_lines), va="top", family="monospace", fontsize=9
    )
    metadata = sample.get("metadata", {})
    fig.suptitle(
        f"{metadata.get('suite', '?')} / episode {metadata.get('episode_id', '?')} / "
        f"frame {metadata.get('frame_index', '?')}\n{sample.get('language', '')}",
        fontsize=11,
    )
    return fig


def _build_paired_sample_figure(
    *,
    sample: dict[str, Any],
    predictions: dict[str, dict[str, np.ndarray]],
    labels: Sequence[str],
    metrics: dict[str, Any],
    figsize: tuple[float, float] = (22.0, 13.0),
    dpi: float | None = None,
) -> Any:
    """Build a fixed-layout single-sample GT/v1/v2 figure."""

    return _build_paired_figure(
        sample=sample,
        predictions=predictions,
        labels=labels,
        metrics=metrics,
        figsize=figsize,
        dpi=dpi,
    )


def _build_paired_episode_figure(
    *,
    sample: dict[str, Any],
    predictions: dict[str, dict[str, np.ndarray]],
    labels: Sequence[str],
    metrics: dict[str, Any],
    episode_context: EpisodeCurveContext,
    episode_index: int,
    figsize: tuple[float, float] = (22.0, 13.0),
    dpi: float | None = None,
) -> Any:
    """Build a fixed-layout frame with immutable full-episode curves."""

    return _build_paired_figure(
        sample=sample,
        predictions=predictions,
        labels=labels,
        metrics=metrics,
        figsize=figsize,
        dpi=dpi,
        episode_context=episode_context,
        episode_index=episode_index,
    )


def save_paired_sample_figure(
    path: str | Path,
    *,
    sample: dict[str, Any],
    predictions: dict[str, dict[str, np.ndarray]],
    labels: Sequence[str],
    metrics: dict[str, Any],
) -> Path:
    """Render GT geometry beside two predictions for one paired sample."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = _build_paired_sample_figure(
        sample=sample,
        predictions=predictions,
        labels=labels,
        metrics=metrics,
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def render_paired_sample_frame(
    *,
    sample: dict[str, Any],
    predictions: dict[str, dict[str, np.ndarray]],
    labels: Sequence[str],
    metrics: dict[str, Any],
) -> np.ndarray:
    """Render the paired static layout into an even-sized RGB video frame."""

    import matplotlib.pyplot as plt

    fig = _build_paired_sample_figure(
        sample=sample,
        predictions=predictions,
        labels=labels,
        metrics=metrics,
        figsize=(16.5, 9.75),
        dpi=80,
    )
    try:
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgb = np.ascontiguousarray(rgba[..., :3])
        height = rgb.shape[0] - rgb.shape[0] % 2
        width = rgb.shape[1] - rgb.shape[1] % 2
        return rgb[:height, :width]
    finally:
        plt.close(fig)


def render_paired_episode_frame(
    *,
    sample: dict[str, Any],
    predictions: dict[str, dict[str, np.ndarray]],
    labels: Sequence[str],
    metrics: dict[str, Any],
    episode_context: EpisodeCurveContext,
    episode_index: int,
) -> np.ndarray:
    """Render one fixed-layout frame with persistent full-episode curves."""

    import matplotlib.pyplot as plt

    fig = _build_paired_episode_figure(
        sample=sample,
        predictions=predictions,
        labels=labels,
        metrics=metrics,
        episode_context=episode_context,
        episode_index=episode_index,
        figsize=(16.5, 9.75),
        dpi=80,
    )
    try:
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgb = np.ascontiguousarray(rgba[..., :3])
        height = rgb.shape[0] - rgb.shape[0] % 2
        width = rgb.shape[1] - rgb.shape[1] % 2
        return rgb[:height, :width]
    finally:
        plt.close(fig)


def save_summary(
    path: str | Path,
    *,
    config: dict[str, Any],
    latency: dict[str, Any],
    metrics: dict[str, Any],
    samples: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "samples": int(samples),
        "latency": latency,
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path
