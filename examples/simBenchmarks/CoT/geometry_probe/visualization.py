"""Visualization and artifact writers for geometry probe samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


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
