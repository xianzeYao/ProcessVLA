"""Offline analysis helpers for RoboCasa rollout representations and traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def uvd_hidden_summary(hidden: np.ndarray) -> np.ndarray:
    """Pool V2 time-major, hand-interleaved UVD tokens without flattening time."""
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[1] < 2 or hidden.shape[1] % 2:
        raise ValueError(f"expected [N, even_tokens, hidden], got {hidden.shape}")
    return np.concatenate(
        [
            hidden.mean(axis=1),
            hidden[:, -2] - hidden[:, 0],
            hidden[:, -1] - hidden[:, 1],
        ],
        axis=-1,
    )


def action_motion_statistics(actions: np.ndarray) -> dict[str, np.ndarray]:
    """Measure per-chunk arm and hand articulation without assuming open/close sign."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[-1] < 26:
        raise ValueError(f"expected [N, horizon, >=26] actions, got {actions.shape}")
    delta = np.diff(actions, axis=1)

    def path_length(start: int, stop: int) -> np.ndarray:
        return np.linalg.norm(delta[..., start:stop], axis=-1).sum(axis=1)

    left_arm = path_length(0, 7)
    right_arm = path_length(7, 14)
    left_hand = path_length(14, 20)
    right_hand = path_length(20, 26)
    return {
        "left_arm_motion": left_arm,
        "right_arm_motion": right_arm,
        "left_hand_motion": left_hand,
        "right_hand_motion": right_hand,
        "arm_motion": left_arm + right_arm,
        "hand_motion": left_hand + right_hand,
        "active_hand": np.where(right_arm > left_arm, 1, 0).astype(np.int64),
    }


def classify_control_phases(actions: np.ndarray) -> np.ndarray:
    """Assign sign-agnostic control phases using run-level motion quantiles."""
    stats = action_motion_statistics(actions)
    arm = stats["arm_motion"]
    hand = stats["hand_motion"]
    arm_threshold = float(np.quantile(arm, 0.4))
    hand_threshold = float(np.quantile(hand, 0.7))
    phase = np.full(actions.shape[0], "settle", dtype="U20")
    phase[arm >= arm_threshold] = "arm_motion"
    phase[hand >= hand_threshold] = "hand_transition"
    return phase


def trace_error_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand valid direct trace errors into offset/hand rows with physical units."""
    direct = record.get("direct") or {}
    offsets = list(direct.get("offsets") or [])
    errors = np.asarray(direct.get("abs_error") or [], dtype=np.float64)
    valid = np.asarray(direct.get("valid") or [], dtype=bool)
    if errors.size == 0:
        return []
    if errors.shape != (*valid.shape, 3) or valid.shape[0] != len(offsets):
        raise ValueError(
            f"invalid trace shapes: offsets={len(offsets)}, errors={errors.shape}, valid={valid.shape}"
        )
    image_size = float(record.get("image_size", 224))
    rows: list[dict[str, Any]] = []
    for offset_index, offset in enumerate(offsets):
        for hand in range(valid.shape[1]):
            if not valid[offset_index, hand]:
                continue
            du, dv, depth = errors[offset_index, hand]
            rows.append(
                {
                    "task_index": int(record.get("task_index", -1)),
                    "episode_index": int(record["episode_index"]),
                    "decision_index": int(record["decision_index"]),
                    "episode_success": bool(record.get("episode_success", False)),
                    "offset": int(offset),
                    "hand": int(hand),
                    "uv_l2_px": float(np.hypot(du, dv) * image_size),
                    "depth_abs_mm": float(abs(depth) * 1000.0),
                }
            )
    return rows


def _task_center(
    train: np.ndarray,
    test: np.ndarray,
    train_tasks: np.ndarray,
    test_tasks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    centered_train = np.empty_like(train, dtype=np.float64)
    centered_test = np.empty_like(test, dtype=np.float64)
    means: dict[int, np.ndarray] = {}
    global_mean = train.mean(axis=0)
    for task in np.unique(train_tasks):
        task = int(task)
        mean = train[train_tasks == task].mean(axis=0)
        means[task] = mean
        centered_train[train_tasks == task] = train[train_tasks == task] - mean
    for task in np.unique(test_tasks):
        task = int(task)
        mean = means.get(task, global_mean)
        centered_test[test_tasks == task] = test[test_tasks == task] - mean
    return centered_train, centered_test, means


def _random_project(
    train: np.ndarray,
    test: np.ndarray,
    *,
    max_features: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train.shape[1] <= max_features:
        return train, test
    rng = np.random.default_rng(seed)
    projection = rng.normal(
        scale=1.0 / np.sqrt(max_features),
        size=(train.shape[1], max_features),
    )
    return train @ projection, test @ projection


def episode_split_ridge_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    task_index: np.ndarray,
    episode_index: np.ndarray,
    alpha: float = 1.0,
    max_features: int = 128,
    seed: int = 7,
) -> dict[str, Any]:
    """Fit a task-residualized ridge probe with deterministic held-out episodes."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    tasks = np.asarray(task_index, dtype=np.int64).reshape(-1)
    episodes = np.asarray(episode_index, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or y.ndim == 1:
        y = y.reshape(y.shape[0], -1)
    if x.ndim != 2 or y.ndim != 2 or not (len(x) == len(y) == len(tasks) == len(episodes)):
        raise ValueError(
            f"incompatible probe inputs: x={x.shape}, y={y.shape}, tasks={tasks.shape}, episodes={episodes.shape}"
        )
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    test_mask = finite & ((episodes % 5) == 0)
    train_mask = finite & ~test_mask
    if train_mask.sum() < 2 or test_mask.sum() < 2:
        raise ValueError(
            f"not enough train/test samples: train={train_mask.sum()}, test={test_mask.sum()}"
        )

    x_train, x_test, _ = _task_center(
        x[train_mask], x[test_mask], tasks[train_mask], tasks[test_mask]
    )
    y_train, y_test_residual, y_means = _task_center(
        y[train_mask], y[test_mask], tasks[train_mask], tasks[test_mask]
    )
    x_scale = x_train.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    x_train = x_train / x_scale
    x_test = x_test / x_scale
    x_train, x_test = _random_project(
        x_train, x_test, max_features=max_features, seed=seed
    )
    gram = x_train.T @ x_train
    weights = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]),
        x_train.T @ y_train,
    )
    prediction_residual = x_test @ weights
    test_tasks = tasks[test_mask]
    y_prediction = np.empty_like(prediction_residual)
    y_true = y[test_mask]
    global_y_mean = y[train_mask].mean(axis=0)
    for task in np.unique(test_tasks):
        task_int = int(task)
        mean = y_means.get(task_int, global_y_mean)
        y_prediction[test_tasks == task] = prediction_residual[test_tasks == task] + mean

    residual = np.sum((y_true - y_prediction) ** 2, axis=0)
    total = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    informative = total > 1e-12
    r2 = float(np.mean(1.0 - residual[informative] / total[informative]))
    true_flat = y_true.reshape(y_true.shape[0], -1)
    pred_flat = y_prediction.reshape(y_prediction.shape[0], -1)
    cosine = np.sum(true_flat * pred_flat, axis=1) / (
        np.linalg.norm(true_flat, axis=1) * np.linalg.norm(pred_flat, axis=1) + 1e-12
    )
    return {
        "r2": r2,
        "cosine_mean": float(np.mean(cosine)),
        "train_count": int(train_mask.sum()),
        "test_count": int(test_mask.sum()),
        "projected_features": int(x_train.shape[1]),
        "alpha": float(alpha),
    }


def _load_task_names(run_dir: Path) -> dict[int, str]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return {int(task["task_index"]): str(task["env_name"]) for task in manifest["tasks"]}


def _task_category(env_name: str) -> str:
    stem = env_name.rsplit("/", 1)[-1]
    if stem.startswith("PnP") and "Close" in stem:
        return "close_tasks"
    for source in ("Cuttingboard", "Placemat", "Plate", "Tray"):
        if f"From{source}" in stem:
            return f"novel_from_{source.lower()}"
    return "other"


def load_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Load and concatenate all completed task NPZ/JSONL artifacts in a run."""
    run_dir = Path(run_dir)
    task_names = _load_task_names(run_dir)
    feature_files = sorted((run_dir / "rollout_features").glob("task_*.npz"))
    if not feature_files:
        raise FileNotFoundError(f"no rollout feature artifacts under {run_dir}")
    feature_chunks: dict[str, list[np.ndarray]] = {}
    for path in feature_files:
        with np.load(path, allow_pickle=False) as payload:
            for key in payload.files:
                feature_chunks.setdefault(key, []).append(np.asarray(payload[key]))
    features = {key: np.concatenate(value, axis=0) for key, value in feature_chunks.items()}
    trace_by_id: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    for path in sorted((run_dir / "trace_consistency").glob("task_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (
                int(record.get("task_index", -1)),
                int(record["episode_index"]),
                int(record["decision_index"]),
            )
            trace_by_id[key] = record
    ids = list(
        zip(
            features["task_index"].astype(int),
            features["episode_index"].astype(int),
            features["decision_index"].astype(int),
        )
    )
    trace_records = [trace_by_id.get(key) for key in ids]
    features["task_name"] = np.asarray([task_names[int(task)] for task in features["task_index"]])
    features["task_category"] = np.asarray(
        [_task_category(name) for name in features["task_name"]]
    )
    features["trace_records"] = trace_records
    return features


def _numeric_trace_feature(
    records: Sequence[Mapping[str, Any] | None],
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    keep: list[bool] = []
    expected_shape: tuple[int, ...] | None = None
    for record in records:
        value = None if record is None else record.get(key)
        if value is None:
            keep.append(False)
            rows.append(np.empty(0, dtype=np.float32))
            continue
        array = np.asarray(value, dtype=np.float32)
        expected_shape = expected_shape or array.shape
        if array.shape != expected_shape or not np.isfinite(array).all():
            keep.append(False)
            rows.append(np.empty(0, dtype=np.float32))
            continue
        keep.append(True)
        rows.append(array.reshape(-1))
    if expected_shape is None:
        raise ValueError(f"no finite {key} traces were found")
    width = int(np.prod(expected_shape))
    output = np.zeros((len(rows), width), dtype=np.float32)
    for index, (valid, row) in enumerate(zip(keep, rows)):
        if valid:
            output[index] = row
    return output, np.asarray(keep, dtype=bool)


def representation_probe_results(
    data: Mapping[str, Any],
    *,
    alpha: float = 1.0,
    max_features: int = 128,
) -> list[dict[str, Any]]:
    """Compare action decodability from numerical traces and hidden representations."""
    actions = np.asarray(data["predicted_action"], dtype=np.float32)
    action_delta = (actions - actions[:, :1]).reshape(actions.shape[0], -1)
    predicted_trace, predicted_keep = _numeric_trace_feature(
        data["trace_records"], "predicted_trace_uvd"
    )
    realized_trace, realized_keep = _numeric_trace_feature(
        [None if record is None else record.get("direct") for record in data["trace_records"]],
        "realized_uvd",
    )
    representations = {
        "predicted_uvd": (predicted_trace, predicted_keep),
        "realized_uvd": (realized_trace, realized_keep),
        "uvd_hidden": (uvd_hidden_summary(data["uvd_hidden"]), np.ones(len(actions), bool)),
        "image_hidden": (np.asarray(data["image_hidden_mean"]), np.ones(len(actions), bool)),
        "native_hidden": (np.asarray(data["native_hidden_mean"]), np.ones(len(actions), bool)),
    }
    representations["native_plus_uvd"] = (
        np.concatenate(
            [representations["native_hidden"][0], representations["uvd_hidden"][0]],
            axis=1,
        ),
        np.ones(len(actions), bool),
    )
    results: list[dict[str, Any]] = []
    for name, (feature, keep) in representations.items():
        probe = episode_split_ridge_probe(
            feature[keep],
            action_delta[keep],
            task_index=np.asarray(data["task_index"])[keep],
            episode_index=np.asarray(data["episode_index"])[keep],
            alpha=alpha,
            max_features=max_features,
        )
        probe["representation"] = name
        results.append(probe)
    return results


def phase_trace_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = np.asarray(data["predicted_action"])
    phases = classify_control_phases(actions)
    active_hands = action_motion_statistics(actions)["active_hand"]
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(data["trace_records"]):
        if record is None:
            continue
        for row in trace_error_rows(record):
            row.update(
                {
                    "phase": str(phases[index]),
                    "hand_role": "active" if row["hand"] == active_hands[index] else "inactive",
                    "task_name": str(data["task_name"][index]),
                    "task_category": str(data["task_category"][index]),
                }
            )
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    sem = 0.0 if array.size == 1 else float(array.std(ddof=1) / np.sqrt(array.size))
    return float(array.mean()), sem


def write_analysis_plots(
    output_dir: str | Path,
    probe_rows: Sequence[Mapping[str, Any]],
    phase_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Render compact probe, phase/hand, and success-conditioned figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    models = list(dict.fromkeys(str(row["model"]) for row in probe_rows))
    representations = list(
        dict.fromkeys(str(row["representation"]) for row in probe_rows)
    )

    fig, ax = plt.subplots(figsize=(max(8, len(representations) * 1.35), 4.8))
    x = np.arange(len(representations), dtype=np.float64)
    width = 0.8 / max(1, len(models))
    for model_index, model in enumerate(models):
        values = []
        for representation in representations:
            matches = [
                float(row["r2"])
                for row in probe_rows
                if str(row["model"]) == model
                and str(row["representation"]) == representation
            ]
            values.append(matches[0] if matches else np.nan)
        ax.bar(
            x + (model_index - (len(models) - 1) / 2) * width,
            values,
            width=width,
            label=model,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Held-out action-delta R²")
    ax.set_xticks(x, representations, rotation=25, ha="right")
    ax.set_title("Action structure decodable from rollout representations")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "representation_probes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    phases = ["arm_motion", "hand_transition", "settle"]
    roles = ["active", "inactive"]
    metrics = [("uv_l2_px", "UV L2 error (px)"), ("depth_abs_mm", "Depth error (mm)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, (metric, ylabel) in zip(axes, metrics):
        for model_index, model in enumerate(models):
            for role_index, role in enumerate(roles):
                means = []
                sems = []
                for phase in phases:
                    mean, sem = _mean_sem(
                        [
                            float(row[metric])
                            for row in phase_rows
                            if str(row["model"]) == model
                            and str(row["phase"]) == phase
                            and str(row["hand_role"]) == role
                        ]
                    )
                    means.append(mean)
                    sems.append(sem)
                axis.errorbar(
                    np.arange(len(phases)),
                    means,
                    yerr=sems,
                    marker="o" if role_index == 0 else "s",
                    linestyle="-" if role_index == 0 else "--",
                    label=f"{model} / {role}",
                )
        axis.set_xticks(np.arange(len(phases)), phases, rotation=20)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].set_title("Phase-conditioned active/inactive trace error")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "phase_active_inactive_trace.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    episode_groups: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for row in phase_rows:
        key = (
            str(row["model"]),
            str(row["task_category"]),
            int(row["task_index"]),
            int(row["episode_index"]),
            bool(row["episode_success"]),
        )
        group = episode_groups.setdefault(key, {metric: [] for metric, _ in metrics})
        for metric, _ in metrics:
            group[metric].append(float(row[metric]))
    episode_rows = []
    for key, values in episode_groups.items():
        model, category, task, episode, success = key
        episode_rows.append(
            {
                "model": model,
                "task_category": category,
                "task_index": task,
                "episode_index": episode,
                "episode_success": success,
                **{metric: float(np.mean(values[metric])) for metric, _ in metrics},
            }
        )
    categories = list(
        dict.fromkeys(str(row["task_category"]) for row in episode_rows)
    )
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(categories) * 1.8), 4.8))
    for axis, (metric, ylabel) in zip(axes, metrics):
        for model in models:
            for success in (False, True):
                means = [
                    _mean_sem(
                        [
                            float(row[metric])
                            for row in episode_rows
                            if str(row["model"]) == model
                            and str(row["task_category"]) == category
                            and bool(row["episode_success"]) == success
                        ]
                    )[0]
                    for category in categories
                ]
                axis.plot(
                    np.arange(len(categories)),
                    means,
                    marker="o" if success else "x",
                    linestyle="-" if success else "--",
                    label=f"{model} / {'success' if success else 'failure'}",
                )
        axis.set_xticks(np.arange(len(categories)), categories, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].set_title("Episode trace error versus rollout outcome")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "success_trace_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    offset_categories = list(
        dict.fromkeys(str(row["task_category"]) for row in phase_rows)
    )
    offsets = sorted({int(row["offset"]) for row in phase_rows})
    fig, axes = plt.subplots(
        len(offset_categories),
        2,
        figsize=(12, max(4, 3.2 * len(offset_categories))),
        squeeze=False,
        sharex=True,
    )
    for category_index, category in enumerate(offset_categories):
        for metric_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[category_index, metric_index]
            for model in models:
                means = []
                sems = []
                for offset in offsets:
                    mean, sem = _mean_sem(
                        [
                            float(row[metric])
                            for row in phase_rows
                            if str(row["model"]) == model
                            and str(row["task_category"]) == category
                            and int(row["offset"]) == offset
                        ]
                    )
                    means.append(mean)
                    sems.append(sem)
                axis.errorbar(offsets, means, yerr=sems, marker="o", label=model)
            axis.set_ylabel(ylabel)
            axis.set_title(category)
            axis.grid(alpha=0.2)
            if category_index == len(offset_categories) - 1:
                axis.set_xlabel("Executed-action offset")
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("All six UVD offsets by RoboCasa task category")
    fig.tight_layout()
    path = output_dir / "offset_trace_by_category.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Completed rollout run; repeat for model comparisons.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-features", type=int, default=128)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_probes: list[dict[str, Any]] = []
    all_phase_rows: list[dict[str, Any]] = []
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run must be LABEL=PATH, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        data = load_run_artifacts(raw_path)
        for result in representation_probe_results(
            data, alpha=args.alpha, max_features=args.max_features
        ):
            result["model"] = label
            all_probes.append(result)
        for row in phase_trace_rows(data):
            row["model"] = label
            all_phase_rows.append(row)

    _write_csv(args.output_dir / "representation_probes.csv", all_probes)
    _write_csv(args.output_dir / "phase_trace_rows.csv", all_phase_rows)
    plot_paths = write_analysis_plots(args.output_dir, all_probes, all_phase_rows)
    summary = {
        "schema_version": 1,
        "runs": args.run,
        "probe_results": all_probes,
        "phase_trace_row_count": len(all_phase_rows),
        "plots": [str(path) for path in plot_paths],
        "notes": {
            "split": "episode_index % 5 == 0 held out within every task",
            "phase": "sign-agnostic arm_motion / hand_transition / settle",
            "trace_gt": "on-policy realized UVD, not expert demonstration UVD",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
