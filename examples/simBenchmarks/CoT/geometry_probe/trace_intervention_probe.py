"""Offline execution and statistics for hidden-geometry action interventions."""

from __future__ import annotations

import gc
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from examples.simBenchmarks.CoT.geometry_probe.paired_probe import (
    build_geometry_example,
    load_materialized_sample,
)


ROBOCASA_ACTION_GROUPS = {
    "left_arm": (0, 7),
    "right_arm": (7, 14),
    "left_hand": (14, 20),
    "right_hand": (20, 26),
    "waist": (26, 29),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _nested_config_value(root: Any, *keys: str) -> Any:
    current = root
    for key in keys:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _optional_config_bool(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} must be boolean when present, got {value!r}")


def load_sample_identity(path: str | Path) -> dict[str, Any]:
    """Read only stable identity metadata from a materialized paired-probe NPZ."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))

    task_id = metadata.get("task_id")
    if task_id is None:
        task_id = metadata.get("task")
    if task_id is None:
        suite = metadata.get("suite")
        language = metadata.get("language")
        language = None if language is None else " ".join(str(language).split())
        if suite is not None and language:
            task_id = f"{suite}:{language}"
        elif language:
            task_id = language
        else:
            task_id = suite
    if task_id is None:
        raise ValueError(
            f"materialized sample lacks task_id/task or suite/language identity: {path}"
        )

    episode_id = metadata.get("episode_id", metadata.get("episode_index"))
    if episode_id is None:
        raise ValueError(f"materialized sample lacks episode identity: {path}")
    frame_index = metadata.get("frame_index", metadata.get("base_frame_index"))
    if frame_index is None:
        raise ValueError(f"materialized sample lacks frame identity: {path}")
    task_id = str(task_id)
    return {
        "path": str(path),
        "task_id": task_id,
        "episode_id": int(episode_id),
        "frame_index": int(frame_index),
        "cluster_id": f"{task_id}:{int(episode_id)}",
    }


def _task_count_fits_batches(
    count: int,
    *,
    batch_count: int,
    max_per_batch: int,
) -> bool:
    """Whether a task can occupy future batches without singleton groups."""

    if count == 0:
        return True
    if batch_count <= 0:
        return False
    minimum_occupied_batches = (count + max_per_batch - 1) // max_per_batch
    maximum_occupied_batches = min(batch_count, count // 2)
    return minimum_occupied_batches <= maximum_occupied_batches


def _choose_batch_task_counts(
    remaining: Mapping[str, int],
    *,
    batch_size: int,
    batches_after: int,
) -> dict[str, int] | None:
    """Choose one full batch with valid within- and cross-task permutations."""

    max_per_task = batch_size // 2
    task_ids = sorted(remaining, key=lambda task_id: (-remaining[task_id], task_id))
    options: dict[str, tuple[int, ...]] = {}
    for task_id in task_ids:
        count = int(remaining[task_id])
        choices = [
            take
            for take in range(min(count, max_per_task), 1, -1)
            if _task_count_fits_batches(
                count - take,
                batch_count=batches_after,
                max_per_batch=max_per_task,
            )
        ]
        if _task_count_fits_batches(
            count,
            batch_count=batches_after,
            max_per_batch=max_per_task,
        ):
            choices.append(0)
        if not choices:
            return None
        options[task_id] = tuple(choices)

    suffix_capacity = [0] * (len(task_ids) + 1)
    for index in range(len(task_ids) - 1, -1, -1):
        suffix_capacity[index] = (
            suffix_capacity[index + 1] + max(options[task_ids[index]])
        )

    memo: dict[tuple[int, int], tuple[int, ...] | None] = {}

    def search(index: int, needed: int) -> tuple[int, ...] | None:
        if needed < 0 or needed > suffix_capacity[index]:
            return None
        if index == len(task_ids):
            return () if needed == 0 else None
        key = (index, needed)
        if key in memo:
            return memo[key]
        task_id = task_ids[index]
        for take in options[task_id]:
            tail = search(index + 1, needed - take)
            if tail is not None:
                memo[key] = (take, *tail)
                return memo[key]
        memo[key] = None
        return None

    selected = search(0, batch_size)
    if selected is None:
        return None
    return {
        task_id: take
        for task_id, take in zip(task_ids, selected, strict=True)
        if take
    }


def build_intervention_batches(
    sample_paths: Sequence[str | Path],
    *,
    batch_size: int,
) -> list[list[Path]]:
    """Build full batches supporting valid within- and cross-task permutations."""

    batch_size = int(batch_size)
    if batch_size < 4 or batch_size % 2:
        raise ValueError(
            f"intervention batch_size must be even and at least 4, got {batch_size}"
        )
    paths_by_task: dict[str, list[Path]] = defaultdict(list)
    for raw_path in sample_paths:
        path = Path(raw_path)
        identity = load_sample_identity(path)
        paths_by_task[identity["task_id"]].append(path)
    if not paths_by_task:
        raise ValueError("no materialized samples were provided")
    if len(paths_by_task) < 2:
        raise ValueError("intervention batches require samples from distinct tasks")
    for task_id, paths in paths_by_task.items():
        if len(paths) < 2:
            raise ValueError(
                f"task {task_id!r} has {len(paths)} sample; at least two are required"
            )

    sample_count = sum(len(paths) for paths in paths_by_task.values())
    if sample_count % batch_size:
        raise ValueError(
            f"{sample_count} samples cannot form full batches of {batch_size}"
        )
    batch_count = sample_count // batch_size
    max_per_task = batch_size // 2
    remaining = {
        task_id: len(paths)
        for task_id, paths in sorted(paths_by_task.items())
    }
    for task_id, count in remaining.items():
        if not _task_count_fits_batches(
            count,
            batch_count=batch_count,
            max_per_batch=max_per_task,
        ):
            raise ValueError(
                f"task {task_id!r} with {count} samples cannot be distributed over "
                f"{batch_count} cross-task-valid batches"
            )

    queues = {
        task_id: list(paths)
        for task_id, paths in sorted(paths_by_task.items())
    }
    batches: list[list[Path]] = []
    for batch_index in range(batch_count):
        batches_after = batch_count - batch_index - 1
        allocation = _choose_batch_task_counts(
            {task_id: count for task_id, count in remaining.items() if count},
            batch_size=batch_size,
            batches_after=batches_after,
        )
        if allocation is None:
            unresolved = {
                task_id: count
                for task_id, count in remaining.items()
                if count
            }
            raise ValueError(
                "cannot form a full cross-task-valid batch from remaining task "
                f"counts: {unresolved}"
            )
        batch: list[Path] = []
        for task_id in sorted(allocation):
            take = allocation[task_id]
            batch.extend(queues[task_id][:take])
            del queues[task_id][:take]
            remaining[task_id] -= take
        batches.append(batch)
    return batches


def _vector_effect_metrics(
    reference: np.ndarray,
    alternative: np.ndarray,
    *,
    axes: int | tuple[int, ...],
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    difference = alternative - reference
    difference_l2 = np.linalg.norm(difference, axis=axes)
    reference_l2 = np.linalg.norm(reference, axis=axes)
    alternative_l2 = np.linalg.norm(alternative, axis=axes)
    dot = np.sum(reference * alternative, axis=axes)
    denominator = reference_l2 * alternative_l2
    cosine = np.divide(
        dot,
        denominator,
        out=np.zeros_like(dot, dtype=np.float64),
        where=denominator > eps,
    )
    reference_zero = reference_l2 <= eps
    alternative_zero = alternative_l2 <= eps
    cosine = np.where(reference_zero & alternative_zero, 1.0, cosine)
    cosine = np.clip(cosine, -1.0, 1.0)
    return {
        "l2": difference_l2,
        "rms": np.sqrt(np.mean(np.square(difference), axis=axes)),
        "reference_l2": reference_l2,
        "relative_l2": difference_l2 / np.maximum(reference_l2, eps),
        "cosine": cosine,
    }


def _full_effect_metrics(reference: np.ndarray, alternative: np.ndarray) -> dict[str, np.ndarray]:
    output = _vector_effect_metrics(reference, alternative, axes=(1, 2))
    per_time = _vector_effect_metrics(reference, alternative, axes=2)
    output.update({f"per_time_{name}": value for name, value in per_time.items()})
    return output


def effect_metrics(
    reference: Any,
    alternative: Any,
    action_groups: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    """Return per-sample norm/RMS effects without conflating group widths."""

    reference = _to_numpy(reference).astype(np.float64, copy=False)
    alternative = _to_numpy(alternative).astype(np.float64, copy=False)
    if reference.shape != alternative.shape or reference.ndim != 3:
        raise ValueError(
            "reference and alternative must share shape [B, horizon, action_dim], "
            f"got {reference.shape} and {alternative.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(alternative).all():
        raise ValueError("effect inputs must contain only finite values")
    output: dict[str, Any] = {**_full_effect_metrics(reference, alternative), "groups": {}}
    action_dim = int(reference.shape[2])
    for name, bounds in action_groups.items():
        start, stop = (int(bounds[0]), int(bounds[1]))
        if not 0 <= start < stop <= action_dim:
            raise ValueError(
                f"action group {name!r} has invalid bounds {(start, stop)} for dim {action_dim}"
            )
        output["groups"][str(name)] = _full_effect_metrics(
            reference[:, :, start:stop],
            alternative[:, :, start:stop],
        )
    return output


def binary_action_flip_metrics(
    reference: Any,
    alternative: Any,
    *,
    indices: Sequence[int],
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Return discrete threshold changes independently from continuous effects."""

    reference = _to_numpy(reference).astype(np.float64, copy=False)
    alternative = _to_numpy(alternative).astype(np.float64, copy=False)
    if reference.shape != alternative.shape or reference.ndim != 3:
        raise ValueError(
            "reference and alternative must share shape [B, horizon, action_dim], "
            f"got {reference.shape} and {alternative.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(alternative).all():
        raise ValueError("binary effect inputs must contain only finite values")
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError(f"binary threshold must be finite, got {threshold}")
    normalized_indices = tuple(int(index) for index in indices)
    if not normalized_indices or len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError("binary action indices must be non-empty and unique")
    action_dim = int(reference.shape[2])
    if any(index < 0 or index >= action_dim for index in normalized_indices):
        raise ValueError(
            f"binary action indices {normalized_indices} are invalid for dim {action_dim}"
        )
    reference_binary = reference[:, :, normalized_indices] >= threshold
    alternative_binary = alternative[:, :, normalized_indices] >= threshold
    flips = reference_binary != alternative_binary
    per_time_flip = np.mean(flips, axis=2, dtype=np.float64)
    return {
        "flip_rate": np.mean(flips, axis=(1, 2), dtype=np.float64),
        "any_flip": np.any(flips, axis=(1, 2)).astype(np.float64),
        "per_time_flip": per_time_flip,
    }


def unnormalize_action_batch(actions: Any, action_norm_stats: Mapping[str, Any]) -> np.ndarray:
    """Unnormalize a `[B,T,D]` action batch with checkpoint dataset statistics."""

    from starVLA.model.tools import FrameworkTools

    actions = _to_numpy(actions).astype(np.float64, copy=False)
    if actions.ndim != 3 or not np.isfinite(actions).all():
        raise ValueError(
            f"normalized actions must be finite with shape [B, horizon, dim], got {actions.shape}"
        )
    flat = actions.reshape(-1, actions.shape[-1])
    unnormalized = FrameworkTools.unnormalize_actions(
        flat,
        dict(action_norm_stats),
        gripper_channel_idx=6 if actions.shape[-1] == 7 else -1,
    )
    return np.asarray(unnormalized, dtype=np.float64).reshape(actions.shape)


def cluster_bootstrap_mean_ci(
    values: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    seed: int,
    resamples: int = 2000,
) -> tuple[float, float]:
    """Bootstrap whole episode clusters and return a percentile 95% CI."""

    values_array = np.asarray(values, dtype=np.float64)
    cluster_ids = tuple(str(cluster_id) for cluster_id in cluster_ids)
    if values_array.ndim != 1 or len(values_array) != len(cluster_ids) or not len(values_array):
        raise ValueError("values and cluster_ids must be non-empty one-dimensional peers")
    if not np.isfinite(values_array).all():
        raise ValueError("bootstrap values must contain only finite values")
    resamples = int(resamples)
    if resamples < 1:
        raise ValueError(f"resamples must be positive, got {resamples}")
    unique_clusters = tuple(sorted(set(cluster_ids)))
    positions = {
        cluster_id: np.asarray(
            [index for index, value in enumerate(cluster_ids) if value == cluster_id],
            dtype=np.int64,
        )
        for cluster_id in unique_clusters
    }
    rng = np.random.default_rng(int(seed))
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        sampled = rng.integers(0, len(unique_clusters), size=len(unique_clusters))
        sampled_positions = np.concatenate(
            [positions[unique_clusters[int(index)]] for index in sampled]
        )
        bootstrap_means[sample_index] = float(np.mean(values_array[sampled_positions]))
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(low), float(high)


def _metric_summary(
    values: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    seed: int,
    resamples: int,
) -> dict[str, float]:
    values_array = np.asarray(values, dtype=np.float64)
    low, high = cluster_bootstrap_mean_ci(
        values_array,
        cluster_ids,
        seed=seed,
        resamples=resamples,
    )
    return {
        "count": int(values_array.size),
        "mean": float(np.mean(values_array)),
        "median": float(np.median(values_array)),
        "std": float(np.std(values_array)),
        "ci95_low": low,
        "ci95_high": high,
    }


_EFFECT_SCALAR_METRICS = ("l2", "rms", "reference_l2", "relative_l2", "cosine")
_EFFECT_TIME_METRICS = tuple(f"per_time_{name}" for name in _EFFECT_SCALAR_METRICS)


def _sample_effect(metrics: dict[str, Any], sample_index: int) -> dict[str, Any]:
    output = {name: float(metrics[name][sample_index]) for name in _EFFECT_SCALAR_METRICS}
    output.update(
        {name: metrics[name][sample_index].tolist() for name in _EFFECT_TIME_METRICS}
    )
    output["groups"] = {}
    for name, group in metrics["groups"].items():
        group_output = {
            metric_name: float(group[metric_name][sample_index])
            for metric_name in _EFFECT_SCALAR_METRICS
        }
        group_output.update(
            {
                metric_name: group[metric_name][sample_index].tolist()
                for metric_name in _EFFECT_TIME_METRICS
            }
        )
        output["groups"][name] = group_output
    return output


def _summarize_effect_nodes(
    nodes: Sequence[dict[str, Any]],
    cluster_ids: Sequence[str],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    output = {}
    for metric_name in _EFFECT_SCALAR_METRICS:
        output[metric_name] = _metric_summary(
            [node[metric_name] for node in nodes],
            cluster_ids,
            seed=seed,
            resamples=resamples,
        )
    for metric_name in _EFFECT_TIME_METRICS:
        width = len(nodes[0][metric_name])
        output[metric_name] = [
            _metric_summary(
                [node[metric_name][index] for node in nodes],
                cluster_ids,
                seed=seed,
                resamples=resamples,
            )
            for index in range(width)
        ]
    output["groups"] = {}
    for group_name in nodes[0].get("groups", {}):
        output["groups"][group_name] = _summarize_effect_nodes(
            [node["groups"][group_name] for node in nodes],
            cluster_ids,
            seed=seed,
            resamples=resamples,
        )
    return output


def _sample_binary_effect(metrics: dict[str, np.ndarray], sample_index: int) -> dict[str, Any]:
    return {
        "flip_rate": float(metrics["flip_rate"][sample_index]),
        "any_flip": float(metrics["any_flip"][sample_index]),
        "per_time_flip": metrics["per_time_flip"][sample_index].tolist(),
    }


def _summarize_binary_nodes(
    nodes: Sequence[dict[str, Any]],
    cluster_ids: Sequence[str],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    output = {
        name: _metric_summary(
            [node[name] for node in nodes], cluster_ids, seed=seed, resamples=resamples
        )
        for name in ("flip_rate", "any_flip")
    }
    output["per_time_flip"] = [
        _metric_summary(
            [node["per_time_flip"][index] for node in nodes],
            cluster_ids,
            seed=seed,
            resamples=resamples,
        )
        for index in range(len(nodes[0]["per_time_flip"]))
    ]
    return output


def _diagnostic_tensor(diagnostics: Sequence[Any], field: str) -> np.ndarray:
    return np.stack([_to_numpy(getattr(step, field)) for step in diagnostics], axis=0)


def _action_groups(action_dim: int) -> dict[str, tuple[int, int]]:
    if int(action_dim) == 29:
        return dict(ROBOCASA_ACTION_GROUPS)
    if int(action_dim) == 7:
        return {"arm": (0, 6), "hand": (6, 7)}
    return {"all": (0, int(action_dim))}


def write_probe_artifacts(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
    per_sample: Sequence[dict[str, Any]],
    trajectories: Mapping[str, np.ndarray],
) -> dict[str, Path]:
    """Write the four recomputable Stage A artifact files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": output_dir / "config.json",
        "summary": output_dir / "summary.json",
        "per_sample": output_dir / "per_sample.jsonl",
        "trajectories": output_dir / "trajectories.npz",
    }
    paths["config"].write_text(
        json.dumps(config, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["summary"].write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    with paths["per_sample"].open("w", encoding="utf-8") as handle:
        for row in per_sample:
            handle.write(json.dumps(row, default=_json_default) + "\n")
    np.savez_compressed(
        paths["trajectories"],
        **{str(name): np.asarray(value) for name, value in trajectories.items()},
    )
    return paths


def write_probe_figures(
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Render compact figures entirely from summary.json-compatible data."""

    import matplotlib.pyplot as plt

    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    velocity = summary["velocity_effect"]
    variants = list(velocity)
    step_names = sorted(
        {step for variant in variants for step in velocity[variant]},
        key=lambda name: int(str(name).removeprefix("step_")),
    )
    matrix = np.asarray(
        [
            [float(velocity[variant][step]["l2"]["mean"]) for step in step_names]
            for variant in variants
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(
        figsize=(max(5.0, 1.2 * len(step_names)), max(3.0, 0.55 * len(variants) + 1.5))
    )
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(step_names)), step_names)
    axis.set_yticks(range(len(variants)), variants)
    axis.set_xlabel("Euler intervention step")
    axis.set_title("Local velocity effect (mean L2)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.3g}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="normalized-space L2")
    figure.tight_layout()
    path = figure_dir / "velocity_effect_heatmap.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    final = summary["final_action_effect"]
    final_means = [float(final[variant]["all"]["l2"]["mean"]) for variant in variants]
    final_low = [float(final[variant]["all"]["l2"]["ci95_low"]) for variant in variants]
    final_high = [float(final[variant]["all"]["l2"]["ci95_high"]) for variant in variants]
    x_positions = np.arange(len(variants))
    figure, axis = plt.subplots(figsize=(max(6.0, 1.25 * len(variants)), 4.0))
    axis.bar(x_positions, final_means)
    axis.vlines(x_positions, final_low, final_high, colors="black", linewidth=1.0)
    cap_width = 0.08
    axis.hlines(final_low, x_positions - cap_width, x_positions + cap_width, colors="black")
    axis.hlines(final_high, x_positions - cap_width, x_positions + cap_width, colors="black")
    axis.set_xticks(x_positions, variants, rotation=30, ha="right")
    axis.set_ylabel("normalized-space L2")
    axis.set_title("Final action effect, intervention at all Euler steps")
    figure.tight_layout()
    path = figure_dir / "final_action_effect.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    group_names = list(final[variants[0]]["all"]["groups"])
    figure, axis = plt.subplots(figsize=(max(6.0, 1.4 * len(group_names)), 4.0))
    width = 0.8 / max(len(variants), 1)
    group_positions = np.arange(len(group_names))
    for variant_index, variant in enumerate(variants):
        values = [
            float(final[variant]["all"]["groups"][group]["l2"]["mean"])
            for group in group_names
        ]
        offset = (variant_index - (len(variants) - 1) / 2.0) * width
        axis.bar(group_positions + offset, values, width=width, label=variant)
    axis.set_xticks(group_positions, group_names, rotation=25, ha="right")
    axis.set_ylabel("normalized-space L2")
    axis.set_title("Final action effect by action group")
    axis.legend()
    figure.tight_layout()
    path = figure_dir / "action_group_effect.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)
    return outputs


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_trace_intervention_checkpoint(
    checkpoint: str | Path,
    *,
    sample_paths: Sequence[str | Path],
    output_dir: str | Path,
    batch_size: int,
    seed: int,
    device: str,
    variants: Sequence[str],
    bootstrap_resamples: int = 2000,
    repeat_tolerance: float = 1e-6,
    config_path: str | Path | None = None,
    unnorm_key: str | None = None,
    framework_loader: Any | None = None,
) -> dict[str, Any]:
    """Load one V2 checkpoint and write a complete offline intervention probe."""

    repeat_tolerance = float(repeat_tolerance)
    if not np.isfinite(repeat_tolerance) or repeat_tolerance < 0.0:
        raise ValueError(
            f"repeat_tolerance must be finite and non-negative, got {repeat_tolerance}"
        )
    if framework_loader is None:
        from starVLA.model.framework.base_framework import baseframework

        framework_loader = baseframework.from_pretrained
    checkpoint = str(checkpoint)
    batches = build_intervention_batches(sample_paths, batch_size=batch_size)
    execution_paths = [path for batch in batches for path in batch]
    identities = [load_sample_identity(path) for path in execution_paths]
    framework = framework_loader(checkpoint)
    if not callable(getattr(framework, "predict_action_interventions", None)):
        raise TypeError("checkpoint framework does not implement predict_action_interventions")
    norm_stats = getattr(framework, "norm_stats", None)
    action_norm_stats = None
    resolved_unnorm_key = None
    if norm_stats:
        from starVLA.model.tools import FrameworkTools

        resolved_unnorm_key = FrameworkTools.check_unnorm_key(norm_stats, unnorm_key)
        action_norm_stats = FrameworkTools.get_action_stats(
            norm_stats,
            resolved_unnorm_key,
        )
    elif unnorm_key is not None:
        raise ValueError("unnorm_key was provided but the checkpoint has no norm_stats")
    checkpoint_include_state = _optional_config_bool(
        _nested_config_value(
            getattr(framework, "config", None),
            "datasets",
            "vla_data",
            "include_state",
        ),
        name="checkpoint datasets.vla_data.include_state",
    )
    checkpoint_state_dim_value = _nested_config_value(
        getattr(framework, "config", None),
        "framework",
        "action_model",
        "state_dim",
    )
    checkpoint_state_dim = (
        None if checkpoint_state_dim_value is None else int(checkpoint_state_dim_value)
    )
    framework = framework.to(device).eval()

    action_dim = int(framework.action_model.action_dim)
    action_groups = _action_groups(action_dim)
    per_sample: list[dict[str, Any]] = []
    trajectory_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    repeat_errors: list[float] = []
    observed_dtypes: set[str] = set()
    observed_devices: set[str] = set()
    observed_example_keys: set[tuple[str, ...]] = set()
    try:
        for batch_index, batch in enumerate(batches):
            batch_samples = [load_materialized_sample(path) for path in batch]
            examples = [build_geometry_example(sample) for sample in batch_samples]
            batch_example_keys = {tuple(sorted(example)) for example in examples}
            if len(batch_example_keys) != 1:
                raise ValueError(
                    f"batch {batch_index} has inconsistent input example keys: "
                    f"{sorted(batch_example_keys)}"
                )
            observed_example_keys.update(batch_example_keys)
            batch_has_state = "state" in next(iter(batch_example_keys))
            if (
                checkpoint_include_state is not None
                and batch_has_state != checkpoint_include_state
            ):
                raise ValueError(
                    "checkpoint state-input contract mismatch: "
                    f"datasets.vla_data.include_state={checkpoint_include_state}, "
                    f"but batch {batch_index} state presence is {batch_has_state}"
                )
            batch_identities = [load_sample_identity(path) for path in batch]
            task_ids = tuple(identity["task_id"] for identity in batch_identities)
            noise_generator = torch.Generator(device="cpu").manual_seed(
                int(seed) + batch_index
            )
            initial_actions = torch.randn(
                (len(examples), int(framework.action_model.action_horizon), action_dim),
                dtype=torch.float32,
                generator=noise_generator,
            )
            result = framework.predict_action_interventions(
                examples,
                variants=tuple(variants),
                task_ids=task_ids,
                initial_actions=initial_actions,
                seed=int(seed) + batch_index,
                rollout_steps=None,
            )
            initial_tensor = result["initial_actions"]
            if isinstance(initial_tensor, torch.Tensor):
                observed_dtypes.add(str(initial_tensor.dtype))
                observed_devices.add(str(initial_tensor.device))
            else:
                observed_dtypes.add(str(np.asarray(initial_tensor).dtype))
                observed_devices.add(str(device))
            batch_size_actual = len(batch)
            batch_offset = len(per_sample)
            repeat_error = float(result["correct"]["repeat_max_abs_error"])
            repeat_errors.append(repeat_error)
            if not np.isfinite(repeat_error) or repeat_error > repeat_tolerance:
                raise RuntimeError(
                    "correct-condition repeat sanity check failed for "
                    f"batch {batch_index}: max_abs_error={repeat_error} exceeds "
                    f"tolerance={repeat_tolerance}"
                )
            correct_actions = _to_numpy(result["correct"]["actions"])
            correct_actions_unnormalized = (
                None
                if action_norm_stats is None
                else unnormalize_action_batch(correct_actions, action_norm_stats)
            )
            correct_x_before = _diagnostic_tensor(
                result["correct"]["diagnostics"], "x_before"
            )
            correct_velocity = _diagnostic_tensor(
                result["correct"]["diagnostics"], "pred_velocity"
            )
            correct_x_after = _diagnostic_tensor(
                result["correct"]["diagnostics"], "x_after"
            )
            trajectory_chunks["initial_actions"].append(_to_numpy(result["initial_actions"]))
            trajectory_chunks["correct_actions"].append(correct_actions)
            trajectory_chunks["correct_repeat_actions"].append(
                _to_numpy(result["correct"]["repeat_actions"])
            )
            trajectory_chunks["correct_x_before"].append(
                np.swapaxes(correct_x_before, 0, 1)
            )
            trajectory_chunks["correct_velocity"].append(
                np.swapaxes(correct_velocity, 0, 1)
            )
            trajectory_chunks["correct_x_after"].append(
                np.swapaxes(correct_x_after, 0, 1)
            )

            batch_records = []
            for sample_index, identity in enumerate(batch_identities):
                batch_records.append(
                    {
                        "sample_index": batch_offset + sample_index,
                        "batch_index": batch_index,
                        **identity,
                        "correct_repeat_max_abs_error": float(
                            result["correct"]["repeat_max_abs_error"]
                        ),
                        "interventions": {},
                    }
                )

            for variant in variants:
                intervention = result["interventions"][variant]
                local_velocities = _to_numpy(intervention["local_velocities"])
                trajectory_chunks[f"{variant}__local_velocity"].append(
                    np.swapaxes(local_velocities, 0, 1)
                )
                permutation = intervention["permutation"]
                if permutation is None:
                    donor_indices = np.full(batch_size_actual, -1, dtype=np.int64)
                else:
                    donor_indices = _to_numpy(permutation).astype(np.int64) + batch_offset
                trajectory_chunks[f"{variant}__permutation"].append(donor_indices)

                local_metrics = [
                    effect_metrics(
                        correct_velocity[step_index],
                        local_velocities[step_index],
                        action_groups,
                    )
                    for step_index in range(correct_velocity.shape[0])
                ]
                rollout_metrics: dict[str, dict[str, Any]] = {}
                rollout_unnormalized_metrics: dict[str, dict[str, Any]] = {}
                hand_metrics: dict[str, dict[str, np.ndarray]] = {}
                for rollout_name, rollout in intervention["rollouts"].items():
                    rollout_actions = _to_numpy(rollout["actions"])
                    rollout_metrics[rollout_name] = effect_metrics(
                        correct_actions,
                        rollout_actions,
                        action_groups,
                    )
                    if correct_actions_unnormalized is not None:
                        rollout_unnormalized_metrics[rollout_name] = effect_metrics(
                            correct_actions_unnormalized,
                            unnormalize_action_batch(
                                rollout_actions,
                                action_norm_stats,
                            ),
                            action_groups,
                        )
                    if action_dim == 7:
                        hand_metrics[rollout_name] = binary_action_flip_metrics(
                            correct_actions,
                            rollout_actions,
                            indices=(6,),
                            threshold=0.5,
                        )
                    trajectory_chunks[f"{variant}__{rollout_name}__actions"].append(
                        rollout_actions
                    )
                    for field in ("x_before", "pred_velocity", "x_after"):
                        values = _diagnostic_tensor(rollout["diagnostics"], field)
                        key_field = "velocity" if field == "pred_velocity" else field
                        trajectory_chunks[
                            f"{variant}__{rollout_name}__{key_field}"
                        ].append(np.swapaxes(values, 0, 1))

                for sample_index, record in enumerate(batch_records):
                    donor_index = int(donor_indices[sample_index])
                    record["interventions"][variant] = {
                        "donor_sample_index": None if donor_index < 0 else donor_index,
                        "diagnostic_counterfactual": bool(
                            intervention["diagnostic_counterfactual"]
                        ),
                        "local_velocity": {
                            f"step_{step_index}": _sample_effect(metrics, sample_index)
                            for step_index, metrics in enumerate(local_metrics)
                        },
                        "final_action": {
                            rollout_name: _sample_effect(metrics, sample_index)
                            for rollout_name, metrics in rollout_metrics.items()
                        },
                    }
                    if hand_metrics:
                        record["interventions"][variant]["hand_discrete"] = {
                            rollout_name: _sample_binary_effect(metrics, sample_index)
                            for rollout_name, metrics in hand_metrics.items()
                        }
                    if rollout_unnormalized_metrics:
                        record["interventions"][variant]["final_action_unnormalized"] = {
                            rollout_name: _sample_effect(metrics, sample_index)
                            for rollout_name, metrics in rollout_unnormalized_metrics.items()
                        }
            per_sample.extend(batch_records)
    finally:
        del framework
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cluster_ids = [record["cluster_id"] for record in per_sample]
    summary: dict[str, Any] = {
        "sample_count": len(per_sample),
        "batch_count": len(batches),
        "correct_repeat_max_abs_error": max(repeat_errors, default=0.0),
        "correct_repeat_tolerance": repeat_tolerance,
        "correct_repeat_passed": True,
        "velocity_effect": {},
        "final_action_effect": {},
        "final_action_effect_unnormalized": {},
        "hand_discrete_effect": {},
    }
    step_count = len(per_sample[0]["interventions"][variants[0]]["local_velocity"])
    for variant_index, variant in enumerate(variants):
        summary["velocity_effect"][variant] = {}
        for step_index in range(step_count):
            nodes = [
                record["interventions"][variant]["local_velocity"][f"step_{step_index}"]
                for record in per_sample
            ]
            summary["velocity_effect"][variant][f"step_{step_index}"] = _summarize_effect_nodes(
                nodes,
                cluster_ids,
                seed=int(seed) + variant_index * 100 + step_index,
                resamples=bootstrap_resamples,
            )
        summary["final_action_effect"][variant] = {}
        rollout_names = tuple(per_sample[0]["interventions"][variant]["final_action"])
        for rollout_index, rollout_name in enumerate(rollout_names):
            nodes = [
                record["interventions"][variant]["final_action"][rollout_name]
                for record in per_sample
            ]
            summary["final_action_effect"][variant][rollout_name] = _summarize_effect_nodes(
                nodes,
                cluster_ids,
                seed=int(seed) + variant_index * 1000 + rollout_index,
                resamples=bootstrap_resamples,
            )
            if action_norm_stats is not None:
                nodes = [
                    record["interventions"][variant]["final_action_unnormalized"][rollout_name]
                    for record in per_sample
                ]
                summary["final_action_effect_unnormalized"].setdefault(variant, {})[
                    rollout_name
                ] = _summarize_effect_nodes(
                    nodes,
                    cluster_ids,
                    seed=int(seed) + variant_index * 1000 + rollout_index,
                    resamples=bootstrap_resamples,
                )
            if action_dim == 7:
                nodes = [
                    record["interventions"][variant]["hand_discrete"][rollout_name]
                    for record in per_sample
                ]
                summary["hand_discrete_effect"].setdefault(variant, {})[rollout_name] = (
                    _summarize_binary_nodes(
                        nodes,
                        cluster_ids,
                        seed=int(seed) + variant_index * 1000 + rollout_index,
                        resamples=bootstrap_resamples,
                    )
                )

    trajectories = {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in trajectory_chunks.items()
    }
    if len(observed_example_keys) != 1:
        raise ValueError(
            f"probe batches used inconsistent input example keys: {sorted(observed_example_keys)}"
        )
    input_example_keys = list(next(iter(observed_example_keys)))
    proprioceptive_state_present = "state" in input_example_keys
    state_conditioning_used = proprioceptive_state_present and (
        checkpoint_state_dim is None or checkpoint_state_dim > 0
    )
    config = {
        "git_commit": _git_commit(),
        "checkpoint": checkpoint,
        "config_yaml": None if config_path is None else str(config_path),
        "sample_paths": [str(path) for path in execution_paths],
        "sample_identities": identities,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "batch_count": len(batches),
        "variants": list(variants),
        "inference_steps": int(trajectories["correct_velocity"].shape[1]),
        "device": (
            next(iter(observed_devices))
            if len(observed_devices) == 1
            else sorted(observed_devices)
        ),
        "requested_device": str(device),
        "dtype": (
            next(iter(observed_dtypes))
            if len(observed_dtypes) == 1
            else sorted(observed_dtypes)
        ),
        "action_horizon": int(trajectories["correct_actions"].shape[1]),
        "action_dim": action_dim,
        "action_groups": {name: list(bounds) for name, bounds in action_groups.items()},
        "normalized_actions": True,
        "unnormalized_action_metrics": action_norm_stats is not None,
        "unnorm_key": resolved_unnorm_key,
        "bootstrap_resamples": int(bootstrap_resamples),
        "correct_repeat_tolerance": repeat_tolerance,
        "input_example_keys": input_example_keys,
        "proprioceptive_state_present": proprioceptive_state_present,
        "checkpoint_include_state": checkpoint_include_state,
        "checkpoint_state_dim": checkpoint_state_dim,
        "state_conditioning_used": state_conditioning_used,
        "intervention_permutations": {
            variant: trajectories[f"{variant}__permutation"].tolist()
            for variant in variants
        },
    }
    write_probe_artifacts(
        output_dir,
        config=config,
        summary=summary,
        per_sample=per_sample,
        trajectories=trajectories,
    )
    write_probe_figures(summary, output_dir)
    return summary
