"""Render offline attention diagnostics for a dual-view V2 q0+depth checkpoint."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from examples.simBenchmarks.CoT.attention_probe.v2_attention_probe import (
    DiTCrossAttentionCollector,
    QwenFullAttentionCollector,
    aggregate_attention_records,
    attention_to_patch_map,
    build_condition_groups,
    modality_attention_mass,
)
from examples.simBenchmarks.CoT.geometry_probe.paired_probe import (
    build_geometry_example,
    load_materialized_sample,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _squeeze_map(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"depth map must resolve to [H,W], got {array.shape}")
    return array


def _canonical_uvd(
    value: Any,
    *,
    points_per_hand: int,
    hand_count: int,
) -> np.ndarray:
    """Restore V2 time-major UVD tokens to ``[time, hand, uvd]``."""

    array = _to_numpy(value).astype(np.float32, copy=False)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    expected = (int(points_per_hand), int(hand_count), 3)
    if array.shape == expected:
        return array
    if array.shape == (expected[0] * expected[1], 3):
        return array.reshape(expected)
    raise ValueError(f"UVD must resolve to {expected} or flat time-major tokens, got {array.shape}")


def _display_image(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _overlay_attention(axis, image: np.ndarray, heatmap: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.imshow(
        heatmap,
        cmap="magma",
        alpha=0.55,
        interpolation="bilinear",
        extent=(0, image.shape[1], image.shape[0], 0),
    )
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _plot_uvd(axis, image: np.ndarray, uvd: np.ndarray, title: str) -> None:
    axis.imshow(image)
    points = np.asarray(uvd)
    if points.ndim == 2 and points.shape[-1] == 3:
        points = points[:, None, :]
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"UVD plot expects [time,hand,3], got {points.shape}")
    colors = ("cyan", "lime", "orange", "magenta")
    hand_names = ("left", "right")
    for hand_index in range(points.shape[1]):
        track = points[:, hand_index]
        color = colors[hand_index % len(colors)]
        label = hand_names[hand_index] if hand_index < len(hand_names) else f"hand {hand_index}"
        axis.plot(
            track[:, 0] * image.shape[1],
            track[:, 1] * image.shape[0],
            "o-",
            color=color,
            markeredgecolor="black",
            linewidth=2,
            markersize=5,
            label=label,
        )
        for index, point in enumerate(track):
            axis.text(
                point[0] * image.shape[1],
                point[1] * image.shape[0],
                str(index),
                color="white",
                fontsize=7,
            )
    if points.shape[1] > 1:
        axis.legend(loc="best", fontsize=7)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _jsonable_groups(groups: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    return {name: np.asarray(indices, dtype=np.int64).tolist() for name, indices in groups.items()}


def _optional_state(framework: Any, example: Mapping[str, Any], condition: torch.Tensor):
    """Mirror V2 inference: use state only when both model and example provide it."""

    state_dim = int(framework.config.framework.action_model.get("state_dim", 0))
    if not state_dim or "state" not in example:
        return None
    state = torch.as_tensor(
        example["state"], device=condition.device, dtype=condition.dtype
    )
    if state.ndim == 1:
        state = state[None, None, :]
    elif state.ndim == 2:
        state = state[None, ...]
    return state[..., :state_dim]


def _render_report(
    path: Path,
    *,
    sample: Mapping[str, Any],
    predicted: Mapping[str, np.ndarray],
    qwen_maps: Mapping[str, Mapping[str, np.ndarray]],
    action_maps: Mapping[str, np.ndarray],
    action_mass: Mapping[str, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = [_display_image(image) for image in sample["images"]]
    if len(images) not in (1, 2):
        raise ValueError(f"V2 probe supports one or two images, got {len(images)}")
    agentview = images[0]
    wrist = images[1] if len(images) == 2 else None
    figure, axes = plt.subplots(4, 4, figsize=(18, 16), constrained_layout=True)

    axes[0, 0].imshow(agentview)
    axes[0, 0].set_title("agentview RGB")
    if wrist is not None:
        axes[0, 1].imshow(wrist)
        axes[0, 1].set_title("wrist RGB")
    else:
        _plot_uvd(axes[0, 1], agentview, predicted["uvd"], "pred two-hand UVD")
    axes[0, 2].imshow(predicted["depth_current"], cmap="viridis")
    axes[0, 2].set_title("pred current depth")
    axes[0, 3].imshow(predicted["depth_future"], cmap="viridis")
    axes[0, 3].set_title("pred future depth")

    axes[1, 0].imshow(_squeeze_map(sample["depth_current"]), cmap="viridis")
    axes[1, 0].set_title("GT current depth")
    axes[1, 1].imshow(_squeeze_map(sample["depth_future"]), cmap="viridis")
    axes[1, 1].set_title("GT future depth")
    _plot_uvd(
        axes[1, 2],
        agentview,
        predicted["uvd"] if wrist is not None else sample["uvd"],
        "pred UVD on agentview" if wrist is not None else "GT two-hand UVD",
    )
    names = list(action_mass)
    axes[1, 3].barh(names, [action_mass[name] for name in names])
    axes[1, 3].set_xlim(0.0, max(0.01, max(action_mass.values(), default=0.01) * 1.15))
    axes[1, 3].set_title("Action Expert attention mass")
    axes[1, 3].invert_yaxis()

    query_names = ("depth_current", "depth_future", "uvd")
    for column, query_name in enumerate(query_names):
        _overlay_attention(
            axes[2, column], agentview, qwen_maps[query_name]["agentview"],
            f"Qwen {query_name} → agentview",
        )
    _overlay_attention(axes[2, 3], agentview, action_maps["agentview"], "Action → agentview")
    if wrist is not None:
        for column, query_name in enumerate(query_names):
            _overlay_attention(
                axes[3, column], wrist, qwen_maps[query_name]["wrist"],
                f"Qwen {query_name} → wrist",
            )
        _overlay_attention(axes[3, 3], wrist, action_maps["wrist"], "Action → wrist")
    else:
        pred_uvd = np.asarray(predicted["uvd"])
        gt_uvd = np.asarray(sample["uvd"])
        hand_names = ("left", "right")
        for hand_index in range(pred_uvd.shape[1]):
            label = hand_names[hand_index] if hand_index < len(hand_names) else f"hand {hand_index}"
            axes[3, 0].plot(pred_uvd[:, hand_index, 2], "o-", label=f"pred {label}")
            axes[3, 0].plot(gt_uvd[:, hand_index, 2], "--", label=f"GT {label}")
        axes[3, 0].set_title("UVD depth over time")
        axes[3, 0].set_xlabel("trajectory point")
        axes[3, 0].set_ylabel("depth (m)")
        axes[3, 0].legend(fontsize=7)
        axes[3, 1].text(
            0.5, 0.5,
            "Single-view RoboCasa input\nNo wrist tokens are present or fabricated.",
            ha="center", va="center", fontsize=12,
        )
        for axis in axes[3, 1:]:
            axis.axis("off")
    for axis in axes[:2, :3].flat:
        axis.axis("off")
    figure.suptitle(str(sample["language"]), fontsize=12)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    from starVLA.model.framework.base_framework import baseframework

    checkpoint = args.checkpoint.resolve()
    sample_path = args.sample_npz.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    sample = load_materialized_sample(sample_path)
    example = build_geometry_example(sample)
    if len(example["image"]) not in (1, 2):
        raise ValueError(
            f"this probe supports one or two image views; sample has {len(example['image'])}"
        )

    framework = baseframework.from_pretrained(str(checkpoint)).to(args.device).eval()
    if type(framework).__name__ != "Qwen_GR00T_CoT_V2":
        raise TypeError(f"expected Qwen_GR00T_CoT_V2, got {type(framework).__name__}")
    if not bool(framework.include_depth_in_action_condition):
        raise ValueError("checkpoint is V2 but does not feed depth tokens to Action Expert")

    qwen_inputs, native_mask = framework._build_native_inputs([example], inference=True)
    native_count = int(qwen_inputs["input_ids"].shape[1])
    geometry_count = int(framework.geometry_layout.geometry_token_count)
    geometry_indices = torch.arange(native_count, native_count + geometry_count)
    language_model = framework.qwen_vl_interface.model.model.language_model
    qwen_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda")
        else nullcontext()
    )
    with torch.inference_mode(), QwenFullAttentionCollector(
        language_model, geometry_indices
    ) as qwen_collector, qwen_context:
        split = framework._run_geometry_backbone(qwen_inputs)
        depth_current, depth_future, uvd = framework._decode_geometry(split, qwen_inputs)

    condition, condition_mask = framework._build_action_condition(
        split, native_attention_mask=native_mask
    )
    action_dtype = next(framework.action_model.parameters()).dtype
    condition = condition.to(dtype=action_dtype)
    state = _optional_state(framework, example, condition)
    torch.manual_seed(int(args.seed))
    with torch.inference_mode(), DiTCrossAttentionCollector(
        framework.action_model.model
    ) as action_collector:
        actions = framework.action_model.predict_action(
            condition, state, encoder_attention_mask=condition_mask
        )

    qwen_summary = aggregate_attention_records(qwen_collector.records)
    action_summary = aggregate_attention_records(action_collector.records)
    input_ids = qwen_inputs["input_ids"][0].detach().cpu()
    native_mask_row = None if native_mask is None else native_mask[0].detach().cpu()
    image_token_id = int(framework.qwen_vl_interface.model.config.image_token_id)
    action_groups = build_condition_groups(
        input_ids,
        image_token_id=image_token_id,
        native_attention_mask=native_mask_row,
        depth_query_count=int(framework.geometry_layout.depth_query_count),
        uvd_token_count=int(framework.geometry_layout.uvd_token_count),
        include_depth=True,
    )
    qwen_groups = build_condition_groups(
        input_ids,
        image_token_id=image_token_id,
        native_attention_mask=native_mask_row,
        depth_query_count=int(framework.geometry_layout.depth_query_count),
        uvd_token_count=int(framework.geometry_layout.uvd_token_count),
        include_depth=True,
    )
    view_names = tuple(
        name for name in ("agentview", "wrist") if name in action_groups
    )

    depth_count = int(framework.geometry_layout.depth_query_count)
    qwen_attention = qwen_summary["mean"]
    query_slices = {
        "depth_current": slice(0, depth_count),
        "depth_future": slice(depth_count, 2 * depth_count),
        "uvd": slice(2 * depth_count, geometry_count),
    }
    qwen_maps: dict[str, dict[str, np.ndarray]] = {}
    for query_name, query_slice in query_slices.items():
        qwen_maps[query_name] = {}
        for view_name in view_names:
            qwen_maps[query_name][view_name] = attention_to_patch_map(
                qwen_attention[query_slice], key_indices=qwen_groups[view_name]
            )
    action_maps = {
        view_name: attention_to_patch_map(
            action_summary["mean"], key_indices=action_groups[view_name]
        )
        for view_name in view_names
    }
    action_mass = modality_attention_mass(action_summary["mean"], action_groups)
    points_per_hand = int(framework.geometry_layout.uvd_points_per_hand)
    hand_count = int(framework.geometry_layout.hand_count)
    predicted = {
        "depth_current": _squeeze_map(depth_current),
        "depth_future": _squeeze_map(depth_future),
        "uvd": _canonical_uvd(
            uvd, points_per_hand=points_per_hand, hand_count=hand_count
        ),
        "actions": _to_numpy(actions)[0],
    }
    sample = dict(sample)
    sample["uvd"] = _canonical_uvd(
        sample["uvd"], points_per_hand=points_per_hand, hand_count=hand_count
    )

    report_path = output_dir / "report.png"
    raw_path = output_dir / "raw_attention.npz"
    summary_path = output_dir / "summary.json"
    _render_report(
        report_path,
        sample=sample,
        predicted=predicted,
        qwen_maps=qwen_maps,
        action_maps=action_maps,
        action_mass=action_mass,
    )
    raw_payload: dict[str, np.ndarray] = {
        "qwen_attention_records": qwen_summary["records"].numpy(),
        "qwen_layers": qwen_summary["layers"].numpy(),
        "action_attention_records": action_summary["records"].numpy(),
        "action_layers": action_summary["layers"].numpy(),
        "action_denoise_calls": action_summary["calls"].numpy(),
        **{f"pred_{name}": value for name, value in predicted.items()},
    }
    for query_name, view_maps in qwen_maps.items():
        for view_name, value in view_maps.items():
            raw_payload[f"qwen_{query_name}_to_{view_name}"] = value
    for view_name, value in action_maps.items():
        raw_payload[f"action_to_{view_name}"] = value
    np.savez_compressed(raw_path, **raw_payload)

    summary = {
        "checkpoint": str(checkpoint),
        "sample_npz": str(sample_path),
        "language": str(sample["language"]),
        "seed": int(args.seed),
        "device": str(args.device),
        "native_token_count": native_count,
        "geometry_token_count": geometry_count,
        "views": list(view_names),
        "uvd_layout": {
            "time_points": points_per_hand,
            "hand_count": hand_count,
            "order": "time_major",
        },
        "qwen_full_attention_layers": qwen_summary["layers"].tolist(),
        "action_cross_attention_layers": sorted(set(action_summary["layers"].tolist())),
        "action_denoise_steps": int(action_summary["calls"].max().item()) + 1,
        "condition_groups": _jsonable_groups(action_groups),
        "action_attention_mass": action_mass,
        "outputs": {
            "report_png": str(report_path),
            "raw_npz": str(raw_path),
            "summary_json": str(summary_path),
        },
        "note": (
            "Attention is descriptive, not causal. Qwen3.5 linear-attention layers do not "
            "define a dense token-to-token matrix, so only full-attention layers are included."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    summary = run_probe(build_parser().parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
