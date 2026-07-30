"""Per-objective gradient attribution for geometric CoT V2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ObjectiveMap = Mapping[str, tuple[torch.Tensor, float]]


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a potentially multi-gigabyte checkpoint without loading it into RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def select_probe_sample_indices(
    *,
    dataset_length: int,
    sample_count: int,
    seed: int,
    sample_indices: str | Sequence[int] | None = None,
) -> list[int]:
    """Choose deterministic unique dataset indices or validate an explicit list."""

    dataset_length = int(dataset_length)
    sample_count = int(sample_count)
    if dataset_length < 1 or sample_count < 1:
        raise ValueError("dataset_length and sample_count must be positive")
    if sample_indices is not None:
        if isinstance(sample_indices, str):
            selected = [int(value.strip()) for value in sample_indices.split(",") if value.strip()]
        else:
            selected = [int(value) for value in sample_indices]
        if len(selected) != sample_count:
            raise ValueError(
                f"sample_indices contains {len(selected)} entries, expected {sample_count}"
            )
        if len(set(selected)) != len(selected):
            raise ValueError("sample_indices must be unique")
        if any(index < 0 or index >= dataset_length for index in selected):
            raise ValueError(f"sample_indices must be within [0, {dataset_length})")
        return selected
    if sample_count > dataset_length:
        raise ValueError(
            f"cannot sample {sample_count} unique examples from dataset length {dataset_length}"
        )
    rng = np.random.default_rng(int(seed))
    return [int(index) for index in rng.choice(dataset_length, size=sample_count, replace=False)]


def summarize_probe_batches(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recursively summarize numeric leaves over probe batches."""

    if not records:
        raise ValueError("probe records must not be empty")

    def summarize(values: Sequence[Any]) -> Any:
        dictionaries = [value for value in values if isinstance(value, Mapping)]
        if dictionaries:
            keys = sorted({key for value in dictionaries for key in value})
            return {
                key: summarize([value[key] for value in dictionaries if key in value])
                for key in keys
            }
        numeric = [
            float(value)
            for value in values
            if isinstance(value, (bool, int, float, np.number))
        ]
        if not numeric:
            return None
        array = np.asarray(numeric, dtype=np.float64)
        finite_mask = np.isfinite(array)
        finite_values = array[finite_mask]
        if finite_values.size == 0:
            raise ValueError("probe summary received no finite values")
        return {
            "mean": float(finite_values.mean()),
            "std": float(finite_values.std()),
            "median": float(np.median(finite_values)),
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
            "count": int(array.size),
            "valid_count": int(finite_values.size),
            "finite": bool(finite_mask.all()),
            "nonzero_count": int(np.count_nonzero(finite_values)),
        }

    return summarize(records)


def _unwrap_module(model: nn.Module) -> nn.Module:
    while isinstance(getattr(model, "module", None), nn.Module):
        model = model.module
    return model


def extract_v2_objectives(
    output: Mapping[str, torch.Tensor],
    model: Any,
) -> dict[str, tuple[torch.Tensor, float]]:
    """Return the five non-overlapping V2 objectives and effective weights."""

    key_map = {
        "action": "action_loss",
        "depth_current": "depth_current_loss",
        "depth_future": "depth_future_loss",
        "uvd_absolute": "uvd_absolute_loss",
        "uvd_relative": "uvd_relative_loss",
    }
    missing = [key for key in key_map.values() if key not in output]
    if missing:
        raise KeyError(f"V2 gradient probe requires objective keys: {missing}")
    weights = {
        "action": float(model.lambda_action),
        "depth_current": float(model.lambda_depth_current),
        "depth_future": float(model.lambda_depth_future),
        "uvd_absolute": float(model.lambda_uvd),
        "uvd_relative": float(model.lambda_uvd) * float(model.lambda_uvd_relative),
    }
    objectives = {}
    for name, key in key_map.items():
        loss = output[key]
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise TypeError(f"{key} must be a scalar tensor")
        if not bool(torch.isfinite(loss.detach()).item()):
            raise ValueError(f"{key} is non-finite")
        objectives[name] = (loss, weights[name])
    return objectives


def select_shared_parameter_groups(
    model: nn.Module,
    *,
    qwen_tail_layers: int = 2,
) -> tuple[dict[str, list[nn.Parameter]], dict[str, dict[str, Any]]]:
    """Select trainable geometry tokens and final Qwen transformer layers."""

    model = _unwrap_module(model)
    geometry_module = getattr(model, "geometry_tokens", None)
    if not isinstance(geometry_module, nn.Module):
        raise ValueError("V2 model has no geometry_tokens module")

    groups: dict[str, list[nn.Parameter]] = {}
    metadata: dict[str, dict[str, Any]] = {}

    geometry_named = [
        (f"geometry_tokens.{name}", parameter)
        for name, parameter in geometry_module.named_parameters()
        if parameter.requires_grad
    ]
    if not geometry_named:
        raise ValueError("geometry_tokens contains no trainable parameters")
    groups["geometry_tokens"] = [parameter for _, parameter in geometry_named]
    metadata["geometry_tokens"] = {
        "parameter_names": [name for name, _ in geometry_named],
        "parameter_count": len(geometry_named),
        "numel": sum(parameter.numel() for _, parameter in geometry_named),
    }

    qwen_tail_layers = int(qwen_tail_layers)
    if qwen_tail_layers < 0:
        raise ValueError(f"qwen_tail_layers must be non-negative, got {qwen_tail_layers}")
    if qwen_tail_layers:
        try:
            layers = model.qwen_vl_interface.model.model.language_model.layers
        except AttributeError as exc:
            raise ValueError(
                "cannot resolve qwen_vl_interface.model.model.language_model.layers"
            ) from exc
        if qwen_tail_layers > len(layers):
            raise ValueError(
                f"requested {qwen_tail_layers} Qwen tail layers, model has {len(layers)}"
            )
        start = len(layers) - qwen_tail_layers
        qwen_named = []
        for layer_index in range(start, len(layers)):
            for name, parameter in layers[layer_index].named_parameters():
                if parameter.requires_grad:
                    qwen_named.append(
                        (
                            f"qwen_vl_interface.model.model.language_model.layers.{layer_index}.{name}",
                            parameter,
                        )
                    )
        if not qwen_named:
            raise ValueError("selected Qwen tail contains no trainable parameters")
        groups["qwen_tail"] = [parameter for _, parameter in qwen_named]
        metadata["qwen_tail"] = {
            "parameter_names": [name for name, _ in qwen_named],
            "parameter_count": len(qwen_named),
            "numel": sum(parameter.numel() for _, parameter in qwen_named),
        }
    return groups, metadata


def _group_statistics(
    gradients: Sequence[torch.Tensor],
    action_gradients: Sequence[torch.Tensor],
    indices: Sequence[int],
) -> tuple[float, float, float, bool]:
    norm_square = torch.zeros((), device=gradients[0].device, dtype=torch.float64)
    action_square = torch.zeros_like(norm_square)
    dot = torch.zeros_like(norm_square)
    for index in indices:
        gradient = gradients[index]
        action = action_gradients[index]
        norm_square += gradient.double().square().sum()
        action_square += action.double().square().sum()
        dot += (gradient.double() * action.double()).sum()
    norm = norm_square.sqrt()
    action_norm = action_square.sqrt()
    valid = bool((norm > 0).item() and (action_norm > 0).item())
    cosine = dot / (norm * action_norm) if valid else dot.new_zeros(())
    return float(norm.item()), float(action_norm.item()), float(cosine.item()), valid


def measure_objective_gradients(
    objectives: ObjectiveMap,
    parameter_groups: Mapping[str, Sequence[nn.Parameter]],
) -> dict[str, Any]:
    """Measure per-loss norms/cosines and the weighted auxiliary sum."""

    if "action" not in objectives:
        raise KeyError("objectives must contain action")
    if not parameter_groups:
        raise ValueError("parameter_groups must not be empty")

    unique_parameters: list[nn.Parameter] = []
    parameter_indices: dict[int, int] = {}
    group_indices: dict[str, list[int]] = {}
    for group_name, parameters in parameter_groups.items():
        indices = []
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            identity = id(parameter)
            if identity not in parameter_indices:
                parameter_indices[identity] = len(unique_parameters)
                unique_parameters.append(parameter)
            indices.append(parameter_indices[identity])
        if not indices:
            raise ValueError(f"parameter group {group_name!r} is empty")
        group_indices[group_name] = indices
    group_indices["shared_total"] = list(range(len(unique_parameters)))
    if not unique_parameters:
        raise ValueError("selected shared parameter scope is empty")

    def gradient_vector(loss: torch.Tensor) -> list[torch.Tensor]:
        raw = torch.autograd.grad(
            loss,
            unique_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        return [
            torch.zeros_like(parameter, dtype=torch.float32)
            if gradient is None
            else gradient.detach().float()
            for parameter, gradient in zip(unique_parameters, raw)
        ]

    action_loss, action_weight = objectives["action"]
    action_gradients = gradient_vector(action_loss)
    combined_aux = [torch.zeros_like(gradient) for gradient in action_gradients]
    output = {
        "groups": {
            group_name: {"objectives": {}}
            for group_name in group_indices
        }
    }

    ordered_names = ["action"] + [name for name in objectives if name != "action"]
    for objective_name in ordered_names:
        loss, weight = objectives[objective_name]
        weight = float(weight)
        if not torch.isfinite(torch.tensor(weight)) or weight < 0.0:
            raise ValueError(f"objective weight for {objective_name} must be finite and non-negative")
        gradients = action_gradients if objective_name == "action" else gradient_vector(loss)
        if objective_name != "action":
            for combined, gradient in zip(combined_aux, gradients):
                combined.add_(gradient, alpha=weight)
        for group_name, indices in group_indices.items():
            raw_norm, _, cosine, cosine_valid = _group_statistics(
                gradients,
                action_gradients,
                indices,
            )
            output["groups"][group_name]["objectives"][objective_name] = {
                "raw_loss": float(loss.detach().item()),
                "weight": weight,
                "weighted_loss": float(loss.detach().item()) * weight,
                "raw_grad_norm": raw_norm,
                "weighted_grad_norm": raw_norm * abs(weight),
                "cosine_with_action": cosine,
                "cosine_valid": cosine_valid,
            }
        if objective_name != "action":
            del gradients

    for group_name, indices in group_indices.items():
        aux_norm, action_norm, cosine, cosine_valid = _group_statistics(
            combined_aux,
            action_gradients,
            indices,
        )
        weighted_action_norm = action_norm * abs(float(action_weight))
        output["groups"][group_name]["combined_aux"] = {
            "weighted_grad_norm": aux_norm,
            "weighted_action_grad_norm": weighted_action_norm,
            "aux_to_action_grad_norm": (
                aux_norm / weighted_action_norm if weighted_action_norm > 0.0 else 0.0
            ),
            "cosine_with_action": cosine,
            "cosine_valid": cosine_valid,
        }
    return output
