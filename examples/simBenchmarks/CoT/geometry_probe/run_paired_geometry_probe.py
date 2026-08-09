#!/usr/bin/env python3
"""Run a fixed, episode-balanced geometry comparison for LIBERO or RoboCasa."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from examples.simBenchmarks.CoT.geometry_probe.dataset_probe import (
    LiberoRerenderStore,
    RoboCasaRerenderStore,
)
from examples.simBenchmarks.CoT.geometry_probe.paired_probe import (
    load_materialized_sample,
    load_prediction,
    materialize_samples,
    run_checkpoint,
    write_paired_results,
)
from examples.simBenchmarks.CoT.geometry_probe.probe_utils import (
    SampleRef,
    build_episode_balanced_sample_plan,
)
from examples.simBenchmarks.CoT.geometry_probe.visualization import (
    save_paired_sample_figure,
)
from starVLA.dataloader.robocasa_fourier_tasks import fourier_dataset_names


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_root: Path
    groups: tuple[str, ...]
    samples_per_group: int
    horizon: int
    uvd_num_points: int
    hand_count: int
    checkpoint_a: Path
    checkpoint_b: Path
    label_a: str
    label_b: str

    @property
    def sample_count(self) -> int:
        return len(self.groups) * self.samples_per_group


def benchmark_spec(name: str) -> BenchmarkSpec:
    """Return the agreed 240-sample comparison definition."""

    if name == "libero":
        return BenchmarkSpec(
            name="libero",
            dataset_root=Path("/root/data/yxz/datasets/libero_rerender"),
            groups=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
            samples_per_group=60,
            horizon=8,
            uvd_num_points=4,
            hand_count=1,
            checkpoint_a=Path(
                "/root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v1_5_60k_4gpu/"
                "checkpoints/steps_60000_pytorch_model.pt"
            ),
            checkpoint_b=Path(
                "/root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v2_8gpu_bs16/"
                "checkpoints/steps_60000_pytorch_model.pt"
            ),
            label_a="v1_5",
            label_b="v2",
        )
    if name == "robocasa":
        return BenchmarkSpec(
            name="robocasa",
            dataset_root=Path("/root/data/yxz/datasets/robocasa_fourier_rerender"),
            groups=tuple(fourier_dataset_names()),
            samples_per_group=10,
            horizon=16,
            uvd_num_points=6,
            hand_count=2,
            checkpoint_a=Path(
                "/root/data/yxz/outputs/qwen35_gr00t_robocasa_fourier_CoT_v1/"
                "checkpoints/steps_100000_pytorch_model.pt"
            ),
            checkpoint_b=Path(
                "/root/data/yxz/outputs/qwen35_gr00t_robocasa_fourier_CoT_v2_8gpu_bs16/"
                "checkpoints/steps_100000_pytorch_model.pt"
            ),
            label_a="v1",
            label_b="v2",
        )
    raise ValueError(f"unknown benchmark: {name!r}")


def _build_store(spec: BenchmarkSpec, dataset_root: Path, video_backend: str) -> Any:
    common = {
        "image_size": 224,
        "horizon": spec.horizon,
        "uvd_num_points": spec.uvd_num_points,
        "video_backend": str(video_backend),
    }
    if spec.name == "libero":
        return LiberoRerenderStore(dataset_root, spec.groups, **common)
    return RoboCasaRerenderStore(dataset_root, spec.groups, **common)


def _save_manifest(
    path: Path,
    *,
    spec: BenchmarkSpec,
    dataset_root: Path,
    seed: int,
    plan: Sequence[SampleRef],
) -> None:
    payload = {
        "benchmark": spec.name,
        "dataset_root": str(dataset_root),
        "seed": int(seed),
        "samples_per_group": spec.samples_per_group,
        "horizon": spec.horizon,
        "uvd_num_points": spec.uvd_num_points,
        "groups": list(spec.groups),
        "samples": [asdict(ref) for ref in plan],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_manifest(path: Path, spec: BenchmarkSpec) -> list[SampleRef]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark") != spec.name:
        raise ValueError(
            f"manifest benchmark mismatch: {payload.get('benchmark')!r} vs {spec.name!r}"
        )
    plan = [SampleRef(**row) for row in payload["samples"]]
    expected = spec.sample_count
    if len(plan) != expected:
        raise ValueError(f"manifest has {len(plan)} samples, expected {expected}")
    counts = {group: sum(ref.suite == group for ref in plan) for group in spec.groups}
    expected_counts = {group: spec.samples_per_group for group in spec.groups}
    if counts != expected_counts:
        raise ValueError(f"manifest is not evenly balanced: {counts}")
    return plan


def _materialize_or_reuse(store: Any, plan: Sequence[SampleRef], directory: Path) -> list[Path]:
    expected = [directory / f"sample_{index:04d}.npz" for index in range(len(plan))]
    if expected and all(path.exists() for path in expected):
        return expected
    return materialize_samples(store, plan, directory)


def _prediction_paths(directory: Path, label: str, count: int) -> list[Path]:
    return [directory / label / f"sample_{index:04d}.npz" for index in range(count)]


def _select_figure_indices(summary: dict[str, Any], groups: Sequence[str], budget: int) -> list[int]:
    budget = max(int(budget), 0)
    if budget == 0:
        return []
    selected: list[int] = []
    for group in groups:
        match = next(
            (row["sample_index"] for row in summary["samples"] if row.get("suite") == group),
            None,
        )
        if match is not None and match not in selected:
            selected.append(int(match))
            if len(selected) >= budget:
                return selected
    labels = summary["labels"]
    remaining = sorted(
        summary["samples"],
        key=lambda row: max(
            float(row["metrics"][label].get("uvd_uv_ade_px", float("-inf")))
            for label in labels
        ),
        reverse=True,
    )
    for row in remaining:
        index = int(row["sample_index"])
        if index not in selected:
            selected.append(index)
            if len(selected) >= budget:
                break
    return selected


def _write_figures(
    *,
    summary: dict[str, Any],
    sample_paths: Sequence[Path],
    prediction_paths: dict[str, Sequence[Path]],
    groups: Sequence[str],
    output_dir: Path,
    budget: int,
) -> list[Path]:
    labels = list(summary["labels"])
    row_by_index = {int(row["sample_index"]): row for row in summary["samples"]}
    figure_dir = output_dir / "figures"
    outputs: list[Path] = []
    for index in _select_figure_indices(summary, groups, budget):
        sample = load_materialized_sample(sample_paths[index])
        predictions = {
            label: load_prediction(prediction_paths[label][index]) for label in labels
        }
        path = figure_dir / f"sample_{index:04d}.png"
        outputs.append(
            save_paired_sample_figure(
                path,
                sample=sample,
                predictions=predictions,
                labels=labels,
                metrics=row_by_index[index]["metrics"],
            )
        )
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, choices=("libero", "robocasa"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint-a", type=Path)
    parser.add_argument("--checkpoint-b", type=Path)
    parser.add_argument("--label-a")
    parser.add_argument("--label-b")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path, help="Reuse an existing fixed sample manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-figures", type=int, default=24)
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=("pyav", "decord", "torchvision_av"),
        help="pyav is the portable default; torchvision_av requires torchvision VideoReader",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and validate only the manifest")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = benchmark_spec(args.bench)
    dataset_root = args.dataset_root or spec.dataset_root
    checkpoint_a = args.checkpoint_a or spec.checkpoint_a
    checkpoint_b = args.checkpoint_b or spec.checkpoint_b
    label_a = args.label_a or spec.label_a
    label_b = args.label_b or spec.label_b
    if label_a == label_b:
        raise ValueError("paired labels must be distinct")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(
        f"/root/data/yxz/outputs/paired_geometry_probe_{spec.name}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    store = _build_store(spec, dataset_root, args.video_backend)
    if args.manifest:
        plan = _load_manifest(args.manifest, spec)
    else:
        plan = build_episode_balanced_sample_plan(
            store.episode_refs(),
            samples_per_group=spec.samples_per_group,
            horizon=spec.horizon,
            seed=args.seed,
        )
    manifest_path = output_dir / "manifest.json"
    _save_manifest(
        manifest_path, spec=spec, dataset_root=dataset_root, seed=args.seed, plan=plan
    )
    counts = {group: sum(ref.suite == group for ref in plan) for group in spec.groups}
    print(json.dumps({"benchmark": spec.name, "samples": len(plan), "counts": counts}, indent=2))
    if args.dry_run:
        print(f"validated manifest: {manifest_path}")
        return

    for checkpoint in (checkpoint_a, checkpoint_b):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    sample_paths = _materialize_or_reuse(store, plan, output_dir / "samples")
    predictions_dir = output_dir / "predictions"
    paths_a = _prediction_paths(predictions_dir, label_a, len(plan))
    if not all(path.exists() for path in paths_a):
        paths_a = run_checkpoint(
            checkpoint_a,
            label=label_a,
            sample_paths=sample_paths,
            output_dir=predictions_dir,
            device=args.device,
        )
    paths_b = _prediction_paths(predictions_dir, label_b, len(plan))
    if not all(path.exists() for path in paths_b):
        paths_b = run_checkpoint(
            checkpoint_b,
            label=label_b,
            sample_paths=sample_paths,
            output_dir=predictions_dir,
            device=args.device,
        )
    prediction_paths = {label_a: paths_a, label_b: paths_b}
    summary = write_paired_results(
        sample_paths=sample_paths,
        prediction_paths=prediction_paths,
        labels=[label_a, label_b],
        output_dir=output_dir / "results",
        image_size=224,
    )
    figures = _write_figures(
        summary=summary,
        sample_paths=sample_paths,
        prediction_paths=prediction_paths,
        groups=spec.groups,
        output_dir=output_dir,
        budget=args.num_figures,
    )
    run_config = {
        "spec": {
            **asdict(spec),
            "dataset_root": str(dataset_root),
            "checkpoint_a": str(checkpoint_a),
            "checkpoint_b": str(checkpoint_b),
            "label_a": label_a,
            "label_b": label_b,
        },
        "seed": args.seed,
        "device": args.device,
        "video_backend": args.video_backend,
        "manifest": str(manifest_path),
        "figures": [str(path) for path in figures],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], indent=2))
    print(f"artifacts: {output_dir}")


if __name__ == "__main__":
    main()
