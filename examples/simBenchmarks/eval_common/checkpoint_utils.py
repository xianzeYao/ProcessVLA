#!/usr/bin/env python3
"""Resolve a StarVLA training output directory to a usable checkpoint."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


STEP_PATTERN = re.compile(r"^steps_(?P<step>\d+)_pytorch_model\.pt$")


def resolve_checkpoint(model_dir: str | Path, checkpoint: str | None = None) -> Path:
    """Return a checkpoint and verify its sidecar config/statistics files.

    With no explicit checkpoint, the numerically largest ``steps_*`` file is
    selected. If no numbered checkpoint exists, ``final_model/pytorch_model.pt``
    is used. Relative overrides are resolved inside ``model_dir``.
    """
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")

    if checkpoint:
        candidate = Path(checkpoint).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
    else:
        candidates = []
        checkpoint_dir = root / "checkpoints"
        if checkpoint_dir.is_dir():
            for path in checkpoint_dir.glob("steps_*_pytorch_model.pt"):
                match = STEP_PATTERN.fullmatch(path.name)
                if match:
                    candidates.append((int(match.group("step")), path))
        if candidates:
            candidate = max(candidates, key=lambda item: item[0])[1]
        else:
            candidate = root / "final_model/pytorch_model.pt"

    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {candidate}")
    run_dir = candidate.parents[1]
    for sidecar in ("config.yaml", "dataset_statistics.json"):
        sidecar_path = run_dir / sidecar
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                f"{sidecar} is required next to checkpoint run directory {run_dir}: {sidecar_path}"
            )
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    print(resolve_checkpoint(args.model_dir, args.checkpoint))


if __name__ == "__main__":
    main()
