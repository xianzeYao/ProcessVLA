from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Union

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image


def _ensure_liv_import_paths(repo_root: Union[str, Path]) -> Path:
    resolved_root = Path(repo_root).expanduser().resolve()
    if not resolved_root.exists():
        raise ImportError(f"LIV repo root does not exist: {resolved_root}")
    root_str = str(resolved_root)
    clip_str = str((resolved_root / "liv/models/clip").resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if clip_str not in sys.path:
        sys.path.insert(0, clip_str)
    return resolved_root


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_image(image_path: Union[str, Path]) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _runtime_info(*, repo_root: Path, device: str) -> dict:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torchvision_version": getattr(torchvision, "__version__", "unknown"),
        "repo_root": str(repo_root),
        "device": device,
    }


def main() -> None:
    args = _parse_args()
    resolved_root = _ensure_liv_import_paths(args.repo_root)

    import clip
    from liv import load_liv

    device = args.device
    liv_model = load_liv()
    liv_model = liv_model.eval()
    transform = T.Compose([T.ToTensor()])
    runtime = _runtime_info(repo_root=resolved_root, device=device)
    print(json.dumps({"ready": True, "runtime": runtime}), flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            image_paths = request.get("image_paths", None)
            instructions = request.get("instructions", None)
            if image_paths is None or instructions is None:
                image_paths = [request["image_path"]]
                instructions = [request["instruction"]]
            if len(image_paths) != len(instructions):
                raise ValueError(
                    f"LIV subprocess batch size mismatch: {len(image_paths)} image_paths vs {len(instructions)} instructions."
                )

            image_tensor = torch.stack(
                [transform(_load_image(image_path)) for image_path in image_paths],
                dim=0,
            ).to(device)
            text_tokens = clip.tokenize([str(instruction) for instruction in instructions]).to(device)
            with torch.no_grad():
                img_embedding = liv_model(input=image_tensor, modality="vision")
                text_embedding = liv_model(input=text_tokens, modality="text")
                sim_model = liv_model.module if isinstance(liv_model, torch.nn.DataParallel) else liv_model
                signal = sim_model.sim(img_embedding, text_embedding)

            signal_tensor = torch.as_tensor(signal, dtype=torch.float32).reshape(-1)
            if signal_tensor.numel() == 1 and len(instructions) > 1:
                signal_tensor = signal_tensor.expand(len(instructions))
            if signal_tensor.numel() != len(instructions):
                raise ValueError(
                    f"LIV subprocess output size mismatch: got {signal_tensor.numel()}, expected {len(instructions)}."
                )
            signal_values = [float(item) for item in signal_tensor.detach().cpu().tolist()]
            print(json.dumps({"signals": signal_values, "runtime": runtime}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc), "runtime": runtime}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "runtime": {
                        "python_executable": sys.executable,
                        "python_version": sys.version.split()[0],
                    },
                }
            ),
            flush=True,
        )
