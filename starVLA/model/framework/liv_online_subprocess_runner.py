from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Union

import torch
from PIL import Image


def _ensure_liv_import_paths(repo_root: Union[str, Path]) -> None:
    resolved_root = Path(repo_root).expanduser().resolve()
    if not resolved_root.exists():
        raise ImportError(f"LIV repo root does not exist: {resolved_root}")
    root_str = str(resolved_root)
    clip_str = str((resolved_root / "liv/models/clip").resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if clip_str not in sys.path:
        sys.path.insert(0, clip_str)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_image(image_path: Union[str, Path]) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _pil_to_float_tensor(image: Image.Image, *, device: torch.device) -> torch.Tensor:
    image = image.convert("RGB")
    channels = len(image.getbands())
    tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = tensor.view(image.size[1], image.size[0], channels)
    tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=torch.float32).div(255.0)


def main() -> None:
    args = _parse_args()
    _ensure_liv_import_paths(args.repo_root)

    import clip
    from liv import load_liv

    device = torch.device(args.device)
    liv_model = load_liv()
    if isinstance(liv_model, torch.nn.DataParallel):
        liv_model = liv_model.module
    liv_model = liv_model.to(device).eval()

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
            images = [_load_image(image_path) for image_path in image_paths]
            image_tensor = torch.stack(
                [_pil_to_float_tensor(image, device=device) for image in images],
                dim=0,
            )
            text_tokens = clip.tokenize([str(instruction) for instruction in instructions]).to(device=device)
            with torch.no_grad():
                img_embedding = liv_model(input=image_tensor, modality="vision")
                text_embedding = liv_model(input=text_tokens, modality="text")
                signal = liv_model.sim(img_embedding, text_embedding)
            signal_tensor = torch.as_tensor(signal, dtype=torch.float32).reshape(-1)
            if signal_tensor.numel() == 1 and len(instructions) > 1:
                signal_tensor = signal_tensor.expand(len(instructions))
            if signal_tensor.numel() != len(instructions):
                raise ValueError(
                    f"LIV subprocess output size mismatch: got {signal_tensor.numel()}, expected {len(instructions)}."
                )
            signal_values = [float(item) for item in signal_tensor.detach().cpu().tolist()]
            print(json.dumps({"signals": signal_values}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
