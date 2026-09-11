"""Frozen Depth Anything 3 feature targets for geometry-token alignment."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from starVLA.model.modules.da3_addict_compat import ensure_addict_compatibility



_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _image_to_chw_float(image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"), copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    elif isinstance(image, np.ndarray):
        array = np.array(image, copy=True)
        if array.ndim != 3:
            raise ValueError(f"future image must have three dimensions, got {array.shape}")
        tensor = torch.from_numpy(array)
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
        elif tensor.shape[0] in (1, 3, 4):
            tensor = tensor[:3]
        else:
            raise ValueError(f"cannot identify future-image channels in shape {array.shape}")
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(
                f"future image tensor must have three dimensions, got {tuple(tensor.shape)}"
            )
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
        elif tensor.shape[0] in (1, 3, 4):
            tensor = tensor[:3]
        else:
            raise ValueError(
                f"cannot identify future-image channels in shape {tuple(tensor.shape)}"
            )
    else:
        raise TypeError(f"unsupported future image type: {type(image)!r}")

    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    if tensor.shape[0] != 3:
        raise ValueError(f"future image must contain RGB channels, got {tuple(tensor.shape)}")
    tensor = tensor.to(torch.float32)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError("future image must be non-empty and finite")
    if float(tensor.min()) < 0.0:
        raise ValueError("future image values must be non-negative")
    if float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    if float(tensor.max()) > 1.0:
        raise ValueError("future image values must lie in [0, 1] or [0, 255]")
    return tensor


def preprocess_future_images(
    images: Sequence[Image.Image | np.ndarray | torch.Tensor],
    *,
    image_size: int = 224,
) -> torch.Tensor:
    """Convert future RGB images to ImageNet-normalized ``[B,3,H,W]`` tensors."""

    if not images:
        raise ValueError("at least one future image is required")
    image_size = int(image_size)
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    batch = torch.stack([_image_to_chw_float(image) for image in images], dim=0)
    if tuple(batch.shape[-2:]) != (image_size, image_size):
        batch = F.interpolate(
            batch,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    mean = batch.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = batch.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (batch - mean) / std


def pool_da3_patch_features(
    features: torch.Tensor,
    *,
    patch_hw: tuple[int, int] = (16, 16),
    pool_hw: tuple[int, int] = (2, 4),
) -> torch.Tensor:
    """Pool raster-ordered DA3 patch features into raster-ordered latent targets."""

    if features.ndim != 3:
        raise ValueError(
            f"DA3 patch features must have shape [B,N,C], got {tuple(features.shape)}"
        )
    patch_h, patch_w = (int(value) for value in patch_hw)
    pool_h, pool_w = (int(value) for value in pool_hw)
    if patch_h * patch_w != int(features.shape[1]):
        raise ValueError(
            f"patch grid {patch_hw} has {patch_h * patch_w} cells but features have "
            f"{features.shape[1]} tokens"
        )
    if min(patch_h, patch_w, pool_h, pool_w) <= 0:
        raise ValueError("patch and pooling grid dimensions must be positive")
    feature_map = features.reshape(
        features.shape[0], patch_h, patch_w, features.shape[-1]
    ).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(feature_map, (pool_h, pool_w))
    return pooled.permute(0, 2, 3, 1).reshape(
        features.shape[0], pool_h * pool_w, features.shape[-1]
    )


def cosine_feature_alignment_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    """Mean token-wise cosine distance in float32 for numerical stability."""

    if tuple(student.shape) != tuple(teacher.shape):
        raise ValueError(
            f"student and teacher feature shape mismatch: "
            f"{tuple(student.shape)} != {tuple(teacher.shape)}"
        )
    if student.ndim != 3:
        raise ValueError(
            f"aligned features must have shape [B,K,C], got {tuple(student.shape)}"
        )
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("student and teacher features must be finite")
    similarity = F.cosine_similarity(student.float(), teacher.float(), dim=-1)
    return (1.0 - similarity).mean()


class _DA3CheckpointRoot(nn.Module):
    """Match the ``model.*`` prefix used by the local Hugging Face checkpoint."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


class _DA3Layer23Extractor(nn.Module):
    def __init__(self, backbone: nn.Module, input_norm: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_norm = input_norm

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features, _ = self.backbone(images[:, None])
        layer23 = features[-1][0]
        return self.input_norm(layer23)


class FrozenDA3FeatureTeacher:
    """Lazy frozen DA3 teacher deliberately kept outside the student module tree."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        source_path: str | Path,
        selected_layer: int,
        feature_dim: int,
        image_size: int,
        pool_hw: tuple[int, int],
    ) -> None:
        self.model_path = Path(model_path)
        self.source_path = Path(source_path)
        self.selected_layer = int(selected_layer)
        self.feature_dim = int(feature_dim)
        self.image_size = int(image_size)
        self.pool_hw = tuple(int(value) for value in pool_hw)
        self._extractor: _DA3Layer23Extractor | None = None

    @property
    def loaded(self) -> bool:
        return self._extractor is not None

    def _resolve_source_root(self) -> Path:
        candidates = (self.source_path, self.source_path / "src")
        for candidate in candidates:
            if (candidate / "depth_anything_3").is_dir():
                return candidate.resolve()
        raise FileNotFoundError(
            "DA3 source package is missing. Expected a depth_anything_3 directory "
            f"under {self.source_path} or {self.source_path / 'src'}"
        )

    def _checkpoint_file(self) -> Path:
        candidate = (
            self.model_path / "model.safetensors"
            if self.model_path.is_dir()
            else self.model_path
        )
        if not candidate.is_file():
            raise FileNotFoundError(
                f"DA3 checkpoint is missing: expected {candidate}"
            )
        return candidate

    def _load(self, device: torch.device) -> _DA3Layer23Extractor:
        source_root = self._resolve_source_root()
        checkpoint_file = self._checkpoint_file()
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        ensure_addict_compatibility()

        cfg_module = importlib.import_module("depth_anything_3.cfg")
        registry_module = importlib.import_module("depth_anything_3.registry")
        config = cfg_module.load_config(registry_module.MODEL_REGISTRY["da3-large"])
        network = cfg_module.create_object(config)
        configured_layers = [int(value) for value in network.backbone.out_layers]
        if configured_layers[-1] != self.selected_layer:
            raise ValueError(
                f"DA3 last configured layer is {configured_layers[-1]}, expected "
                f"{self.selected_layer}"
            )
        if int(network.head.norm.normalized_shape[0]) != self.feature_dim:
            raise ValueError(
                f"DA3 head input dimension is {network.head.norm.normalized_shape[0]}, "
                f"expected {self.feature_dim}"
            )

        checkpoint_root = _DA3CheckpointRoot(network)
        load_model = importlib.import_module("safetensors.torch").load_model
        load_model(checkpoint_root, checkpoint_file, strict=True, device="cpu")
        extractor = _DA3Layer23Extractor(
            checkpoint_root.model.backbone,
            checkpoint_root.model.head.norm,
        )
        extractor.requires_grad_(False)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        extractor.to(device=device, dtype=dtype)
        extractor.eval()
        self._extractor = extractor
        return extractor

    def extract(
        self,
        images: Sequence[Image.Image | np.ndarray | torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        extractor = self._extractor or self._load(device)
        extractor.eval()
        if any(parameter.requires_grad for parameter in extractor.parameters()):
            raise RuntimeError("DA3 teacher parameters must remain frozen")
        batch = preprocess_future_images(images, image_size=self.image_size).to(
            device=device,
            dtype=next(extractor.parameters()).dtype,
        )
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = extractor(batch)
            else:
                raw = extractor(batch)
        expected_raw = (len(images), 1, 256, self.feature_dim)
        if tuple(raw.shape) != expected_raw:
            raise ValueError(
                f"DA3 layer-{self.selected_layer} output shape {tuple(raw.shape)}, "
                f"expected {expected_raw}"
            )
        patches = raw[:, 0]
        if not torch.isfinite(patches).all():
            raise ValueError("DA3 patch target must be finite")
        pooled = pool_da3_patch_features(
            patches,
            patch_hw=(16, 16),
            pool_hw=self.pool_hw,
        )
        expected_pooled = (len(images), 8, self.feature_dim)
        if tuple(pooled.shape) != expected_pooled:
            raise ValueError(
                f"pooled DA3 target shape {tuple(pooled.shape)}, expected {expected_pooled}"
            )
        return pooled.detach()
