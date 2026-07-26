"""Shared image-space geometry for the RoboCasa replay renderer."""

from __future__ import annotations

import numpy as np


DIAL_RENDER_WIDTH = 1280
DIAL_RENDER_HEIGHT = 800
DIAL_CROP_X0 = 110
DIAL_CROP_X1 = 1130
DIAL_CROP_Y0 = 310
DIAL_CROP_Y1 = 770
DIAL_RESIZED_WIDTH = 720
DIAL_RESIZED_HEIGHT = 480
DIAL_PAD_Y = 120


def dial_content_region_bounds(output_size: int = 256) -> tuple[float, float, float, float]:
    """Return the continuous output-pixel rectangle containing real DIAL content."""
    if output_size < 2:
        raise ValueError(f"output_size must be at least 2, got {output_size}")
    scale = float(output_size) / float(DIAL_RESIZED_WIDTH)
    return (
        0.0,
        float(output_size),
        float(DIAL_PAD_Y) * scale,
        float(DIAL_PAD_Y + DIAL_RESIZED_HEIGHT) * scale,
    )


def dial_content_region_mask(uvd_pixels: np.ndarray, valid_mask: np.ndarray, *, output_size: int = 256) -> np.ndarray:
    """Mask projected points that fall in DIAL's black padding."""
    uvd = np.asarray(uvd_pixels, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if uvd.shape[-1] != 3:
        raise ValueError(f"uvd_pixels must end in 3, got {uvd.shape}")
    if valid.shape != uvd.shape[:-1]:
        raise ValueError(f"valid_mask must have shape {uvd.shape[:-1]}, got {valid.shape}")
    x0, x1, y0, y1 = dial_content_region_bounds(output_size)
    inside = (
        np.isfinite(uvd).all(axis=-1)
        & (uvd[..., 0] >= x0)
        & (uvd[..., 0] < x1)
        & (uvd[..., 1] >= y0)
        & (uvd[..., 1] < y1)
    )
    return valid & inside


def dial_image_transform_matrix(
    source_width: int = DIAL_RENDER_WIDTH,
    source_height: int = DIAL_RENDER_HEIGHT,
    output_size: int = 256,
) -> np.ndarray:
    """Return the homogeneous pixel transform used by DIAL's crop pipeline."""
    if (source_width, source_height) != (DIAL_RENDER_WIDTH, DIAL_RENDER_HEIGHT):
        raise ValueError(
            "DIAL camera geometry expects raw render size 1280x800, "
            f"got {source_width}x{source_height}"
        )
    if output_size < 2:
        raise ValueError(f"output_size must be at least 2, got {output_size}")

    # raw (x, y) -> vertical flip -> crop -> resize -> square padding -> output
    x_scale = output_size / float(DIAL_CROP_X1 - DIAL_CROP_X0)
    y_scale = output_size / float(DIAL_RESIZED_WIDTH)
    y_crop_scale = DIAL_RESIZED_HEIGHT / float(DIAL_CROP_Y1 - DIAL_CROP_Y0)
    y_scale *= y_crop_scale
    x_offset = -DIAL_CROP_X0 * x_scale
    y_offset = (
        (source_height - 1 - DIAL_CROP_Y0) * y_scale
        + DIAL_PAD_Y * output_size / float(DIAL_RESIZED_WIDTH)
    )
    return np.array(
        [[x_scale, 0.0, x_offset], [0.0, -y_scale, y_offset], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def transform_intrinsic(
    camera_k: np.ndarray,
    source_width: int = DIAL_RENDER_WIDTH,
    source_height: int = DIAL_RENDER_HEIGHT,
    output_size: int = 256,
) -> np.ndarray:
    """Transform a pinhole intrinsic matrix with the RGB/depth pixel map."""
    matrix = dial_image_transform_matrix(source_width, source_height, output_size)
    k = np.asarray(camera_k, dtype=np.float32)
    if k.shape[-2:] != (3, 3):
        raise ValueError(f"camera_k must end in (3, 3), got {k.shape}")
    return np.einsum("ij,...jk->...ik", matrix, k).astype(np.float32)


def canonical_dial_image_transform_matrix(
    source_width: int = DIAL_RENDER_WIDTH,
    source_height: int = DIAL_RENDER_HEIGHT,
    output_size: int = 256,
) -> np.ndarray:
    """Return DIAL's crop/pad map for an already vertically flipped image.

    RoboCasa's ``get_basic_observation`` flips MuJoCo's bottom-up render before
    ``process_img_cotrain`` applies the crop.  RoboSuite's intrinsic matrix is
    also defined in this canonical, top-down image coordinate system, so this
    affine map must keep a positive image-space v axis.
    """
    if (source_width, source_height) != (DIAL_RENDER_WIDTH, DIAL_RENDER_HEIGHT):
        raise ValueError(
            "DIAL camera geometry expects raw render size 1280x800, "
            f"got {source_width}x{source_height}"
        )
    if output_size < 2:
        raise ValueError(f"output_size must be at least 2, got {output_size}")

    x_scale = output_size / float(DIAL_CROP_X1 - DIAL_CROP_X0)
    y_scale = output_size / float(DIAL_RESIZED_WIDTH)
    y_scale *= DIAL_RESIZED_HEIGHT / float(DIAL_CROP_Y1 - DIAL_CROP_Y0)
    return np.array(
        [
            [x_scale, 0.0, -DIAL_CROP_X0 * x_scale],
            [0.0, y_scale, -DIAL_CROP_Y0 * y_scale + DIAL_PAD_Y * output_size / float(DIAL_RESIZED_WIDTH)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def transform_canonical_intrinsic(
    camera_k: np.ndarray,
    source_width: int = DIAL_RENDER_WIDTH,
    source_height: int = DIAL_RENDER_HEIGHT,
    output_size: int = 256,
) -> np.ndarray:
    """Transform a RoboSuite K defined on the vertically flipped image."""
    matrix = canonical_dial_image_transform_matrix(source_width, source_height, output_size)
    k = np.asarray(camera_k, dtype=np.float32)
    if k.shape[-2:] != (3, 3):
        raise ValueError(f"camera_k must end in (3, 3), got {k.shape}")
    return np.einsum("ij,...jk->...ik", matrix, k).astype(np.float32)


def transformed_agentview_intrinsic(
    camera_utils: Any,
    sim: Any,
    camera_name: str = "egoview",
    output_size: int = 256,
) -> np.ndarray:
    """Read RoboSuite K with its (height, width) API order and apply DIAL."""
    raw_k = np.asarray(
        camera_utils.get_camera_intrinsic_matrix(
            sim,
            camera_name,
            DIAL_RENDER_HEIGHT,
            DIAL_RENDER_WIDTH,
        ),
        dtype=np.float32,
    )
    return transform_canonical_intrinsic(raw_k, DIAL_RENDER_WIDTH, DIAL_RENDER_HEIGHT, output_size)


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize with OpenCV area filtering, with a NumPy fallback for unit tests."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - production environment has OpenCV
        y = np.rint(np.linspace(0, image.shape[0] - 1, height)).astype(np.int64)
        x = np.rint(np.linspace(0, image.shape[1] - 1, width)).astype(np.int64)
        if image.ndim == 2:
            return image[np.ix_(y, x)]
        return image[np.ix_(y, x, np.arange(image.shape[2]))]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def apply_image_affine(image: np.ndarray, output_size: int = 256) -> np.ndarray:
    """Apply DIAL's exact raw render transform to RGB or metric depth."""
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"image must be HxW or HxWxC, got {array.shape}")
    if array.shape[0] != DIAL_RENDER_HEIGHT or array.shape[1] != DIAL_RENDER_WIDTH:
        raise ValueError(
            "DIAL camera geometry expects raw render size (800, 1280), "
            f"got {array.shape[:2]}"
        )

    flipped = array[::-1, ...]
    cropped = flipped[DIAL_CROP_Y0:DIAL_CROP_Y1, DIAL_CROP_X0:DIAL_CROP_X1]
    resized = _resize(cropped, DIAL_RESIZED_HEIGHT, DIAL_RESIZED_WIDTH)
    if resized.ndim == 2:
        padded = np.pad(
            resized,
            ((DIAL_PAD_Y, DIAL_PAD_Y), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        padded = np.pad(
            resized,
            ((DIAL_PAD_Y, DIAL_PAD_Y), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    output = _resize(padded, output_size, output_size)
    return np.ascontiguousarray(output.astype(array.dtype, copy=False))


def dial_image_transform(image: np.ndarray, output_size: int = 256) -> np.ndarray:
    """Alias used by tests and the renderer for the DIAL camera transform."""
    return apply_image_affine(image, output_size=output_size)
