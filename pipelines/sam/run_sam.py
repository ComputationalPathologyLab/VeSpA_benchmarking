from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency at import time
    torch = None


DEFAULT_SAM_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")
DEFAULT_SAM_MAX_DIM = int(os.getenv("SAM_MAX_DIM", "1024"))
DEFAULT_SAM_MAX_MASK_RATIO = float(os.getenv("SAM_MAX_MASK_RATIO", "0.25"))
DEFAULT_SAM_CHECKPOINT = os.getenv(
    "SAM_CHECKPOINT",
    str(Path("models") / f"sam_{DEFAULT_SAM_TYPE}_01ec64.pth"),
)
SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

_SAM_GENERATOR = None
_SAM_STATE = None


def _set_reproducible_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _resolve_checkpoint_path(model_type: str, checkpoint_path: str | None) -> Path:
    if checkpoint_path:
        return Path(checkpoint_path)
    return Path(DEFAULT_SAM_CHECKPOINT)


def _load_generator(model_type: str = DEFAULT_SAM_TYPE, checkpoint_path: str | None = None):
    global _SAM_GENERATOR, _SAM_STATE

    resolved_checkpoint = _resolve_checkpoint_path(model_type, checkpoint_path)
    state = (model_type, str(resolved_checkpoint.resolve()) if resolved_checkpoint.exists() else str(resolved_checkpoint))
    if _SAM_GENERATOR is not None and _SAM_STATE == state:
        return _SAM_GENERATOR

    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:  # pragma: no cover - import depends on environment
        raise ImportError(
            "segment_anything is not installed. Install dependencies from requirements.txt first."
        ) from exc

    if not resolved_checkpoint.exists():
        download_url = SAM_CHECKPOINT_URLS.get(model_type, "the Meta SAM release URL")
        raise FileNotFoundError(
            f"SAM checkpoint not found at '{resolved_checkpoint}'. "
            f"Download it first, for example from {download_url}."
        )

    if torch is None:
        raise ImportError("torch is required to run SAM.")

    _set_reproducible_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(resolved_checkpoint))
    sam.to(device=device)

    _SAM_GENERATOR = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=64,
    )
    _SAM_STATE = state
    return _SAM_GENERATOR


def combine_masks(masks: Iterable[np.ndarray], image_shape: tuple[int, int]) -> np.ndarray:
    combined = np.zeros(image_shape, dtype=np.uint8)
    for mask in masks:
        combined = np.maximum(combined, mask.astype(np.uint8) * 255)
    return combined


def _resize_for_sam(image_rgb: np.ndarray, max_dim: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image_rgb.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dim:
        return image_rgb, (height, width)

    scale = max_dim / float(longest_side)
    resized = cv2.resize(
        image_rgb,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, (height, width)


def run_model(
    image_path: str,
    model_type: str = DEFAULT_SAM_TYPE,
    checkpoint_path: str | None = None,
    max_dim: int = DEFAULT_SAM_MAX_DIM,
    max_mask_ratio: float = DEFAULT_SAM_MAX_MASK_RATIO,
) -> np.ndarray:
    """
    Returns binary segmentation mask (H x W), dtype uint8.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    processed_rgb, original_shape = _resize_for_sam(image_rgb, max_dim=max_dim)
    generator = _load_generator(model_type=model_type, checkpoint_path=checkpoint_path)
    generated_masks = generator.generate(processed_rgb)

    if not generated_masks:
        return np.zeros(original_shape, dtype=np.uint8)

    image_area = processed_rgb.shape[0] * processed_rgb.shape[1]
    max_allowed_area = image_area * max_mask_ratio
    filtered_masks = [
        entry["segmentation"]
        for entry in generated_masks
        if "segmentation" in entry and entry.get("area", image_area) <= max_allowed_area
    ]
    if not filtered_masks:
        filtered_masks = [entry["segmentation"] for entry in generated_masks if "segmentation" in entry]

    combined = combine_masks(filtered_masks, processed_rgb.shape[:2])
    if combined.shape != original_shape:
        combined = cv2.resize(combined, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return combined.astype(np.uint8)
