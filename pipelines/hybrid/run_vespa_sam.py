from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency at import time
    torch = None


DEFAULT_SAM_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")
DEFAULT_SAM_MAX_DIM = int(os.getenv("SAM_MAX_DIM", "1024"))
DEFAULT_SAM_CHECKPOINT = os.getenv(
    "SAM_CHECKPOINT",
    str(Path("models") / f"sam_{DEFAULT_SAM_TYPE}_01ec64.pth"),
)
SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

_SAM_PREDICTOR = None
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


def _load_predictor(model_type: str = DEFAULT_SAM_TYPE, checkpoint_path: str | None = None):
    """Load SAM predictor (not the automatic mask generator)."""
    global _SAM_PREDICTOR, _SAM_STATE

    resolved_checkpoint = _resolve_checkpoint_path(model_type, checkpoint_path)
    state = (model_type, str(resolved_checkpoint.resolve()) if resolved_checkpoint.exists() else str(resolved_checkpoint))
    if _SAM_PREDICTOR is not None and _SAM_STATE == state:
        return _SAM_PREDICTOR

    try:
        from segment_anything import SamPredictor, sam_model_registry
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

    _SAM_PREDICTOR = SamPredictor(sam)
    _SAM_STATE = state
    return _SAM_PREDICTOR


def _extract_stain_mask(
    image_bgr: np.ndarray,
    blur_kernel: int = 5,
    closing_kernel: int = 5,
) -> np.ndarray:
    bgrdash = image_bgr.astype(np.float64) / 255.0

    K = 1 - np.max(bgrdash, axis=2)
    denom = 1 - K
    denom[denom == 0] = 1e-8

    C = (1 - bgrdash[..., 2] - K) / denom
    M = (1 - bgrdash[..., 1] - K) / denom
    Y = (1 - bgrdash[..., 0] - K) / denom

    CMYK = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)
    Y_channel = CMYK[:, :, 2]

    Y_blurred = cv2.GaussianBlur(Y_channel, (blur_kernel, blur_kernel), 0)
    _, stain_mask = cv2.threshold(Y_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    closing_kernel_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_CLOSE, closing_kernel_elem)
    return stain_mask


def _extract_prompts_from_mask(
    mask: np.ndarray,
    min_component_area: int = 50,
    max_component_area: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    valid_points = []
    valid_labels = []
    valid_boxes = []

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue
        if max_component_area is not None and area > max_component_area:
            continue

        cx, cy = centroids[label_id]
        valid_points.append([cx, cy])
        valid_labels.append(1)

        x, y, w, h = stats[label_id, cv2.CC_STAT_LEFT : cv2.CC_STAT_LEFT + 4]
        valid_boxes.append((x, y, x + w, y + h))

    if not valid_points:
        return np.array([], dtype=np.float32).reshape(0, 2), np.array([], dtype=np.int32), []

    point_coords = np.array(valid_points, dtype=np.float32)
    point_labels = np.array(valid_labels, dtype=np.int32)
    return point_coords, point_labels, valid_boxes


def _resize_for_sam(image_rgb: np.ndarray, max_dim: int) -> tuple[np.ndarray, tuple[int, int], float]:
    height, width = image_rgb.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_dim:
        return image_rgb, (height, width), 1.0

    scale = max_dim / float(longest_side)
    resized = cv2.resize(
        image_rgb,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, (height, width), scale


def _refine_masks_with_sam(
    image_rgb: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    boxes: list[tuple[int, int, int, int]] | None,
    predictor,
) -> list[np.ndarray]:
    predictor.set_image(image_rgb)

    refined_masks = []
    for i, (point, label) in enumerate(zip(point_coords, point_labels)):
        try:
            predict_kwargs = {
                "point_coords": np.array([point], dtype=np.float32),
                "point_labels": np.array([label], dtype=np.int32),
                "multimask_output": False,
            }
            if boxes is not None and i < len(boxes):
                predict_kwargs["box"] = np.array(boxes[i], dtype=np.float32)

            masks, scores, _ = predictor.predict(**predict_kwargs)
            mask_array = np.asarray(masks)
            if mask_array.ndim == 3:
                if mask_array.shape[0] == 1:
                    mask_array = mask_array[0]
                else:
                    best_idx = int(np.argmax(np.asarray(scores)))
                    mask_array = mask_array[best_idx]
            mask_2d = np.squeeze(mask_array).astype(bool)
            if mask_2d.ndim != 2:
                raise ValueError(f"Expected a 2D mask from SAM, got shape {mask_2d.shape}")
            refined_masks.append(mask_2d)
        except Exception as exc:
            print(f"Warning: SAM prediction failed for point {i}: {exc}")
            continue

    return refined_masks


def _combine_refined_masks(
    masks: list[np.ndarray],
    image_shape: tuple[int, int],
) -> np.ndarray:
    if not masks:
        return np.zeros(image_shape, dtype=np.uint8)

    combined = np.zeros(image_shape, dtype=bool)
    for mask in masks:
        combined = np.logical_or(combined, np.squeeze(mask).astype(bool))

    combined_uint8 = combined.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined_uint8, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def run_model(
    image_path: str,
    model_type: str = DEFAULT_SAM_TYPE,
    checkpoint_path: str | None = None,
    max_dim: int = DEFAULT_SAM_MAX_DIM,
    min_stain_area: int = 50,
    max_stain_area: Optional[int] = None,
    blur_kernel: int = 5,
    closing_kernel: int = 5,
) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    original_shape = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    stain_mask = _extract_stain_mask(image_bgr, blur_kernel=blur_kernel, closing_kernel=closing_kernel)
    point_coords, point_labels, boxes = _extract_prompts_from_mask(
        stain_mask,
        min_component_area=min_stain_area,
        max_component_area=max_stain_area,
    )
    if len(point_coords) == 0:
        return np.zeros(original_shape, dtype=np.uint8)

    processed_rgb, _, scale_factor = _resize_for_sam(image_rgb, max_dim=max_dim)
    processed_height, processed_width = processed_rgb.shape[:2]

    scaled_point_coords = point_coords * scale_factor
    if len(scaled_point_coords):
        scaled_point_coords[:, 0] = np.clip(scaled_point_coords[:, 0], 0, processed_width - 1)
        scaled_point_coords[:, 1] = np.clip(scaled_point_coords[:, 1], 0, processed_height - 1)

    scaled_boxes = None
    if boxes:
        scaled_boxes = []
        for x0, y0, x1, y1 in boxes:
            scaled_boxes.append(
                (
                    int(np.clip(round(x0 * scale_factor), 0, processed_width - 1)),
                    int(np.clip(round(y0 * scale_factor), 0, processed_height - 1)),
                    int(np.clip(round(x1 * scale_factor), 0, processed_width - 1)),
                    int(np.clip(round(y1 * scale_factor), 0, processed_height - 1)),
                )
            )

    predictor = _load_predictor(model_type=model_type, checkpoint_path=checkpoint_path)
    refined_masks = _refine_masks_with_sam(
        processed_rgb,
        scaled_point_coords,
        point_labels,
        scaled_boxes,
        predictor,
    )

    if scale_factor != 1.0:
        resized_masks = []
        h, w = original_shape[:2]
        for mask in refined_masks:
            mask_uint8 = np.squeeze(mask).astype(np.uint8) * 255
            resized = cv2.resize(mask_uint8, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized > 127)
        refined_masks = resized_masks

    combined = _combine_refined_masks(refined_masks, original_shape)
    return combined.astype(np.uint8)
