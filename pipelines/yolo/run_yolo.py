from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency at import time
    torch = None


DEFAULT_YOLO_WEIGHTS = os.getenv(
    "YOLO_SEG_WEIGHTS",
    str(Path(__file__).resolve().parents[2] / "models" / "yolov8n-seg.pt"),
)
YOLO_CONFIG_DIR = os.getenv("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / ".cache" / "ultralytics"))
YOLO_CONFIDENCE_THRESHOLDS = (0.25, 0.15, 0.05, 0.01)
_YOLO_MODEL = None
_YOLO_WEIGHTS = None


def _set_reproducible_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _load_model(weights_path: str = DEFAULT_YOLO_WEIGHTS):
    global _YOLO_MODEL, _YOLO_WEIGHTS

    if _YOLO_MODEL is not None and _YOLO_WEIGHTS == weights_path:
        return _YOLO_MODEL

    try:
        os.environ.setdefault("YOLO_CONFIG_DIR", YOLO_CONFIG_DIR)
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - import depends on environment
        raise ImportError("ultralytics is not installed. Install dependencies from requirements.txt first.") from exc

    _set_reproducible_seeds()
    _YOLO_MODEL = YOLO(weights_path)
    _YOLO_WEIGHTS = weights_path
    return _YOLO_MODEL


def _predict_with_confidence_fallback(
    model,
    image_path: str,
    imgsz: int,
    device: str | int,
    confidence_thresholds: tuple[float, ...] = YOLO_CONFIDENCE_THRESHOLDS,
):
    for conf in confidence_thresholds:
        results = model.predict(
            source=image_path,
            imgsz=imgsz,
            retina_masks=True,
            verbose=False,
            device=device,
            conf=conf,
        )
        if not results:
            continue

        result = results[0]
        masks = getattr(result, "masks", None)
        if masks is not None and getattr(masks, "data", None) is not None:
            return results

        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            return results

    return []


def run_model(image_path: str, weights_path: str = DEFAULT_YOLO_WEIGHTS, imgsz: int = 1024) -> np.ndarray:
    """
    Returns binary segmentation mask (H x W), dtype uint8.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image_bgr.shape[:2]
    model = _load_model(weights_path=weights_path)
    device = 0 if torch is not None and torch.cuda.is_available() else "cpu"
    results = _predict_with_confidence_fallback(
        model=model,
        image_path=image_path,
        imgsz=imgsz,
        device=device,
    )

    combined = np.zeros((height, width), dtype=np.uint8)
    if not results:
        return combined

    result = results[0]
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None:
        return combined

    mask_array = masks.data.detach().cpu().numpy()
    for mask in mask_array:
        resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        combined = np.maximum(combined, (resized > 0.5).astype(np.uint8) * 255)

    return combined
