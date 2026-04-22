from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency at import time
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import compute_all_metrics
from pipelines.VeSpA.run_vespa import run_model as run_vespa
from pipelines.sam.run_sam import run_model as run_sam
from pipelines.yolo.run_yolo import run_model as run_yolo
from pipelines.hybrid.run_vespa_sam import run_model as run_vespa_sam_hybrid


DATASET_DIR = REPO_ROOT / "datasets"
IMAGES_DIR = DATASET_DIR / "images"
GROUND_TRUTH_DIR = DATASET_DIR / "groundthruth"
RESULTS_DIR = REPO_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTION_DIR = RESULTS_DIR / "prediction"
RESULTS_CSV = METRICS_DIR / "results.csv"
SUMMARY_CSV = METRICS_DIR / "summary.csv"
RUNTIME_CSV = METRICS_DIR / "runtime_summary.csv"


def set_reproducible_seeds(seed: int = 42) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_output_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load ground truth mask: {mask_path}")
    return ((mask > 0).astype(np.uint8) * 255)


def resize_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return ((resized > 0).astype(np.uint8) * 255)


def build_ground_truth_path(image_path: Path) -> Path:
    return GROUND_TRUTH_DIR / f"{image_path.stem}_mask.png"


def create_overlay(image: np.ndarray, prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    base = image.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    else:
        base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

    pred = np.squeeze(prediction) > 0
    gt = np.squeeze(ground_truth) > 0

    overlay = base.astype(np.float32)
    overlay[gt] = overlay[gt] * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
    overlay[pred] = overlay[pred] * 0.5 + np.array([255, 0, 0], dtype=np.float32) * 0.5
    overlap = np.logical_and(pred, gt)
    overlay[overlap] = overlay[overlap] * 0.3 + np.array([255, 255, 0], dtype=np.float32) * 0.7
    return overlay.astype(np.uint8)


def save_prediction_artifacts(
    image_path: Path,
    model_name: str,
    image_bgr: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    model_dir = PREDICTION_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    pred_path = model_dir / f"{image_path.stem}_pred.png"
    overlay_path = model_dir / f"{image_path.stem}_overlay.png"

    cv2.imwrite(str(pred_path), prediction)
    overlay = create_overlay(image_bgr, prediction, ground_truth)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def generate_summary_plots(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return

    plt.figure(figsize=(10, 6))
    results_df.boxplot(column="dice", by="model")
    plt.title("Dice Score by Model")
    plt.suptitle("")
    plt.xlabel("Model")
    plt.ylabel("Dice")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "dice_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    summary = results_df.groupby("model", as_index=False)["iou"].mean()
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"][:len(summary)]
    plt.bar(summary["model"], summary["iou"], color=colors)
    plt.title("Mean IoU by Model")
    plt.xlabel("Model")
    plt.ylabel("Mean IoU")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "iou_comparison.png", dpi=200)
    plt.close()
    
    # Additional comprehensive comparison plot
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ["dice", "iou", "precision", "recall"]
    summary_all = results_df.groupby("model", as_index=False)[metrics_to_plot].mean()
    
    x = np.arange(len(summary_all))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i * width, summary_all[metric], width, label=metric.capitalize())
    
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Comprehensive Metric Comparison")
    plt.xticks(x + width * 1.5, summary_all["model"], rotation=45, ha='right')
    plt.legend()
    plt.ylim([0, 1.0])
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "metrics_comparison.png", dpi=200)
    plt.close()


def build_model_registry() -> dict[str, callable]:
    return {
        "VeSpA": run_vespa,
        "SAM": run_sam,
        "YOLOv8-seg": run_yolo,
        "VeSpA+SAM (Hybrid)": run_vespa_sam_hybrid,
    }


def run_benchmark() -> pd.DataFrame:
    set_reproducible_seeds()
    ensure_output_dirs()

    model_registry = build_model_registry()
    image_paths = sorted(IMAGES_DIR.glob("*.tif*"))
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {IMAGES_DIR}")

    rows: list[dict[str, float | str]] = []
    runtime_rows: list[dict[str, float | str]] = []

    for image_path in image_paths:
        ground_truth_path = build_ground_truth_path(image_path)
        ground_truth = load_mask(ground_truth_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        for model_name, runner in model_registry.items():
            start = time.perf_counter()
            prediction = runner(str(image_path))
            runtime_seconds = time.perf_counter() - start

            prediction = resize_binary_mask(prediction, ground_truth.shape)
            metrics = compute_all_metrics(prediction, ground_truth)
            save_prediction_artifacts(image_path, model_name, image_bgr, prediction, ground_truth)

            rows.append(
                {
                    "image_name": image_path.name,
                    "model": model_name,
                    **metrics,
                }
            )
            runtime_rows.append(
                {
                    "image_name": image_path.name,
                    "model": model_name,
                    "runtime_seconds": runtime_seconds,
                }
            )
            print(
                f"{image_path.name} | {model_name} | "
                f"Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f} "
                f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
                f"Runtime={runtime_seconds:.2f}s"
            )

    results_df = pd.DataFrame(rows)
    runtime_df = pd.DataFrame(runtime_rows)
    results_df.to_csv(RESULTS_CSV, index=False)

    summary_df = (
        results_df.groupby("model", as_index=False)[["dice", "iou", "precision", "recall"]]
        .mean()
        .sort_values(by="dice", ascending=False)
    )
    summary_df.to_csv(SUMMARY_CSV, index=False)

    runtime_summary = runtime_df.groupby("model", as_index=False)["runtime_seconds"].mean()
    runtime_summary.to_csv(RUNTIME_CSV, index=False)

    generate_summary_plots(results_df)

    best_model = summary_df.iloc[0]["model"]
    print("\nAverage Dice per model:")
    for _, row in summary_df.iterrows():
        print(f"- {row['model']}: {row['dice']:.4f}")
    print(f"Best performing method: {best_model}")
    print("Runtime comparison (mean seconds/image):")
    for _, row in runtime_summary.iterrows():
        print(f"- {row['model']}: {row['runtime_seconds']:.2f}s")

    return results_df


if __name__ == "__main__":
    run_benchmark()
