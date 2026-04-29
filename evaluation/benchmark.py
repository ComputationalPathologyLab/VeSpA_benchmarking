from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Literal

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


# Ground truth categories
GTCategory = Literal["category_1_observers", "category_2_intersection", "category_3_union"]
Observer = Literal["SLR", "GG"]


class GTManager:
    """Manages ground truth mask organization across observer and derived categories."""

    def __init__(self, gt_base_dir: Path):
        """
        Initialize GTManager.

        Args:
            gt_base_dir: Base directory for all ground truth masks
        """
        self.base_dir = Path(gt_base_dir)
        self.cat1_dir = self.base_dir / "category_1_observers"
        self.cat2_dir = self.base_dir / "category_2_intersection"
        self.cat3_dir = self.base_dir / "category_3_union"

        # Ensure directories exist
        for d in [self.cat1_dir, self.cat2_dir, self.cat3_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load mask from file."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")
        return ((mask > 0).astype(np.uint8) * 255)

    def get_observer_mask_path(self, image_stem: str, observer: Observer) -> Path:
        """Get path to observer-specific mask."""
        return self.cat1_dir / f"{image_stem}_{observer}_mask.png"

    def get_intersection_mask_path(self, image_stem: str) -> Path:
        """Get path to intersection mask."""
        return self.cat2_dir / f"{image_stem}_mask.png"

    def get_union_mask_path(self, image_stem: str) -> Path:
        """Get path to union mask."""
        return self.cat3_dir / f"{image_stem}_mask.png"

    def load_gt_mask(self, image_stem: str, category: GTCategory) -> np.ndarray:
        """
        Load ground truth mask for a specific category.

        Args:
            image_stem: Stem of image file (without extension)
            category: Ground truth category to load from

        Returns:
            Binary mask array
        """
        if category == "category_1_observers":
            raise ValueError("category_1_observers requires specifying an observer. Use load_observer_mask instead.")

        if category == "category_2_intersection":
            mask_path = self.get_intersection_mask_path(image_stem)
        elif category == "category_3_union":
            mask_path = self.get_union_mask_path(image_stem)
        else:
            raise ValueError(f"Unknown category: {category}")

        return self._load_mask(mask_path)

    def load_observer_mask(self, image_stem: str, observer: Observer) -> np.ndarray:
        """Load observer-specific mask from category_1."""
        mask_path = self.get_observer_mask_path(image_stem, observer)
        return self._load_mask(mask_path)


DATASET_DIR = REPO_ROOT / "datasets"
IMAGES_DIR = DATASET_DIR / "images"
GROUND_TRUTH_DIR = DATASET_DIR / "groundthruth"
RESULTS_DIR = REPO_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTION_DIR = RESULTS_DIR / "prediction"

# Ground truth category suffixes for CSV files
GT_CATEGORY_SUFFIX_MAP = {
    "category_1_observers": "_cat1",
    "category_2_intersection": "_cat2",
    "category_3_union": "_cat3",
}


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
    """Build legacy ground truth path (for backward compatibility)."""
    return GROUND_TRUTH_DIR / f"{image_path.stem}_mask.png"


def load_gt_mask_by_category(
    image_stem: str,
    category: GTCategory,
    observer: str | None = None,
    gt_manager: GTManager | None = None,
) -> np.ndarray:
    """
    Load ground truth mask for a specific category.

    Args:
        image_stem: Image stem without extension
        category: Ground truth category
        observer: Observer name ("SLR" or "GG") for category_1_observers
        gt_manager: Optional GTManager instance

    Returns:
        Ground truth mask array
    """
    if gt_manager is None:
        gt_manager = GTManager(GROUND_TRUTH_DIR)

    if category == "category_1_observers":
        if observer is None:
            raise ValueError("observer must be specified for category_1_observers")
        return gt_manager.load_observer_mask(image_stem, observer)
    else:
        return gt_manager.load_gt_mask(image_stem, category)


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

    # Get the desired model order from registry
    model_order = list(build_model_registry().keys())

    # Ensure results_df has models in the correct order for boxplot
    results_df["model"] = pd.Categorical(results_df["model"], categories=model_order, ordered=True)

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
    # Sort summary to match model registry order
    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary = summary.sort_values("model")
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
    # Sort summary_all to match model registry order
    summary_all["model"] = pd.Categorical(summary_all["model"], categories=model_order, ordered=True)
    summary_all = summary_all.sort_values("model")

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
        "VeSpA+SAM (Hybrid)": run_vespa_sam_hybrid,
        "SAM": run_sam,
        "YOLOv8-seg": run_yolo,
    }


def run_benchmark(
    gt_category: GTCategory = "category_2_intersection",
    observer: str | None = None,
) -> pd.DataFrame:
    """
    Run benchmarking pipeline with specified ground truth category.

    Args:
        gt_category: Which ground truth category to evaluate against
                    - "category_1_observers": Individual observer masks (requires observer parameter)
                    - "category_2_intersection": Intersection of SLR and GG masks
                    - "category_3_union": Union of SLR and GG masks
        observer: Observer name ("SLR" or "GG") - required if gt_category is "category_1_observers"

    Returns:
        DataFrame with results
    """
    set_reproducible_seeds()
    ensure_output_dirs()

    # Setup GT manager
    gt_manager = GTManager(GROUND_TRUTH_DIR)

    # Validate observer parameter for category_1
    if gt_category == "category_1_observers":
        if observer is None:
            raise ValueError("observer parameter required for category_1_observers")
        if observer not in ["SLR", "GG"]:
            raise ValueError("observer must be 'SLR' or 'GG'")

    model_registry = build_model_registry()
    image_paths = sorted(IMAGES_DIR.glob("*.tif*"))
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {IMAGES_DIR}")

    rows: list[dict[str, float | str]] = []
    runtime_rows: list[dict[str, float | str]] = []

    # Add category info to output suffix
    category_suffix = GT_CATEGORY_SUFFIX_MAP.get(gt_category, "")
    observer_suffix = f"_{observer}" if gt_category == "category_1_observers" else ""
    output_suffix = f"{category_suffix}{observer_suffix}"

    print(f"\n{'='*70}")
    print(f"Running benchmark with GT category: {gt_category}")
    if observer:
        print(f"Observer: {observer}")
    print(f"{'='*70}\n")

    for image_path in image_paths:
        try:
            # Load image
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")

            # Load ground truth for the specified category
            ground_truth = load_gt_mask_by_category(
                image_path.stem,
                gt_category,
                observer=observer,
                gt_manager=gt_manager,
            )

        except FileNotFoundError as e:
            print(f"⚠ Skipping {image_path.name}: {e}")
            continue

        for model_name, runner in model_registry.items():
            try:
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
                        "gt_category": gt_category,
                        **([{"observer": observer}] if observer else [{}])[0],
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
            except Exception as e:
                print(f"✗ Error processing {image_path.name} with {model_name}: {e}")
                continue

    if not rows:
        print("No results generated. Check that ground truth masks exist in the specified category.")
        return pd.DataFrame()

    results_df = pd.DataFrame(rows)
    runtime_df = pd.DataFrame(runtime_rows)

    # Save results with category suffix
    results_csv = METRICS_DIR / f"results{output_suffix}.csv"
    summary_csv = METRICS_DIR / f"summary{output_suffix}.csv"
    runtime_csv = METRICS_DIR / f"runtime_summary{output_suffix}.csv"

    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to {results_csv.name}")

    summary_df = (
        results_df.groupby("model", as_index=False)[["dice", "iou", "precision", "recall"]]
        .mean()
        .sort_values(by="dice", ascending=False)
    )
    summary_df.to_csv(summary_csv, index=False)

    runtime_summary = runtime_df.groupby("model", as_index=False)["runtime_seconds"].mean()
    runtime_summary.to_csv(runtime_csv, index=False)

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


def run_all_benchmarks() -> None:
    """Run benchmarks for all GT categories and observers."""
    all_results = []

    # Run for category 2 (intersection)
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Category 2 (Intersection of SLR and GG)")
    print("=" * 70)
    results_cat2 = run_benchmark(gt_category="category_2_intersection")
    all_results.append(results_cat2)

    # Run for category 3 (union)
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Category 3 (Union of SLR and GG)")
    print("=" * 70)
    results_cat3 = run_benchmark(gt_category="category_3_union")
    all_results.append(results_cat3)

    # Run for category 1 with each observer
    for observer in ["SLR", "GG"]:
        print("\n" + "=" * 70)
        print(f"BENCHMARK {2 + ['SLR', 'GG'].index(observer) + 1}: Category 1 - Observer {observer}")
        print("=" * 70)
        results_observer = run_benchmark(gt_category="category_1_observers", observer=observer)
        all_results.append(results_observer)

    # Combine and save all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_csv = METRICS_DIR / "results_all_categories.csv"
        combined_results.to_csv(combined_csv, index=False)
        print(f"\n✓ All results combined and saved to {combined_csv.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run VeSpA benchmarking pipeline with different GT categories"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="category_2_intersection",
        choices=["category_1_observers", "category_2_intersection", "category_3_union"],
        help="Ground truth category to use for evaluation",
    )
    parser.add_argument(
        "--observer",
        type=str,
        default=None,
        choices=["SLR", "GG"],
        help="Observer name for category_1_observers (required if category is category_1_observers)",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Run benchmarks for all categories and observers",
    )

    args = parser.parse_args()

    if args.all_categories:
        run_all_benchmarks()
    else:
        run_benchmark(gt_category=args.category, observer=args.observer)
