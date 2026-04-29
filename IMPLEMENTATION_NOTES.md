# VeSpA Benchmarking Implementation

## Overview

This repository benchmarks four vessel segmentation methods on histological images:
- **VeSpA**: Color-based segmentation using CMYK decomposition
- **SAM**: Segment Anything Model with automatic mask generation
- **YOLOv8-seg**: Pre-trained object detection and segmentation
- **VeSpA+SAM (Hybrid)**: Novel combination of color priors with foundation models

Evaluation uses ground truth masks from two observers (SLR and GG) across three categories:
- **Category 1**: Individual observer masks (8 total)
- **Category 2**: Intersection (consensus regions, 4 masks)
- **Category 3**: Union (inclusive regions, 4 masks)

## Repository Structure

```
datasets/
├── images/                         # Input histology images (ROI_1.tiff to ROI_4.tiff)
└── groundthruth/                   # Ground truth masks
    ├── category_1_observers/       # Individual observer annotations
    ├── category_2_intersection/    # Consensus regions (AND)
    └── category_3_union/           # Inclusive regions (OR)

pipelines/
├── VeSpA/                          # Color-based segmentation
├── sam/                            # SAM automatic segmentation
├── yolo/                           # YOLOv8-seg
└── hybrid/                         # VeSpA+SAM hybrid

evaluation/
├── benchmark.py                    # Main benchmarking script
└── metrics.py                      # Evaluation metrics

results/
├── metrics/                        # CSV results and plots
└── prediction/                     # Generated masks and overlays
```

## How to Run

### Basic Benchmark
```bash
python evaluation/benchmark.py
```
Runs all methods against intersection GT (Category 2).

### Run Specific Category
```bash
# Against individual observers
python evaluation/benchmark.py --category category_1_observers --observer SLR
python evaluation/benchmark.py --category category_1_observers --observer GG

# Against consensus
python evaluation/benchmark.py --category category_2_intersection

# Against union
python evaluation/benchmark.py --category category_3_union
```

### Run All Categories
```bash
python evaluation/benchmark.py --all-categories
```

### Check GT Status
```bash
python scripts/setup_gt_masks.py
```
Shows the current organization of ground truth masks across all categories.

## Output Files

Results saved to `results/metrics/`:
- `results_cat*.csv` - Per-image metrics
- `summary_cat*.csv` - Mean metrics per model
- `runtime_summary_cat*.csv` - Speed comparison
- `*.png` - Comparison plots

Predictions saved to `results/prediction/{model}/`:
- `*_pred.png` - Binary predictions
- `*_overlay.png` - Ground truth overlays

## Key Features

- **Multi-method comparison**: 4 segmentation approaches
- **Multi-observer evaluation**: Assess inter-observer variability
- **Consensus analysis**: Intersection vs union ground truth
- **Comprehensive metrics**: Dice, IoU, Precision, Recall, Runtime
- **Visualization**: Automatic plots and overlays