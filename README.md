# VeSpA Benchmarking Repository

Comprehensive benchmarking of vessel segmentation pipelines for histology images with DAB immunostaining.

## Overview

This repository compares three vessel segmentation methods on histological images with brown DAB (diaminobenzidine) immunostaining marking endothelial cells:

1. **VeSpA** - Color-based vessel segmentation using CMYK decomposition
2. **SAM** - Segment Anything Model with automatic mask generation
3. **YOLOv8-seg** - Pre-trained object detection and segmentation
4. **VeSpA+SAM (Hybrid)** - Novel hybrid approach combining color priors with foundation models

## Quick Start

### Download the SAM model:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mkdir -p models
mv sam_vit_b_01ec64.pth models/
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the benchmarking pipeline:

```bash
python evaluation/benchmark.py
```

This will:
- Process all images in `datasets/images/`
- Compare all four methods
- Generate metrics (Dice, IoU, Precision, Recall)
- Create comparison plots in `results/metrics/`
- Save predictions in `results/prediction/`

---

## Methods

### 1. VeSpA (Vessel Segmentation Pipeline Algorithm)

**Approach**: Color-based segmentation using CMYK decomposition

**Key Steps**:
- Convert BGR → CMYK color space
- Extract Yellow channel (inverse of brown stain)
- Apply Otsu threshold
- Morphological operations (blur, close, dilate, erode)

**Advantages**:
- Domain-specific (targets brown immunostain)
- Fast and deterministic
- No model weights required

**Limitations**:
- Sensitive to morphological parameter tuning
- Sometimes imperfect boundary definition

### 2. SAM (Segment Anything Model)

**Approach**: Foundation model with automatic mask generation

**Key Steps**:
- Uniform 16×16 grid of prompt points
- SAM prediction at each point
- Filtering by IoU and stability scores
- Mask combination

**Advantages**:
- Automatic (no prompting required)
- Detects diverse objects
- Strong generalization

**Limitations**:
- Not histology-aware
- Many false positives from non-vessel structures
- Grid-based sampling may miss small vessels

### 3. YOLOv8-seg

**Approach**: Pre-trained instance segmentation

**Key Steps**:
- Object detection + segmentation head
- Non-maximum suppression
- Mask refinement

**Advantages**:
- Fast inference (nano model)
- Well-optimized

**Limitations**:
- Trained on general objects
- May not generalize well to histology

### 4. VeSpA+SAM (Hybrid) - **NEW**

**Approach**: Combines color priors with foundation model refinement

**Key Steps**:
1. **Stain Detection** (VeSpA): Extract brown stain using CMYK + Otsu
2. **Prompt Generation**: Find connected components, compute centroids
3. **SAM Refinement**: Use centroids as point prompts to SamPredictor
4. **Post-processing**: Combine masks, remove artifacts

**Advantages**:
- ✓ Domain-aware (targets brown stain)
- ✓ Boundary refinement (SAM correction)
- ✓ Reduced false positives (constrained to stain regions)
- ✓ Handles staining variability

**Expected Improvements over SAM alone**:
- Higher precision (fewer non-vessel detections)
- Better boundary accuracy
- Robustness to stain intensity variations

**Expected Improvements over VeSpA alone**:
- Superior boundary definition
- Less sensitive to morphological parameters
- Foundation model reliability

---

## Pipeline Architecture

### File Structure

```
pipelines/
├── sam/
│   └── run_sam.py                 # Automatic SAM segmentation
├── yolo/
│   └── run_yolo.py                # YOLOv8-seg segmentation
├── VeSpA/
│   └── Segmentation + Measurements.py  # Color-based segmentation
└── hybrid/
    ├── __init__.py
    └── run_vespa_sam.py           # Novel hybrid approach (NEW)

evaluation/
├── benchmark.py                    # Main benchmarking script
└── metrics.py                      # Metric computation

datasets/
├── images/                         # Input histology images
├── groundtruth/                    # Manual annotations
└── masks/                          # Predicted masks
```

### Configuration

All pipelines support configuration via environment variables:

```bash
# SAM configuration
export SAM_MODEL_TYPE="vit_b"          # vit_b, vit_l, vit_h
export SAM_MAX_DIM="1024"              # Input resize dimension
export SAM_CHECKPOINT="models/sam_vit_b_01ec64.pth"

# YOLO configuration
export YOLO_SEG_WEIGHTS="models/yolov8n-seg.pt"

# VeSpA/Hybrid configuration (morphological parameters)
export BLUR_KERNEL="5"
export CLOSING_KERNEL="5"
```

---

## Evaluation Metrics

All methods evaluated on:

- **Dice Coefficient**: 2·|A∩B| / (|A| + |B|)
- **IoU (Intersection over Union)**: |A∩B| / |A∪B|
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **Runtime**: Seconds per image

### Results Output

Results saved to:
- `results/metrics/results.csv` - Per-image metrics
- `results/metrics/summary.csv` - Mean metrics per model
- `results/metrics/runtime_summary.csv` - Runtime comparison
- `results/metrics/*.png` - Comparison plots

Predictions saved to:
- `results/prediction/{model_name}/*_pred.png` - Binary predictions
- `results/prediction/{model_name}/*_overlay.png` - Overlay visualization

---

## Technical Documentation

See [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md) for detailed explanation of:
- SAM automatic vs. prompt-guided segmentation
- Why histology-specific prompts improve segmentation
- Hybrid pipeline algorithm description
- Expected advantages and improvements

---

## Citation

If you use this benchmarking repository or the hybrid VeSpA+SAM pipeline, please cite:

```bibtex
@misc{vespa_benchmark_2024,
  title={VeSpA Benchmarking: Hybrid Foundation Model and Color-Prior Vessel Segmentation},
  author={[Your Name]},
  year={2024},
  howpublished={GitHub},
  url={https://github.com/[your-repo]/VeSpA_benchmarking}
}
```

---

## Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- OpenCV
- NumPy, Pandas, Matplotlib
- segment-anything
- ultralytics (YOLO)
- scikit-image

See [requirements.txt](requirements.txt) for full dependency list.

---

## License

[Add appropriate license]

---

## Contact

For questions or issues, please open an issue on GitHub.
