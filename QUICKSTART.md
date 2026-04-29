# Quick Start Guide: Running the VeSpA Benchmarking Pipeline

## Prerequisites

1. **Python 3.8+**
   ```bash
   python --version
   ```

2. **SAM Model** (if not already downloaded)
   ```bash
   mkdir -p models
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/sam_vit_b_01ec64.pth
   ```

3. **Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Full Benchmark

### Basic Benchmark (All 4 Methods)

```bash
# From repository root
python evaluation/benchmark.py
```

**What this does**:
1. Processes all images in `datasets/images/`
2. Runs VeSpA segmentation
3. Runs SAM automatic segmentation
4. Runs YOLOv8-seg segmentation
5. Runs VeSpA+SAM hybrid segmentation ✓ NEW
6. Computes metrics (Dice, IoU, Precision, Recall, Runtime)
7. Generates comparison plots
8. Saves results to CSVs

**Expected output**:
```
ROI_1.tiff | VeSpA | Dice=0.6776 IoU=0.5124 Precision=0.8171 Recall=0.6712 Runtime=2.57s
ROI_1.tiff | SAM | Dice=0.3385 IoU=0.2037 Precision=0.6509 Recall=0.1914 Runtime=23.35s
...
ROI_1.tiff | VeSpA+SAM (Hybrid) | Dice=? IoU=? Precision=? Recall=? Runtime=?s

Average Dice per model:
- VeSpA: X.XXXX
- SAM: X.XXXX
- YOLOv8-seg: X.XXXX
- VeSpA+SAM (Hybrid): X.XXXX (NEW)
```

---

## Output Files

### Metrics (CSV)

**`results/metrics/results_cat2.csv`** - All per-image results (default: intersection GT)
```
image_name,model,gt_category,dice,iou,precision,recall
ROI_1.tiff,VeSpA,category_2_intersection,0.6935,0.5309,0.7198,0.6692
ROI_1.tiff,SAM,category_2_intersection,0.3709,0.2276,0.6660,0.2570
...
```

**`results/metrics/summary_cat2.csv`** - Mean metrics per model
```
model,dice,iou,precision,recall
VeSpA,0.7417,0.5909,0.7054,0.7914
SAM,0.3239,0.1961,0.5768,0.2259
...
```

**`results/metrics/runtime_summary_cat2.csv`** - Speed comparison
```
model,runtime_seconds
VeSpA,2.43
SAM,66.30
...
```

**Category-specific files:**
- `results_cat1_SLR.csv` / `results_cat1_GG.csv` - Individual observer results
- `results_cat2.csv` - Intersection GT results (default)
- `results_cat3.csv` - Union GT results
- `results_all_categories.csv` - Combined results from all categories
```
model,runtime_seconds
YOLOv8-seg,0.59
VeSpA,2.57
SAM,23.35
VeSpA+SAM (Hybrid),?
```

### Plots (PNG)

**`results/metrics/dice_boxplot.png`** - Distribution of Dice scores
**`results/metrics/iou_comparison.png`** - Mean IoU bar chart
**`results/metrics/metrics_comparison.png`** - NEW: Comprehensive 4-metric comparison

### Predictions

**`results/prediction/{model_name}/{image_name}_pred.png`** - Binary prediction mask
**`results/prediction/{model_name}/{image_name}_overlay.png`** - Overlay visualization

Colors in overlays:
- 🟢 Green = Ground truth vessel
- 🔴 Red = Predicted vessel
- 🟡 Yellow = True positive (both)

---

## Configuration

### Environment Variables

**SAM Configuration**:
```bash
export SAM_MODEL_TYPE="vit_b"    # Options: vit_b (default), vit_l, vit_h
export SAM_MAX_DIM="1024"         # Resize max dimension
export SAM_CHECKPOINT="models/sam_vit_b_01ec64.pth"
```

**YOLO Configuration**:
```bash
export YOLO_SEG_WEIGHTS="models/yolov8n-seg.pt"
export YOLO_CONFIG_DIR=".cache/ultralytics"
```

**Hybrid Configuration** (VeSpA+SAM):
```bash
export MIN_STAIN_AREA="50"        # Filter tiny components
export BLUR_KERNEL="5"            # Stain smoothing
export CLOSING_KERNEL="5"         # Stain connectivity
```

**Example with custom config**:
```bash
export SAM_MODEL_TYPE="vit_l"
export SAM_MAX_DIM="2048"
python evaluation/benchmark.py
```

---

## Testing Individual Methods

### Test VeSpA
```python
from pipelines.VeSpA.run_vespa import run_model
mask = run_model("datasets/images/ROI_1.tiff")
print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
```

### Test SAM
```python
from pipelines.sam.run_sam import run_model
mask = run_model("datasets/images/ROI_1.tiff")
print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
```

### Test YOLO
```python
from pipelines.yolo.run_yolo import run_model
mask = run_model("datasets/images/ROI_1.tiff")
print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
```

### Test Hybrid (NEW)
```python
from pipelines.hybrid.run_vespa_sam import run_model
mask = run_model("datasets/images/ROI_1.tiff")
print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
python evaluation/benchmark.py
```

### SAM Model Not Found
```bash
# Verify download
ls -lh models/sam_vit_b_01ec64.pth

# Re-download if needed
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/
```

### No Images in Dataset
```bash
# Check dataset structure
ls -la datasets/images/
ls -la datasets/groundthruth/

# Verify naming convention
# Image: datasets/images/ROI_*.tiff
# Mask:  datasets/groundthruth/ROI_*_mask.png
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify segment-anything is installed
python -c "from segment_anything import sam_model_registry; print('OK')"
```

---

## Advanced Usage

### Benchmark Single Method

```python
from evaluation.benchmark import set_reproducible_seeds, load_mask, build_ground_truth_path
from evaluation.metrics import compute_all_metrics
from pipelines.hybrid.run_vespa_sam import run_model
from pathlib import Path
import cv2

image_path = Path("datasets/images/ROI_1.tiff")
gt_path = Path("datasets/groundthruth/ROI_1_mask.png")

image_bgr = cv2.imread(str(image_path))
gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

# Run hybrid pipeline
pred_mask = run_model(str(image_path))

# Compute metrics
metrics = compute_all_metrics(pred_mask, gt_mask)
print(f"Dice: {metrics['dice']:.4f}")
print(f"IoU: {metrics['iou']:.4f}")
```

### Batch Processing

```python
import time
from pathlib import Path
from pipelines.hybrid.run_vespa_sam import run_model

images_dir = Path("datasets/images")
for image_path in sorted(images_dir.glob("*.tiff")):
    start = time.perf_counter()
    mask = run_model(str(image_path))
    elapsed = time.perf_counter() - start
    print(f"{image_path.name}: {elapsed:.2f}s, mask shape: {mask.shape}")
```

---

## Expected Results

Based on baseline benchmarks:

| Model | Dice | IoU | Precision | Recall | Runtime |
|-------|------|-----|-----------|--------|---------|
| VeSpA | ~0.73 | ~0.58 | ~0.82 | ~0.67 | ~2.6s |
| SAM | ~0.30 | ~0.18 | ~0.65 | ~0.19 | ~23.3s |
| YOLOv8-seg | ~0.03 | ~0.02 | ~0.86 | ~0.02 | ~0.6s |
| **VeSpA+SAM** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

The hybrid method is expected to outperform automatic SAM while maintaining VeSpA's specificity.

---

## Documentation

For detailed technical information, see:
- **README.md** - Project overview
- **TECHNICAL_NOTES.md** - SAM prompting strategies
- **IMPLEMENTATION_SUMMARY.md** - Complete implementation details
- **manuscript/method.tex** - Formal methodology

---

## Next Steps

1. ✓ Install dependencies
2. ✓ Download SAM model
3. ✓ Run benchmark
4. ✓ Review results in `results/metrics/`
5. ✓ Update manuscript with results
6. ✓ Prepare figures for publication

---

**For questions or issues**: Refer to technical documentation files or review the code comments in implementation files.
