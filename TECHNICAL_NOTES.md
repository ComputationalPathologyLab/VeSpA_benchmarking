# Technical Notes: SAM Prompting Strategies and Hybrid Approaches

## Executive Summary

This document explains the difference between **automatic SAM segmentation** (current approach) and **prompt-guided SAM segmentation** (proposed enhancement), with particular application to histology image segmentation with DAB immunostaining.

---

## 1. SAM Automatic Mask Generation (Current Implementation)

### Overview
The current `run_sam.py` uses `SamAutomaticMaskGenerator`, which applies a **grid-based prompting strategy** without domain knowledge:

```python
SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,         # 16×16 = 256 total prompt points
    pred_iou_thresh=0.8,        # IoU confidence threshold
    stability_score_thresh=0.9, # Stability threshold
    crop_n_layers=0,            # No hierarchical crops
    min_mask_region_area=64,    # Minimum connected component size
)
```

### Mechanism
- Distributes 16×16 points uniformly across the image
- Runs SAM prediction at each point
- Uses IoU and stability scores to filter predictions
- Combines all masks into a single binary output
- **No awareness of image domain (histology) or stain properties**

### Advantages
✓ Automatic—no manual prompting required  
✓ Detects diverse objects  
✓ Works on general images  

### Limitations
✗ **Non-specific**: Detects all objects (noise, debris, overlapping structures)  
✗ **Domain-agnostic**: Ignores immunostain signal (brown DAB)  
✗ **False positives**: May segment non-vessel structures  
✗ **Boundary quality**: Produces jagged boundaries due to grid-based sampling  

---

## 2. Prompt-Guided SAM Segmentation (Proposed Enhancement)

### Why Prompts Matter

SAM supports **three types of prompts**:

| Prompt Type | Usage | Benefit |
|---|---|---|
| **Point prompts** | Positive: (x, y) with label=1 | Targets specific region centers |
| **Box prompts** | Bounding box: (x_min, y_min, x_max, y_max) | Restricts search region |
| **Mask prompts** | Binary mask conditioning | Guides segmentation from preliminary mask |

### SamPredictor API

```python
from segment_anything import SamPredictor

predictor = SamPredictor(sam)
predictor.set_image(image_rgb)  # Set image once

# Point-based prompting
masks, scores, logits = predictor.predict(
    point_coords=np.array([[x1, y1], [x2, y2]]),
    point_labels=np.array([1, 1]),          # 1=foreground, 0=background
    multimask_output=True                   # Returns 3 candidates
)
# Select mask: masks[np.argmax(scores)]
```

### Key Differences

| Aspect | Automatic | Prompt-Guided |
|---|---|---|
| **Approach** | Grid-based coverage | Target-specific regions |
| **Specificity** | All objects | Only prompts regions |
| **Domain knowledge** | None | Can incorporate priors |
| **Boundary quality** | Coarse grid sampling | Refined from prompts |
| **False positives** | Higher | Lower (constrained to prompts) |
| **Inference cost** | Single pass per scale | Per-prompt prediction |

---

## 3. Histology-Specific Prompting Strategy

### Problem Domain
- **Images**: Histological sections with DAB immunostaining
- **Target**: Endothelial vessel networks (stained brown)
- **Challenge**: Varying staining intensity, overlapping structures, image artifacts

### Solution: Brown Stain Detection as Prompt Source

#### Step 1: Detect Brown Stain (VeSpA Color Prior)
Extract brown-colored pixels using color space analysis:

```python
# Convert to CMYK (captures stain)
bgr_normalized = image_bgr / 255.0
K = 1 - np.max([bgr_normalized[..., 0], bgr_normalized[..., 1], bgr_normalized[..., 2]], axis=0)
# ... CMYK extraction
Y_channel = CMYK[..., 2]

# Otsu threshold on yellow channel
_, stain_mask = cv2.threshold(Y_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

**Result**: Binary mask of detected stain regions

#### Step 2: Generate Prompts from Stain Mask

```python
# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(stain_mask)

# Extract prompts
point_prompts = centroids[1:]  # Skip background (label 0)
point_labels = np.ones(len(point_prompts))  # All positive prompts

# Optional: Bounding boxes
boxes = []
for i in range(1, num_labels):
    x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_WIDTH+cv2.CC_STAT_HEIGHT+1]
    boxes.append([x, y, x+w, y+h])
```

**Result**: Points and boxes targeting brown-stained regions

#### Step 3: SAM Refinement with Prompts

```python
predictor.set_image(image_rgb)

# Predict for each vessel (brown component)
refined_masks = []
for i, point in enumerate(point_prompts):
    mask, score, _ = predictor.predict(
        point_coords=np.array([point], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=False
    )
    refined_masks.append(mask)

# Combine
combined = np.logical_or.reduce(refined_masks)
```

**Result**: SAM-refined boundaries using brown stain guidance

---

## 4. Why This Approach Works

### Advantages of Hybrid VeSpA+SAM

| Property | Benefit |
|---|---|
| **Color prior** | Focuses on histologically relevant structures (brown stain) |
| **Foundation model** | SAM's universality refines VeSpA's sometimes-imperfect boundaries |
| **Specificity** | Reduces false positives vs. automatic SAM |
| **Robustness** | Handles staining variability through two-stage process |
| **Modularity** | Can swap SAM variants (ViT-B, ViT-L, ViT-H) without changing pipeline |

### Expected Improvements

Compared to automatic SAM:
- ↑ **Precision**: Fewer non-vessel detections
- ↑ **Recall**: Better detection of all stained regions
- ↑ **Boundary quality**: Refined contours from prompts
- ↓ **False positives**: Constrained to stain regions

Compared to VeSpA alone:
- ↑ **Boundary accuracy**: SAM correction of morphological artifacts
- ↑ **Flexibility**: Adapts to SAM foundation model improvements
- ↑ **Consistency**: Less sensitive to kernel size parameter tuning

---

## 5. Implementation Notes

### File: `pipelines/hybrid/run_vespa_sam.py`

The hybrid pipeline implements:

1. **VeSpA component initialization** (reusable CMYK extraction)
2. **Stain-based prompt generation** (connected components)
3. **SAM predictor setup** (singleton pattern for efficiency)
4. **Per-vessel refinement** (prompt-guided prediction)
5. **Post-processing** (morphological cleanup, small object removal)

### Configuration Parameters

Key tunable hyperparameters:
- `min_stain_region_area`: Threshold to ignore tiny components
- `max_stain_region_area`: Threshold to ignore huge background regions
- `morphology_kernel_size`: Post-processing refinement
- `use_multimask_output`: Whether to select best SAM candidate

---

## 6. Benchmark Methodology

All methods evaluated on:
- **Metric**: Dice coefficient, IoU, Precision, Recall
- **Dataset**: Histology images with DAB immunostaining
- **Ground truth**: Manual annotations of endothelial networks
- **Variants tested**:
  - SAM (automatic, current)
  - YOLOv8-seg (pre-trained)
  - VeSpA (color-based, current)
  - **VeSpA+SAM (hybrid, proposed)** ← NEW

---

## 7. Future Extensions

1. **Negative prompts**: Mark non-vessel regions to exclude
2. **Iterative prompting**: Refine prompts based on SAM corrections
3. **Stain-adaptive thresholding**: Learn threshold per image
4. **SAM model ensembling**: Combine predictions from ViT-B/L/H
5. **Real-time feedback**: Interactive prompt adjustment

---

## References

- **SAM**: Kirillov et al. "Segment Anything" (https://arxiv.org/abs/2304.02643)
- **VeSpA**: Original CMYK-based vessel segmentation (from this repository)
- **DAB Immunostaining**: Standard histology technique for endothelial markers (CD31, PECAM-1)
