import os
import cv2
import numpy as np
from skimage.measure import label, regionprops_table, regionprops
import pandas as pd
from pathlib import Path
import re


# ─────────────────────────────────────────────
#  TUNABLE PARAMETERS
# ─────────────────────────────────────────────

# Lumen detection
LUMEN_AREA_MIN        = 200       # px²  – ignore tiny noise holes
LUMEN_AREA_MAX        = 80_000    # px²  – ignore artefactually large holes
LUMEN_CIRCULARITY_MIN = 0.20      # 0–1  – low to allow elongated/irregular shapes
LUMEN_ECCENTRICITY_MAX = 0.97     # reject near-perfect lines (fragmentation artefacts)

# Wall-repair closing kernel (used before lumen detection)
WALL_CLOSE_KSIZE = 28             # px   – increase if vessel walls are very fragmented
WALL_CLOSE_ITER  = 2              # increase to improve wall closing

# Initial morphological cleanup
DILATE_KSIZE  = 21                # px   – dilation kernel for initial binary cleanup
DILATE_ITER   = 1
ERODE_KSIZE   = 3                 # px   – erosion kernel for initial binary cleanup
ERODE_ITER    = 2

# Lumen expansion (merges lumen onto inner wall boundary after detection)
LUMEN_EXPAND_KSIZE = 5            # px   – expansion kernel size
LUMEN_EXPAND_ITER  = 3            # increase to bridge larger inner-wall gaps

# Minimum vessel area to keep after all filtering
VESSEL_AREA_MIN = 500             # px²


# ─────────────────────────────────────────────
#  CORE LUMEN-FILLING LOGIC
# ─────────────────────────────────────────────

def fill_vessel_lumens(binary_walls: np.ndarray) -> np.ndarray:
    """
    Detect and fill vessel lumens in four steps:
      1. Repair fragmented walls via morphological closing (lumen detection only).
      2. Flood-fill from the image border to identify true background;
         candidate lumens are everything that is neither background nor wall.
      3. Filter candidates by area, eccentricity, and circularity.
      4. Expand validated lumens by a few px and merge onto the original
         (pre-repair) wall mask to preserve contour precision.

    Returns the filled mask.
    """
    h, w = binary_walls.shape

    # ── 1. Repair fragmented walls ─────────────────────────────────────
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (WALL_CLOSE_KSIZE, WALL_CLOSE_KSIZE))
    repaired = cv2.morphologyEx(binary_walls, cv2.MORPH_CLOSE, kernel,
                                iterations=WALL_CLOSE_ITER)

    # ── 2. Identify candidate lumen regions ────────────────────────────
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h+1, 1:w+1] = repaired
    cv2.floodFill(padded, None, (0, 0), 128)   # 128 = "background" label
    background      = padded[1:h+1, 1:w+1] == 128
    candidate_holes = (~background & ~repaired.astype(bool)).astype(np.uint8) * 255

    # ── 3. Filter candidates by shape ─────────────────────────────────
    labeled    = label(candidate_holes)
    lumen_mask = np.zeros((h, w), dtype=np.uint8)
    for region in regionprops(labeled):
        if not (LUMEN_AREA_MIN <= region.area <= LUMEN_AREA_MAX):
            continue
        if region.eccentricity > LUMEN_ECCENTRICITY_MAX:
            continue
        perim = region.perimeter_crofton
        circ  = (4 * np.pi * region.area) / (perim ** 2) if perim > 1e-6 else 0.0
        if circ < LUMEN_CIRCULARITY_MIN:
            continue
        lumen_mask[labeled == region.label] = 255

    # ── 4. Expand and merge onto original walls ────────────────────────
    expand_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (LUMEN_EXPAND_KSIZE, LUMEN_EXPAND_KSIZE))
    lumen_expanded = cv2.dilate(lumen_mask, expand_kernel, iterations=LUMEN_EXPAND_ITER)
    return cv2.bitwise_or(binary_walls, lumen_expanded)


# ─────────────────────────────────────────────
#  PER-IMAGE PROCESSING
# ─────────────────────────────────────────────

def process_image(input_path: str, output_dir: str,
                  threshold_mode: str = "otsu",
                  percentile: int | None = None) -> int:
    """
    Process a single image: segment vessels and fill their lumens.
    Returns the number of vessels detected.
    """
    base_name        = Path(input_path).stem
    image_output_dir = Path(output_dir) / base_name
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load image once; derive all needed data from it ────────
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {input_path}")

    # ── Step 2: Convert to CMYK Yellow channel ─────────────────────────
    img_f = img_bgr.astype(np.float32) / 255.0
    K         = 1 - np.max(img_f, axis=2)
    Y_channel = ((1 - img_f[:, :, 0] - K) / (1 - K + 1e-10) * 255).astype(np.uint8)

    # ── Step 3: Threshold ──────────────────────────────────────────────
    if threshold_mode == "otsu":
        _, binary = cv2.threshold(Y_channel, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print("  Thresholding: Otsu")
    elif threshold_mode == "percentile":
        if percentile is None:
            raise ValueError("threshold_mode is 'percentile' but no percentile value was provided.")
        thresh_value = int(np.percentile(Y_channel, percentile))
        _, binary    = cv2.threshold(Y_channel, thresh_value, 255,
                                     cv2.THRESH_BINARY)
        print(f"  Thresholding: {percentile}th percentile (value={thresh_value})")
    else:
        raise ValueError(f"Invalid threshold_mode: '{threshold_mode}'.")

    # ── Step 4: Initial morphological cleanup ─────────────────────────
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (DILATE_KSIZE, DILATE_KSIZE))
    dilated  = cv2.dilate(binary, kernel_d, iterations=DILATE_ITER)

    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (ERODE_KSIZE, ERODE_KSIZE))
    eroded   = cv2.erode(dilated, kernel_e, iterations=ERODE_ITER)

    # ── Step 5: Contour refinement (keep large vessels only) ───────────
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w      = eroded.shape[:2]
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) < VESSEL_AREA_MIN:
            continue
        cv2.drawContours(wall_mask, [cnt], -1, 255, -1)

    # ── Step 6: Fill lumens ────────────────────────────────────────────
    print("  Filling vessel lumens …")
    filled_mask = fill_vessel_lumens(wall_mask)

    # ── Step 7: Measurements on filled mask ───────────────────────────
    label_img = label(filled_mask)

    props = regionprops_table(
        label_img,
        properties=["area", "axis_major_length", "axis_minor_length",
                    "eccentricity", "orientation"]
    )
    measurements_df = pd.DataFrame(props)

    csv_path = image_output_dir / f"{base_name}_measurements.csv"
    measurements_df.to_csv(csv_path, index=True)
    print(f"Saved measurements: {csv_path}")

    # ── Step 8: Save filled binary mask ───────────────────────────────
    binary_path = image_output_dir / f"{base_name}_binary.png"
    if not cv2.imwrite(str(binary_path), filled_mask):
        raise IOError(f"Failed to write binary mask: {binary_path}")
    print(f"Saved filled binary mask: {binary_path}")

    # ── Step 9: Colour overlays ────────────────────────────────────────
    # Green = full vessel (walls + filled lumens)
    overlay = np.zeros_like(img_bgr)
    overlay[:, :, 1] = filled_mask   # green channel → full vessel (same in BGR and RGB)
    blended = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)

    overlay_path = image_output_dir / f"{base_name}_overlay.png"
    if not cv2.imwrite(str(overlay_path), blended):
        raise IOError(f"Failed to write overlay: {overlay_path}")
    print(f"Saved overlay (green=vessel): {overlay_path}")

    # ── Step 10: Statistics ────────────────────────────────────────────
    n_vessels = len(measurements_df)
    print(f"\nVessel Statistics for {base_name}:")
    print(f"  Total vessels    : {n_vessels}")
    print(f"  Average area     : {measurements_df['area'].mean():.2f}")
    print(f"  Total area       : {measurements_df['area'].sum():.2f}")
    print(f"  Avg eccentricity : {measurements_df['eccentricity'].mean():.3f}\n")

    return n_vessels


# ─────────────────────────────────────────────
#  FOLDER-LEVEL PROCESSING
# ─────────────────────────────────────────────

def natural_sort_key(path: Path) -> list:
    """Sort paths so that e.g. image_2.png comes before image_10.png."""
    parts = re.split(r'(\d+)', path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def process_folder(input_folder: str, output_folder: str,
                   threshold_mode: str = "otsu",
                   percentile: int | None = None) -> None:
    """Process all PNG files in the input folder."""
    png_files = sorted(Path(input_folder).glob("*.png"), key=natural_sort_key)

    if not png_files:
        print(f"No PNG files found in {input_folder}")
        return

    mode_label = f"percentile ({percentile}th)" if threshold_mode == "percentile" else threshold_mode
    print(f"Found {len(png_files)} PNG files to process")
    print(f"Threshold mode: {mode_label}\n" + "=" * 60)

    total_vessels = 0
    for i, png_file in enumerate(png_files, 1):
        print(f"\nProcessing [{i}/{len(png_files)}]: {png_file.name}")
        print("-" * 60)
        try:
            total_vessels += process_image(str(png_file), output_folder,
                                           threshold_mode, percentile)
        except (ValueError, IOError) as e:
            print(f"Error processing {png_file.name}: {e}")
        except Exception:
            print(f"Unexpected error processing {png_file.name}")
            raise

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Total vessels detected across all images: {total_vessels}")
    print(f"Output saved to: {output_folder}")


# ─────────────────────────────────────────────
#  run_model() — compatibility entry point
# ─────────────────────────────────────────────

def run_model(image_path: str) -> np.ndarray:
    """
    Run the VeSpA segmentation pipeline for a single image.

    Parameters
    ----------
    image_path : str
        Path to the input image.

    Returns
    -------
    np.ndarray
        Binary segmentation mask (H x W), dtype uint8, values in {0, 255}.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # CMYK Yellow channel
    img_f = img_bgr.astype(np.float32) / 255.0
    K         = 1 - np.max(img_f, axis=2)
    Y_channel = ((1 - img_f[:, :, 0] - K) / (1 - K + 1e-10) * 255).astype(np.uint8)

    # Otsu threshold
    _, binary = cv2.threshold(Y_channel, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Initial morphological cleanup
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KSIZE, DILATE_KSIZE))
    dilated  = cv2.dilate(binary, kernel_d, iterations=DILATE_ITER)

    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_KSIZE, ERODE_KSIZE))
    eroded   = cv2.erode(dilated, kernel_e, iterations=ERODE_ITER)

    # Keep large vessels only
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = eroded.shape[:2]
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= VESSEL_AREA_MIN:
            cv2.drawContours(wall_mask, [cnt], -1, 255, -1)

    # Fill lumens and return
    return fill_vessel_lumens(wall_mask)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def get_threshold_mode() -> tuple[str, int | None]:
    """
    Prompt the user to choose a thresholding method.
    Returns (mode, percentile) where percentile is None for Otsu mode.
    """
    def prompt_percentile(default: int = 10) -> int:
        """Prompt for a percentile value in 1–99, with a suggested default."""
        while True:
            raw = input(f"  Enter percentile (1–99) [suggested: {default}]: ").strip()
            if raw == "":
                print(f"→ Selected: {default}th percentile thresholding\n")
                return default
            if raw.isdigit() and 1 <= int(raw) <= 99:
                value = int(raw)
                print(f"→ Selected: {value}th percentile thresholding\n")
                return value
            print("  Invalid input. Please enter a whole number between 1 and 99.")

    print("\n" + "=" * 60)
    print("SELECT THRESHOLDING METHOD")
    print("=" * 60)
    print("  [1] Otsu thresholding (automatic)")
    print("  [2] Percentile thresholding (manual)")
    print("=" * 60)

    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            print("→ Selected: Otsu thresholding\n")
            return "otsu", None
        if choice == "2":
            return "percentile", prompt_percentile()
        print("  Invalid input. Please enter 1 or 2.")


def main() -> None:
    # ============ SET YOUR PATHS HERE ============
    input_folder  = '/Users/giuliagrion/Downloads/General/Ricerca/APA/Codes/Inputs for seg'
    output_folder = '/Users/giuliagrion/Downloads/General/Ricerca/APA/Codes/outputs for valid'
    # ============================================

    threshold_mode, percentile = get_threshold_mode()   # UI stays in main()
    process_folder(input_folder, output_folder, threshold_mode, percentile)


if __name__ == "__main__":
    main()