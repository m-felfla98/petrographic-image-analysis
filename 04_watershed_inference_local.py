# =============================================================================
# Watershed Grain Segmentation — Local Environment
# =============================================================================
# Usage:
#   1. Fill in the three paths below
#   2. Adjust parameters in the PARAMETERS section if needed
#   3. Run:  python 04_watershed_inference_local.py
# =============================================================================

import cv2
import numpy as np
import json
import os

# ==============================================================================
# PATHS  ← fill these in before running
# ==============================================================================
IMAGE_PATH  = ""          # e.g.  r"C:\data\thin_section.jpg"
OUTPUT_DIR  = ""          # e.g.  r"C:\data\watershed_output"
# Optional: path to an existing JSON annotation file for overlay visualisation
# Leave as "" to skip overlay and run segmentation only
JSON_PATH   = ""

# ==============================================================================
# PARAMETERS  ← adjust to tune segmentation quality
# ==============================================================================
# Gaussian blur kernel size (must be odd).  Larger → smoother, less noise.
BLUR_KERNEL         = 5          # recommended range: 3–11

# Morphological opening iterations — removes small foreground noise
MORPH_OPEN_ITER     = 2          # recommended range: 1–4

# Morphological dilation iterations — expands sure-foreground region
MORPH_DILATE_ITER   = 3          # recommended range: 1–6

# Distance transform threshold (fraction of max distance).
# Higher → only very central pixels treated as definite grain interiors.
DIST_THRESHOLD      = 0.5        # recommended range: 0.3–0.7

# Overlay transparency (0 = fully transparent, 1 = fully opaque)
OVERLAY_ALPHA       = 0.6        # recommended range: 0.4–0.8

# Minimum contour area in pixels — grains smaller than this are ignored
MIN_GRAIN_AREA      = 200        # recommended range: 50–1000

# ==============================================================================
# HELPERS
# ==============================================================================

def run_watershed(image_bgr):
    """Apply watershed segmentation and return (labels, num_grains)."""
    gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)

    # Otsu threshold → binary foreground
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                               kernel, iterations=MORPH_OPEN_ITER)

    # Sure background (dilation)
    sure_bg = cv2.dilate(opening, kernel, iterations=MORPH_DILATE_ITER)

    # Sure foreground (distance transform)
    dist    = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist,
                               DIST_THRESHOLD * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(image_bgr, markers)
    return markers, num_labels - 1   # subtract background label


def build_overlay(image_bgr, markers):
    """Colour each grain with a gradient overlay; return blended image."""
    unique = [m for m in np.unique(markers) if m > 1]   # skip bg & border
    overlay = np.zeros_like(image_bgr)

    pale = np.array([255, 220, 180])   # BGR — light blue
    deep = np.array([180,   0,   0])   # BGR — deep blue
    n    = max(len(unique) - 1, 1)

    for i, label in enumerate(unique):
        mask = (markers == label).astype(np.uint8)
        # Filter small regions
        if mask.sum() < MIN_GRAIN_AREA:
            continue
        t = i / n
        color = ((1 - t) * pale + t * deep).astype(np.uint8).tolist()
        overlay[markers == label] = color

    return cv2.addWeighted(overlay, OVERLAY_ALPHA,
                           image_bgr, 1 - OVERLAY_ALPHA, 0)


def extract_grain_polygons(markers):
    """Return a list of grain dicts with polygon and area."""
    grains = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = (markers == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_GRAIN_AREA:
                continue
            polygon = cnt.reshape(-1, 2).tolist()
            grains.append({"label": int(label),
                            "area_px": float(area),
                            "polygon": polygon})
    return grains


def load_json_overlay(image_bgr, json_path):
    """Overlay polygons from an existing JSON annotation file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Support both list-of-grains and COCO-style formats
    grains = data if isinstance(data, list) else data.get("annotations", [])
    overlay = image_bgr.copy()

    pale = np.array([255, 220, 180])
    deep = np.array([180,   0,   0])
    n    = max(len(grains) - 1, 1)

    for i, grain in enumerate(grains):
        polygon = grain.get("polygon", None)
        if polygon is None:
            continue
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        t = i / n
        color = ((1 - t) * pale + t * deep).astype(np.uint8).tolist()
        cv2.fillPoly(overlay, [pts], color)

    return cv2.addWeighted(overlay, OVERLAY_ALPHA,
                           image_bgr, 1 - OVERLAY_ALPHA, 0)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # --- validate inputs ---
    assert IMAGE_PATH, "IMAGE_PATH is empty — fill it in at the top of the script."
    assert OUTPUT_DIR, "OUTPUT_DIR is empty — fill it in at the top of the script."
    assert os.path.exists(IMAGE_PATH), f"Image not found: {IMAGE_PATH}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = cv2.imread(IMAGE_PATH)
    assert image is not None, f"Could not read image: {IMAGE_PATH}"
    h, w = image.shape[:2]
    print(f"Image loaded: {w}×{h} px")

    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    # ── Mode A: JSON overlay (visualisation only) ──────────────────────────
    if JSON_PATH:
        assert os.path.exists(JSON_PATH), f"JSON not found: {JSON_PATH}"
        print("JSON_PATH provided — running overlay visualisation only.")
        blended = load_json_overlay(image, JSON_PATH)
        out_path = os.path.join(OUTPUT_DIR, f"{base}_json_overlay.jpg")
        cv2.imwrite(out_path, blended)
        print(f"Overlay saved → {out_path}")
        cv2.imshow("JSON Overlay", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── Mode B: Full watershed segmentation ────────────────────────────────
    print("Running watershed segmentation …")
    markers, num_grains = run_watershed(image)
    print(f"Grains detected (before area filter): {num_grains}")

    # Coloured overlay
    blended  = build_overlay(image, markers)
    out_img  = os.path.join(OUTPUT_DIR, f"{base}_watershed_overlay.jpg")
    cv2.imwrite(out_img, blended)
    print(f"Overlay saved → {out_img}")

    # Border image (watershed lines = -1)
    border_img = image.copy()
    border_img[markers == -1] = [0, 0, 255]
    out_border = os.path.join(OUTPUT_DIR, f"{base}_watershed_borders.jpg")
    cv2.imwrite(out_border, border_img)
    print(f"Borders saved → {out_border}")

    # JSON export
    grains     = extract_grain_polygons(markers)
    out_json   = os.path.join(OUTPUT_DIR, f"{base}_grains.json")
    with open(out_json, "w") as f:
        json.dump(grains, f, indent=2)
    print(f"Polygons saved → {out_json}")
    print(f"Grains exported (after area filter ≥{MIN_GRAIN_AREA} px): {len(grains)}")

    # Summary statistics
    areas = [g["area_px"] for g in grains]
    if areas:
        print(f"\n── Grain Statistics ──────────────────")
        print(f"  Count        : {len(areas)}")
        print(f"  Mean area    : {np.mean(areas):.1f} px²")
        print(f"  Median area  : {np.median(areas):.1f} px²")
        print(f"  Min / Max    : {np.min(areas):.0f} / {np.max(areas):.0f} px²")
        img_area = h * w
        covered  = sum(areas) / img_area * 100
        print(f"  Coverage     : {covered:.1f} % of image")

    # Display
    cv2.namedWindow("Watershed Overlay", cv2.WINDOW_NORMAL)
    cv2.imshow("Watershed Overlay", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
