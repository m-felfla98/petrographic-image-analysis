# =============================================================================
# Watershed Grain Segmentation — Google Colab Version
# =============================================================================
# Instructions:
#   Copy each CELL block into a separate Colab cell and run in order.
#   You do NOT need to install anything manually — Cell 1 handles it.
# =============================================================================


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install dependencies                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""
!pip install opencv-python-headless numpy matplotlib -q
print("✅ Dependencies ready.")
"""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Parameters  ← adjust here before running             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""
# ── Segmentation parameters ────────────────────────────────────────
BLUR_KERNEL        = 5      # Gaussian blur kernel (odd number, 3–11)
MORPH_OPEN_ITER    = 2      # Morphological opening iterations (1–4)
MORPH_DILATE_ITER  = 3      # Dilation iterations for sure-background (1–6)
DIST_THRESHOLD     = 0.5    # Distance transform threshold (0.3–0.7)
OVERLAY_ALPHA      = 0.6    # Overlay transparency (0.0–1.0)
MIN_GRAIN_AREA     = 200    # Minimum grain area in pixels to keep

print("✅ Parameters set.")
"""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Upload your image                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""
from google.colab import files
import cv2
import numpy as np
import os, json

print("📂 Upload your thin-section image (JPG or PNG):")
uploaded = files.upload()

IMAGE_NAME  = list(uploaded.keys())[0]
IMAGE_PATH  = f"/content/{IMAGE_NAME}"
OUTPUT_DIR  = "/content/watershed_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

image = cv2.imread(IMAGE_PATH)
h, w  = image.shape[:2]
print(f"✅ Image loaded: {w}×{h} px")
"""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Run watershed segmentation                            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def run_watershed(img):
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel   = np.ones((3, 3), np.uint8)
    opening  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                                kernel, iterations=MORPH_OPEN_ITER)
    sure_bg  = cv2.dilate(opening, kernel, iterations=MORPH_DILATE_ITER)
    dist     = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, DIST_THRESHOLD * dist.max(), 255, 0)
    sure_fg  = np.uint8(sure_fg)
    unknown  = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers  = markers + 1
    markers[unknown == 255] = 0
    markers  = cv2.watershed(img, markers)
    return markers, n - 1

def build_overlay(img, markers):
    unique  = [m for m in np.unique(markers) if m > 1]
    overlay = np.zeros_like(img)
    pale = np.array([255, 220, 180])
    deep = np.array([180,   0,   0])
    n    = max(len(unique) - 1, 1)
    for i, label in enumerate(unique):
        if (markers == label).sum() < MIN_GRAIN_AREA:
            continue
        t = i / n
        color = ((1-t)*pale + t*deep).astype(np.uint8).tolist()
        overlay[markers == label] = color
    return cv2.addWeighted(overlay, OVERLAY_ALPHA, img, 1-OVERLAY_ALPHA, 0)

def extract_grains(markers):
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
            grains.append({"label": int(label),
                            "area_px": float(area),
                            "polygon": cnt.reshape(-1, 2).tolist()})
    return grains

# ── Run ────────────────────────────────────────────────────────────
print("⚙️  Running watershed segmentation …")
markers, num_raw = run_watershed(image)
blended          = build_overlay(image, markers)
grains           = extract_grains(markers)

print(f"✅ Done — {len(grains)} grains detected (after area filter ≥{MIN_GRAIN_AREA} px)")

# ── Statistics ─────────────────────────────────────────────────────
areas = [g["area_px"] for g in grains]
if areas:
    print(f"\n── Grain Statistics ──────────────────")
    print(f"  Count        : {len(areas)}")
    print(f"  Mean area    : {np.mean(areas):.1f} px²")
    print(f"  Median area  : {np.median(areas):.1f} px²")
    print(f"  Min / Max    : {np.min(areas):.0f} / {np.max(areas):.0f} px²")
    covered = sum(areas) / (h*w) * 100
    print(f"  Coverage     : {covered:.1f} % of image")

# ── Display ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].imshow(cv2.cvtColor(image,   cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image",   fontsize=13)
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Watershed Overlay  ({len(grains)} grains)", fontsize=13)
axes[1].axis("off")
plt.tight_layout()
plt.show()
"""


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Save results and download                             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""
base = os.path.splitext(IMAGE_NAME)[0]

# Overlay image
out_img = os.path.join(OUTPUT_DIR, f"{base}_watershed_overlay.jpg")
cv2.imwrite(out_img, blended)

# Watershed borders
border_img = image.copy()
border_img[markers == -1] = [0, 0, 255]
out_border = os.path.join(OUTPUT_DIR, f"{base}_watershed_borders.jpg")
cv2.imwrite(out_border, border_img)

# JSON polygons
out_json = os.path.join(OUTPUT_DIR, f"{base}_grains.json")
with open(out_json, "w") as f:
    json.dump(grains, f, indent=2)

print(f"✅ Files saved in {OUTPUT_DIR}:")
print(f"   {os.path.basename(out_img)}")
print(f"   {os.path.basename(out_border)}")
print(f"   {os.path.basename(out_json)}")

# Download all outputs
from google.colab import files as colab_files
colab_files.download(out_img)
colab_files.download(out_border)
colab_files.download(out_json)
print("⬇️  Download started.")
"""
