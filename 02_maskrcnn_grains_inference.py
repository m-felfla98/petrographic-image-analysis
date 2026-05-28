"""
=============================================================================
Mask R-CNN Grain Instance Segmentation — Inference Script
=============================================================================
Reference:
    Felfla, M. Sh. (2026). Generational shift in computer vision for fully
    automated petrographic image analysis.

Trained model weights:
    https://drive.google.com/file/d/1mplGIrgKFumBl1wXMaVcgbRjhBOgsp30/

Dependencies:
    pip install torch torchvision opencv-python matplotlib Pillow numpy

Usage:
    Set the three paths below, then run:
        python 02_maskrcnn_grains_inference.py
=============================================================================
"""

import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import cv2
from PIL import Image

# =============================================================================
#                          *** USER SETTINGS ***
#          Fill in the three paths below before running the script
# =============================================================================

MODEL_PATH  = ""          # Path to the downloaded .pth / .pth.tar model file
                          # Example: "C:/models/maskrcnn_grains.pth"

IMAGE_PATH  = ""          # Path to the input thin-section image
                          # Example: "C:/images/sample_01.jpg"

OUTPUT_DIR  = ""          # Directory where results will be saved
                          # Example: "C:/results/maskrcnn_grains/"

# =============================================================================
#              Advanced settings (change only if needed)
# =============================================================================

SCORE_THRESHOLD = 0.50    # Minimum confidence score to keep a detection (0–1)
MASK_THRESHOLD  = 0.50    # Threshold for converting soft masks to binary
NUM_CLASSES     = 2       # Background (0) + Grain (1)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
#                         Model Builder
# =============================================================================

def build_maskrcnn(num_classes, device):
    """
    Build a Mask R-CNN model with ResNet-50 + FPN backbone and
    replace the classification / mask heads for the target number of classes.
    """
    model = maskrcnn_resnet50_fpn(weights=None)

    # Replace box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)

    return model.to(device)


# =============================================================================
#                         Helper Functions
# =============================================================================

def load_model(path, num_classes, device):
    model = build_maskrcnn(num_classes, device)
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict",
            ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    return model


def generate_palette(n):
    """Return n visually distinct RGB colours (float, 0–1)."""
    base = []
    for cmap in [plt.cm.tab20, plt.cm.Set1, plt.cm.Dark2, plt.cm.Paired]:
        base.extend(list(cmap.colors))
    base = base * (n // len(base) + 1)
    np.random.seed(42)
    np.random.shuffle(base)
    return base[:n]


# =============================================================================
#                         Main Inference Function
# =============================================================================

def run_inference():

    # --- Validate user paths ---
    for label, path in [("MODEL_PATH", MODEL_PATH),
                         ("IMAGE_PATH", IMAGE_PATH),
                         ("OUTPUT_DIR", OUTPUT_DIR)]:
        if not path:
            raise ValueError(
                f"{label} is empty. Please fill in the path at the top of the script.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load model ---
    print(f"[1/4] Loading Mask R-CNN model from:\n      {MODEL_PATH}")
    model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
    print(f"      Device: {DEVICE}")

    # --- Load image ---
    print(f"[2/4] Reading image from:\n      {IMAGE_PATH}")
    original = Image.open(IMAGE_PATH).convert("RGB")
    W, H     = original.size

    img_tensor = TF.to_tensor(original).to(DEVICE)  # shape [3, H, W], values 0–1

    # --- Predict ---
    print("[3/4] Running instance segmentation …")
    with torch.no_grad():
        outputs = model([img_tensor])

    boxes   = outputs[0]["boxes"].cpu().numpy()
    scores  = outputs[0]["scores"].cpu().numpy()
    masks   = outputs[0]["masks"].cpu().numpy()   # shape [N, 1, H, W]

    # Keep only high-confidence predictions
    keep    = scores >= SCORE_THRESHOLD
    boxes   = boxes[keep]
    scores  = scores[keep]
    masks   = masks[keep]
    n_grains = len(scores)

    print(f"[4/4] Detected {n_grains} grains (score ≥ {SCORE_THRESHOLD}). Saving results …")

    # --- Build composite visualisation ---
    colours   = generate_palette(max(n_grains, 1))
    fig, axes = plt.subplots(1, 2, figsize=(W * 2 / 100, H / 100), dpi=100)

    # Panel 1 — original image
    axes[0].imshow(original)
    axes[0].set_title("Input Image", fontsize=10)
    axes[0].axis("off")

    # Panel 2 — instance overlay
    axes[1].imshow(original)
    axes[1].set_title(f"Mask R-CNN — {n_grains} grains detected", fontsize=10)
    axes[1].axis("off")

    img_np  = np.array(original)
    for i, (mask, score, colour) in enumerate(zip(masks, scores, colours)):
        binary = (mask[0] > MASK_THRESHOLD).astype(np.uint8)  # [H, W]

        # Semi-transparent colour fill
        overlay        = np.zeros((*binary.shape, 4), dtype=np.float32)
        overlay[binary == 1] = (*colour[:3], 0.55)
        axes[1].imshow(overlay)

        # Bounding box
        x1, y1, x2, y2 = boxes[i]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0.8, edgecolor=colour[:3], facecolor="none")
        axes[1].add_patch(rect)

        # Grain ID label at centroid
        cy_c, cx_c = np.where(binary)
        if len(cx_c):
            axes[1].text(
                cx_c.mean(), cy_c.mean(), str(i + 1),
                color="white", fontsize=6, fontweight="bold",
                ha="center", va="center",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none",
                          boxstyle="round,pad=0.2"))

    plt.tight_layout(pad=0.5)

    base        = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    fig_path    = os.path.join(OUTPUT_DIR, f"{base}_grains_maskrcnn.png")

    # Binary instance-label map (each grain has a unique integer label)
    label_map   = np.zeros((H, W), dtype=np.uint16)
    for i, mask in enumerate(masks):
        binary              = (mask[0] > MASK_THRESHOLD)
        label_map[binary]   = i + 1
    label_path  = os.path.join(OUTPUT_DIR, f"{base}_labelmap_maskrcnn.png")

    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    cv2.imwrite(label_path, label_map.astype(np.uint16))

    print("\n  Results saved:")
    print(f"  • Overlay image  : {fig_path}")
    print(f"  • Instance labels: {label_path}")
    print(f"  • Total grains   : {n_grains}")
    print("  Done.")


# =============================================================================

if __name__ == "__main__":
    run_inference()
