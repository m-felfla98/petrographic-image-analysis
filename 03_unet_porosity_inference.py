"""
=============================================================================
U-Net Porosity Mapping — Inference Script
=============================================================================
Reference:
    Felfla, M. Sh. (2026). Generational shift in computer vision for fully
    automated petrographic image analysis.

Trained model weights:
    https://drive.google.com/file/d/1aZar1y3bLAR58Tdd5rcNn9-TPiXnCQlM/

Dependencies:
    pip install torch torchvision opencv-python matplotlib Pillow numpy

Usage:
    Set the three paths below, then run:
        python 03_unet_porosity_inference.py
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
from PIL import Image

# =============================================================================
#                          *** USER SETTINGS ***
#          Fill in the three paths below before running the script
# =============================================================================

MODEL_PATH  = ""          # Path to the downloaded .pth.tar model file
                          # Example: "C:/models/unet_porosity.pth.tar"

IMAGE_PATH  = ""          # Path to the input thin-section image
                          # Example: "C:/images/sample_01.jpg"

OUTPUT_DIR  = ""          # Directory where results will be saved
                          # Example: "C:/results/unet_porosity/"

# =============================================================================
#              Advanced settings (change only if needed)
# =============================================================================

THRESHOLD   = 0.30        # Pore-detection threshold (0.0 – 1.0).
                          # Lower = more pores detected; higher = fewer false positives.
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
#                         Model Architecture
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool   = nn.MaxPool2d(2, 2)
        self.downs  = nn.ModuleList()
        self.ups    = nn.ModuleList()

        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-(i // 2 + 1)]
            if x.shape != s.shape:
                x = TF.resize(x, s.shape[2:])
            x = self.ups[i + 1](torch.cat([s, x], dim=1))
        return self.final_conv(x)


# =============================================================================
#                         Helper Functions
# =============================================================================

def load_model(path, device):
    model = UNet().to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    return model


def compute_porosity_stats(mask, prob_map):
    """Return a dictionary of porosity metrics."""
    total_pixels = mask.size
    pore_pixels  = int(mask.sum())
    porosity_pct = pore_pixels / total_pixels * 100.0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8, cv2.CV_32S)
    n_pores = int(n_labels - 1)

    pore_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    mean_area  = float(np.mean(pore_areas))  if pore_areas else 0.0
    max_area   = float(np.max(pore_areas))   if pore_areas else 0.0

    return {
        "porosity_%"   : round(porosity_pct, 4),
        "pore_count"   : n_pores,
        "mean_pore_area_px": round(mean_area, 2),
        "max_pore_area_px" : round(max_area, 2),
        "mean_confidence"  : round(float(prob_map[mask == 1].mean()) * 100, 2)
                             if pore_pixels else 0.0,
    }


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
    print(f"[1/4] Loading porosity model from:\n      {MODEL_PATH}")
    model = load_model(MODEL_PATH, DEVICE)
    print(f"      Device: {DEVICE}")

    # --- Load image ---
    print(f"[2/4] Reading image from:\n      {IMAGE_PATH}")
    original = Image.open(IMAGE_PATH).convert("RGB")
    W, H     = original.size
    resized  = original.resize((512, 512))

    tensor   = TF.to_tensor(resized)
    tensor   = TF.normalize(tensor, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    tensor   = tensor.unsqueeze(0).to(DEVICE)

    # --- Predict ---
    print("[3/4] Running porosity segmentation …")
    with torch.no_grad():
        logits   = model(tensor)
        probs    = torch.sigmoid(logits)
        mask_t   = (probs > THRESHOLD).float()

    prob_512 = probs[0, 0].cpu().numpy()       # probability map  (512 × 512)
    mask_512 = mask_t[0, 0].cpu().numpy()      # binary mask      (512 × 512)

    # Resize back to original resolution
    prob_full = cv2.resize(prob_512, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_full = cv2.resize(mask_512, (W, H), interpolation=cv2.INTER_NEAREST)

    # --- Statistics ---
    stats = compute_porosity_stats(mask_full, prob_full)
    print(f"      Porosity : {stats['porosity_%']} %")
    print(f"      Pores    : {stats['pore_count']}")

    # --- Four-panel figure ---
    print("[4/4] Building figure and saving results …")
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 4, figure=fig)

    img_np = np.array(original)

    # Panel A — original
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(original)
    ax0.set_title("(a) Input Image", fontsize=11, fontweight="bold")
    ax0.axis("off")

    # Panel B — probability heatmap
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(prob_full, cmap="magma", vmin=0, vmax=1)
    ax1.set_title("(b) Pore Probability Map", fontsize=11, fontweight="bold")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Confidence")

    # Panel C — binary pore mask
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(mask_full, cmap="gray")
    ax2.set_title(
        f"(c) Pore Mask  (threshold = {THRESHOLD})", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # Panel D — overlay
    pore_rgba                  = np.zeros((*mask_full.shape, 4), dtype=np.float32)
    pore_rgba[mask_full == 1]  = [0.0, 0.6, 1.0, 0.55]   # cyan-blue pores

    ax3 = fig.add_subplot(gs[3])
    ax3.imshow(original)
    ax3.imshow(pore_rgba)
    ax3.set_title("(d) Overlay", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # Statistics annotation on Panel D
    stat_text = (
        f"Porosity: {stats['porosity_%']} %\n"
        f"Pore count: {stats['pore_count']}\n"
        f"Mean pore area: {stats['mean_pore_area_px']} px²"
    )
    ax3.text(0.02, 0.02, stat_text, transform=ax3.transAxes,
             fontsize=8, color="white", va="bottom",
             bbox=dict(facecolor="black", alpha=0.6, edgecolor="none",
                       boxstyle="round,pad=0.4"))

    # --- Save ---
    base      = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    fig_path  = os.path.join(OUTPUT_DIR, f"{base}_porosity_unet.png")
    mask_path = os.path.join(OUTPUT_DIR, f"{base}_pore_mask.png")
    stat_path = os.path.join(OUTPUT_DIR, f"{base}_porosity_stats.txt")

    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    cv2.imwrite(mask_path, (mask_full * 255).astype(np.uint8))

    with open(stat_path, "w") as fh:
        fh.write(f"Image       : {IMAGE_PATH}\n")
        fh.write(f"Threshold   : {THRESHOLD}\n")
        for k, v in stats.items():
            fh.write(f"{k:<28}: {v}\n")

    print("\n  Results saved:")
    print(f"  • Four-panel figure : {fig_path}")
    print(f"  • Binary pore mask  : {mask_path}")
    print(f"  • Statistics report : {stat_path}")
    print("  Done.")


# =============================================================================

if __name__ == "__main__":
    run_inference()
