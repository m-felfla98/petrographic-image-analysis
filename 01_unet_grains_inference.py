"""
=============================================================================
U-Net Grain Segmentation — Inference Script
=============================================================================
Reference:
    Felfla, M. Sh. (2026). Generational shift in computer vision for fully
    automated petrographic image analysis.

Trained model weights:
    https://drive.google.com/file/d/1p20k7MoDK3uSMsntd77F0f_6TMJ4fuAO/

Dependencies:
    pip install torch torchvision opencv-python matplotlib Pillow numpy

Usage:
    Set the three paths below, then run:
        python 01_unet_grains_inference.py
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
from PIL import Image

# =============================================================================
#                          *** USER SETTINGS ***
#          Fill in the three paths below before running the script
# =============================================================================

MODEL_PATH  = ""          # Path to the downloaded .pth.tar model file
                          # Example: "C:/models/mah_net_epoch_100.pth.tar"

IMAGE_PATH  = ""          # Path to the input thin-section image
                          # Example: "C:/images/sample_01.jpg"

OUTPUT_DIR  = ""          # Directory where results will be saved
                          # Example: "C:/results/unet_grains/"

# =============================================================================
#              Advanced settings (change only if needed)
# =============================================================================

THRESHOLD   = 0.15        # Detection sensitivity (0.0 – 1.0). Lower = more sensitive.
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


def generate_palette(n):
    """Build a high-contrast RGBA colour palette for n grain instances."""
    base = []
    for cmap in [plt.cm.tab20, plt.cm.Set1, plt.cm.Dark2, plt.cm.Paired]:
        base.extend([mcolors.to_rgb(c) for c in cmap.colors])
    colours = (np.array(base) * 255).astype(np.uint8)
    if n > len(colours):
        colours = np.tile(colours, (n // len(colours) + 1, 1))
    np.random.seed(42)
    np.random.shuffle(colours)
    fg    = colours[:n]
    bg    = np.array([[0, 0, 0]], dtype=np.uint8)
    alpha_fg = np.full((n, 1), 160, np.uint8)
    alpha_bg = np.zeros((1, 1), np.uint8)
    return np.concatenate([
        np.concatenate([bg,  alpha_bg], axis=1),
        np.concatenate([fg,  alpha_fg], axis=1),
    ], axis=0)


# =============================================================================
#                         Main Inference Function
# =============================================================================

def run_inference():

    # --- Validate user paths ---
    for label, path in [("MODEL_PATH", MODEL_PATH),
                         ("IMAGE_PATH", IMAGE_PATH),
                         ("OUTPUT_DIR", OUTPUT_DIR)]:
        if not path:
            raise ValueError(f"{label} is empty. Please fill in the path at the top of the script.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load model ---
    print(f"[1/4] Loading model from:\n      {MODEL_PATH}")
    model = load_model(MODEL_PATH, DEVICE)
    print(f"      Device: {DEVICE}")

    # --- Load image ---
    print(f"[2/4] Reading image from:\n      {IMAGE_PATH}")
    original = Image.open(IMAGE_PATH).convert("RGB")
    W, H     = original.size
    resized  = original.resize((512, 512))

    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    tensor = tensor.unsqueeze(0).to(DEVICE)

    # --- Predict ---
    print("[3/4] Running segmentation …")
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits)
        mask   = (probs > THRESHOLD).float()

    mask_512 = mask[0, 0].cpu().numpy()
    mask_full = cv2.resize(mask_512, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_u8   = (mask_full * 255).astype(np.uint8)

    # --- Geometric separation of touching grains ---
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cores   = cv2.erode(mask_u8, kernel, iterations=3)
    inv     = cv2.bitwise_not(cores)
    _, vor  = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5,
                                               labelType=cv2.DIST_LABEL_PIXEL)
    L = vor
    bounds  = (
        (L != np.roll(L, 1, 0)) | (L != np.roll(L, -1, 0)) |
        (L != np.roll(L, 1, 1)) | (L != np.roll(L, -1, 1))
    ) & (mask_u8 == 255)
    separated        = mask_u8.copy()
    separated[bounds] = 0

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        separated, 8, cv2.CV_32S)
    grain_count = n_labels - 1

    # --- Colourised overlay figure ---
    print(f"[4/4] Detected {grain_count} grains. Saving results …")
    palette = generate_palette(n_labels + 1)
    overlay = palette[labels]

    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(original)
    ax.imshow(overlay)
    for i in range(1, n_labels):
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        ax.text(cx, cy, str(i), color="white", fontsize=7, fontweight="bold",
                ha="center", va="center",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none",
                          boxstyle="round,pad=0.2"))

    base   = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    fig_path  = os.path.join(OUTPUT_DIR, f"{base}_grains_unet.png")
    mask_path = os.path.join(OUTPUT_DIR, f"{base}_mask_unet.png")

    plt.savefig(fig_path,  dpi=dpi, pad_inches=0)
    cv2.imwrite(mask_path, mask_u8)
    plt.close(fig)

    print("\n  Results saved:")
    print(f"  • Overlay image : {fig_path}")
    print(f"  • Binary mask   : {mask_path}")
    print(f"  • Total grains  : {grain_count}")
    print("  Done.")


# =============================================================================

if __name__ == "__main__":
    run_inference()
