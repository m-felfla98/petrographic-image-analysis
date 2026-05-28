# ===========================================================================
# U-Net Porosity Mapping — Google Colab Version
# Copy each "CELL" block into a separate Colab cell and run in order.
# ===========================================================================

# ═══════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies  (run once per session)
# ═══════════════════════════════════════════════════════════════
!pip install torch torchvision opencv-python matplotlib Pillow numpy gdown -q


# ═══════════════════════════════════════════════════════════════
# CELL 2 — Download the trained model from Google Drive
# ═══════════════════════════════════════════════════════════════
import gdown, os

MODEL_FILE = "/content/unet_porosity.pth.tar"

if not os.path.exists(MODEL_FILE):
    print("Downloading U-Net Porosity model …")
    gdown.download(
        "https://drive.google.com/uc?id=1aZar1y3bLAR58Tdd5rcNn9-TPiXnCQlM",
        MODEL_FILE, quiet=False)
    print("Download complete.")
else:
    print("Model file already present, skipping download.")


# ═══════════════════════════════════════════════════════════════
# CELL 3 — Upload your thin-section image
# ═══════════════════════════════════════════════════════════════
from google.colab import files

print("Please select a thin-section image from your computer …")
uploaded   = files.upload()
IMAGE_FILE = list(uploaded.keys())[0]
IMAGE_PATH = f"/content/{IMAGE_FILE}"
print(f"Image ready: {IMAGE_PATH}")


# ═══════════════════════════════════════════════════════════════
# CELL 4 — Run porosity mapping
# ═══════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
from PIL import Image

# ── Settings ──────────────────────────────────────────────────
OUTPUT_DIR = "/content/results_unet_porosity"
THRESHOLD  = 0.30    # pore detection sensitivity
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model architecture ─────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[64,128,256,512]):
        super().__init__()
        self.pool  = nn.MaxPool2d(2,2)
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        for f in features:
            self.downs.append(DoubleConv(in_ch, f)); in_ch = f
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-(i//2+1)]
            if x.shape != s.shape:
                x = TF.resize(x, s.shape[2:])
            x = self.ups[i+1](torch.cat([s, x], dim=1))
        return self.final_conv(x)

# ── Load model ─────────────────────────────────────────────────
print(f"Loading porosity model on {DEVICE} …")
model = UNet().to(DEVICE)
ckpt  = torch.load(MODEL_FILE, map_location=DEVICE)
model.load_state_dict(ckpt.get("state_dict", ckpt))
model.eval()

# ── Load & prepare image ───────────────────────────────────────
original = Image.open(IMAGE_PATH).convert("RGB")
W, H     = original.size
resized  = original.resize((512, 512))
tensor   = TF.to_tensor(resized)
tensor   = TF.normalize(tensor, [0.0,0.0,0.0], [1.0,1.0,1.0])
tensor   = tensor.unsqueeze(0).to(DEVICE)

# ── Predict ────────────────────────────────────────────────────
print("Running porosity segmentation …")
with torch.no_grad():
    probs  = torch.sigmoid(model(tensor))
    binary = (probs > THRESHOLD).float()

prob_512  = probs[0,0].cpu().numpy()
mask_512  = binary[0,0].cpu().numpy()
prob_full = cv2.resize(prob_512, (W,H), interpolation=cv2.INTER_LINEAR)
mask_full = cv2.resize(mask_512, (W,H), interpolation=cv2.INTER_NEAREST)

# ── Statistics ─────────────────────────────────────────────────
pore_px      = int(mask_full.sum())
porosity_pct = round(pore_px / mask_full.size * 100, 4)
n_lab, _, st, _ = cv2.connectedComponentsWithStats(
    mask_full.astype(np.uint8), 8, cv2.CV_32S)
n_pores   = n_lab - 1
areas     = [int(st[i, cv2.CC_STAT_AREA]) for i in range(1, n_lab)]
mean_area = round(float(np.mean(areas)), 2) if areas else 0.0

print(f"\nPorosity   : {porosity_pct} %")
print(f"Pore count : {n_pores}")
print(f"Mean pore area : {mean_area} px²")

# ── Four-panel figure ──────────────────────────────────────────
fig = plt.figure(figsize=(16, 8), constrained_layout=True)
gs  = gridspec.GridSpec(1, 4, figure=fig)

ax0 = fig.add_subplot(gs[0])
ax0.imshow(original)
ax0.set_title("(a) Input Image", fontsize=11, fontweight="bold")
ax0.axis("off")

ax1 = fig.add_subplot(gs[1])
im1 = ax1.imshow(prob_full, cmap="magma", vmin=0, vmax=1)
ax1.set_title("(b) Pore Probability Map", fontsize=11, fontweight="bold")
ax1.axis("off")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Confidence")

ax2 = fig.add_subplot(gs[2])
ax2.imshow(mask_full, cmap="gray")
ax2.set_title(f"(c) Pore Mask  (thr={THRESHOLD})", fontsize=11, fontweight="bold")
ax2.axis("off")

pore_rgba = np.zeros((*mask_full.shape, 4), dtype=np.float32)
pore_rgba[mask_full == 1] = [0.0, 0.6, 1.0, 0.55]

ax3 = fig.add_subplot(gs[3])
ax3.imshow(original); ax3.imshow(pore_rgba)
ax3.set_title("(d) Overlay", fontsize=11, fontweight="bold")
ax3.axis("off")
ax3.text(0.02, 0.02,
         f"Porosity: {porosity_pct} %\nPores: {n_pores}\nMean area: {mean_area} px²",
         transform=ax3.transAxes, fontsize=8, color="white", va="bottom",
         bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.4"))

plt.savefig(f"{OUTPUT_DIR}/porosity_result.png", dpi=150, bbox_inches="tight")
cv2.imwrite(f"{OUTPUT_DIR}/pore_mask.png", (mask_full*255).astype(np.uint8))

with open(f"{OUTPUT_DIR}/porosity_stats.txt", "w") as fh:
    fh.write(f"Image         : {IMAGE_PATH}\n")
    fh.write(f"Threshold     : {THRESHOLD}\n")
    fh.write(f"Porosity (%)  : {porosity_pct}\n")
    fh.write(f"Pore count    : {n_pores}\n")
    fh.write(f"Mean pore area: {mean_area} px2\n")

plt.show()
print(f"\nResults saved in: {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════════════
# CELL 5 — (Optional) Download results to your computer
# ═══════════════════════════════════════════════════════════════
files.download(f"{OUTPUT_DIR}/porosity_result.png")
files.download(f"{OUTPUT_DIR}/porosity_stats.txt")
