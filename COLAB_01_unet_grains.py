# ===========================================================================
# U-Net Grain Segmentation — Google Colab Version
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

MODEL_FILE = "/content/unet_grains.pth.tar"

if not os.path.exists(MODEL_FILE):
    print("Downloading U-Net Grains model …")
    gdown.download(
        "https://drive.google.com/uc?id=1p20k7MoDK3uSMsntd77F0f_6TMJ4fuAO",
        MODEL_FILE, quiet=False)
    print("Download complete.")
else:
    print("Model file already present, skipping download.")


# ═══════════════════════════════════════════════════════════════
# CELL 3 — Upload your thin-section image
# ═══════════════════════════════════════════════════════════════
from google.colab import files

print("Please select a thin-section image from your computer …")
uploaded = files.upload()

IMAGE_FILE = list(uploaded.keys())[0]
IMAGE_PATH = f"/content/{IMAGE_FILE}"
print(f"Image ready: {IMAGE_PATH}")


# ═══════════════════════════════════════════════════════════════
# CELL 4 — Run segmentation
# ═══════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
from PIL import Image

# ── Settings ──────────────────────────────────────────────────
OUTPUT_DIR  = "/content/results_unet_grains"
THRESHOLD   = 0.15       # lower = more sensitive
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
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
        self.pool  = nn.MaxPool2d(2, 2)
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
print(f"Loading model on {DEVICE} …")
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
print("Running segmentation …")
with torch.no_grad():
    probs  = torch.sigmoid(model(tensor))
    binary = (probs > THRESHOLD).float()

mask_512  = binary[0,0].cpu().numpy()
mask_full = cv2.resize(mask_512, (W, H), interpolation=cv2.INTER_NEAREST)
mask_u8   = (mask_full * 255).astype(np.uint8)

# ── Separate touching grains ───────────────────────────────────
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
cores  = cv2.erode(mask_u8, kernel, iterations=3)
inv    = cv2.bitwise_not(cores)
_, vor = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5,
                                          labelType=cv2.DIST_LABEL_PIXEL)
L      = vor
bounds = ((L != np.roll(L,1,0))|(L != np.roll(L,-1,0))|
          (L != np.roll(L,1,1))|(L != np.roll(L,-1,1))) & (mask_u8==255)
sep    = mask_u8.copy(); sep[bounds] = 0

n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(sep, 8, cv2.CV_32S)
grain_count = n_lab - 1

# ── Colour palette ─────────────────────────────────────────────
def palette(n):
    base = []
    for cm in [plt.cm.tab20, plt.cm.Set1, plt.cm.Dark2, plt.cm.Paired]:
        base.extend([mcolors.to_rgb(c) for c in cm.colors])
    cols = (np.array(base)*255).astype(np.uint8)
    if n > len(cols): cols = np.tile(cols,(n//len(cols)+1,1))
    np.random.seed(42); np.random.shuffle(cols)
    bg   = np.array([[0,0,0]],np.uint8)
    fg   = cols[:n]
    a_bg = np.zeros((1,1),np.uint8)
    a_fg = np.full((n,1),160,np.uint8)
    return np.concatenate([np.hstack([bg,a_bg]), np.hstack([fg,a_fg])], axis=0)

pal     = palette(n_lab+1)
overlay = pal[labels]

# ── Display & save ─────────────────────────────────────────────
dpi = 100
fig = plt.figure(figsize=(W/dpi, H/dpi), dpi=dpi)
ax  = fig.add_axes([0,0,1,1]); ax.axis("off")
ax.imshow(original); ax.imshow(overlay)
for i in range(1, n_lab):
    cx, cy = int(centroids[i][0]), int(centroids[i][1])
    ax.text(cx, cy, str(i), color="white", fontsize=7, fontweight="bold",
            ha="center", va="center",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2"))

base      = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
fig_path  = f"{OUTPUT_DIR}/{base}_grains_unet.png"
mask_path = f"{OUTPUT_DIR}/{base}_mask_unet.png"

plt.savefig(fig_path, dpi=dpi, pad_inches=0)
cv2.imwrite(mask_path, mask_u8)
plt.show()

print(f"\nTotal grains detected : {grain_count}")
print(f"Overlay image saved   : {fig_path}")
print(f"Binary mask saved     : {mask_path}")


# ═══════════════════════════════════════════════════════════════
# CELL 5 — (Optional) Download results to your computer
# ═══════════════════════════════════════════════════════════════
files.download(fig_path)
files.download(mask_path)
