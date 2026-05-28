# ===========================================================================
# Mask R-CNN Grain Instance Segmentation — Google Colab Version
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

MODEL_FILE = "/content/maskrcnn_grains.pth"

if not os.path.exists(MODEL_FILE):
    print("Downloading Mask R-CNN Grains model …")
    gdown.download(
        "https://drive.google.com/uc?id=1mplGIrgKFumBl1wXMaVcgbRjhBOgsp30",
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
# CELL 4 — Run instance segmentation
# ═══════════════════════════════════════════════════════════════
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import cv2
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn  import MaskRCNNPredictor

# ── Settings ──────────────────────────────────────────────────
OUTPUT_DIR      = "/content/results_maskrcnn_grains"
SCORE_THRESHOLD = 0.50   # minimum confidence to accept a detection
MASK_THRESHOLD  = 0.50   # threshold for soft → binary mask
NUM_CLASSES     = 2      # background + grain
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Build model ────────────────────────────────────────────────
def build_maskrcnn(num_classes, device):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_box  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, num_classes)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
    return model.to(device)

print(f"Loading Mask R-CNN on {DEVICE} …")
model = build_maskrcnn(NUM_CLASSES, DEVICE)
ckpt  = torch.load(MODEL_FILE, map_location=DEVICE)
state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
model.load_state_dict(state)
model.eval()

# ── Load image ─────────────────────────────────────────────────
original   = Image.open(IMAGE_PATH).convert("RGB")
W, H       = original.size
img_tensor = TF.to_tensor(original).to(DEVICE)

# ── Predict ────────────────────────────────────────────────────
print("Running instance segmentation …")
with torch.no_grad():
    outputs = model([img_tensor])

boxes  = outputs[0]["boxes"].cpu().numpy()
scores = outputs[0]["scores"].cpu().numpy()
masks  = outputs[0]["masks"].cpu().numpy()

keep   = scores >= SCORE_THRESHOLD
boxes, scores, masks = boxes[keep], scores[keep], masks[keep]
n_grains = len(scores)
print(f"Detected {n_grains} grains (confidence ≥ {SCORE_THRESHOLD})")

# ── Colour palette ─────────────────────────────────────────────
def get_colours(n):
    base = []
    for cm in [plt.cm.tab20, plt.cm.Set1, plt.cm.Dark2, plt.cm.Paired]:
        base.extend(list(cm.colors))
    base = base * (n // len(base) + 1)
    np.random.seed(42); np.random.shuffle(base)
    return base[:n]

colours = get_colours(max(n_grains, 1))

# ── Figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(W*2/100, H/100), dpi=100)
axes[0].imshow(original); axes[0].set_title("Input Image"); axes[0].axis("off")
axes[1].imshow(original); axes[1].set_title(f"Mask R-CNN — {n_grains} grains"); axes[1].axis("off")

for i, (mask, score, colour) in enumerate(zip(masks, scores, colours)):
    binary = (mask[0] > MASK_THRESHOLD).astype(np.uint8)
    ov = np.zeros((*binary.shape, 4), dtype=np.float32)
    ov[binary == 1] = (*colour[:3], 0.55)
    axes[1].imshow(ov)
    x1, y1, x2, y2 = boxes[i]
    axes[1].add_patch(mpatches.Rectangle(
        (x1,y1), x2-x1, y2-y1, linewidth=0.8,
        edgecolor=colour[:3], facecolor="none"))
    cy_c, cx_c = np.where(binary)
    if len(cx_c):
        axes[1].text(cx_c.mean(), cy_c.mean(), str(i+1),
                     color="white", fontsize=6, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(facecolor="black", alpha=0.5,
                               edgecolor="none", boxstyle="round,pad=0.2"))

plt.tight_layout(pad=0.5)

base     = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
fig_path = f"{OUTPUT_DIR}/{base}_grains_maskrcnn.png"

# Instance label map
label_map = np.zeros((H, W), dtype=np.uint16)
for i, mask in enumerate(masks):
    label_map[(mask[0] > MASK_THRESHOLD)] = i + 1
label_path = f"{OUTPUT_DIR}/{base}_labelmap_maskrcnn.png"

plt.savefig(fig_path, dpi=100, bbox_inches="tight")
cv2.imwrite(label_path, label_map.astype(np.uint16))
plt.show()

print(f"\nTotal grains detected  : {n_grains}")
print(f"Overlay image saved    : {fig_path}")
print(f"Instance label map     : {label_path}")


# ═══════════════════════════════════════════════════════════════
# CELL 5 — (Optional) Download results to your computer
# ═══════════════════════════════════════════════════════════════
files.download(fig_path)
