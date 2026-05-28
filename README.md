# petrographic-image-analysis
Inference code for automated petrographic grain segmentation
# Petrographic Image Analysis — Trained Models & Inference Code

Companion code for:

> Felfla, M. Sh. (2026). "Generational shift in computer vision for fully
> automated petrographic image analysis."

---

## Repository Contents

| File | Description |
|------|-------------|
| `01_unet_grains_inference.py` | Grain segmentation using U-Net (semantic) |
| `02_maskrcnn_grains_inference.py` | Grain instance segmentation using Mask R-CNN |
| `03_unet_porosity_inference.py` | Porosity mapping using U-Net (semantic) |

---

## Trained Model Weights

Download the three model files from Google Drive before running the scripts:

| Model | Link |
|-------|------|
| U-Net Grains | https://drive.google.com/file/d/1p20k7MoDK3uSMsntd77F0f_6TMJ4fuAO/view?usp=drive_link |
| Mask R-CNN Grains | https://drive.google.com/file/d/1mplGIrgKFumBl1wXMaVcgbRjhBOgsp30/view?usp=drive_link |
| U-Net Porosity | https://drive.google.com/file/d/1aZar1y3bLAR58Tdd5rcNn9-TPiXnCQlM/view?usp=drive_link |

---

## ⚠️ Environment Compatibility Notice

The dependencies required by these scripts (PyTorch, torchvision, OpenCV)
are **primarily supported on Linux-based environments**.
Native installation on Windows may require additional configuration steps
and is not guaranteed to work out of the box.

**Recommended environments:**

| Platform | Notes |
|----------|-------|
| **Google Colab** (recommended) | Free, browser-based, no local installation needed. GPU available. See the Colab section below. |
| **Linux** (Ubuntu 20.04 / 22.04) | Full native support. Standard `pip install` works as expected. |
| **Windows** | May work via WSL2 (Windows Subsystem for Linux) or Anaconda, but is not the primary tested environment. |

---

## Option A — Run on Google Colab (Easiest)

No installation required. Open a new Colab notebook at
[colab.research.google.com](https://colab.research.google.com) and run the
following cells:

**Cell 1 — Install dependencies**
```python
!pip install torch torchvision opencv-python matplotlib Pillow numpy
```

**Cell 2 — Download a model from Google Drive**
```python
!pip install gdown -q
!gdown "1p20k7MoDK3uSMsntd77F0f_6TMJ4fuAO" -O unet_grains.pth.tar      # U-Net Grains
# !gdown "1mplGIrgKFumBl1wXMaVcgbRjhBOgsp30" -O maskrcnn_grains.pth     # Mask R-CNN Grains
# !gdown "1aZar1y3bLAR58Tdd5rcNn9-TPiXnCQlM" -O unet_porosity.pth.tar   # U-Net Porosity
```

**Cell 3 — Upload your image and run the script**
```python
from google.colab import files
uploaded = files.upload()          # select your thin-section image

# Then set the three paths in the script:
# MODEL_PATH = "/content/unet_grains.pth.tar"
# IMAGE_PATH = "/content/your_image.jpg"
# OUTPUT_DIR = "/content/results/"

!python 01_unet_grains_inference.py
```

---

## Option B — Run on Linux (Local)

**Install dependencies**
```bash
pip install torch torchvision opencv-python matplotlib Pillow numpy
```

**Fill in paths at the top of the chosen script**
```python
MODEL_PATH = ""   # <- full path to the downloaded .pth.tar file
IMAGE_PATH = ""   # <- full path to the input thin-section image
OUTPUT_DIR = ""   # <- directory where results will be saved
```

**Run**
```bash
python 01_unet_grains_inference.py
python 02_maskrcnn_grains_inference.py
python 03_unet_porosity_inference.py
```

---

## Computational Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU | Not required (CPU mode available) | NVIDIA GPU with CUDA |
| Storage | 2 GB free | 5 GB free |
| Python | 3.8+ | 3.10 |

---

## Contact

Mahmoud Sh. Felfla
Geology Department, Faculty of Science, Damietta University
m.felfla@du.edu.eg
