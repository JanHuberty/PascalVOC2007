![CI](https://github.com/JanHuberty/PascalVOC2007/actions/workflows/python-ci.yml/badge.svg)
![Links](https://github.com/JanHuberty/PascalVOC2007/actions/workflows/links.yml/badge.svg)
![Auto format](https://github.com/JanHuberty/PascalVOC2007/actions/workflows/auto-format.yml/badge.svg)
![License](https://img.shields.io/github/license/JanHuberty/PascalVOC2007)


# Pascal VOC 2007 – Faster R-CNN (PyTorch)

Object detection pipeline on VOC2007 with torchvision’s Faster R-CNN.  
**Stack:** PyTorch, torchvision, Python 3.x

## Overview
- Custom dataset + transforms for VOC XML
- Training loop with checkpoints
- Evaluation (mAP, PR); sample detections

## Quick Start
```bash
# 1) Create & activate a virtual env
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# 2) Install deps (CPU-friendly)
pip install -r requirements.txt

# 3) Configure dataset path (choose one)
#   Option A: Edit configs/default.yaml (recommended)
#   Option B: Pass as an arg, e.g.:
# python src/train.py --data_dir /path/to/VOCdevkit/VOC2007 --epochs 1 --batch-size 2 --seed 42

# 4) Evaluate (example)
# python src/eval.py --checkpoint checkpoints/best.pth --data_dir /path/to/VOCdevkit/VOC2007

Tips you can add right below:
- If your scripts live elsewhere, change `src/train.py`/`src/eval.py` to your actual paths.
- Keep **epochs=1** and **batch-size small** so a recruiter can run it quickly.

---

## 1.8 Add sample visuals (screenshots) and embed them
**Clicks to upload:** Repo → **Add file → Upload files** → in the filename box type `docs/` then drop images (e.g., `docs/detection_1.jpg`, `docs/detection_2.jpg`) → **Commit**.

**Embed in README** (where you want them to appear):
```md
## Results
- mAP@0.5: **…**  (put your number when you have it)

<!-- Single images -->
![Detection 1](docs/detection_1.jpg)
![Detection 2](docs/detection_2.jpg)

<!-- Optional: two images side-by-side using HTML -->
<p float="left">
  <img src="docs/detection_1.jpg" width="48%" />
  <img src="docs/detection_2.jpg" width="48%" />
</p>
