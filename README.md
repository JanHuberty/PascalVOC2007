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

# 3) Run the script
# (adjust args if your script expects dataset paths)
python code.py
# e.g.: python code.py --data_dir /path/to/VOCdevkit/VOC2007 --epochs 1 --batch-size 2

> If `code.py` currently doesn’t accept CLI args, leave the plain `python code.py`. You can add args later.

---

# C) Results images (use your existing `Docs/` folder)

You already created `Docs/`. Let’s embed sample detections.

1) If your images are there (e.g., `Docs/detection_1.jpg`, `Docs/detection_2.jpg`), add this to the README under a “Results” heading:

```md
## Results
- mAP@0.5: **…** (fill in when you have it)

![Detection 1](Docs/detection_1.jpg)
![Detection 2](Docs/detection_2.jpg)

<!-- Optional: side-by-side -->
<p float="left">
  <img src="Docs/detection_1.jpg" width="48%" />
  <img src="Docs/detection_2.jpg" width="48%" />
</p>
