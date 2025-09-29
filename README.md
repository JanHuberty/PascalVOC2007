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
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Set dataset path in script/config, then:
# python src/train.py --epochs 1 --batch-size 2 --seed 42
# python src/eval.py --checkpoint checkpoints/best.pth

### 1.8 Add sample visuals (screenshots)
- **Add file → Upload files** → create folder `docs/` → upload 2 detection images, e.g.:
  - `docs/detection_1.jpg`
  - `docs/detection_2.jpg`
- Embed in README:
```md
## Results
![Detection 1](docs/detection_1.jpg)
![Detection 2](docs/detection_2.jpg)
