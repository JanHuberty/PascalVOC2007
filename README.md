# Pascal VOC 2007 – Faster R-CNN (PyTorch)

Object detection pipeline on VOC2007 with torchvision’s Faster R-CNN.  
**Stack:** PyTorch, torchvision, Python 3.x

## Overview
- Custom dataset + transforms for VOC XML
- Training loop with checkpoints
- Evaluation (mAP, PR); sample detections

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
