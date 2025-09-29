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
- Evaluation (mAP, PR), sample detections
 
## Project structure (target)
src/
voc_frcnn/
datasets/voc.py # (coming next)
engine/train.py # (coming next)
engine/evaluate.py # (coming next)
cli.py # (coming next)
scripts/
train.sh eval.sh download_pascal_voc.sh
tests/
.github/workflows/

## Setup

### 1) Create & activate a virtual env
**Windows (Command Prompt)**
```bat
py -3 -m venv .venv
.\.venv\Scripts\activate.bat