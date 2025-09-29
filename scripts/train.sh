#!/usr/bin/env bash
set -euo pipefail
python -m voc_frcnn.cli train --data ./data/VOC2007 --epochs 1 --lr 0.005
