#!/usr/bin/env bash
set -euo pipefail
python -m voc_frcnn.cli evaluate --data ./data/VOC2007 --split test
