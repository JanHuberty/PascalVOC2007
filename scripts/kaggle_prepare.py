"""
Kaggle convenience script to copy VOC07 from an input dataset to /kaggle/working/data/VOC2007.
Adjust 'SOURCE' below to match your Kaggle dataset path.
"""

import shutil
from pathlib import Path

SOURCE = Path("/kaggle/input/pascal-voc-2007/VOCdevkit/VOC2007")
DEST = Path("/kaggle/working/data/VOC2007")

print("Source:", SOURCE)
print("Dest  :", DEST)

if DEST.exists():
    print("Destination already exists. Skipping.")
else:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    # copytree requires dest not to exist
    shutil.copytree(SOURCE, DEST)
    print("Copied!")

print("Done.")
