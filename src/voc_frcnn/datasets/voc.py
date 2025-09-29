from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

Transform = Callable[[Image.Image, dict[str, Tensor]], tuple[Tensor, dict[str, Tensor]]]


# VOC has 20 classes; model expects background=0, so we map 1..20 to real classes
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
CLASS_TO_IDX: dict[str, int] = {c: i + 1 for i, c in enumerate(VOC_CLASSES)}  # 0 is background


class VocDataset(Dataset):
    """
    Expected structure:
      root/
        JPEGImages/*.jpg
        Annotations/*.xml
        ImageSets/Main/{train,val,test,trainval}.txt
    """

    def __init__(self, root: str | Path, split: str, transforms: Transform | None = None) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        split_list = self.root / "ImageSets" / "Main" / f"{split}.txt"
        if not split_list.exists():
            raise FileNotFoundError(f"Missing split list: {split_list}")

        self.ids: list[str] = [
            line.strip() for line in split_list.read_text().splitlines() if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        img_id = self.ids[idx]
        img_path = self.root / "JPEGImages" / f"{img_id}.jpg"
        ann_path = self.root / "Annotations" / f"{img_id}.xml"

        image = Image.open(img_path).convert("RGB")
        boxes, labels = self._parse_xml(ann_path)

        target: dict = {
            "boxes": boxes,  # FloatTensor [N,4] (xmin,ymin,xmax,ymax)
            "labels": labels,  # LongTensor [N]
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return self._to_tensor(image), target

    @staticmethod
    def _to_tensor(img: Image.Image) -> Tensor:
        # Minimal ToTensor to avoid extra deps; values in [0,1]
        arr = torch.frombuffer(img.tobytes(), dtype=torch.uint8)
        arr = arr.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float() / 255.0
        return arr

    def _parse_xml(self, xml_path: Path) -> tuple[Tensor, Tensor]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes: list[list[float]] = []
        labels: list[int] = []

        for obj in root.findall("object"):
            name = obj.findtext("name")
            b = obj.find("bndbox")
            xmin = float(b.findtext("xmin"))
            ymin = float(b.findtext("ymin"))
            xmax = float(b.findtext("xmax"))
            ymax = float(b.findtext("ymax"))
            if xmax <= xmin or ymax <= ymin:
                # Skip degenerate boxes
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name])

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
