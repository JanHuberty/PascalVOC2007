from __future__ import annotations

from typing import Any


def detection_collate(batch: list[tuple[Any, dict]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=False)
    return list(images), list(targets)
