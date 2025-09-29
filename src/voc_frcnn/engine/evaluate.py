from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from voc_frcnn.models.build import build_model
from voc_frcnn.utils.collate import detection_collate


def evaluate(
    dataset: TorchDataset,
    weights: Path | None = None,
    batch_size: int = 2,
    num_workers: int = 2,
    score_thresh: float = 0.5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=21).to(device).eval()

    if weights is not None and Path(weights).exists():
        state = torch.load(weights, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights: {weights}")
    else:
        print(
            "No weights provided/found; running with randomly initialized model (for sanity only)."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate,
    )

    with torch.no_grad():
        for images, _ in loader:
            images = [img.to(device) for img in images]
            outputs: list[dict] = model(images)
            for out in outputs:
                keep = out["scores"] >= score_thresh
                print(
                    f"detections >= {score_thresh:.2f}: {int(keep.sum())} "
                    f"(max score {out['scores'].max().item():.3f} if any)"
                )
    print("Evaluation finished.")
