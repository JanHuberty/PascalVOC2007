from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from voc_frcnn.models.build import build_model
from voc_frcnn.utils.collate import detection_collate


@dataclass
class TrainConfig:
    data_root: Path
    split: str = "train"
    epochs: int = 1
    batch_size: int = 2
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 2
    print_every: int = 50
    out_dir: Path = Path("models")


def train(dataset: TorchDataset, cfg: TrainConfig) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=21).to(device)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=detection_collate,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )

    model.train()
    step = 0
    for epoch in range(cfg.epochs):
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % cfg.print_every == 0:
                print(
                    f"epoch={epoch} step={step} "
                    + " ".join(f"{k}:{v.item():.4f}" for k, v in loss_dict.items())
                )
            step += 1

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    save_path = cfg.out_dir / "fasterrcnn_voc07.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete → {save_path}")
    return save_path
