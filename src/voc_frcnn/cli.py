from __future__ import annotations

import argparse
from pathlib import Path

from voc_frcnn.datasets.voc import VocDataset
from voc_frcnn.engine.evaluate import evaluate
from voc_frcnn.engine.train import TrainConfig, train


def main() -> None:
    p = argparse.ArgumentParser("VOC07 Faster R-CNN")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--data", type=Path, required=True, help="Path to VOC2007 root")
    p_train.add_argument("--split", choices=["train", "trainval"], default="train")
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--lr", type=float, default=0.005)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--num-workers", type=int, default=2)

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model")
    p_eval.add_argument("--data", type=Path, required=True)
    p_eval.add_argument("--split", choices=["test", "val"], default="test")
    p_eval.add_argument("--weights", type=Path, default=Path("models/fasterrcnn_voc07.pth"))
    p_eval.add_argument("--batch-size", type=int, default=2)
    p_eval.add_argument("--num-workers", type=int, default=2)
    p_eval.add_argument("--score-thresh", type=float, default=0.5)

    args = p.parse_args()

    if args.cmd == "train":
        ds = VocDataset(args.data, args.split)
        cfg = TrainConfig(
            data_root=args.data,
            split=args.split,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        train(ds, cfg)
    elif args.cmd == "evaluate":
        ds = VocDataset(args.data, args.split)
        evaluate(
            ds,
            weights=args.weights,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            score_thresh=args.score_thresh,
        )


if __name__ == "__main__":
    main()
