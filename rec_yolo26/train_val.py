from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from rec_yolo26.model import RecYOLO26Model


def main(args):
    cfg_path = RecYOLO26Model.resolve_model_cfg(args.model, args.task)
    model = YOLO(str(cfg_path), task=args.task)
    if args.weights:
        model = model.load(args.weights)

    save_dir = Path(args.save_dir)
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
    )
    metrics = getattr(results, "results_dict", {}) or {}
    print(json.dumps({k: float(v) for k, v in metrics.items()}, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Ultralytics-aligned YOLO26 / YOLO26-OBB train+val entrypoint.")
    parser.add_argument("--task", choices=["detect", "obb"], required=True)
    parser.add_argument("--model", default="yolo26")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--save_dir", default="runs/rec_yolo26")
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
