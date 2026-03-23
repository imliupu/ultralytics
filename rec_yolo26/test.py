from __future__ import annotations

import argparse
import json

import torch

from rec_yolo26.dataset import create_train_val_dataloaders, load_data_config
from rec_yolo26.metrics import EvalConfig, evaluate_model
from rec_yolo26.model import RecYOLO26Model


def main(args):
    device = torch.device(args.device)
    data = load_data_config(args.data)
    model = RecYOLO26Model.build(task=args.task, nc=data["nc"], model=args.model, verbose=not args.quiet)
    model.load(args.weights, strict=False)
    model.to(device)
    model.names = data["names"]
    _, _, dataloader = create_train_val_dataloaders(args.data, args.task, args.imgsz, args.batch, args.workers, eval_split=args.split)
    metrics = evaluate_model(model, dataloader, device, args.task, data["names"], EvalConfig(conf=args.conf, iou=args.iou, max_det=args.max_det))
    print(json.dumps({k: float(v) for k, v in metrics.items()}, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone self-contained YOLO26 / YOLO26-OBB test entrypoint.")
    parser.add_argument("--task", choices=["detect", "obb"], required=True)
    parser.add_argument("--model", default="yolo26")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--split", default="test")
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
