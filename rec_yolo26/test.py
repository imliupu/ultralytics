from __future__ import annotations

import argparse
import json

from ultralytics import YOLO

from rec_yolo26.model import RecYOLO26Model


def main(args):
    cfg_path = RecYOLO26Model.resolve_model_cfg(args.model, args.task)
    model = YOLO(str(cfg_path), task=args.task).load(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        split=args.split,
    )
    results = getattr(metrics, "results_dict", {}) or {}
    print(json.dumps({k: float(v) for k, v in results.items()}, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Ultralytics-aligned YOLO26 / YOLO26-OBB test entrypoint.")
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
