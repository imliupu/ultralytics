from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from rec_yolo26.dataset import letterbox, load_data_config, normalize_imgsz
from rec_yolo26.model import RecYOLO26Model


def parse_imgsz(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError(f"--imgsz expects one int or two ints, got: {values}")


def infer_nc_and_names(weights: str, data_yaml: Optional[str]) -> tuple[int, dict[int, str]]:
    if data_yaml:
        data = load_data_config(data_yaml)
        return int(data["nc"]), {int(k): str(v) for k, v in data["names"].items()}
    ckpt = torch.load(weights, map_location="cpu")
    names = ckpt.get("names")
    if isinstance(names, dict) and names:
        norm = {int(k): str(v) for k, v in names.items()}
        return len(norm), norm
    if isinstance(names, list) and names:
        norm = {i: str(v) for i, v in enumerate(names)}
        return len(norm), norm
    return 1, {0: "0"}


def xywha_to_polygon(x: float, y: float, w: float, h: float, a: float):
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = w / 2.0, h / 2.0
    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    return [(x + px * ca - py * sa, y + px * sa + py * ca) for px, py in corners]


@torch.inference_mode()
def main(args):
    imgsz = parse_imgsz(args.imgsz)
    nc, names = infer_nc_and_names(args.weights, args.data)
    device = torch.device(args.device)

    model = RecYOLO26Model.build(task=args.task, nc=nc, model=args.model, verbose=not args.quiet)
    model.load(args.weights, strict=False)
    model.to(device).eval()
    model.names = names

    image = Image.open(args.source).convert("RGB")
    original = image.copy()
    image_lb, ratio, pad = letterbox(image, normalize_imgsz(imgsz))
    arr = np.asarray(image_lb, dtype=np.uint8).transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(device).float() / 255.0

    raw = model(tensor)
    preds = model.postprocess(raw, conf=args.conf, iou=args.iou, max_det=args.max_det)[0]

    draw = ImageDraw.Draw(original)
    padw, padh = pad
    for box, conf, cls in zip(preds["bboxes"], preds["conf"], preds["cls"]):
        cls_i = int(cls.item())
        label = f"{names.get(cls_i, str(cls_i))} {float(conf):.3f}"
        if args.task == "obb" and box.numel() >= 5:
            x, y, w, h, a = [float(v) for v in box[:5]]
            x, y, w, h = (x - padw) / ratio, (y - padh) / ratio, w / ratio, h / ratio
            poly = xywha_to_polygon(x, y, w, h, a)
            draw.polygon(poly, outline="red", width=2)
            draw.text((poly[0][0], poly[0][1]), label, fill="red")
        else:
            x, y, w, h = [float(v) for v in box[:4]]
            x, y, w, h = (x - padw) / ratio, (y - padh) / ratio, w / ratio, h / ratio
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
            draw.text((x1, y1), label, fill="red")

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    original.save(out)
    if not args.quiet:
        print(f"Saved prediction visualization to {out}")


def build_parser():
    parser = argparse.ArgumentParser(description="Single-image prediction/visualization for rec_yolo26.")
    parser.add_argument("--task", choices=["detect", "obb"], required=True)
    parser.add_argument("--model", default="yolo26")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Path to one image.")
    parser.add_argument("--data", default=None, help="Optional dataset yaml for class names/nc.")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640], help="Image size as single int or 'h w'.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", default="runs/rec_yolo26/predict.jpg")
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
