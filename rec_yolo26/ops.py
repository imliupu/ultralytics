from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torchvision.ops import box_iou, nms

from .modules import C2PSA, C3k2, Concat, Conv, Detect, OBB, SPPF

MODULES = {
    "Conv": Conv,
    "C3k2": C3k2,
    "SPPF": SPPF,
    "C2PSA": C2PSA,
    "Concat": Concat,
    "Detect": Detect,
    "OBB": OBB,
    "nn.Upsample": nn.Upsample,
}


def yaml_load(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_divisible(x: float, divisor: int) -> int:
    return int(math.ceil(x / divisor) * divisor)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def xywhr_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return xywh_to_xyxy(boxes[..., :4])


def make_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()


def build_targets_from_batch(batch: dict[str, torch.Tensor], task: str) -> list[dict[str, torch.Tensor]]:
    targets = []
    batch_size = batch["img"].shape[0]
    for i in range(batch_size):
        idx = batch["batch_idx"] == i
        targets.append({"cls": batch["cls"][idx].view(-1), "bboxes": batch["bboxes"][idx].view(-1, 5 if task == 'obb' else 4)})
    return targets


def non_max_suppression(predictions: list[dict[str, torch.Tensor]], conf_thres: float, iou_thres: float, task: str):
    outputs = []
    for pred in predictions:
        scores = pred["scores"].sigmoid()
        conf, cls = scores.max(dim=-1)
        keep = conf >= conf_thres
        if keep.sum() == 0:
            outputs.append({"bboxes": pred["boxes"].new_zeros((0, 5 if task == 'obb' else 4)), "conf": conf[:0], "cls": cls[:0].float()})
            continue
        boxes = pred["boxes"][keep]
        conf = conf[keep]
        cls = cls[keep].float()
        if task == "obb":
            keep_idx = nms(xywhr_to_xyxy(boxes), conf, iou_thres)
            outputs.append({"bboxes": boxes[keep_idx], "conf": conf[keep_idx], "cls": cls[keep_idx]})
        else:
            keep_idx = nms(xywh_to_xyxy(boxes), conf, iou_thres)
            outputs.append({"bboxes": xywh_to_xyxy(boxes)[keep_idx], "conf": conf[keep_idx], "cls": cls[keep_idx]})
    return outputs


def build_model_from_yaml(cfg: dict[str, Any], task: str, scale: str = "n", ch: int = 3, nc: int | None = None):
    cfg = deepcopy(cfg)
    depth, width, max_channels = cfg.get("scales", {}).get(scale, cfg.get("scales", {}).get("n", [1.0, 1.0, 1024]))
    if nc is not None:
        cfg["nc"] = nc
    layers = []
    save = []
    channels = [ch]

    def ch_lookup(index: int) -> int:
        return channels[index + 1] if index != -1 else channels[-1]

    definitions = cfg["backbone"] + cfg["head"]
    for i, (f, n, module_name, args) in enumerate(definitions):
        module_cls = MODULES[module_name]
        repeats = max(round(n * depth), 1) if n > 1 else n
        if module_name == "nn.Upsample":
            module = module_cls(scale_factor=args[1], mode=args[2])
            c2 = ch_lookup(f if isinstance(f, int) else f[0])
        elif module_name == "Concat":
            module = module_cls(*args)
            c2 = sum(ch_lookup(x) for x in f)
        elif module_name in {"Detect", "OBB"}:
            from_channels = [ch_lookup(x) for x in f]
            module = module_cls(cfg["nc"], args[1], tuple(from_channels)) if module_name == "OBB" else module_cls(cfg["nc"], tuple(from_channels))
            c2 = sum(from_channels)
        else:
            c1 = ch_lookup(f) if isinstance(f, int) else sum(ch_lookup(x) for x in f)
            c2 = args[0]
            if c2 != cfg["nc"]:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if module_name == "C3k2":
                module = nn.Sequential(*[module_cls(c1 if j == 0 else c2, c2, *args[1:]) for j in range(repeats)])
            elif module_name == "C2PSA":
                module = nn.Sequential(*[module_cls(c1 if j == 0 else c2, c2) for j in range(repeats)])
            else:
                module = module_cls(c1, c2, *args[1:])
        module.f = f
        module.i = i
        layers.append(module)
        channels.append(c2)
        if isinstance(f, list):
            save.extend(x for x in f if x != -1)
        elif f != -1:
            save.append(f)
    return nn.ModuleList(layers), sorted(set(save))
