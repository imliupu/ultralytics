from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union
import torch
import torch.nn as nn
import yaml

from .modules import C2PSA, C3, C3k, C3k2, Concat, Conv, Detect, OBB, OBB26, SPPF

MODULES = {
    "Conv": Conv,
    "C3": C3,
    "C3k": C3k,
    "C3k2": C3k2,
    "SPPF": SPPF,
    "C2PSA": C2PSA,
    "Concat": Concat,
    "Detect": Detect,
    "OBB": OBB,
    "OBB26": OBB26,
    "nn.Upsample": nn.Upsample,
}


def yaml_load(path: Union[str, Path]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_divisible(x: float, divisor: int) -> int:
    return int(math.ceil(x / divisor) * divisor)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


def xywhr_to_xyxyxyxy(boxes: torch.Tensor) -> torch.Tensor:
    ctr = boxes[..., :2]
    wh = boxes[..., 2:4] / 2
    angle = boxes[..., 4:5]
    cos, sin = angle.cos(), angle.sin()
    rot = torch.stack((
        torch.cat((cos, -sin), dim=-1),
        torch.cat((sin, cos), dim=-1),
    ), dim=-2)
    corners = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=boxes.dtype, device=boxes.device)
    corners = corners * wh.unsqueeze(-2)
    return corners @ rot.transpose(-1, -2) + ctr.unsqueeze(-2)


def xywhr_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    corners = xywhr_to_xyxyxyxy(boxes)
    mins = corners.amin(dim=-2)
    maxs = corners.amax(dim=-2)
    return torch.cat((mins, maxs), dim=-1)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    a1, a2 = box1[:, None, :2], box1[:, None, 2:]
    b1, b2 = box2[None, :, :2], box2[None, :, 2:]
    inter = (torch.minimum(a2, b2) - torch.maximum(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, CIoU: bool = False, eps: float = 1e-7):
    if xywh:
        box1, box2 = xywh_to_xyxy(box1), xywh_to_xyxy(box2)
    inter = (
        (torch.minimum(box1[..., 2:], box2[..., 2:]) - torch.maximum(box1[..., :2], box2[..., :2])).clamp(0).prod(-1)
    )
    area1 = (box1[..., 2] - box1[..., 0]).clamp(0) * (box1[..., 3] - box1[..., 1]).clamp(0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(0) * (box2[..., 3] - box2[..., 1]).clamp(0)
    union = area1 + area2 - inter + eps
    iou = inter / union
    if not CIoU:
        return iou.unsqueeze(-1)
    c_x1y1 = torch.minimum(box1[..., :2], box2[..., :2])
    c_x2y2 = torch.maximum(box1[..., 2:], box2[..., 2:])
    c2 = ((c_x2y2 - c_x1y1) ** 2).sum(-1) + eps
    rho2 = ((xyxy_to_xywh(box1)[..., :2] - xyxy_to_xywh(box2)[..., :2]) ** 2).sum(-1)
    w1, h1 = xyxy_to_xywh(box1)[..., 2:].unbind(-1)
    w2, h2 = xyxy_to_xywh(box2)[..., 2:].unbind(-1)
    v = (4 / math.pi**2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)
    return (iou - (rho2 / c2 + v * alpha)).unsqueeze(-1)


def _get_covariance_matrix(obb: torch.Tensor):
    w, h, a = obb[..., 2:3], obb[..., 3:4], obb[..., 4:5]
    cos, sin = a.cos(), a.sin()
    a_term = (w.pow(2) * cos.pow(2) + h.pow(2) * sin.pow(2)) / 12
    b_term = (w.pow(2) * sin.pow(2) + h.pow(2) * cos.pow(2)) / 12
    c_term = (w.pow(2) - h.pow(2)) * sin * cos / 12
    return a_term, b_term, c_term


def probiou(obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = ((((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)) + eps).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if not CIoU:
        return iou
    w1, h1 = obb1[..., 2:4].split(1, dim=-1)
    w2, h2 = obb2[..., 2:4].split(1, dim=-1)
    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)
    return iou - v * alpha


def batch_probiou(obb1: torch.Tensor, obb2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))
    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = ((((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)) + eps).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def make_anchors(feats: list[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5):
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, feat in enumerate(feats):
        h, w = feat.shape[2:]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), strides[i], dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1) -> torch.Tensor:
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: Optional[int] = None) -> torch.Tensor:
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    return dist.clamp_(0, reg_max - 0.01) if reg_max is not None else dist


def dist2rbox(pred_dist: torch.Tensor, pred_angle: torch.Tensor, anchor_points: torch.Tensor, dim: int = -1) -> torch.Tensor:
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat((x, y), dim=dim) + anchor_points
    return torch.cat((xy, lt + rb), dim=dim)


def rbox2dist(target_bboxes: torch.Tensor, anchor_points: torch.Tensor, target_angle: torch.Tensor, reg_max: Optional[int] = None):
    xy, wh = target_bboxes.split(2, dim=-1)
    offset = xy - anchor_points
    offset_x, offset_y = offset.split(1, dim=-1)
    cos, sin = torch.cos(target_angle), torch.sin(target_angle)
    xf = offset_x * cos + offset_y * sin
    yf = -offset_x * sin + offset_y * cos
    w, h = wh.split(1, dim=-1)
    dist = torch.cat((w / 2 - xf, h / 2 - yf, w / 2 + xf, h / 2 + yf), dim=-1)
    return dist.clamp_(0, reg_max - 0.01) if reg_max is not None else dist


def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_thres]
    return torch.stack(keep)


def nms_rotated(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = batch_probiou(boxes[i : i + 1], boxes[rest]).squeeze(0)
        order = rest[iou <= iou_thres]
    return torch.stack(keep)


def non_max_suppression(prediction: torch.Tensor, conf_thres=0.25, iou_thres=0.45, multi_label=False, max_det=300, nc=0, rotated=False, end2end=False):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if prediction.shape[-1] == 6 or prediction.shape[-1] == 7 or end2end:
        return [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4 - (1 if rotated else 0))
    extra = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres
    prediction = prediction.transpose(-1, -2)
    if not rotated:
        prediction[..., :4] = xywh_to_xyxy(prediction[..., :4])
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        box, cls, extra_info = x.split((4, nc, extra), 1)
        if multi_label and nc > 1:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), extra_info[i]), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), extra_info), 1)[conf.view(-1) > conf_thres]
        if not x.shape[0]:
            continue
        scores = x[:, 4]
        if rotated:
            keep = nms_rotated(torch.cat((x[:, :4], x[:, -1:]), dim=-1), scores, iou_thres)
        else:
            keep = nms_xyxy(x[:, :4], scores, iou_thres)
        output[xi] = x[keep[:max_det]]
    return output


def build_model_from_yaml(cfg: dict[str, Any], task: str, scale: str = "n", ch: int = 3, nc: Optional[int] = None):
    cfg = deepcopy(cfg)
    depth, width, max_channels = cfg.get("scales", {}).get(scale, cfg.get("scales", {}).get("n", [1.0, 1.0, 1024]))
    if nc is not None:
        cfg["nc"] = nc
    layers, save, channels = [], [], [ch]
    reg_max = int(cfg.get("reg_max", 1))
    end2end = bool(cfg.get("end2end", False))

    def ch_lookup(index: int) -> int:
        return channels[index + 1] if index != -1 else channels[-1]

    for i, (f, n, module_name, args) in enumerate(cfg["backbone"] + cfg["head"]):
        module_cls = MODULES[module_name]
        repeats = max(round(n * depth), 1) if n > 1 else n
        if module_name == "nn.Upsample":
            module = module_cls(scale_factor=args[1], mode=args[2])
            c2 = ch_lookup(f if isinstance(f, int) else f[0])
        elif module_name == "Concat":
            module = module_cls(*args)
            c2 = sum(ch_lookup(x) for x in f)
        elif module_name in {"Detect", "OBB", "OBB26"}:
            from_channels = tuple(ch_lookup(x) for x in f)
            if module_name == "Detect":
                module = module_cls(cfg["nc"], reg_max, end2end, from_channels)
            else:
                module = module_cls(cfg["nc"], args[1], reg_max, end2end, from_channels)
            c2 = sum(from_channels)
        else:
            c1 = ch_lookup(f) if isinstance(f, int) else sum(ch_lookup(x) for x in f)
            c2 = args[0]
            if c2 != cfg["nc"]:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if module_name == "C3k2":
                module = module_cls(c1, c2, repeats, *args[1:])
            elif module_name == "C2PSA":
                module = module_cls(c1, c2, repeats)
            elif module_name == "SPPF":
                module = module_cls(c1, c2, *args[1:])
            else:
                module = module_cls(c1, c2, *args[1:])
        module.f = f
        module.i = i
        module.type = module_name
        module.np = sum(p.numel() for p in module.parameters())
        layers.append(module)
        channels.append(c2)
        if isinstance(f, list):
            save.extend(x for x in f if x != -1)
        elif f != -1:
            save.append(f)
    return nn.ModuleList(layers), sorted(set(save)), cfg
