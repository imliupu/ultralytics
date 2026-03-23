from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


class Conv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k2(nn.Module):
    """Simplified YOLO26 CSP-style block used by the standalone project."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, large_kernel: bool = False):
        super().__init__()
        c_ = int(c2 * e)
        k = 5 if large_kernel else 3
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.blocks = nn.Sequential(*[Bottleneck(c_, c_, shortcut=shortcut, e=1.0) for _ in range(2)])
        self.cv3 = Conv(2 * c_, c2, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SimpleAttention(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (c // heads) ** -0.5
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        y = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        y = y.reshape(b, c, h, w)
        return self.proj(y) + x


class C2PSA(nn.Module):
    def __init__(self, c1: int, c2: int | None = None):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.cv1 = Conv(c1, c2, 1, 1)
        self.attn = SimpleAttention(c2)
        self.ffn = nn.Sequential(Conv(c2, c2, 1, 1), Conv(c2, c2, 3, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        x = self.attn(x)
        return x + self.ffn(x)


class Concat(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.dim)


class Detect(nn.Module):
    def __init__(self, nc: int, ch: tuple[int, ...]):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_channels = 4
        self.box_head = nn.ModuleList(nn.Conv2d(c, self.reg_channels, 1) for c in ch)
        self.cls_head = nn.ModuleList(nn.Conv2d(c, nc, 1) for c in ch)

    def forward_head(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        boxes, scores = [], []
        for feature, box_layer, cls_layer in zip(features, self.box_head, self.cls_head):
            b, _, h, w = feature.shape
            boxes.append(box_layer(feature).view(b, 4, h * w))
            scores.append(cls_layer(feature).view(b, self.nc, h * w))
        return {"boxes": torch.cat(boxes, dim=-1), "scores": torch.cat(scores, dim=-1), "feats": features}


class OBB(Detect):
    def __init__(self, nc: int, ne: int, ch: tuple[int, ...]):
        super().__init__(nc=nc, ch=ch)
        self.angle_head = nn.ModuleList(nn.Conv2d(c, ne, 1) for c in ch)

    def forward_head(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        preds = super().forward_head(features)
        angles = []
        for feature, angle_layer in zip(features, self.angle_head):
            b, _, h, w = feature.shape
            angles.append(angle_layer(feature).view(b, 1, h * w))
        preds["angle"] = torch.cat(angles, dim=-1)
        return preds
