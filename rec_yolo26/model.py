from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .modules import Detect, OBB
from .ops import build_model_from_yaml, make_grid, non_max_suppression, yaml_load

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
MODEL_ALIASES = {
    "yolo26": CONFIG_DIR / "yolo26.yaml",
    "yolo26-obb": CONFIG_DIR / "yolo26-obb.yaml",
    "yolo26obb": CONFIG_DIR / "yolo26-obb.yaml",
}


@dataclass
class ModelBuildConfig:
    task: str
    nc: int
    model: str = "yolo26"
    scale: str = "n"
    ch: int = 3


class RecYOLO26Model(nn.Module):
    """Standalone YOLO26 / YOLO26-OBB model without Ultralytics runtime dependency."""

    def __init__(self, layers: nn.ModuleList, save: list[int], task: str, nc: int, ch: int = 3):
        super().__init__()
        self.model = layers
        self.save_indices = save
        self.task = task
        self.nc = nc
        self.ch = ch
        self.head = self.model[-1]
        self.names = {i: str(i) for i in range(nc)}
        self.stride = self._infer_stride(ch=ch)

    @staticmethod
    def resolve_model_cfg(model: str, task: str) -> Path:
        key = model.lower().replace("_", "-")
        if key in MODEL_ALIASES:
            return MODEL_ALIASES[key]
        return CONFIG_DIR / ("yolo26-obb.yaml" if task == "obb" or key.endswith("-obb") else "yolo26.yaml")

    @classmethod
    def build(
        cls,
        task: str,
        nc: int,
        model: str = "yolo26",
        scale: str = "n",
        ch: int = 3,
        verbose: bool = False,
        args_overrides: dict[str, Any] | None = None,
    ) -> "RecYOLO26Model":
        cfg = yaml_load(cls.resolve_model_cfg(model, task))
        layers, save = build_model_from_yaml(cfg=cfg, task=task, scale=scale, ch=ch, nc=nc)
        return cls(layers=layers, save=save, task=task, nc=nc, ch=ch)

    def _infer_stride(self, ch: int) -> torch.Tensor:
        with torch.no_grad():
            raw = self.forward(torch.zeros(1, ch, 256, 256))
        strides = []
        for feat in raw["feats"]:
            strides.append(256 / feat.shape[-1])
        return torch.tensor(strides, dtype=torch.float32)

    def save_checkpoint(self, path: str | Path, **extra: Any) -> None:
        torch.save({"model": self.state_dict(), "task": self.task, "names": self.names, **extra}, path)

    def save(self, path: str | Path, **extra: Any) -> None:
        self.save_checkpoint(path, **extra)

    def load(self, path: str | Path, strict: bool = True) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()
        self.load_state_dict(state_dict, strict=strict)
        self.names = checkpoint.get("names", self.names)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = []
        x = images
        for module in self.model:
            if module.f != -1:
                x = outputs[module.f] if isinstance(module.f, int) else [x if j == -1 else outputs[j] for j in module.f]
            if module is self.head:
                return module.forward_head(x if isinstance(x, list) else [x])
            x = module(x)
            outputs.append(x)
        raise RuntimeError("Head layer was not reached.")

    def decode_outputs(self, raw_preds: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        feats = raw_preds["feats"]
        boxes = raw_preds["boxes"]
        scores = raw_preds["scores"]
        angles = raw_preds.get("angle")
        start = 0
        decoded_levels = []
        for level, feat in enumerate(feats):
            bs, _, h, w = feat.shape
            count = h * w
            grid = make_grid(h, w, feat.device).unsqueeze(0)
            stride = self.stride[level].to(feat.device)
            level_boxes = boxes[:, :, start : start + count].permute(0, 2, 1)
            level_scores = scores[:, :, start : start + count].permute(0, 2, 1)
            xy = (level_boxes[..., :2].sigmoid() * 2.0 - 0.5 + grid) * stride
            wh = (level_boxes[..., 2:4].sigmoid() * 2.0).pow(2.0) * stride
            decoded = torch.cat((xy, wh), dim=-1)
            level_pred = {"boxes": decoded, "scores": level_scores}
            if angles is not None:
                level_pred["angles"] = angles[:, :, start : start + count].permute(0, 2, 1).tanh() * (torch.pi / 2)
            decoded_levels.append(level_pred)
            start += count

        merged = []
        for b in range(boxes.shape[0]):
            sample_boxes = torch.cat([x["boxes"][b] for x in decoded_levels], dim=0)
            sample_scores = torch.cat([x["scores"][b] for x in decoded_levels], dim=0)
            sample = {"boxes": sample_boxes, "scores": sample_scores}
            if angles is not None:
                sample["boxes"] = torch.cat((sample_boxes, torch.cat([x["angles"][b] for x in decoded_levels], dim=0)), dim=-1)
            merged.append(sample)
        return merged

    @torch.inference_mode()
    def postprocess(self, raw_preds: dict[str, torch.Tensor], conf: float = 0.25, iou: float = 0.7) -> list[dict[str, torch.Tensor]]:
        decoded = self.decode_outputs(raw_preds)
        return non_max_suppression(decoded, conf_thres=conf, iou_thres=iou, task=self.task)
