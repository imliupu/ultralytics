from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from .loss import build_criterion
from .ops import build_model_from_yaml, dist2bbox, dist2rbox, make_anchors, non_max_suppression, yaml_load

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
    def __init__(self, layers: nn.ModuleList, save: list[int], task: str, cfg: dict[str, Any], cfg_path: Path, ch: int = 3):
        super().__init__()
        self.model = layers
        self.save_layers = save
        self.task = task
        self.cfg = cfg
        self.cfg_path = Path(cfg_path)
        self.ch = ch
        self.nc = int(cfg["nc"])
        self.names = {i: str(i) for i in range(self.nc)}
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, angle=1.0, epochs=100)
        self.end2end = bool(getattr(self.model[-1], "end2end", False))
        self.criterion = None
        self.stride = self._infer_stride(ch)
        self.model[-1].stride = self.stride

    @staticmethod
    def resolve_model_cfg(model: str, task: str) -> Path:
        key = model.lower().replace("_", "-")
        if key in MODEL_ALIASES:
            return MODEL_ALIASES[key]
        if Path(model).suffix in {".yaml", ".yml"}:
            return Path(model)
        return CONFIG_DIR / ("yolo26-obb.yaml" if task == "obb" or key.endswith("-obb") else "yolo26.yaml")

    @classmethod
    def build(cls, task: str, nc: int, model: str = "yolo26", scale: str = "n", ch: int = 3, verbose: bool = False, args_overrides: Optional[dict[str, Any]] = None):
        cfg_path = cls.resolve_model_cfg(model, task)
        cfg = yaml_load(cfg_path)
        layers, save, cfg = build_model_from_yaml(cfg=cfg, task=task, scale=scale, ch=ch, nc=nc)
        instance = cls(layers=layers, save=save, task=task, cfg=cfg, cfg_path=cfg_path, ch=ch)
        if args_overrides:
            for k, v in args_overrides.items():
                setattr(instance.args, k, v)
        if verbose:
            print(f"Built {task} model from {cfg_path} with {sum(p.numel() for p in instance.parameters()):,} params")
        return instance

    def _predict_once(self, x: torch.Tensor):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save_layers else None)
        return x

    def forward(self, x):
        if isinstance(x, dict):
            return self.loss(x)
        return self._predict_once(x)

    def loss(self, batch: dict[str, torch.Tensor], preds=None):
        if self.criterion is None:
            self.criterion = build_criterion(self, self.task)
        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def _infer_stride(self, ch: int) -> torch.Tensor:
        training = self.training
        self.train()
        with torch.no_grad():
            raw = self._predict_once(torch.zeros(1, ch, 256, 256))
        preds = raw["one2many"] if isinstance(raw, dict) and "one2many" in raw else raw
        feats = preds["feats"]
        stride = torch.tensor([256 / feat.shape[-2] for feat in feats], dtype=torch.float32)
        self.train(training)
        return stride

    def decode_predictions(self, preds: dict[str, torch.Tensor]) -> torch.Tensor:
        head = self.model[-1]
        pred_boxes = preds["boxes"]
        pred_scores = preds["scores"]
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        if head.reg_max > 1:
            b, _, a = pred_boxes.shape
            pred_boxes = pred_boxes.view(b, 4, head.reg_max, a).transpose(2, 1).softmax(1)
            proj = torch.arange(head.reg_max, dtype=pred_boxes.dtype, device=pred_boxes.device)
            pred_boxes = pred_boxes.matmul(proj).view(b, 4, a)
        if self.task == "obb":
            angle = preds["angle"]
            dbox = dist2rbox(pred_boxes.transpose(1, 2), angle.transpose(1, 2), anchor_points).transpose(1, 2)
            return torch.cat((dbox * stride_tensor.T, pred_scores.sigmoid(), angle), 1)
        dbox = dist2bbox(pred_boxes.transpose(1, 2), anchor_points, xywh=True).transpose(1, 2)
        return torch.cat((dbox * stride_tensor.T, pred_scores.sigmoid()), 1)

    @torch.inference_mode()
    def postprocess(self, raw_preds, conf: float = 0.25, iou: float = 0.7, max_det: int = 300):
        preds = raw_preds["one2one"] if isinstance(raw_preds, dict) and "one2one" in raw_preds else raw_preds
        decoded = self.decode_predictions(preds)
        outputs = non_max_suppression(
            decoded,
            conf_thres=conf,
            iou_thres=iou,
            multi_label=True,
            max_det=max_det,
            nc=self.nc,
            rotated=self.task == "obb",
            end2end=False,
        )
        formatted = []
        for x in outputs:
            extra = x[:, 6:] if x.shape[1] > 6 else x[:, 6:]
            bboxes = torch.cat((x[:, :4], extra), dim=-1) if self.task == "obb" else x[:, :4]
            formatted.append({"bboxes": bboxes, "conf": x[:, 4], "cls": x[:, 5]})
        return formatted

    def load(self, path: Union[str, Path], strict: bool = True) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()
        self.load_state_dict(state_dict, strict=strict)
        self.names = checkpoint.get("names", self.names)

    def save_checkpoint(self, path: Union[str, Path], **extra: Any) -> None:
        torch.save({"model": self.state_dict(), "task": self.task, "names": self.names, **extra}, path)

    def save(self, path: Union[str, Path], **extra: Any) -> None:
        self.save_checkpoint(path, **extra)
