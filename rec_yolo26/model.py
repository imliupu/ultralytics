from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from .loss import build_criterion
from .modules import Detect
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
        self._initialize_modules()
        self._init_head_biases()

    def _initialize_modules(self) -> None:
        """Align module defaults with Ultralytics initialize_weights behavior."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
                m.inplace = True

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
        if args_overrides and "end2end" in args_overrides:
            cfg["end2end"] = bool(args_overrides["end2end"])
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

    def fuse(self):
        """Fuse Conv2d + BatchNorm2d layers for inference, matching Ultralytics eval/infer behavior."""
        bn_count = sum(isinstance(m, nn.BatchNorm2d) for m in self.modules())
        if bn_count < 10:
            return self
        for m in self.modules():
            if hasattr(m, "conv") and hasattr(m, "bn") and hasattr(m, "forward_fuse"):
                m.conv = self._fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
            if hasattr(m, "conv_transpose") and hasattr(m, "bn") and hasattr(m, "forward_fuse"):
                m.conv_transpose = self._fuse_deconv_and_bn(m.conv_transpose, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
            if isinstance(m, Detect) and getattr(m, "end2end", False):
                m.fuse()
        return self

    @staticmethod
    def _fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        fused = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        ).to(conv.weight.device)
        w_conv = conv.weight.view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.running_var + bn.eps)))
        fused.weight.data = torch.mm(w_bn, w_conv).view(fused.weight.shape)
        b_conv = torch.zeros(conv.out_channels, device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fused.bias.data = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
        return fused

    @staticmethod
    def _fuse_deconv_and_bn(deconv: nn.ConvTranspose2d, bn: nn.BatchNorm2d) -> nn.ConvTranspose2d:
        fused = nn.ConvTranspose2d(
            deconv.in_channels,
            deconv.out_channels,
            kernel_size=deconv.kernel_size,
            stride=deconv.stride,
            padding=deconv.padding,
            output_padding=deconv.output_padding,
            dilation=deconv.dilation,
            groups=deconv.groups,
            bias=True,
        ).to(deconv.weight.device)
        w_deconv = deconv.weight.view(deconv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.running_var + bn.eps)))
        fused.weight.data = torch.mm(w_bn, w_deconv).view(fused.weight.shape)
        b_deconv = torch.zeros(deconv.out_channels, device=deconv.weight.device) if deconv.bias is None else deconv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fused.bias.data = torch.mm(w_bn, b_deconv.reshape(-1, 1)).reshape(-1) + b_bn
        return fused

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
        else:
            pred_boxes = torch.nn.functional.softplus(pred_boxes)
        if self.task == "obb":
            angle = preds["angle"]
            dbox = dist2rbox(pred_boxes.transpose(1, 2), angle.transpose(1, 2), anchor_points).transpose(1, 2)
            return torch.cat((dbox * stride_tensor.T, pred_scores.sigmoid(), angle), 1)
        dbox = dist2bbox(pred_boxes.transpose(1, 2), anchor_points, xywh=True).transpose(1, 2)
        return torch.cat((dbox * stride_tensor.T, pred_scores.sigmoid()), 1)

    def _init_head_biases(self) -> None:
        head = self.model[-1]
        if not hasattr(head, "one2many"):
            return

        def init_group(group):
            box_heads, cls_heads = group.get("box_head"), group.get("cls_head")
            if box_heads is None or cls_heads is None:
                return
            for i, (a, b) in enumerate(zip(box_heads, cls_heads)):
                if not hasattr(a[-1], "bias") or not hasattr(b[-1], "bias"):
                    continue
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(5 / max(self.nc, 1) / (640 / float(self.stride[i])) ** 2)

        with torch.no_grad():
            init_group(head.one2many)
            if getattr(head, "end2end", False) and hasattr(head, "one2one"):
                init_group(head.one2one)

    @torch.inference_mode()
    def postprocess(self, raw_preds, conf: float = 0.25, iou: float = 0.7, max_det: int = 300):
        if isinstance(raw_preds, tuple):
            infer, aux = raw_preds
            if self.end2end:
                outputs = []
                for bi in range(infer.shape[0]):
                    x = infer[bi]
                    keep = x[:, 4] > conf
                    x = x[keep]
                    if x.numel() == 0:
                        outputs.append({"bboxes": x[:, :5] if self.task == "obb" else x[:, :4], "conf": x[:, 4], "cls": x[:, 5]})
                        continue
                    bboxes = torch.cat((x[:, :4], x[:, 6:]), dim=-1) if self.task == "obb" else x[:, :4]
                    outputs.append({"bboxes": bboxes, "conf": x[:, 4], "cls": x[:, 5]})
                return outputs
            raw_preds = aux
        preds = raw_preds["one2one"] if isinstance(raw_preds, dict) and "one2one" in raw_preds else raw_preds
        decoded = self.decode_predictions(preds)
        if self.end2end:
            decoded = decoded.transpose(1, 2)  # [B, N, 4 + nc (+1 for obb angle)]
            extra_dims = 1 if self.task == "obb" else 0
            boxes = decoded[..., :4]
            scores = decoded[..., 4 : 4 + self.nc]
            extra = decoded[..., 4 + self.nc : 4 + self.nc + extra_dims]
            max_scores, cls_idx = scores.max(dim=-1)
            topk = min(max_det, decoded.shape[1])
            topk_scores, topk_indices = max_scores.topk(topk, dim=1)
            gathered_boxes = boxes.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, 4))
            gathered_cls = cls_idx.gather(1, topk_indices).float()
            gathered_extra = extra.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, extra_dims)) if extra_dims else None
            outputs = []
            for bi in range(decoded.shape[0]):
                keep = topk_scores[bi] > conf
                b = gathered_boxes[bi][keep]
                c = topk_scores[bi][keep]
                k = gathered_cls[bi][keep]
                if self.task == "obb":
                    e = gathered_extra[bi][keep]
                    b = torch.cat((b, e), dim=-1)
                outputs.append({"bboxes": b, "conf": c, "cls": k})
            return outputs
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
        self.model.names = self.names

    def save_checkpoint(self, path: Union[str, Path], **extra: Any) -> None:
        torch.save({"model": self.state_dict(), "task": self.task, "names": self.names, **extra}, path)

    def save(self, path: Union[str, Path], **extra: Any) -> None:
        self.save_checkpoint(path, **extra)
