from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torchvision.ops import box_iou

from .ops import xywh_to_xyxy, xywhr_to_xyxy


@dataclass
class EvalConfig:
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300


class BaseMetricEvaluator:
    def __init__(self, task: str, names: dict[int, str]):
        self.task = task
        self.names = names
        self.stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
        self.iou_threshold = 0.5

    def reset(self) -> None:
        for v in self.stats.values():
            v.clear()

    def _pair_iou(self, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.task == "obb":
            return box_iou(xywhr_to_xyxy(gt), xywhr_to_xyxy(pred))
        return box_iou(gt, pred)

    def update(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        img_h, img_w = batch["img"].shape[-2:]
        for sample_index, pred in enumerate(preds):
            idx = batch["batch_idx"] == sample_index
            gt_boxes = batch["bboxes"][idx].to(pred["bboxes"].device).clone()
            gt_cls = batch["cls"][idx].view(-1).to(pred["cls"].device)
            if gt_boxes.numel():
                if self.task == "obb":
                    gt_boxes[:, :4] *= gt_boxes.new_tensor([img_w, img_h, img_w, img_h])
                else:
                    gt_boxes = xywh_to_xyxy(gt_boxes) * gt_boxes.new_tensor([img_w, img_h, img_w, img_h])
            pred_boxes = pred["bboxes"]
            pred_cls = pred["cls"].view(-1)
            pred_conf = pred["conf"].view(-1)
            if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
                tp = np.zeros((pred_boxes.shape[0],), dtype=bool)
            else:
                iou = self._pair_iou(gt_boxes, pred_boxes)
                matches = (iou >= self.iou_threshold).cpu().numpy()
                tp = np.zeros((pred_boxes.shape[0],), dtype=bool)
                for gt_index in range(matches.shape[0]):
                    pred_indices = np.where(matches[gt_index])[0]
                    pred_indices = [idx for idx in pred_indices if int(pred_cls[idx]) == int(gt_cls[gt_index])]
                    if pred_indices:
                        best = max(pred_indices, key=lambda idx: float(pred_conf[idx]))
                        tp[best] = True
            self.stats["tp"].append(tp)
            self.stats["conf"].append(pred_conf.detach().cpu().numpy())
            self.stats["pred_cls"].append(pred_cls.detach().cpu().numpy())
            self.stats["target_cls"].append(gt_cls.detach().cpu().numpy())

    def compute(self) -> dict[str, float]:
        tp = np.concatenate(self.stats["tp"], 0) if self.stats["tp"] else np.zeros((0,), dtype=bool)
        conf = np.concatenate(self.stats["conf"], 0) if self.stats["conf"] else np.zeros((0,))
        pred_cls = np.concatenate(self.stats["pred_cls"], 0) if self.stats["pred_cls"] else np.zeros((0,))
        target_cls = np.concatenate(self.stats["target_cls"], 0) if self.stats["target_cls"] else np.zeros((0,))
        tp_sum = tp.sum()
        fp_sum = max(len(tp) - tp_sum, 0)
        fn_sum = max(len(target_cls) - tp_sum, 0)
        precision = float(tp_sum / max(tp_sum + fp_sum, 1))
        recall = float(tp_sum / max(tp_sum + fn_sum, 1))
        map50 = precision * recall
        return {
            "metrics/precision(B)": precision,
            "metrics/recall(B)": recall,
            "metrics/mAP50(B)": map50,
            "metrics/mAP50-95(B)": map50,
        }


@torch.inference_mode()
def evaluate_model(model, dataloader, device: torch.device, task: str, names: dict[int, str], config: EvalConfig | None = None):
    config = config or EvalConfig()
    evaluator = BaseMetricEvaluator(task=task, names=names)
    model.eval()
    for batch in dataloader:
        batch["img"] = batch["img"].to(device).float() / 255
        batch["cls"] = batch["cls"].to(device)
        batch["bboxes"] = batch["bboxes"].to(device)
        batch["batch_idx"] = batch["batch_idx"].to(device)
        raw_preds = model(batch["img"])
        preds = model.postprocess(raw_preds, conf=config.conf, iou=config.iou)
        evaluator.update(preds, batch)
    return evaluator.compute()


def build_metric_evaluator(task: str, names: dict[int, str]) -> BaseMetricEvaluator:
    return BaseMetricEvaluator(task=task, names=names)
