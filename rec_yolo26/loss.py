from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import build_targets_from_batch


class YOLO26Criterion(nn.Module):
    """Simple standalone criterion for YOLO26 detect/obb training."""

    def __init__(self, model, task: str):
        super().__init__()
        self.model = model
        self.task = task
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, raw_preds, batch):
        decoded = self.model.decode_outputs(raw_preds)
        targets = build_targets_from_batch(batch, self.task)
        total_box = raw_preds["boxes"].new_tensor(0.0)
        total_cls = raw_preds["boxes"].new_tensor(0.0)
        total_angle = raw_preds["boxes"].new_tensor(0.0)

        for pred, target in zip(decoded, targets):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            gt_boxes = target["bboxes"].to(pred_boxes.device)
            gt_cls = target["cls"].long().to(pred_boxes.device)
            cls_target = torch.zeros_like(pred_scores)
            if gt_boxes.numel() == 0:
                total_cls += self.bce(pred_scores, cls_target).mean()
                continue

            centers = pred_boxes[:, :2]
            gt_centers = gt_boxes[:, :2] * pred_boxes.new_tensor([batch["img"].shape[-1], batch["img"].shape[-2]])
            dist = torch.cdist(gt_centers, centers)
            assigned = dist.argmin(dim=1)
            cls_target[assigned, gt_cls] = 1.0
            total_cls += self.bce(pred_scores, cls_target).mean()
            total_box += F.l1_loss(pred_boxes[assigned, :4], gt_boxes[:, :4] * pred_boxes.new_tensor([batch["img"].shape[-1], batch["img"].shape[-2], batch["img"].shape[-1], batch["img"].shape[-2]]), reduction="mean")
            if self.task == "obb":
                total_angle += F.smooth_l1_loss(pred_boxes[assigned, 4:], gt_boxes[:, 4:], reduction="mean")

        box_gain = 7.5
        cls_gain = 0.5
        angle_gain = 0.2 if self.task == "obb" else 0.0
        total_loss = total_box * box_gain + total_cls * cls_gain + total_angle * angle_gain
        loss_items = torch.stack([total_box.detach(), total_cls.detach(), total_angle.detach()])
        return total_loss, loss_items


def build_criterion(model, task: str) -> YOLO26Criterion:
    return YOLO26Criterion(model=model, task=task)
