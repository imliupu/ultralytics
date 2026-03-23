from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .ops import batch_probiou, box_iou, xywh_to_xyxy


@dataclass
class EvalConfig:
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300


class Metric:
    def __init__(self) -> None:
        self.p = []
        self.r = []
        self.f1 = []
        self.all_ap = []
        self.ap_class_index = []
        self.nc = 0

    @property
    def ap50(self):
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self) -> float:
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self) -> float:
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self) -> float:
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self) -> float:
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        return [self.mp, self.mr, self.map50, self.map]

    def fitness(self) -> float:
        return float((np.nan_to_num(np.array(self.mean_results())) * [0.0, 0.0, 0.0, 1.0]).sum())

    def update(self, results: tuple):
        self.p, self.r, self.f1, self.all_ap, self.ap_class_index, *_rest = results


class DetMetrics:
    def __init__(self, names: dict[int, str] = {}):
        self.names = names
        self.box = Metric()
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.nt_per_class = None
        self.nt_per_image = None

    def update_stats(self, stat: dict[str, Any]):
        for k in self.stats.keys():
            self.stats[k].append(stat[k])

    def process(self):
        stats = {k: np.concatenate(v, 0) if len(v) else np.zeros((0,)) for k, v in self.stats.items()}
        if stats["target_cls"].size:
            results = ap_per_class(stats["tp"], stats["conf"], stats["pred_cls"], stats["target_cls"], names=self.names)[2:]
            self.box.nc = len(self.names)
            self.box.update(results)
            self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=len(self.names))
            self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=len(self.names))
        return stats

    @property
    def results_dict(self):
        keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"]
        values = [*self.box.mean_results(), self.box.fitness()]
        return dict(zip(keys, [float(v) for v in values]))


class OBBMetrics(DetMetrics):
    pass


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(recall: list[float], precision: list[float]):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, names: dict[int, str] = {}, eps: float = 1e-16):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x = np.linspace(0, 1, 1000)
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    prec_values = []
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l, n_p = nt[ci], i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))
    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    i = smooth(f1_curve.mean(0), 0.1).argmax() if f1_curve.size else 0
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


class DetectionMetricEvaluator:
    def __init__(self, task: str, names: dict[int, str]):
        self.task = task
        self.names = names
        self.metrics = OBBMetrics(names) if task == "obb" else DetMetrics(names)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()

    def _prepare_batch(self, si: int, batch: dict[str, Any]):
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx].clone()
        imgsz = batch["img"].shape[2:]
        if cls.shape[0]:
            if self.task == "obb":
                bbox[..., :4].mul_(torch.tensor(imgsz, device=bbox.device)[[1, 0, 1, 0]])
            else:
                bbox = xywh_to_xyxy(bbox) * torch.tensor(imgsz, device=bbox.device)[[1, 0, 1, 0]]
        return {"cls": cls, "bboxes": bbox}

    def match_predictions(self, pred_classes, true_classes, iou):
        correct = np.zeros((pred_classes.shape[0], self.niou), dtype=bool)
        if iou.shape[0] == 0 or iou.shape[1] == 0:
            return torch.from_numpy(correct)
        correct_class = true_classes[:, None] == pred_classes
        for i, thr in enumerate(self.iouv.cpu().tolist()):
            matches = torch.nonzero((iou >= thr) & correct_class)
            if matches.shape[0]:
                matches = matches.cpu().numpy()
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].cpu().numpy().argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool)

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]):
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        if self.task == "obb":
            iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        else:
            iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def update(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]):
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            cls = pbatch["cls"].cpu().numpy()
            no_pred = pred["cls"].shape[0] == 0
            self.metrics.update_stats({
                **self._process_batch(pred, pbatch),
                "target_cls": cls,
                "target_img": np.unique(cls),
                "conf": np.zeros(0) if no_pred else pred["conf"].detach().cpu().numpy(),
                "pred_cls": np.zeros(0) if no_pred else pred["cls"].detach().cpu().numpy(),
            })


def build_metric_evaluator(task: str, names: dict[int, str]):
    return DetectionMetricEvaluator(task=task, names=names)


@torch.inference_mode()
def evaluate_model(model, dataloader, device: torch.device, task: str, names: dict[int, str], config: Optional[EvalConfig] = None):
    config = config or EvalConfig()
    evaluator = build_metric_evaluator(task, names)
    model.eval()
    for batch in dataloader:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        raw_preds = model(batch["img"])
        preds = model.postprocess(raw_preds, conf=config.conf, iou=config.iou, max_det=config.max_det)
        evaluator.update(preds, batch)
    evaluator.metrics.process()
    return evaluator.metrics.results_dict
