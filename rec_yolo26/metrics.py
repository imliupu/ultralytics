from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from typing import Any, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from .ops import batch_probiou, box_iou, xywh_to_xyxy


@dataclass
class EvalConfig:
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    split: str = "val"
    save_dir: str = "runs/rec_yolo26/val"


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
    func = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    ap = func(np.interp(x, mrec, mpre), x)
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
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}
    i = smooth(f1_curve.mean(0), 0.1).argmax()
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
        self._coco_gt: list[dict[str, Any]] = []
        self._coco_pred: list[dict[str, Any]] = []
        self._image_id_by_file: dict[str, int] = {}
        self._next_image_id = 1
        self._obb_gt: list[dict[str, Any]] = []
        self._obb_pred: list[dict[str, Any]] = []

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
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": batch["ori_shape"][si],
            "ratio_pad": batch["ratio_pad"][si],
            "im_file": batch["im_file"][si],
        }

    @staticmethod
    def _scale_xyxy_to_original(xyxy: torch.Tensor, ratio_pad: tuple[float, tuple[int, int]], ori_shape: tuple[int, int]) -> torch.Tensor:
        ratio, (padw, padh) = ratio_pad
        out = xyxy.clone()
        out[:, [0, 2]] = (out[:, [0, 2]] - float(padw)) / float(ratio)
        out[:, [1, 3]] = (out[:, [1, 3]] - float(padh)) / float(ratio)
        h0, w0 = ori_shape
        out[:, [0, 2]] = out[:, [0, 2]].clamp(0, float(w0))
        out[:, [1, 3]] = out[:, [1, 3]].clamp(0, float(h0))
        return out

    @staticmethod
    def _scale_xywhr_to_original(xywhr: torch.Tensor, ratio_pad: tuple[float, tuple[int, int]], ori_shape: tuple[int, int]) -> torch.Tensor:
        ratio, (padw, padh) = ratio_pad
        out = xywhr.clone()
        out[:, 0] = (out[:, 0] - float(padw)) / float(ratio)
        out[:, 1] = (out[:, 1] - float(padh)) / float(ratio)
        out[:, 2] = out[:, 2] / float(ratio)
        out[:, 3] = out[:, 3] / float(ratio)
        h0, w0 = ori_shape
        out[:, 0] = out[:, 0].clamp(0, float(w0))
        out[:, 1] = out[:, 1].clamp(0, float(h0))
        out[:, 2] = out[:, 2].clamp(0, float(w0))
        out[:, 3] = out[:, 3].clamp(0, float(h0))
        return out

    def _collect_coco_records(self, pred: dict[str, torch.Tensor], pbatch: dict[str, Any]):
        im_file = pbatch["im_file"]
        if im_file not in self._image_id_by_file:
            self._image_id_by_file[im_file] = self._next_image_id
            self._next_image_id += 1
        image_id = self._image_id_by_file[im_file]
        ori_shape = pbatch["ori_shape"]
        ratio_pad = pbatch["ratio_pad"]
        if self.task == "obb":
            gt_cls = pbatch["cls"].detach().cpu().numpy().astype(int)
            gt_xywhr = self._scale_xywhr_to_original(pbatch["bboxes"], ratio_pad, ori_shape).detach().cpu().numpy()
            for c, b in zip(gt_cls.tolist(), gt_xywhr.tolist()):
                cx, cy, w, h, a = b
                self._obb_gt.append({
                    "image_id": image_id,
                    "category_id": int(c),
                    "bbox": [float(cx), float(cy), float(w), float(h), float(a)],
                    "area": float(max(w, 0.0) * max(h, 0.0)),
                })
            if pred["bboxes"].shape[0]:
                pred_xywhr = self._scale_xywhr_to_original(pred["bboxes"], ratio_pad, ori_shape).detach().cpu().numpy()
                pred_cls = pred["cls"].detach().cpu().numpy().astype(int)
                pred_conf = pred["conf"].detach().cpu().numpy()
                for c, s, b in zip(pred_cls.tolist(), pred_conf.tolist(), pred_xywhr.tolist()):
                    cx, cy, w, h, a = b
                    self._obb_pred.append({
                        "image_id": image_id,
                        "category_id": int(c),
                        "bbox": [float(cx), float(cy), float(w), float(h), float(a)],
                        "area": float(max(w, 0.0) * max(h, 0.0)),
                        "score": float(s),
                    })
            return

        if self.task != "detect":
            return

        gt_cls = pbatch["cls"].detach().cpu().numpy().astype(int)
        gt_xyxy = self._scale_xyxy_to_original(pbatch["bboxes"], ratio_pad, ori_shape).detach().cpu().numpy()
        for c, b in zip(gt_cls.tolist(), gt_xyxy.tolist()):
            x1, y1, x2, y2 = b
            self._coco_gt.append({
                "id": len(self._coco_gt) + 1,
                "image_id": image_id,
                "category_id": int(c),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)),
                "iscrowd": 0,
            })

        if pred["bboxes"].shape[0] == 0:
            return
        pred_xyxy = self._scale_xyxy_to_original(pred["bboxes"], ratio_pad, ori_shape).detach().cpu().numpy()
        pred_cls = pred["cls"].detach().cpu().numpy().astype(int)
        pred_conf = pred["conf"].detach().cpu().numpy()
        for c, s, b in zip(pred_cls.tolist(), pred_conf.tolist(), pred_xyxy.tolist()):
            x1, y1, x2, y2 = b
            self._coco_pred.append({
                "image_id": image_id,
                "category_id": int(c),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })

    def match_predictions(self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor):
        correct = np.zeros((pred_classes.shape[0], self.niou), dtype=bool)
        if iou.shape[0] == 0 or iou.shape[1] == 0:
            return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

        iou = (iou * (true_classes[:, None] == pred_classes)).cpu().numpy()
        for i, thr in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= thr)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

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
            self._collect_coco_records(pred, pbatch)
            cls = pbatch["cls"].cpu().numpy()
            no_pred = pred["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(pred, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else pred["conf"].detach().cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else pred["cls"].detach().cpu().numpy(),
                }
            )

    def coco_size_results(self) -> dict[str, float]:
        """Strict COCO semantic AP for small/medium/large (detect task only)."""
        if self.task == "obb":
            return self.obb_coco_size_results()
        if self.task != "detect":
            return {}
        if importlib.util.find_spec("pycocotools") is None:
            return {}
        if not self._coco_gt:
            return {
                "metrics/mAP50-95_small(B)": 0.0,
                "metrics/mAP50-95_medium(B)": 0.0,
                "metrics/mAP50-95_large(B)": 0.0,
            }

        coco_module = importlib.import_module("pycocotools.coco")
        cocoeval_module = importlib.import_module("pycocotools.cocoeval")
        COCO = coco_module.COCO
        COCOeval = cocoeval_module.COCOeval

        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": i, "file_name": f} for f, i in self._image_id_by_file.items()],
            "annotations": self._coco_gt,
            "categories": [{"id": int(i), "name": str(n)} for i, n in self.names.items()],
        }
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(self._coco_pred) if self._coco_pred else coco_gt.loadRes([])
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.maxDets = [1, 10, 100]
        coco_eval.evaluate()
        coco_eval.accumulate()
        # stats: [AP, AP50, AP75, AP_small, AP_medium, AP_large, AR1, AR10, AR100, AR_small, AR_medium, AR_large]
        return {
            "metrics/mAP50-95_small(B)": float(coco_eval.stats[3]) if coco_eval.stats[3] >= 0 else 0.0,
            "metrics/mAP50-95_medium(B)": float(coco_eval.stats[4]) if coco_eval.stats[4] >= 0 else 0.0,
            "metrics/mAP50-95_large(B)": float(coco_eval.stats[5]) if coco_eval.stats[5] >= 0 else 0.0,
        }

    def obb_coco_size_results(self) -> dict[str, float]:
        """COCO-style area-range AP for OBB using rotated IoU and COCO AP integration."""
        if not self._obb_gt:
            return {
                "metrics/mAP50-95_small(OBB)": 0.0,
                "metrics/mAP50-95_medium(OBB)": 0.0,
                "metrics/mAP50-95_large(OBB)": 0.0,
            }

        area_ranges = {
            "small": (0.0, float(32**2)),
            "medium": (float(32**2), float(96**2)),
            "large": (float(96**2), float("inf")),
        }
        eps = 1e-16
        iouv = self.iouv.cpu().numpy()
        results = {}
        classes = [int(k) for k in self.names.keys()]

        # Pre-index records by class+image and precompute IoU matrix once to avoid repeated heavy computation.
        gt_by_class_img: dict[int, dict[int, list[dict[str, Any]]]] = {c: {} for c in classes}
        dt_by_class_img: dict[int, dict[int, list[dict[str, Any]]]] = {c: {} for c in classes}
        for g in self._obb_gt:
            c = int(g["category_id"])
            if c in gt_by_class_img:
                gt_by_class_img[c].setdefault(int(g["image_id"]), []).append(g)
        for d in self._obb_pred:
            c = int(d["category_id"])
            if c in dt_by_class_img:
                dt_by_class_img[c].setdefault(int(d["image_id"]), []).append(d)

        cache: dict[int, dict[int, dict[str, Any]]] = {c: {} for c in classes}
        for c in classes:
            image_ids = sorted(set(gt_by_class_img[c].keys()) | set(dt_by_class_img[c].keys()))
            for img_id in image_ids:
                gt_img = gt_by_class_img[c].get(img_id, [])
                dt_img = dt_by_class_img[c].get(img_id, [])
                dt_img = sorted(dt_img, key=lambda x: x["score"], reverse=True)[:100]

                gt_boxes = torch.tensor([g["bbox"] for g in gt_img], dtype=torch.float32) if gt_img else torch.zeros((0, 5), dtype=torch.float32)
                gt_area = np.array([g["area"] for g in gt_img], dtype=np.float32) if gt_img else np.zeros((0,), dtype=np.float32)
                dt_boxes = torch.tensor([d["bbox"] for d in dt_img], dtype=torch.float32) if dt_img else torch.zeros((0, 5), dtype=torch.float32)
                dt_conf = np.array([d["score"] for d in dt_img], dtype=np.float32) if dt_img else np.zeros((0,), dtype=np.float32)
                ious = batch_probiou(gt_boxes, dt_boxes).cpu().numpy() if (len(gt_img) and len(dt_img)) else np.zeros((len(gt_img), len(dt_img)), dtype=np.float32)
                cache[c][img_id] = {"gt_area": gt_area, "dt_conf": dt_conf, "ious": ious}

        def match_one_image(ious: np.ndarray, gt_ignore: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
            nd = ious.shape[1]
            tp = np.zeros((nd,), dtype=bool)
            fp = np.zeros((nd,), dtype=bool)
            if nd == 0:
                return tp, fp
            if ious.shape[0] == 0:
                fp[:] = True
                return tp, fp

            non_ignore_idx = np.where(~gt_ignore)[0]
            ignore_idx = np.where(gt_ignore)[0]
            matched = np.zeros((ious.shape[0],), dtype=bool)
            for d_idx in range(nd):
                # Match non-ignore GT first.
                if non_ignore_idx.size:
                    cand = non_ignore_idx[~matched[non_ignore_idx]]
                    if cand.size:
                        cand_iou = ious[cand, d_idx]
                        j = cand_iou.argmax()
                        if cand_iou[j] >= thr:
                            g_idx = cand[j]
                            matched[g_idx] = True
                            tp[d_idx] = True
                            continue
                # Match ignore GT (detection ignored, neither TP nor FP).
                if ignore_idx.size:
                    cand = ignore_idx[~matched[ignore_idx]]
                    if cand.size:
                        cand_iou = ious[cand, d_idx]
                        j = cand_iou.argmax()
                        if cand_iou[j] >= thr:
                            matched[cand[j]] = True
                            continue
                fp[d_idx] = True
            return tp, fp

        for size_name, (amin, amax) in area_ranges.items():
            ap_cls = []
            for c in classes:
                npos = 0
                for item in cache[c].values():
                    gt_area = item["gt_area"]
                    npos += int(((gt_area >= amin) & (gt_area < amax)).sum())
                if npos == 0:
                    continue

                all_conf, all_tp, all_fp = [], [], []
                for item in cache[c].values():
                    conf = item["dt_conf"]
                    if conf.size == 0:
                        continue
                    gt_area = item["gt_area"]
                    gt_ignore = ~((gt_area >= amin) & (gt_area < amax))
                    ious = item["ious"]
                    tp = np.zeros((conf.shape[0], self.niou), dtype=bool)
                    fp = np.zeros((conf.shape[0], self.niou), dtype=bool)
                    for t_idx, thr in enumerate(iouv.tolist()):
                        tp[:, t_idx], fp[:, t_idx] = match_one_image(ious, gt_ignore, thr)
                    all_conf.append(conf)
                    all_tp.append(tp)
                    all_fp.append(fp)

                if not all_conf:
                    ap_cls.append(np.zeros(self.niou, dtype=np.float32))
                    continue

                conf = np.concatenate(all_conf, axis=0)
                tp = np.concatenate(all_tp, axis=0)
                fp = np.concatenate(all_fp, axis=0)
                order = np.argsort(-conf)
                tp = tp[order]
                fp = fp[order]

                ap_t = np.zeros(self.niou, dtype=np.float32)
                for t_idx in range(self.niou):
                    tpc = tp[:, t_idx].cumsum(0)
                    fpc = fp[:, t_idx].cumsum(0)
                    recall = tpc / (npos + eps)
                    precision = tpc / (tpc + fpc + eps)
                    ap_t[t_idx], _, _ = compute_ap(recall, precision)
                ap_cls.append(ap_t)

            results[f"metrics/mAP50-95_{size_name}(OBB)"] = float(np.stack(ap_cls, axis=0).mean()) if ap_cls else 0.0

        return results

    def running_results(self) -> dict[str, float]:
        """Compute current metrics from accumulated stats for progress display."""
        stats = {k: np.concatenate(v, 0) if len(v) else np.zeros((0,)) for k, v in self.metrics.stats.items()}
        keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"]
        if not stats["target_cls"].size:
            return dict(zip(keys, [0.0, 0.0, 0.0, 0.0, 0.0]))
        results = ap_per_class(stats["tp"], stats["conf"], stats["pred_cls"], stats["target_cls"], names=self.names)[2:]
        p, r, _f1, all_ap, _cls_idx, *_ = results
        mp = float(p.mean()) if len(p) else 0.0
        mr = float(r.mean()) if len(r) else 0.0
        map50 = float(all_ap[:, 0].mean()) if len(all_ap) else 0.0
        map5095 = float(all_ap.mean()) if len(all_ap) else 0.0
        fitness = map5095
        return dict(zip(keys, [mp, mr, map50, map5095, fitness]))


def build_metric_evaluator(task: str, names: dict[int, str]):
    return DetectionMetricEvaluator(task=task, names=names)


@torch.inference_mode()
def evaluate_model(
    model,
    dataloader,
    device: torch.device,
    task: str,
    names: dict[int, str],
    config: Optional[EvalConfig] = None,
    show_progress: bool = True,
    progress_update_interval: int = 10,
):
    config = config or EvalConfig()
    evaluator = build_metric_evaluator(task, names)
    model.eval()
    pbar = tqdm(dataloader, desc=f"{task} eval", dynamic_ncols=True, leave=False) if show_progress else dataloader
    for i, batch in enumerate(pbar):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        raw_preds = model(batch["img"])
        preds = model.postprocess(raw_preds, conf=config.conf, iou=config.iou, max_det=config.max_det)
        evaluator.update(preds, batch)
        if show_progress and ((i + 1) % max(progress_update_interval, 1) == 0 or (i + 1) == len(dataloader)):
            running = evaluator.running_results()
            pbar.set_postfix({
                "P": f"{running['metrics/precision(B)']:.4f}",
                "R": f"{running['metrics/recall(B)']:.4f}",
                "mAP50": f"{running['metrics/mAP50(B)']:.4f}",
                "mAP50-95": f"{running['metrics/mAP50-95(B)']:.4f}",
            })
    evaluator.metrics.process()
    results = evaluator.metrics.results_dict
    results.update(evaluator.coco_size_results())
    return results
