from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ultralytics.cfg import get_cfg
from ultralytics.models import yolo
from ultralytics.utils import DEFAULT_CFG


@dataclass
class EvalConfig:
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    split: str = "val"
    save_dir: str = "runs/rec_yolo26/val"


def build_metric_evaluator(task: str, names: dict[int, str], config: EvalConfig | None = None):
    """Build the original Ultralytics validator used by YOLO26 detect/obb."""
    config = config or EvalConfig()
    args = get_cfg(
        DEFAULT_CFG,
        overrides={
            "task": task,
            "conf": config.conf,
            "iou": config.iou,
            "max_det": config.max_det,
            "split": config.split,
            "save_dir": config.save_dir,
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "single_cls": False,
            "agnostic_nms": False,
        },
    )
    validator_cls = yolo.obb.OBBValidator if task == "obb" else yolo.detect.DetectionValidator
    validator = validator_cls(dataloader=None, save_dir=Path(config.save_dir), args=args)
    validator.data = {config.split: "", "names": names, "channels": 3}
    return validator


@torch.inference_mode()
def evaluate_model(model, dataloader, device: torch.device, task: str, names: dict[int, str], config: EvalConfig | None = None):
    """Evaluate with the original Ultralytics validator logic and metrics."""
    config = config or EvalConfig()
    validator = build_metric_evaluator(task=task, names=names, config=config)
    validator.device = device
    validator.dataloader = dataloader
    validator.training = False
    validator.args.half = False
    validator.args.plots = False
    validator.args.model = getattr(model, "cfg_path", None) or getattr(getattr(model, "model", None), "yaml_file", None)
    wrapped = getattr(model, "model", model)
    wrapped.eval()
    validator.init_metrics(wrapped)

    for batch in dataloader:
        batch = validator.preprocess(batch)
        preds = wrapped(batch["img"])
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)

    stats = validator.get_stats()
    validator.finalize_metrics()
    return {k: float(v) for k, v in stats.items()}
