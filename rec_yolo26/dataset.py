from __future__ import annotations

import os
from typing import Any

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.dataset import YOLO26Dataset, build_dataloader, build_yolo_dataset
from ultralytics.utils import DEFAULT_CFG

SUPPORTED_TASKS = frozenset({"detect", "obb"})


def load_data_config(data_yaml: str | os.PathLike[str]) -> dict[str, Any]:
    """Load data YAML with the original Ultralytics dataset checker."""
    return check_det_dataset(str(data_yaml), autodownload=False)


def build_dataset(
    data: dict[str, Any],
    split: str,
    task: str,
    imgsz: int,
    augment: bool,
    batch_size: int = 16,
    stride: int = 32,
    args_overrides: dict[str, Any] | None = None,
) -> YOLO26Dataset:
    """Build the original Ultralytics YOLO26 dataset object."""
    if task not in SUPPORTED_TASKS:
        raise NotImplementedError(f"Only {sorted(SUPPORTED_TASKS)} are supported, but got '{task}'.")
    mode = "train" if augment else "val"
    args = get_cfg(
        DEFAULT_CFG,
        overrides={
            "task": task,
            "imgsz": imgsz,
            "workers": 0,
            "batch": batch_size,
            "rect": mode != "train",
            "cache": False,
            **(args_overrides or {}),
        },
    )
    return build_yolo_dataset(args, data[split], batch_size, data, mode=mode, rect=mode != "train", stride=stride)


def create_train_val_dataloaders(
    data_yaml: str | os.PathLike[str],
    task: str,
    imgsz: int,
    batch_size: int,
    workers: int,
    eval_split: str = "val",
    stride: int = 32,
    args_overrides: dict[str, Any] | None = None,
):
    """Create original Ultralytics train/val dataloaders for YOLO26 detect/obb."""
    data = load_data_config(data_yaml)
    train_dataset = build_dataset(
        data,
        "train",
        task,
        imgsz,
        augment=True,
        batch_size=batch_size,
        stride=stride,
        args_overrides={"workers": workers, **(args_overrides or {})},
    )
    split = eval_split if eval_split in data and data.get(eval_split) else "val"
    eval_dataset = build_dataset(
        data,
        split,
        task,
        imgsz,
        augment=False,
        batch_size=batch_size,
        stride=stride,
        args_overrides={"workers": workers, **(args_overrides or {})},
    )
    train_loader = build_dataloader(train_dataset, batch=batch_size, workers=workers, shuffle=True, rank=-1)
    eval_loader = build_dataloader(eval_dataset, batch=batch_size, workers=workers, shuffle=False, rank=-1)
    return data, train_loader, eval_loader
