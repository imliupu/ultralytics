from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .ops import xywh_to_xyxy, yaml_load

SUPPORTED_TASKS = frozenset({"detect", "obb"})
IMG_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class AugmentConfig:
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.5
    flipud: float = 0.0


def load_data_config(data_yaml: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(data_yaml).resolve()
    data = yaml_load(path)
    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    data["path"] = str(root)
    for split in ("train", "val", "test"):
        value = data.get(split)
        if value is None:
            continue
        split_path = Path(value)
        data[split] = str((root / split_path).resolve()) if not split_path.is_absolute() else str(split_path.resolve())
    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    data["names"] = names
    data["nc"] = len(names)
    data.setdefault("channels", 3)
    return data


def list_images(path_like: str) -> list[Path]:
    path = Path(path_like)
    if path.is_file() and path.suffix == ".txt":
        return [Path(x.strip()) for x in path.read_text().splitlines() if x.strip()]
    if path.is_dir():
        return sorted(x for x in path.rglob("*") if x.suffix.lower() in IMG_FORMATS)
    if path.is_file() and path.suffix.lower() in IMG_FORMATS:
        return [path]
    raise FileNotFoundError(f"Unsupported image source: {path_like}")


def letterbox(image: np.ndarray, new_shape: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    shape = image.shape[:2]
    ratio = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw //= 2
    dh //= 2
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, dh, new_shape - new_unpad[1] - dh, dw, new_shape - new_unpad[0] - dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, ratio, (dw, dh)


def augment_hsv(image: np.ndarray, cfg: AugmentConfig) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    gains = np.array([cfg.hsv_h, cfg.hsv_s, cfg.hsv_v]) * np.random.uniform(-1, 1, 3) + 1
    hsv[..., 0] = (hsv[..., 0] * gains[0]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * gains[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * gains[2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def polygon_to_xywhr(points: np.ndarray) -> np.ndarray:
    xs, ys = points[:, 0], points[:, 1]
    cx, cy = xs.mean(), ys.mean()
    w = np.linalg.norm(points[1] - points[0])
    h = np.linalg.norm(points[2] - points[1])
    angle = math.atan2(points[1, 1] - points[0, 1], points[1, 0] - points[0, 0])
    return np.array([cx, cy, w, h, angle], dtype=np.float32)


class YOLO26Dataset(Dataset):
    def __init__(
        self,
        image_root: str,
        task: str,
        imgsz: int,
        augment: bool,
        names: dict[int, str],
        augment_cfg: AugmentConfig | None = None,
    ) -> None:
        if task not in SUPPORTED_TASKS:
            raise NotImplementedError(f"Only {sorted(SUPPORTED_TASKS)} are supported, but got '{task}'.")
        self.task = task
        self.imgsz = imgsz
        self.augment = augment
        self.names = names
        self.augment_cfg = augment_cfg or AugmentConfig()
        self.images = list_images(image_root)
        self.labels = [self._label_path(path) for path in self.images]

    @staticmethod
    def _label_path(image_path: Path) -> Path:
        parts = list(image_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
        label = Path(*parts).with_suffix(".txt")
        return label

    def __len__(self) -> int:
        return len(self.images)

    def _load_labels(self, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
        if not label_path.exists():
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 5 if self.task == 'obb' else 4), dtype=np.float32)
        rows = []
        for line in label_path.read_text().splitlines():
            parts = [float(x) for x in line.strip().split()]
            if not parts:
                continue
            rows.append(parts)
        if not rows:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 5 if self.task == 'obb' else 4), dtype=np.float32)
        arr = np.asarray(rows, dtype=np.float32)
        cls = arr[:, :1]
        if self.task == "detect":
            boxes = arr[:, 1:5]
        else:
            if arr.shape[1] == 6:
                boxes = arr[:, 1:6]
            elif arr.shape[1] == 9:
                boxes = np.stack([polygon_to_xywhr(x.reshape(4, 2)) for x in arr[:, 1:9]], axis=0)
            else:
                raise ValueError(f"Unsupported OBB label format in {label_path}: expected 6 or 9 columns, got {arr.shape[1]}")
        return cls, boxes

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.images[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        original_shape = image.shape[:2]
        cls, boxes = self._load_labels(self.labels[index])
        if self.augment:
            image = augment_hsv(image, self.augment_cfg)
        image, ratio, pad = letterbox(image, self.imgsz)

        if boxes.shape[0]:
            boxes = boxes.copy()
            boxes[:, 0] = boxes[:, 0] * original_shape[1] * ratio + pad[0]
            boxes[:, 1] = boxes[:, 1] * original_shape[0] * ratio + pad[1]
            boxes[:, 2] = boxes[:, 2] * original_shape[1] * ratio
            boxes[:, 3] = boxes[:, 3] * original_shape[0] * ratio
            if self.task == "detect":
                boxes[:, [0, 2]] /= self.imgsz
                boxes[:, [1, 3]] /= self.imgsz
            else:
                boxes[:, :4] /= np.array([self.imgsz, self.imgsz, self.imgsz, self.imgsz], dtype=np.float32)

        if self.augment and random.random() < self.augment_cfg.fliplr:
            image = np.ascontiguousarray(image[:, ::-1])
            if boxes.shape[0]:
                boxes[:, 0] = 1.0 - boxes[:, 0]
                if self.task == "obb":
                    boxes[:, 4] *= -1
        if self.augment and random.random() < self.augment_cfg.flipud:
            image = np.ascontiguousarray(image[::-1])
            if boxes.shape[0]:
                boxes[:, 1] = 1.0 - boxes[:, 1]
                if self.task == "obb":
                    boxes[:, 4] *= -1

        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        return {
            "img": torch.from_numpy(image),
            "cls": torch.from_numpy(cls),
            "bboxes": torch.from_numpy(boxes),
            "batch_idx": torch.zeros((len(cls),), dtype=torch.int64),
            "im_file": str(image_path),
            "ori_shape": original_shape,
            "ratio_pad": (ratio, pad),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated = {"img": torch.stack([x["img"] for x in batch], 0)}
        collated["cls"] = torch.cat([x["cls"] for x in batch], 0) if batch else torch.zeros((0, 1))
        collated["bboxes"] = torch.cat([x["bboxes"] for x in batch], 0) if batch else torch.zeros((0, 4))
        collated["batch_idx"] = torch.cat([
            torch.full((len(x["cls"]),), i, dtype=torch.int64) for i, x in enumerate(batch)
        ], 0) if batch else torch.zeros((0,), dtype=torch.int64)
        collated["im_file"] = [x["im_file"] for x in batch]
        collated["ori_shape"] = [x["ori_shape"] for x in batch]
        collated["ratio_pad"] = [x["ratio_pad"] for x in batch]
        return collated


def build_dataset(data: dict[str, Any], split: str, task: str, imgsz: int, augment: bool) -> YOLO26Dataset:
    return YOLO26Dataset(image_root=data[split], task=task, imgsz=imgsz, augment=augment, names=data["names"])


def build_dataloader(dataset: YOLO26Dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True, collate_fn=dataset.collate_fn)


def create_train_val_dataloaders(
    data_yaml: str | os.PathLike[str],
    task: str,
    imgsz: int,
    batch_size: int,
    workers: int,
    eval_split: str = "val",
) -> tuple[dict[str, Any], DataLoader, DataLoader]:
    data = load_data_config(data_yaml)
    train_loader = build_dataloader(build_dataset(data, "train", task, imgsz, augment=True), batch_size=batch_size, workers=workers, shuffle=True)
    split = eval_split if eval_split in data and data.get(eval_split) else "val"
    eval_loader = build_dataloader(build_dataset(data, split, task, imgsz, augment=False), batch_size=batch_size, workers=workers, shuffle=False)
    return data, train_loader, eval_loader
