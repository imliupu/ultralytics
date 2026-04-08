from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .ops import yaml_load

SUPPORTED_TASKS = frozenset({"detect", "obb"})
IMG_FORMATS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}


@dataclass
class AugmentConfig:
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.5
    flipud: float = 0.0
    bgr: float = 0.0


def _normalize_angle(angle: float) -> float:
    while angle >= 3 * np.pi / 4:
        angle -= np.pi
    while angle < -np.pi / 4:
        angle += np.pi
    return float(angle)


def xyxyr_to_xywhr(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2, angle = boxes.T
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = np.abs(x2 - x1)
    h = np.abs(y2 - y1)
    angles = np.array([_normalize_angle(a) for a in angle], dtype=np.float32)
    return np.stack((cx, cy, w, h, angles), axis=1).astype(np.float32)


def load_data_config(data_yaml: Union[str, os.PathLike]) -> dict[str, Any]:
    path = Path(data_yaml).resolve()
    data = yaml_load(path)
    if "val" not in data and "validation" in data:
        data["val"] = data.pop("validation")
    if "train" not in data or "val" not in data:
        raise SyntaxError(f"{path} must define both 'train' and 'val' splits.")
    if "names" not in data and "nc" not in data:
        raise SyntaxError(f"{path} must define either 'names' or 'nc'.")
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(int(data["nc"]))]
    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    data["path"] = str(root)
    for split in ("train", "val", "test"):
        value = data.get(split)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            resolved = []
            for item in value:
                split_path = Path(item)
                resolved.append(str((root / split_path).resolve()) if not split_path.is_absolute() else str(split_path.resolve()))
            data[split] = resolved
        else:
            split_path = Path(value)
            data[split] = str((root / split_path).resolve()) if not split_path.is_absolute() else str(split_path.resolve())
    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    data["names"] = names
    data["nc"] = len(names)
    data.setdefault("channels", 3)
    data.setdefault("obb_format", "xywhr")
    return data


def list_images(path_like: Union[str, os.PathLike, list[str], tuple[str, ...]]) -> list[Path]:
    if isinstance(path_like, (list, tuple)):
        images = []
        for item in path_like:
            images.extend(list_images(item))
        return sorted(images)
    path = Path(path_like)
    if path.is_file() and path.suffix == ".txt":
        return [Path(x.strip()) for x in path.read_text().splitlines() if x.strip()]
    if path.is_dir():
        return sorted(x for x in path.rglob("*") if x.suffix.lower() in IMG_FORMATS)
    if path.is_file() and path.suffix.lower() in IMG_FORMATS:
        return [path]
    raise FileNotFoundError(f"Unsupported image source: {path_like}")


def normalize_imgsz(imgsz: Union[int, tuple[int, int], list[int]]) -> tuple[int, int]:
    if isinstance(imgsz, int):
        return imgsz, imgsz
    if isinstance(imgsz, (tuple, list)):
        if len(imgsz) == 1:
            return int(imgsz[0]), int(imgsz[0])
        if len(imgsz) == 2:
            return int(imgsz[0]), int(imgsz[1])
    raise ValueError(f"Invalid imgsz={imgsz!r}, expected int or 2-int tuple/list.")


def letterbox(
    image: Image.Image,
    new_shape: Union[int, tuple[int, int]],
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    center: bool = True,
    stride: int = 32,
    padding_value: int = 114,
) -> tuple[Image.Image, float, tuple[int, int]]:
    """Resize and pad image following Ultralytics LetterBox behavior."""
    new_h, new_w = normalize_imgsz(new_shape)
    w, h = image.size
    r = min(new_h / h, new_w / w)
    if not scaleup:
        r = min(r, 1.0)
    ratio = r
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]
    if auto:
        dw, dh = dw % stride, dh % stride
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_w, new_h)
    if center:
        dw /= 2
        dh /= 2
    if image.size != new_unpad:
        image = image.resize(new_unpad, Image.BILINEAR)
    left = round(dw - 0.1) if center else 0
    top = round(dh - 0.1) if center else 0
    right = round(dw + 0.1)
    bottom = round(dh + 0.1)
    canvas = Image.new("RGB", (new_unpad[0] + left + right, new_unpad[1] + top + bottom), (padding_value,) * 3)
    canvas.paste(image, (left, top))
    return canvas, ratio, (left, top)


def augment_hsv(image: Image.Image, cfg: AugmentConfig) -> Image.Image:
    hsv = np.array(image.convert("HSV"), dtype=np.float32)
    gains = np.array([cfg.hsv_h, cfg.hsv_s, cfg.hsv_v]) * np.random.uniform(-1, 1, 3) + 1
    hsv[..., 0] = (hsv[..., 0] * gains[0]) % 255
    hsv[..., 1] = np.clip(hsv[..., 1] * gains[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * gains[2], 0, 255)
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def polygon_to_xywhr(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    try:
        import cv2

        # Align with upstream Ultralytics behavior: use OpenCV minAreaRect to obtain the minimum-area OBB.
        (cx, cy), (w, h), angle = cv2.minAreaRect(points)
        angle = angle / 180.0 * np.pi
        if w < h:
            w, h = h, w
            angle += np.pi / 2
        angle = _normalize_angle(angle)
    except Exception:
        # Fallback for environments without OpenCV: infer from polygon edges.
        center = points.mean(axis=0, keepdims=True)
        order = np.argsort(np.arctan2(points[:, 1] - center[0, 1], points[:, 0] - center[0, 0]))
        points = points[order]
        edges = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(edges, axis=1)
        long_edge = int(lengths.argmax())
        short_edge = (long_edge + 1) % 4
        cx, cy = points.mean(axis=0)
        w = lengths[long_edge]
        h = lengths[short_edge]
        angle = _normalize_angle(np.arctan2(edges[long_edge, 1], edges[long_edge, 0]))
    return np.array([cx, cy, w, h, angle], dtype=np.float32)


class YOLO26Dataset(Dataset):
    def __init__(
        self,
        image_root: Union[str, os.PathLike, list[str], tuple[str, ...]],
        task: str,
        imgsz: Union[int, tuple[int, int], list[int]],
        augment: bool,
        names: dict[int, str],
        augment_cfg: Optional[AugmentConfig] = None,
        obb_format: str = "xywhr",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
    ):
        if task not in SUPPORTED_TASKS:
            raise NotImplementedError(f"Only {sorted(SUPPORTED_TASKS)} are supported, but got '{task}'.")
        self.task = task
        self.imgsz = normalize_imgsz(imgsz)
        self.augment = augment
        self.names = names
        self.augment_cfg = augment_cfg or AugmentConfig()
        self.obb_format = obb_format.lower()
        self.rect = rect
        self.batch_size = max(int(batch_size), 1)
        self.stride = max(int(stride), 1)
        self.pad = float(pad)
        if self.task == "obb" and self.obb_format not in {"xywhr", "xyxyr"}:
            raise ValueError(f"Unsupported obb_format='{obb_format}', expected one of ('xywhr', 'xyxyr').")
        self.images = list_images(image_root)
        self.labels = [self._label_path(path) for path in self.images]
        self.ni = len(self.images)
        self.batch = np.floor(np.arange(self.ni) / self.batch_size).astype(int) if self.ni else np.zeros((0,), dtype=int)
        self.batch_shapes: Optional[np.ndarray] = None
        if self.rect and self.ni:
            self.set_rectangle()

    @staticmethod
    def _label_path(image_path: Path) -> Path:
        parts = list(image_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")

    def __len__(self) -> int:
        return len(self.images)

    def set_rectangle(self) -> None:
        """Sort images by aspect ratio and precompute per-batch target shapes for rectangular batching."""
        shapes = []
        for p in self.images:
            with Image.open(p) as im:
                w, h = im.size
            shapes.append((h, w))
        shapes = np.asarray(shapes, dtype=np.float32)
        ar = shapes[:, 0] / shapes[:, 1]  # h/w
        irect = ar.argsort()
        self.images = [self.images[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)
        nb = bi[-1] + 1 if self.ni else 0
        batch_shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            if len(ari) == 0:
                continue
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                batch_shapes[i] = [maxi, 1]
            elif mini > 1:
                batch_shapes[i] = [1, 1 / mini]
        imgsz = np.array(self.imgsz, dtype=np.float32)
        self.batch_shapes = np.ceil(np.array(batch_shapes) * imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi

    def _load_labels(self, label_path: Path, image_hw: Optional[tuple[int, int]] = None):
        cols = 5 if self.task == "detect" else 6
        if not label_path.exists():
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0, cols - 1), dtype=np.float32)
        rows = [[float(x) for x in line.strip().split()] for line in label_path.read_text().splitlines() if line.strip()]
        if not rows:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0, cols - 1), dtype=np.float32)
        arr = np.asarray(rows, dtype=np.float32)
        cls = arr[:, :1]
        if self.task == "detect":
            boxes = arr[:, 1:5]
        else:
            if arr.shape[1] == 6:
                boxes = arr[:, 1:6]
                if self.obb_format == "xyxyr":
                    boxes = xyxyr_to_xywhr(boxes)
                else:
                    boxes = boxes.copy()
                    boxes[:, 4] = np.array([_normalize_angle(a) for a in boxes[:, 4]], dtype=np.float32)
            elif arr.shape[1] == 9:
                polygons = arr[:, 1:9].reshape(-1, 4, 2).copy()
                if image_hw is not None:
                    h, w = image_hw
                    polygons[..., 0] *= w
                    polygons[..., 1] *= h
                boxes = np.stack([polygon_to_xywhr(poly) for poly in polygons], axis=0)
                if image_hw is not None:
                    h, w = image_hw
                    boxes[:, [0, 2]] /= w
                    boxes[:, [1, 3]] /= h
            else:
                raise ValueError(f"Unsupported OBB label format in {label_path}: expected 6 or 9 columns, got {arr.shape[1]}")
        return cls, boxes

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        original_shape = image.size[1], image.size[0]
        cls, boxes = self._load_labels(self.labels[index], image_hw=original_shape)
        if self.augment:
            image = augment_hsv(image, self.augment_cfg)
        target_shape = tuple(self.batch_shapes[self.batch[index]].tolist()) if self.rect and self.batch_shapes is not None else self.imgsz
        image, ratio, pad = letterbox(
            image,
            target_shape,
            auto=False,
            scale_fill=False,
            scaleup=self.augment,
            center=True,
            stride=self.stride,
            padding_value=114,
        )
        target_h, target_w = normalize_imgsz(target_shape)
        if boxes.shape[0]:
            boxes = boxes.copy()
            boxes[:, 0] = boxes[:, 0] * original_shape[1] * ratio + pad[0]
            boxes[:, 1] = boxes[:, 1] * original_shape[0] * ratio + pad[1]
            boxes[:, 2] = boxes[:, 2] * original_shape[1] * ratio
            boxes[:, 3] = boxes[:, 3] * original_shape[0] * ratio
            boxes[:, :4] /= np.array([target_w, target_h, target_w, target_h], dtype=np.float32)
        if self.augment and random.random() < self.augment_cfg.fliplr:
            image = ImageOps.mirror(image)
            if boxes.shape[0]:
                boxes[:, 0] = 1.0 - boxes[:, 0]
                if self.task == "obb":
                    boxes[:, 4] *= -1
        if self.augment and random.random() < self.augment_cfg.flipud:
            image = ImageOps.flip(image)
            if boxes.shape[0]:
                boxes[:, 1] = 1.0 - boxes[:, 1]
                if self.task == "obb":
                    boxes[:, 4] *= -1
        image = np.asarray(image, dtype=np.uint8)[..., ::-1].transpose(2, 0, 1)  # PIL RGB -> BGR CHW
        if random.uniform(0, 1) > self.augment_cfg.bgr and image.shape[0] == 3:
            image = image[::-1]  # BGR->RGB (default when bgr=0.0), matching Ultralytics Format behavior
        return {
            "img": torch.from_numpy(np.ascontiguousarray(image)),
            "cls": torch.from_numpy(cls),
            "bboxes": torch.from_numpy(boxes),
            "batch_idx": torch.zeros((len(cls),), dtype=torch.int64),
            "im_file": str(image_path),
            "ori_shape": original_shape,
            "ratio_pad": (ratio, pad),
        }

    @staticmethod
    def collate_fn(batch):
        bbox_cols = batch[0]["bboxes"].shape[1] if batch else 4
        return {
            "img": torch.stack([x["img"] for x in batch], 0),
            "cls": torch.cat([x["cls"] for x in batch], 0) if batch else torch.zeros((0, 1)),
            "bboxes": torch.cat([x["bboxes"] for x in batch], 0) if batch else torch.zeros((0, bbox_cols)),
            "batch_idx": torch.cat([torch.full((len(x["cls"]),), i, dtype=torch.int64) for i, x in enumerate(batch)], 0) if batch else torch.zeros((0,), dtype=torch.int64),
            "im_file": [x["im_file"] for x in batch],
            "ori_shape": [x["ori_shape"] for x in batch],
            "ratio_pad": [x["ratio_pad"] for x in batch],
        }


def build_dataset(data: dict[str, Any], split: str, task: str, imgsz: Union[int, tuple[int, int], list[int]], augment: bool):
    return YOLO26Dataset(data[split], task=task, imgsz=imgsz, augment=augment, names=data["names"], obb_format=data.get("obb_format", "xywhr"))


def build_dataloader(dataset: YOLO26Dataset, batch_size: int, workers: int, shuffle: bool):
    batch_size = min(batch_size, max(len(dataset), 1))
    if dataset.rect and shuffle and dataset.batch_shapes is not None and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True, collate_fn=dataset.collate_fn)


def create_train_val_dataloaders(
    data_yaml: Union[str, os.PathLike],
    task: str,
    imgsz: Union[int, tuple[int, int], list[int]],
    batch_size: int,
    workers: int,
    eval_split: str = "val",
    stride: int = 32,
    train_augment: bool = True,
):
    data = load_data_config(data_yaml)
    train_dataset = YOLO26Dataset(
        data["train"],
        task=task,
        imgsz=imgsz,
        augment=train_augment,
        names=data["names"],
        obb_format=data.get("obb_format", "xywhr"),
        rect=False,
        batch_size=batch_size,
        stride=stride,
        pad=0.0,
    )
    split = eval_split if eval_split in data and data.get(eval_split) else "val"
    eval_dataset = YOLO26Dataset(
        data[split],
        task=task,
        imgsz=imgsz,
        augment=False,
        names=data["names"],
        obb_format=data.get("obb_format", "xywhr"),
        rect=True,
        batch_size=batch_size,
        stride=stride,
        pad=0.5,
    )
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, workers=workers, shuffle=True)
    eval_loader = build_dataloader(eval_dataset, batch_size=batch_size, workers=workers, shuffle=False)
    return data, train_loader, eval_loader
