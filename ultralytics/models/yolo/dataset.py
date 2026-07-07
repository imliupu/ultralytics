# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import dataloader, distributed

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import TORCH_2_0

SUPPORTED_TASKS = frozenset({"detect", "obb"})


class YOLO26Dataset(YOLODataset):
    """Slim dataset wrapper for YOLO26 detect and YOLO26-OBB."""

    def __init__(self, *args: Any, task: str = "detect", mode: str = "train", **kwargs: Any) -> None:
        if task not in SUPPORTED_TASKS:
            raise NotImplementedError(f"YOLO26Dataset only supports {sorted(SUPPORTED_TASKS)}, but got '{task}'.")
        self.mode = mode
        super().__init__(*args, task=task, **kwargs)


class InfiniteDataLoader(dataloader.DataLoader):
    """DataLoader that reuses workers for infinite iteration."""

    def __init__(self, *args: Any, **kwargs: Any):
        if not TORCH_2_0:
            kwargs.pop("prefetch_factor", None)
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()
        except Exception:
            pass

    def reset(self):
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """Sampler that repeats forever for infinite iteration."""

    def __init__(self, sampler: Any):
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        while True:
            yield from iter(self.sampler)


class ContiguousDistributedSampler(torch.utils.data.Sampler):
    """Distributed sampler that assigns contiguous batch-aligned chunks to each rank."""

    def __init__(
        self,
        dataset,
        num_replicas: int | None = None,
        batch_size: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if batch_size is None:
            batch_size = getattr(dataset, "batch_size", 1)

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.total_size = len(dataset)
        self.batch_size = 1 if batch_size >= self.total_size else batch_size
        self.num_batches = math.ceil(self.total_size / self.batch_size)

    def _get_rank_indices(self) -> tuple[int, int]:
        batches_per_rank_base = self.num_batches // self.num_replicas
        remainder = self.num_batches % self.num_replicas
        batches_for_this_rank = batches_per_rank_base + (1 if self.rank < remainder else 0)
        start_batch = self.rank * batches_per_rank_base + min(self.rank, remainder)
        end_batch = start_batch + batches_for_this_rank
        start_idx = start_batch * self.batch_size
        end_idx = min(end_batch * self.batch_size, self.total_size)
        return start_idx, end_idx

    def __iter__(self) -> Iterator:
        start_idx, end_idx = self._get_rank_indices()
        indices = list(range(start_idx, end_idx))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]
        return iter(indices)

    def __len__(self) -> int:
        start_idx, end_idx = self._get_rank_indices()
        return end_idx - start_idx

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def seed_worker(worker_id: int) -> None:
    """Set dataloader worker seed for reproducibility across worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
):
    """Build the slim YOLO26/YOLO26-OBB dataset from trainer/validator config."""
    if cfg.task not in SUPPORTED_TASKS:
        raise NotImplementedError(f"Slim YOLO26 dataset only supports {sorted(SUPPORTED_TASKS)}, but got '{cfg.task}'.")

    return YOLO26Dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        mode=mode,
    )


def build_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> InfiniteDataLoader:
    """Create and return the slim YOLO26 dataloader."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()
    nw = min(os.cpu_count() // max(nd, 1), workers)
    sampler = (
        None
        if rank == -1
        else distributed.DistributedSampler(dataset, shuffle=shuffle)
        if shuffle
        else ContiguousDistributedSampler(dataset, batch_size=batch)
    )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=4 if nw > 0 else None,
        pin_memory=nd > 0 and pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
    )
