from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel, OBBModel
from ultralytics.utils import DEFAULT_CFG

CONFIG_DIR = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "26"
MODEL_ALIASES = {
    "yolo26": CONFIG_DIR / "yolo26.yaml",
    "yolo26-obb": CONFIG_DIR / "yolo26-obb.yaml",
    "yolo26obb": CONFIG_DIR / "yolo26-obb.yaml",
}
MODEL_BY_TASK = {"detect": DetectionModel, "obb": OBBModel}


@dataclass
class ModelBuildConfig:
    task: str
    nc: int
    model: str = "yolo26"
    scale: str = "n"
    ch: int = 3


class RecYOLO26Model(nn.Module):
    """Thin wrapper around the original Ultralytics YOLO26/YOLO26-OBB PyTorch models."""

    def __init__(self, core_model: nn.Module, task: str, cfg_path: Path, ch: int = 3):
        super().__init__()
        self.model = core_model
        self.task = task
        self.cfg_path = Path(cfg_path)
        self.ch = ch
        self.nc = getattr(core_model, "nc", core_model.yaml.get("nc", 80))
        self.names = getattr(core_model, "names", {i: str(i) for i in range(self.nc)})
        self.stride = core_model.stride
        self.args = getattr(core_model, "args", None)
        if self.args is None:
            self.set_args(task=task, model=str(self.cfg_path), imgsz=640)

    @staticmethod
    def resolve_model_cfg(model: str, task: str) -> Path:
        key = model.lower().replace("_", "-")
        if key in MODEL_ALIASES:
            return MODEL_ALIASES[key]
        if Path(model).suffix in {".yaml", ".yml"}:
            return Path(model)
        return CONFIG_DIR / ("yolo26-obb.yaml" if task == "obb" or key.endswith("-obb") else "yolo26.yaml")

    @classmethod
    def build(
        cls,
        task: str,
        nc: int,
        model: str = "yolo26",
        scale: str = "n",
        ch: int = 3,
        verbose: bool = False,
        args_overrides: dict[str, Any] | None = None,
    ) -> "RecYOLO26Model":
        cfg_path = cls.resolve_model_cfg(model, task)
        model_cls = MODEL_BY_TASK[task]
        core_model = model_cls(cfg=str(cfg_path), ch=ch, nc=nc, verbose=verbose)
        instance = cls(core_model=core_model, task=task, cfg_path=cfg_path, ch=ch)
        instance.set_args(task=task, model=str(cfg_path), imgsz=640, **(args_overrides or {}))
        return instance

    def set_args(self, **overrides: Any):
        """Attach an Ultralytics config namespace so the original loss/validator stack works unchanged."""
        self.args = get_cfg(DEFAULT_CFG, overrides=overrides)
        self.model.args = self.args
        return self.args

    def forward(self, images: torch.Tensor):
        return self.model(images)

    def loss(self, batch: dict[str, torch.Tensor], preds=None):
        return self.model.loss(batch, preds)

    def warmup(self, imgsz: tuple[int, ...]):
        return self.model.warmup(imgsz=imgsz)

    def load(self, path: str | Path, strict: bool = True) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()
        self.model.load_state_dict(state_dict, strict=strict)
        self.names = checkpoint.get("names", self.names)
        self.model.names = self.names

    def save_checkpoint(self, path: str | Path, **extra: Any) -> None:
        torch.save({"model": self.model.state_dict(), "task": self.task, "names": self.names, **extra}, path)

    def save(self, path: str | Path, **extra: Any) -> None:
        self.save_checkpoint(path, **extra)
