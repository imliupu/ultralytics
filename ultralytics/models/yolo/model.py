# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, OBBModel

SUPPORTED_YOLO26_TASKS = frozenset({"detect", "obb"})
UNSUPPORTED_MODEL_TOKENS = ("-world", "yoloe", "rtdetr", "fastsam", "sam")


class YOLO(Model):
    """Slim YOLO interface that only keeps YOLO26 detect and YOLO26-OBB code paths."""

    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False):
        """Initialize a YOLO26 model with detect/obb-only support."""
        path = Path(model if isinstance(model, (str, Path)) else "")
        if any(token in path.stem.lower() for token in UNSUPPORTED_MODEL_TOKENS):
            raise NotImplementedError(
                "This slimmed repository only keeps YOLO26 detect and YOLO26-OBB support. "
                f"Unsupported model: {path.name or model}."
            )

        super().__init__(model=model, task=task, verbose=verbose)

        if self.task not in SUPPORTED_YOLO26_TASKS:
            raise NotImplementedError(
                "This slimmed repository only supports the 'detect' and 'obb' tasks, "
                f"but got '{self.task}'."
            )

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map the remaining tasks to their model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """Removed in the slim YOLO26/YOLO26-OBB-only build."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("YOLOWorld has been removed from this slim YOLO26-only repository.")


class YOLOE(Model):
    """Removed in the slim YOLO26/YOLO26-OBB-only build."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("YOLOE has been removed from this slim YOLO26-only repository.")
