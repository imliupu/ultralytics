from __future__ import annotations

from typing import Any


def build_criterion(model: Any, task: str | None = None, args_overrides: dict | None = None):
    """Build the original Ultralytics criterion for YOLO26 detect/obb models."""
    wrapper = getattr(model, "model", model)
    if getattr(wrapper, "args", None) is None:
        if hasattr(model, "set_args"):
            model.set_args(task=task or getattr(model, "task", "detect"), **(args_overrides or {}))
        else:
            raise ValueError("Model must expose original Ultralytics args before building the criterion.")
    return wrapper.init_criterion()
