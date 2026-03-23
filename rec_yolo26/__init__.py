"""Standalone simplified YOLO26 / YOLO26-OBB training project."""

from importlib import import_module

_EXPORTS = {
    "YOLO26Dataset": ("rec_yolo26.dataset", "YOLO26Dataset"),
    "build_dataloader": ("rec_yolo26.dataset", "build_dataloader"),
    "build_dataset": ("rec_yolo26.dataset", "build_dataset"),
    "create_train_val_dataloaders": ("rec_yolo26.dataset", "create_train_val_dataloaders"),
    "load_data_config": ("rec_yolo26.dataset", "load_data_config"),
    "build_criterion": ("rec_yolo26.loss", "build_criterion"),
    "build_metric_evaluator": ("rec_yolo26.metrics", "build_metric_evaluator"),
    "evaluate_model": ("rec_yolo26.metrics", "evaluate_model"),
    "RecYOLO26Model": ("rec_yolo26.model", "RecYOLO26Model"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    return getattr(import_module(module_name), attr_name)
