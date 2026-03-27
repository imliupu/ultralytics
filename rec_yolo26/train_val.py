from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from rec_yolo26.dataset import create_train_val_dataloaders
from rec_yolo26.metrics import EvalConfig, evaluate_model
from rec_yolo26.model import RecYOLO26Model
from rec_yolo26.optimizer import build_optimizer


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                v.copy_(src)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_imgsz(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError(f"--imgsz expects one int or two ints, got: {values}")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=device.type == "cuda")
    batch["img"] = batch["img"].float() / 255
    return batch


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch: int,
    epochs: int,
    grad_clip: float = 10.0,
    ema: ModelEMA | None = None,
    base_accumulate: int = 1,
    warmup_steps: int = -1,
    warmup_bias_lr: float = 0.1,
    warmup_momentum: float = 0.8,
    final_momentum: float = 0.9,
):
    model.train()
    running_loss = 0.0
    nb = max(len(dataloader), 1)
    last_opt_step = -1
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(dataloader, desc=f"train {epoch + 1}/{epochs}")
    for step, batch in enumerate(progress, start=1):
        ni = (step - 1) + nb * epoch
        accumulate = base_accumulate
        if warmup_steps >= 0 and ni <= warmup_steps:
            xi = [0, warmup_steps]
            accumulate = max(1, int(np.interp(ni, xi, [1, base_accumulate]).round()))
            for pg in optimizer.param_groups:
                pg["lr"] = float(np.interp(ni, xi, [warmup_bias_lr if pg.get("param_group") == "bias" else 0.0, pg["initial_lr"]]))
                if "momentum" in pg:
                    pg["momentum"] = float(np.interp(ni, xi, [warmup_momentum, final_momentum]))
        batch = move_batch_to_device(batch, device)
        total_loss, _ = model.loss(batch)
        total_loss = total_loss.sum()
        total_loss.backward()
        if ni - last_opt_step >= accumulate:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            last_opt_step = ni
        if model.criterion is not None and hasattr(model.criterion, "update"):
            model.criterion.update()
        running_loss += float(total_loss.detach().item())
        progress.set_postfix(loss=f"{running_loss / step:.4f}")
    return running_loss / max(len(dataloader), 1)


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    imgsz = parse_imgsz(args.imgsz)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = RecYOLO26Model.build(task=args.task, nc=1, model=args.model, verbose=not args.quiet)
    data, train_loader, val_loader = create_train_val_dataloaders(
        args.data,
        args.task,
        imgsz,
        args.batch,
        args.workers,
        stride=int(max(model.stride.max().item(), 32)),
        train_augment=not args.no_aug,
    )
    model = RecYOLO26Model.build(
        task=args.task,
        nc=data["nc"],
        model=args.model,
        verbose=not args.quiet,
        args_overrides={"epochs": args.epochs, "end2end": args.end2end},
    )
    if args.weights:
        model.load(args.weights)
    model.to(device)
    model.names = data["names"]
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
    if ema is not None:
        ema.ema.names = data["names"]
    nbs = max(args.nbs, 1)
    accumulate = max(round(nbs / max(args.batch, 1)), 1)
    scaled_weight_decay = args.weight_decay * args.batch * accumulate / nbs
    iterations = math.ceil(len(train_loader.dataset) / max(args.batch, nbs)) * args.epochs
    optimizer = build_optimizer(
        model=model,
        name=args.optimizer,
        lr=args.lr0,
        momentum=args.momentum,
        decay=scaled_weight_decay,
        iterations=max(iterations, 1),
    )
    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr0 * 0.01)
    best_fitness, history = float("-inf"), []
    nw = max(round(args.warmup_epochs * len(train_loader)), 100) if args.warmup_epochs > 0 else -1
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.epochs,
            grad_clip=args.grad_clip,
            ema=ema,
            base_accumulate=accumulate,
            warmup_steps=nw,
            warmup_bias_lr=args.warmup_bias_lr,
            warmup_momentum=args.warmup_momentum,
            final_momentum=args.momentum,
        )
        eval_model = ema.ema if ema is not None else model
        val_metrics = evaluate_model(eval_model, val_loader, device, args.task, data["names"], EvalConfig(conf=args.conf, iou=args.iou, max_det=args.max_det))
        scheduler.step()
        metrics_row = {"epoch": epoch + 1, "train_loss": train_loss, **{k: float(v) for k, v in val_metrics.items()}}
        history.append(metrics_row)
        print(json.dumps(metrics_row, ensure_ascii=False))
        fitness = metrics_row.get("metrics/mAP50-95(B)", 0.0)
        eval_model.save(save_dir / "last.pt", epoch=epoch + 1, optimizer=optimizer.state_dict(), metrics=metrics_row)
        if fitness >= best_fitness:
            best_fitness = fitness
            eval_model.save(save_dir / "best.pt", epoch=epoch + 1, optimizer=optimizer.state_dict(), metrics=metrics_row)
    (save_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone self-contained YOLO26 / YOLO26-OBB train+val entrypoint.")
    parser.add_argument("--task", choices=["detect", "obb"], required=True)
    parser.add_argument("--model", default="yolo26")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640], help="Image size as single int or 'h w'.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", default="MuSGD")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size for optimizer hyperparameter scaling.")
    parser.add_argument("--warmup_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_momentum", type=float, default=0.8)
    parser.add_argument("--warmup_bias_lr", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Max gradient norm clipping value. Set <=0 to disable.")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--save_dir", default="runs/rec_yolo26")
    parser.add_argument("--no_aug", action="store_true", help="Disable train-time image augmentation for overfit/debug.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable end-to-end dual-head training/inference.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible overfit/debug runs.")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable EMA model for eval/checkpoint smoothing.")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay factor.")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
