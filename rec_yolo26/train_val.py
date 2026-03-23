from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from rec_yolo26.dataset import create_train_val_dataloaders
from rec_yolo26.metrics import EvalConfig, evaluate_model
from rec_yolo26.model import RecYOLO26Model


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=device.type == "cuda")
    batch["img"] = batch["img"].float() / 255
    return batch


def train_one_epoch(model, dataloader, optimizer, device, epoch: int, epochs: int):
    model.train()
    running_loss = 0.0
    progress = tqdm(dataloader, desc=f"train {epoch + 1}/{epochs}")
    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        total_loss, _ = model.loss(batch)
        total_loss = total_loss.sum()
        total_loss.backward()
        optimizer.step()
        if model.criterion is not None and hasattr(model.criterion, "update"):
            model.criterion.update()
        running_loss += float(total_loss.detach().item())
        progress.set_postfix(loss=f"{running_loss / step:.4f}")
    return running_loss / max(len(dataloader), 1)


def main(args):
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = RecYOLO26Model.build(task=args.task, nc=1, model=args.model, verbose=not args.quiet)
    data, train_loader, val_loader = create_train_val_dataloaders(args.data, args.task, args.imgsz, args.batch, args.workers, stride=int(max(model.stride.max().item(), 32)))
    model = RecYOLO26Model.build(task=args.task, nc=data["nc"], model=args.model, verbose=not args.quiet, args_overrides={"epochs": args.epochs})
    if args.weights:
        model.load(args.weights, strict=False)
    model.to(device)
    model.names = data["names"]
    optimizer = optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr0 * 0.01)
    best_fitness, history = float("-inf"), []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_metrics = evaluate_model(model, val_loader, device, args.task, data["names"], EvalConfig(conf=args.conf, iou=args.iou, max_det=args.max_det))
        scheduler.step()
        metrics_row = {"epoch": epoch + 1, "train_loss": train_loss, **{k: float(v) for k, v in val_metrics.items()}}
        history.append(metrics_row)
        print(json.dumps(metrics_row, ensure_ascii=False))
        fitness = metrics_row.get("metrics/mAP50-95(B)", 0.0)
        model.save(save_dir / "last.pt", epoch=epoch + 1, optimizer=optimizer.state_dict(), metrics=metrics_row)
        if fitness >= best_fitness:
            best_fitness = fitness
            model.save(save_dir / "best.pt", epoch=epoch + 1, optimizer=optimizer.state_dict(), metrics=metrics_row)
    (save_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone self-contained YOLO26 / YOLO26-OBB train+val entrypoint.")
    parser.add_argument("--task", choices=["detect", "obb"], required=True)
    parser.add_argument("--model", default="yolo26")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--save_dir", default="runs/rec_yolo26")
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
