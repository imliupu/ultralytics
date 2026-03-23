# rec_yolo26

一个**完全对齐原始 Ultralytics YOLO26 / YOLO26-OBB 训练栈**的轻量封装层。

## 对齐原则

- `model.py`：直接包装原始 `DetectionModel / OBBModel`
- `loss.py`：直接返回原始 `init_criterion()`，即 `E2ELoss / v8DetectionLoss / v8OBBLoss`
- `dataset.py`：直接复用原始 `YOLO26Dataset / build_yolo_dataset / build_dataloader`
- `metrics.py`：直接复用原始 `DetectionValidator / OBBValidator`
- `train_val.py` / `test.py`：直接调用原始 `YOLO(...).train()` / `YOLO(...).val()`

换句话说，`rec_yolo26` 不再维护一套“看起来像 YOLO26”的独立简化实现，而是把 detect / obb 的训练、验证、loss、NMS、metrics 全部对齐到仓库里的原始实现。

## 使用示例

```bash
python -m rec_yolo26.train_val --task detect --model yolo26 --data /path/to/your_detect.yaml
python -m rec_yolo26.train_val --task obb --model yolo26-obb --data /path/to/your_obb.yaml
```

```bash
python -m rec_yolo26.test --task detect --model yolo26 --data /path/to/your_detect.yaml --weights runs/rec_yolo26/weights/best.pt
python -m rec_yolo26.test --task obb --model yolo26-obb --data /path/to/your_obb.yaml --weights runs/rec_yolo26/weights/best.pt
```
