# rec_yolo26

一个**自包含的 YOLO26 / YOLO26-OBB 对齐实现**，不再依赖 `ultralytics` Python 包导入。

## 当前实现原则

- `model.py`：按 `rec_yolo26/configs/*.yaml` 构建本地 `YOLO26` / `YOLO26-OBB` 网络，保留 `end2end=True`、`Detect`、`OBB26` 语义。
- `loss.py`：内置 `TaskAlignedAssigner`、`RotatedTaskAlignedAssigner`、`v8DetectionLoss`、`v8OBBLoss` 和 `E2ELoss`，不再走外部 `ultralytics` 运行时。
- `metrics.py`：本地实现 detect / obb 的 mAP 统计和 IoU / ProbIoU 匹配逻辑。
- `dataset.py`：保留当前 `detect` / `obb` 所需的最小数据读取、letterbox、翻转和标签解析能力，并避免引入 OpenCV / ultralytics 依赖。
- `train_val.py` / `test.py`：回到纯本地训练 / 验证 / 测试流程。

## 使用示例

```bash
python -m rec_yolo26.train_val --task detect --model yolo26 --data /path/to/your_detect.yaml
python -m rec_yolo26.train_val --task obb --model yolo26-obb --data /path/to/your_obb.yaml
```

```bash
python -m rec_yolo26.test --task detect --model yolo26 --data /path/to/your_detect.yaml --weights runs/rec_yolo26/best.pt
python -m rec_yolo26.test --task obb --model yolo26-obb --data /path/to/your_obb.yaml --weights runs/rec_yolo26/best.pt
```
