# rec_yolo26

一个独立的、面向二次开发/QAT 的 YOLO26 / YOLO26-OBB 简化项目骨架。

## 文件说明

- `dataset.py`: 数据集加载、增强、dataloader 构造
- `model.py`: 模型静态构造、forward 输出 raw head、postprocess 独立
- `loss.py`: detect / obb loss
- `metrics.py`: detect / obb 指标与验证逻辑
- `train_val.py`: 自己写的训练/验证循环
- `test.py`: 独立测试脚本

## 使用示例

```bash
python -m rec_yolo26.train_val --task detect --model yolo26 --data /path/to/your_detect.yaml
python -m rec_yolo26.train_val --task obb --model yolo26-obb --data /path/to/your_obb.yaml
```


## 本地快速自测

```bash
python -m rec_yolo26.tools.make_dummy_data --output rec_yolo26/demo_data
python -m rec_yolo26.train_val --task detect --model yolo26 --data rec_yolo26/demo_data/detect.yaml --epochs 1 --batch 2 --workers 0 --imgsz 128
python -m rec_yolo26.train_val --task obb --model yolo26-obb --data rec_yolo26/demo_data/obb.yaml --epochs 1 --batch 2 --workers 0 --imgsz 128
```
