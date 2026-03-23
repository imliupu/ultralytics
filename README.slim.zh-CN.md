# YOLO26 / YOLO26-OBB 精简说明

这个仓库已经按“只保留 YOLO26 检测 + YOLO26-OBB”方向做了一层代码瘦身，目标是减少无关任务入口，方便你继续做结构改造和 QAT。

## 当前保留内容

- `YOLO("yolo26*.pt|yaml")` 的 **detect** 路径
- `YOLO("yolo26*-obb.pt|yaml")` 的 **obb** 路径
- 对应的 `trainer / validator / predictor / model` 映射

## 当前主动移除/禁用内容

- `segment / classify / pose`
- `YOLOWorld / YOLOE / NAS / SAM / FastSAM / RTDETR`
- CLI 默认任务映射中的非 `detect / obb` 项

## 建议的 QAT 切入点

如果你要继续做 QAT，建议优先从下面几层切入：

1. `ultralytics/models/yolo/dataset.py`：现在已经把 YOLO26/YOLO26-OBB 的 dataset + dataloader 抽到一起了
2. `ultralytics/nn/tasks.py` 里的 `DetectionModel / OBBModel`
3. `ultralytics/nn/modules/` 中 backbone、neck、head 模块
4. `ultralytics/models/yolo/detect` 与 `ultralytics/models/yolo/obb` 的 trainer / predictor / validator

## 最小使用方式

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26-obb.yaml", task="obb")
```

```bash
yolo obb train model=yolo26n-obb.pt data=dota8.yaml
```
