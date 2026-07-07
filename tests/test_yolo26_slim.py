# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest

from ultralytics import YOLO


@pytest.mark.parametrize(
    ('model_cfg', 'task'),
    (
        ('ultralytics/cfg/models/26/yolo26.yaml', 'detect'),
        ('ultralytics/cfg/models/26/yolo26-obb.yaml', 'obb'),
    ),
)
def test_yolo26_slim_supported_tasks(model_cfg, task):
    model = YOLO(model_cfg, task=task)
    assert model.task == task
    assert set(model.task_map) == {'detect', 'obb'}


def test_yolo26_slim_rejects_removed_model_families():
    with pytest.raises(NotImplementedError, match='only keeps YOLO26 detect and YOLO26-OBB support'):
        YOLO('yoloe-26.yaml')


def test_yolo26_slim_rejects_removed_tasks():
    with pytest.raises(NotImplementedError, match="only supports the 'detect' and 'obb' tasks"):
        YOLO('ultralytics/cfg/models/26/yolo26-seg.yaml')
