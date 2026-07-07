from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw
import yaml


def rotated_box(cx: float, cy: float, w: float, h: float, angle_deg: float):
    angle = math.radians(angle_deg)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    points = []
    for x, y in corners:
        rx = x * math.cos(angle) - y * math.sin(angle) + cx
        ry = x * math.sin(angle) + y * math.cos(angle) + cy
        points.append((rx, ry))
    return points


def write_detect_sample(image_path: Path, label_path: Path, size: int, cls_id: int):
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = size * 0.2, size * 0.25, size * 0.75, size * 0.8
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    img.save(image_path)
    cx = (x1 + x2) / 2 / size
    cy = (y1 + y2) / 2 / size
    w = (x2 - x1) / size
    h = (y2 - y1) / size
    label_path.write_text(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding='utf-8')


def write_obb_sample(image_path: Path, label_path: Path, size: int, cls_id: int, angle_deg: float):
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    points = rotated_box(size * 0.5, size * 0.5, size * 0.45, size * 0.22, angle_deg)
    draw.polygon(points, outline='blue', width=3)
    img.save(image_path)
    norm = [coord / size for point in points for coord in point]
    label_path.write_text(f"{cls_id} " + ' '.join(f"{x:.6f}" for x in norm) + "\n", encoding='utf-8')


def build_split(root: Path, split: str, task: str, count: int = 4, size: int = 128):
    image_dir = root / task / 'images' / split
    label_dir = root / task / 'labels' / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        image_path = image_dir / f'{split}_{i}.jpg'
        label_path = label_dir / f'{split}_{i}.txt'
        if task == 'detect':
            write_detect_sample(image_path, label_path, size=size, cls_id=i % 2)
        else:
            write_obb_sample(image_path, label_path, size=size, cls_id=i % 2, angle_deg=15 + i * 10)


def write_yaml(root: Path, task: str):
    yaml_path = root / f'{task}.yaml'
    data = {
        'path': str((root / task).resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'class0', 1: 'class1'},
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def main(args):
    root = Path(args.output).resolve()
    for task in ('detect', 'obb'):
        for split in ('train', 'val', 'test'):
            build_split(root=root, split=split, task=task, count=args.count, size=args.size)
        write_yaml(root=root, task=task)
    print(f'Dummy datasets created under {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create tiny local detect/obb demo datasets for rec_yolo26.')
    parser.add_argument('--output', default='rec_yolo26/demo_data')
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--size', type=int, default=128)
    main(parser.parse_args())
