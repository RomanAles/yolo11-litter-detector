"""Train-only small-object crops and paired fine-tuning experiments."""
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import yaml

from finalize_v2 import error_gallery
from litter_detector.artifacts import write_json, write_text
from litter_detector.detector import ROOT, read_image, write_image
from litter_detector.metrics import count_at
from quality_evaluation import OUTPUT, PROFILES, validation_predictions
from quality_training import TABLE, evaluate_experiment, format_row
from v2_pipeline import run


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_boxes(row):
    for label in row['labels']:
        cls, x, y, width, height = map(float, label.split())
        yield int(cls), x, y, width, height


def crop_training_image(row, output):
    source = Path(row['image_path'])
    image = read_image(source)
    height, width = image.shape[:2]
    boxes = list(normalized_boxes(row))
    small = [box for box in boxes if box[3] * box[4] * width * height < 1024
             and min(box[3] * width, box[4] * height) >= 5]
    size = min(512, width, height)
    if not small or size < 128 or (width == size and height == size):
        return None
    _, center_x, center_y, _, _ = max(small, key=lambda box: box[3] * box[4])
    x = min(width - size, max(0, round(center_x * width - size / 2)))
    y = min(height - size, max(0, round(center_y * height - size / 2)))
    labels = []
    for cls, cx, cy, bw, bh in boxes:
        x1 = max(x, (cx - bw / 2) * width)
        x2 = min(x + size, (cx + bw / 2) * width)
        y1 = max(y, (cy - bh / 2) * height)
        y2 = min(y + size, (cy + bh / 2) * height)
        if min(x2 - x1, y2 - y1) < 2:
            continue
        coordinates = ((x1 + x2 - 2 * x) / (2 * size), (y1 + y2 - 2 * y) / (2 * size),
                       (x2 - x1) / size, (y2 - y1) / size)
        labels.append(f'{cls} ' + ' '.join(f'{value:.8f}' for value in coordinates))
    image_path = output / 'images' / 'train' / source.name
    label_path = output / 'labels' / 'train' / source.with_suffix('.txt').name
    write_image(image_path, image[y:y + size, x:x + size])
    write_text(label_path, '\n'.join(labels) + '\n')
    return {'source': str(source), 'group': row['group'], 'image': str(image_path),
            'crop_xyxy': [x, y, x + size, y + size], 'sha256': sha256(image_path),
            'labels': str(label_path), 'label_sha256': sha256(label_path)}


def small_crop_dataset():
    base = ROOT / 'data' / 'v2' / 'land'
    output = ROOT / 'data' / 'quality' / 'land-small-crops'
    source_manifest = base / 'manifest.json'
    summary_path = output / 'manifest.json'
    base_hashes = {name: sha256(base / name) for name in ('manifest.json', 'train.txt', 'val.txt')}
    if summary_path.exists():
        summary = load_json(summary_path)
        if summary['base_sha256'] != base_hashes:
            raise ValueError('Land training data changed after crop preparation')
        for item in summary['crops']:
            if sha256(item['image']) != item['sha256'] or sha256(item['labels']) != item['label_sha256']:
                raise ValueError(f"Modified training crop: {item['image']}")
        return output / 'data.yaml', summary
    source = load_json(source_manifest)['images']
    sizes = {record['image']: (record['width'], record['height']) for record in
             load_json(ROOT / 'outputs' / 'v2' / 'prepared_file_audit.json')['profiles']['land']['records']}
    candidates = []
    for row in source:
        if row['split'] != 'train' or not row['labels']:
            continue
        width, height = sizes[row['image_path']]
        if any(bw * bh * width * height < 1024 and min(bw * width, bh * height) >= 5
               for _, _, _, bw, bh in normalized_boxes(row)):
            candidates.append(row)
    random.Random(42).shuffle(candidates)
    crops = []
    for row in candidates:
        crop = crop_training_image(row, output)
        if crop is not None:
            crops.append(crop)
        if len(crops) == 180:
            break
    if not crops:
        raise ValueError('No train-only small-object crops could be generated')
    lookup = {row['source']: row['image'] for row in crops}
    train = (base / 'train.txt').read_text(encoding='utf-8').splitlines()
    assert set(lookup).issubset(train)
    variant = [lookup.get(path, path) for path in train]
    assert len(variant) == len(train) == len(set(variant))
    write_text(output / 'train.txt', '\n'.join(variant) + '\n')
    write_text(output / 'val.txt', (base / 'val.txt').read_text(encoding='utf-8'))
    config = {'path': str(output), 'train': 'train.txt', 'val': 'val.txt', 'names': {0: 'trash'}}
    write_text(output / 'data.yaml', yaml.safe_dump(config))
    summary = {'base_sha256': base_hashes, 'train_images': len(variant), 'crops': crops,
               'selection': 'seed 42, train only, one crop replaces its original frame'}
    write_json(summary_path, summary)
    return output / 'data.yaml', summary


def diagnose(candidate):
    profile = candidate['profile']
    prediction = validation_predictions(candidate)
    threshold = candidate['operating']['confidence']
    error_gallery(prediction, threshold, OUTPUT / 'validation_errors' / profile)
    records = prediction['records']
    diagnostics = {'all': count_at(records, threshold)}
    if profile == 'underwater':
        brightness = []
        for record in records:
            pixels = read_image(record['file'])
            brightness.append(float(pixels[::8, ::8].mean()))
        quartile = np.quantile(brightness, .25)
        diagnostics['darkest_quartile'] = count_at([record for record, value in zip(records, brightness)
                                                    if value <= quartile], threshold)
        diagnostics['other_frames'] = count_at([record for record, value in zip(records, brightness)
                                                if value > quartile], threshold)
    return diagnostics


def tune_training():
    yolo11 = load_json(OUTPUT / '02_yolo11.json')['selected']
    yolo26 = load_json(OUTPUT / '03_yolo26.json')['profiles']
    outcomes = {}
    for profile in PROFILES:
        candidates = [yolo11[profile], yolo26[profile]['full']]
        start = max(candidates, key=lambda candidate: candidate['ap']['map50_95'])
        diagnosis = diagnose(start)
        options = dict(arch=start['arch'], weights=start['weights'], imgsz=start['imgsz'], batch=8,
                       epochs=12, patience=6, lr0=.0002)
        crop_count = 0
        if profile == 'land':
            data, manifest = small_crop_dataset()
            crop_count = len(manifest['crops'])
            variation = {'data': str(data)}
        else:
            variation = {'augment': json.dumps({'hsv_v': .55})}
        control = run(profile, 'quality_finetune_control', python=sys.executable, **options)
        variant = run(profile, 'quality_finetune_variant', python=sys.executable, **options, **variation)
        outcomes[profile] = {'initial': start, 'diagnosis': diagnosis, 'land_train_crops': crop_count,
                             'control': evaluate_experiment(control), 'variant': evaluate_experiment(variant)}
        write_json(OUTPUT / f'04_tuning_{profile}.json', outcomes[profile])
    write_json(OUTPUT / '04_tuning.json', outcomes)
    lines = ['# Шаг 4. Данные и аугментации', '',
             f"В наземной серии {outcomes['land']['land_train_crops']} кадров train заменены фрагментами вокруг мелких объектов. "
             'Для подводной серии изменён только `hsv_v` до 0,55. Контроль и вариант стартуют с одних весов, 12 эпох максимум.', '', *TABLE]
    for item in outcomes.values():
        lines.extend(format_row(item[key]) for key in ('initial', 'control', 'variant'))
    lines += ['', 'Галерея ошибок validation и показатели по тёмным подводным кадрам сохранены рядом с отчётом.', '']
    write_text(OUTPUT / '04_tuning.md', '\n'.join(lines))


if __name__ == '__main__':
    tune_training()
