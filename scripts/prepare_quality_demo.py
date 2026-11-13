"""Export the frozen selection and assemble original media for the local demo."""
from pathlib import Path
from statistics import median
from time import perf_counter
import shutil

import cv2
import numpy as np
import onnx
from ultralytics import YOLO

from litter_detector.artifacts import write_json, write_text
from litter_detector.detector import ROOT, read_image, write_image
from litter_detector.metrics import count_at, iou_one
from quality_evaluation import OUTPUT, PROFILES, validation_predictions
from quality_tuning import load_json, sha256


ASSETS = ROOT / 'demo-assets'


def video_from_images(records, target, profile):
    if profile == 'underwater':
        first_height, first_width = read_image(records[0]['file']).shape[:2]
        width, height = first_width - first_width % 2, first_height - first_height % 2
    else:
        width, height = 960, 540
    fps = 12
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f'Cannot write demo source video: {target}')
    try:
        frames = 6 * fps
        for index, row in enumerate(records):
            image = read_image(row['file'])
            h, w = image.shape[:2]
            scale = min(width / w, height / h)
            w, h = round(w * scale), round(h * scale)
            image = cv2.resize(image, (w, h))
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            x, y = (width - w) // 2, (height - h) // 2
            canvas[y:y + h, x:x + w] = image
            copies = (index + 1) * frames // len(records) - index * frames // len(records)
            for _ in range(copies):
                writer.write(canvas)
    finally:
        writer.release()


def sample_media(candidate, threshold, profile):
    records = validation_predictions(candidate)['records']
    positive = [row for row in records if row['truth']]
    ranked = []
    for row in positive:
        scores = count_at([row], threshold)
        if scores['tp']:
            ranked.append((scores['f1'], scores['tp'], row))
    ranked.sort(key=lambda item: item[:2], reverse=True)
    if not ranked:
        raise ValueError(f'No positive validation example for {profile}')
    photo = ASSETS / 'examples' / f'{profile}.jpg'
    write_image(photo, read_image(ranked[0][2]['file']))
    video = ASSETS / 'examples' / f'{profile}.mp4'
    if profile == 'land':
        frames = [row for _, _, row in ranked[:4]]
        video_from_images(frames, video, profile)
    else:
        grouped = {}
        for record in records:
            group = Path(record['file']).stem.split('_frame')[0]
            grouped.setdefault(group, []).append(record)
        hit_paths = {row['file'] for _, _, row in ranked}
        frames = max(grouped.values(), key=lambda group: sum(row['file'] in hit_paths for row in group))
        frames = sorted(frames, key=lambda row: row['file'])
        video_from_images(frames, video, profile)
    return {'photo': photo.relative_to(ASSETS).as_posix(), 'video': video.relative_to(ASSETS).as_posix(),
            'photo_source': Path(ranked[0][2]['file']).name,
            'video_sources': [Path(row['file']).name for row in frames]}


def matched_detections(a, b):
    if len(a) != len(b):
        return None
    remaining = list(range(len(b)))
    min_iou, max_conf_error = 1.0, 0.0
    for row in a:
        overlaps = iou_one(row[:4], b[remaining, :4])
        index = int(np.argmax(overlaps))
        match = b[remaining.pop(index)]
        if row[5] != match[5]:
            return None
        min_iou = min(min_iou, float(overlaps[index]))
        max_conf_error = max(max_conf_error, abs(float(row[4] - match[4])))
    return min_iou, max_conf_error


def check_export(model, artifact, candidate, threshold):
    exported = YOLO(str(artifact), task='detect')
    records = validation_predictions(candidate)['records']
    positives = [row for row in records if count_at([row], threshold)['tp']]
    sample = [row['file'] for row in positives[:12]]
    sample.extend(row['file'] for row in records[::max(1, len(records) // 12)][:12])
    sample = list(dict.fromkeys(sample))
    times = []
    unmatched = []
    minimum_iou, max_conf_error = 1.0, 0.0
    for filename in sample:
        image = read_image(filename)
        settings = dict(imgsz=candidate['imgsz'], conf=threshold, iou=.5, device='cpu', verbose=False, rect=True)
        original = model.predict(image, **settings)[0].boxes.data.cpu().numpy()
        started = perf_counter()
        converted = exported.predict(image, **settings)[0].boxes.data.cpu().numpy()
        times.append(1000 * (perf_counter() - started))
        result = matched_detections(original, converted)
        if result is None or result[0] < .98 or result[1] > .01:
            unmatched.append(Path(filename).name)
        else:
            minimum_iou = min(minimum_iou, result[0])
            max_conf_error = max(max_conf_error, result[1])
    return {'images': len(sample), 'differences': unmatched, 'minimum_iou': minimum_iou,
            'max_confidence_error': max_conf_error, 'onnx_cpu_median_ms': median(times[3:] or times)}


def prepare():
    evaluated = load_json(OUTPUT / '06_test.json')
    selection = load_json(OUTPUT / '05_selection.json')
    (ASSETS / 'models').mkdir(parents=True, exist_ok=True)
    (ASSETS / 'examples').mkdir(parents=True, exist_ok=True)
    manifest = {}
    report = {}
    for profile in PROFILES:
        item = selection[profile]
        candidate = item['selected']
        threshold = candidate['chosen_operating']['confidence']
        tested = evaluated[profile]['selected']
        if tested.get('weight_sha256') != candidate['weight_sha256'] or tested['metrics']['confidence'] != threshold:
            raise ValueError(f'Run the test evaluation for the current selection before exporting: {profile}')
        source = Path(candidate['weights'])
        if sha256(source) != candidate['weight_sha256']:
            raise ValueError(f'Changed release weights: {source}')
        folder = OUTPUT / 'exports' / profile
        folder.mkdir(parents=True, exist_ok=True)
        copy = folder / 'export.pt'
        shutil.copy2(source, copy)
        model = YOLO(str(copy))
        artifact = Path(model.export(format='onnx', imgsz=candidate['imgsz'], opset=18, dynamic=True,
                                     simplify=False, batch=1, device='cpu'))
        onnx.checker.check_model(str(artifact))
        parity = check_export(model, artifact, candidate, threshold)
        report[profile] = parity
        write_json(OUTPUT / f'07_export_{profile}.json', parity)
        if len(parity['differences']) > max(1, .05 * parity['images']):
            raise RuntimeError(f'{profile}: ONNX parity check failed on {parity["differences"]}')
        target = ASSETS / 'models' / f'{profile}.onnx'
        shutil.copy2(artifact, target)
        manifest[profile] = {'model': target.relative_to(ASSETS).as_posix(), 'model_sha256': sha256(target),
                             'architecture': candidate['arch'], 'imgsz': candidate['imgsz'], 'confidence': threshold,
                             **sample_media(candidate, threshold, profile)}
    write_json(ASSETS / 'manifest.json', manifest)
    write_json(OUTPUT / '07_export.json', report)
    lines = ['# Шаг 7. Экспорт и демонстрационные файлы', '',
             '| Профиль | Проверено фото | Различия | Медиана ONNX CPU, мс |', '|---|---:|---:|---:|']
    for profile, row in report.items():
        lines.append(f"| {profile} | {row['images']} | {len(row['differences'])} | {row['onnx_cpu_median_ms']:.1f} |")
    lines += ['', 'Сравнение выполнено на одинаковых входах и порогах. Сохранены исходные примеры фото и видео для обоих профилей.', '']
    write_text(OUTPUT / '07_export.md', '\n'.join(lines))


if __name__ == '__main__':
    prepare()
