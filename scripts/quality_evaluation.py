"""Compare saved detectors on the same prepared data and video inputs."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter
import hashlib
import json
import subprocess
import sys

import cv2
import numpy as np
import yaml

from litter_detector.artifacts import write_json, write_text
from litter_detector.detector import ROOT, Detector, read_image
from litter_detector.experiments import file_sha256
from litter_detector.metrics import count_at


OUTPUT = ROOT / 'outputs' / 'quality'
PROFILES = ('land', 'underwater')
BASELINES = {
    'land': ('s_compare', 'YOLO11s'),
    'underwater': ('n960_long', 'YOLO11n'),
}


def video_source(profile):
    if profile == 'underwater':
        return ROOT / 'data' / 'samples' / 'underwater_dataset_sequence.mp4'
    destination = OUTPUT / 'benchmark' / 'land_dataset_sequence.mp4'
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = (ROOT / 'data' / 'v2' / profile / 'val.txt').read_text(encoding='utf-8').splitlines()
    temporary = destination.with_name('land_dataset_sequence.tmp.mp4')
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), 12, (960, 540))
    if not writer.isOpened():
        raise OSError(temporary)
    try:
        for path in paths[:36]:
            image = read_image(path)
            height, width = image.shape[:2]
            factor = min(960 / width, 540 / height)
            image = cv2.resize(image, (round(width * factor), round(height * factor)))
            frame = np.zeros((540, 960, 3), dtype=np.uint8)
            x, y = (960 - image.shape[1]) // 2, (540 - image.shape[0]) // 2
            frame[y:y + image.shape[0], x:x + image.shape[1]] = image
            writer.write(frame)
    finally:
        writer.release()
    temporary.replace(destination)
    return destination


def video_benchmark(profile, weights, size, confidence, device='0'):
    source = video_source(profile)
    detector = Detector(profile, confidence=confidence, model_path=weights, device=device)
    detector.config['imgsz'] = size
    capture = cv2.VideoCapture(str(source))
    timings = []
    try:
        while capture.isOpened():
            started = perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            detector.predict(frame)
            timings.append(perf_counter() - started)
    finally:
        capture.release()
    timings = timings[3:]
    if not timings:
        raise ValueError(f'No benchmark frames after warm-up: {source}')
    return {
        'frames': len(timings), 'fps': len(timings) / sum(timings),
        'median_ms': 1000 * median(timings), 'p95_ms': 1000 * float(np.quantile(timings, .95)),
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'device': device, 'includes': 'video decoding, model inference and drawing',
    }


def run_evaluation(profile, name, weights, size, split='val', sliced=False, tile_size=640):
    path = OUTPUT / 'predictions' / f'{profile}_{name}_{split}.json'
    command = [sys.executable, str(ROOT / 'scripts' / 'predict_v2.py'), str(weights), profile,
               '--split', split, '--imgsz', str(size), '--output', str(path), '--tile-size', str(tile_size)]
    if sliced:
        command.append('--sliced')
    # Release the evaluation process and its CUDA context before the next training run.
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(path.read_text(encoding='utf-8'))


def isolated_video_benchmark(profile, name, weights, size, confidence, device='0'):
    path = OUTPUT / 'benchmark' / f'{profile}_{name}_{device}.json'
    command = [sys.executable, str(Path(__file__).resolve()), '--benchmark', profile, str(weights),
               str(size), str(confidence), device, str(path)]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(path.read_text(encoding='utf-8'))


def compact_result(report, confidence):
    operating = next((row for row in report['thresholds'] if row['confidence'] == confidence), None)
    if operating is None:
        if confidence < .05:
            raise ValueError('Confidence is below the prediction cutoff (0.05)')
        operating = count_at(report['records'], confidence)
    return {
        'weights': report['weights'], 'weight_sha256': report['weight_sha256'],
        'imgsz': report['imgsz'], 'ap': report['ap'], 'operating': operating,
        'image_count': len(report['records']), 'file_list_sha256': report['file_list_sha256'],
        'manifest_sha256': report['manifest_sha256'],
    }


def validation_predictions(candidate):
    path = Path(candidate['prediction_file'])
    report = json.loads(path.read_text(encoding='utf-8'))
    profile = candidate['profile']
    dataset = ROOT / 'data' / 'v2' / profile
    expected = {
        'profile': profile, 'split': 'val', 'sliced': False, 'imgsz': candidate['imgsz'],
        'weight_sha256': file_sha256(candidate['weights']),
        'file_list_sha256': file_sha256(dataset / 'val.txt'),
        'manifest_sha256': file_sha256(dataset / 'manifest.json'),
    }
    if any(report.get(key) != value for key, value in expected.items()) or any(
        candidate[key] != report[key] for key in ('ap', 'weight_sha256', 'file_list_sha256', 'manifest_sha256')
    ):
        raise ValueError(f'Validation predictions do not match the model or dataset: {path}')
    operating = compact_result(report, candidate['operating']['confidence'])['operating']
    if operating != candidate['operating']:
        raise ValueError(f'Validation operating point does not match the predictions: {path}')
    return report


def run_baselines():
    results = {}
    for profile in PROFILES:
        app_config = yaml.safe_load((ROOT / 'models' / 'baseline_v1' / 'configs' / f'{profile}.yaml').read_text())
        experiment, arch = BASELINES[profile]
        for name, weights, size, threshold in (
            ('v1', ROOT / 'models' / 'baseline_v1' / profile / 'weights' / 'best.pt', 640, app_config['confidence']),
            (arch, ROOT / 'models' / 'experiments' / profile / experiment / 'weights' / 'best.pt', 960, None),
        ):
            full = run_evaluation(profile, name, weights, size)
            if threshold is None:
                threshold = max(full['thresholds'], key=lambda row: row['f1'])['confidence']
            row = compact_result(full, threshold)
            row['video'] = isolated_video_benchmark(profile, name, weights, size, threshold)
            results.setdefault(profile, {})[name] = row
    write_json(OUTPUT / '01_baseline.json', results)
    lines = [
        '# Шаг 1. Контрольная оценка YOLO11', '',
        'Для каждого профиля использована одна и та же validation-выборка. Мелкие объекты имеют площадь меньше 32 × 32 пикселей в подготовленном кадре.', '',
        '| Профиль | Модель | Вход | mAP50–95 | Precision | Recall мелких | Ложные рамки на пустых кадрах | FPS |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for profile, models in results.items():
        for name, row in models.items():
            point = row['operating']
            lines.append(f"| {profile} | {name} | {row['imgsz']} | {row['ap']['map50_95']:.3f} | "
                         f"{point['precision']:.3f} | {point['small_recall']:.3f} | "
                         f"{point['negative_false_positives']} / {point['negative_images']} | {row['video']['fps']:.1f} |")
    lines += ['', 'FPS измерен на RTX 4070 SUPER и включает декодирование, распознавание и отрисовку. Рабочая цель — прибавить 5 процентных пунктов полноты на мелких объектах при сопоставимой точности.', '']
    write_text(OUTPUT / '01_baseline.md', '\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', nargs=6, metavar=('PROFILE', 'WEIGHTS', 'SIZE', 'CONF', 'DEVICE', 'OUT'))
    args = parser.parse_args()
    if args.benchmark:
        profile, weights, size, confidence, device, output = args.benchmark
        write_json(output, video_benchmark(profile, weights, int(size), float(confidence), device))
    else:
        run_baselines()
