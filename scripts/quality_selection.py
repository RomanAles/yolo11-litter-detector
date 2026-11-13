"""Select an operating point on validation, then evaluate the frozen result on test."""
import argparse
import json
import subprocess
import sys

import yaml

from finalize_v2 import error_gallery
from litter_detector.artifacts import write_json, write_text
from litter_detector.detector import ROOT
from litter_detector.metrics import count_at
from quality_evaluation import OUTPUT, PROFILES, isolated_video_benchmark, run_evaluation, validation_predictions
from quality_training import baseline_choice, evaluate_experiment
from quality_tuning import load_json, sha256
from v2_pipeline import EXPERIMENT, PYTHON, run


PRECISION_TOLERANCE = .01
AP_TOLERANCE = .005
TARGET_SMALL_GAIN = .05


def at_precision(report, minimum_precision):
    feasible = [point for point in report['thresholds'] if point['precision'] >= minimum_precision
                and point['tp'] > 0 and point['small_recall'] is not None]
    if not feasible:
        return None
    return max(feasible, key=lambda point: (point['small_recall'], point['recall'], point['precision']))


def choose(candidates, reference):
    min_precision = reference['operating']['precision'] - PRECISION_TOLERANCE
    min_ap = reference['ap']['map50_95'] - AP_TOLERANCE
    target = reference['operating']['small_recall'] + TARGET_SMALL_GAIN
    feasible = []
    for candidate in candidates:
        point = at_precision(validation_predictions(candidate), min_precision)
        if point is not None and candidate['ap']['map50_95'] >= min_ap:
            feasible.append({**candidate, 'chosen_operating': point})
    if not feasible:
        raise ValueError('The reference itself must be eligible at its recorded operating threshold')
    meeting_target = [candidate for candidate in feasible if candidate['chosen_operating']['small_recall'] >= target]
    return max(meeting_target or feasible, key=lambda candidate: candidate['ap']['map50_95'])


def repeat_with_seed(candidate):
    profile = candidate['profile']
    name = f"quality_{candidate['name']}_seed123"
    if candidate['name'] == 'v1':
        folder = EXPERIMENT / profile / name
        metadata = folder / 'training_summary.json'
        if not metadata.exists():
            command = [PYTHON, str(ROOT / 'scripts' / 'train_models.py'), profile, '--seed', '123', '--run-name', name]
            if (folder / 'weights' / 'last.pt').exists():
                command.append('--resume')
            log_path = ROOT / '.cache' / f'quality-{profile}-v1-seed123.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        run_metadata = load_json(metadata)
        report = run_evaluation(profile, name, run_metadata['best_weights'], 640)
        return {'name': name, 'ap': report['ap'], 'seed': 123}
    original = candidate['run']
    # Quality fine-tuning and YOLO26 runs use the newer environment.
    python = sys.executable if candidate['name'].startswith('quality_') else PYTHON
    saved_arguments = yaml.safe_load(
        (EXPERIMENT / profile / original['name'] / 'args.yaml').read_text(encoding='utf-8')
    )
    lr0 = saved_arguments['lr0']
    patience = saved_arguments['patience']
    replica = run(profile, name, python=python, arch=original['arch'], imgsz=original['imgsz'],
                  batch=saved_arguments['batch'], epochs=original['epochs_requested'], patience=patience,
                  weights=original['initial_weights'], data=original['data'], lr0=lr0,
                  workers=original.get('workers', 1), seed=123,
                  augment=json.dumps(original['augmentation']))
    return evaluate_experiment(replica)


def select_models():
    yolo11 = load_json(OUTPUT / '02_yolo11.json')['selected']
    yolo26 = load_json(OUTPUT / '03_yolo26.json')['profiles']
    tuning = load_json(OUTPUT / '04_tuning.json')
    selection = {}
    for profile in PROFILES:
        reference = yolo11[profile]
        candidates = [reference, yolo26[profile]['full'], tuning[profile]['control'], tuning[profile]['variant']]
        chosen = choose(candidates, reference)
        repeated = repeat_with_seed(chosen)
        point = chosen['chosen_operating']
        tiled = run_evaluation(profile, 'selected_sahi', chosen['weights'], chosen['imgsz'],
                               sliced=True, tile_size=640 if profile == 'land' else 320)
        sahi_point = at_precision(tiled, reference['operating']['precision'] - PRECISION_TOLERANCE)
        measured = isolated_video_benchmark(profile, chosen['name'], chosen['weights'], chosen['imgsz'],
                                            point['confidence'])
        selection[profile] = {'selected': chosen, 'reference': reference, 'seed123': repeated, 'video': measured,
                              'sahi': {'point': sahi_point, 'ap': tiled['ap'], 'tile_size': tiled['tile_size'],
                                       'median_ms': tiled['median_ms']}}
        write_json(OUTPUT / f'05_selection_{profile}.json', selection[profile])
    write_json(OUTPUT / '05_selection.json', selection)
    lines = ['# Шаг 5. Модель, порог и устойчивость', '',
             'Рабочий порог выбран на validation. Допуск по precision составляет один процентный пункт. '
             'Контрольный ориентир по mAP50–95 допускает снижение до 0,005, когда растёт полнота по мелким объектам.', '',
             '| Профиль | Серия | Порог | Precision | Recall мелких | AP seed 42 / 123 | FPS |',
             '|---|---|---:|---:|---:|---:|---:|']
    for profile, item in selection.items():
        row = item['selected']
        point = row['chosen_operating']
        lines.append(f"| {profile} | {row['name']} | {point['confidence']:.2f} | {point['precision']:.3f} | "
                     f"{point['small_recall']:.3f} | {row['ap']['map50_95']:.4f} / "
                     f"{item['seed123']['ap']['map50_95']:.4f} | {item['video']['fps']:.1f} |")
    for profile, item in selection.items():
        point = item['sahi']['point']
        description = (f"recall мелких {point['small_recall']:.3f} при precision {point['precision']:.3f}"
                       if point else 'при заданной precision подходящий порог не найден')
        lines.append(f"\n{profile}, SAHI {item['sahi']['tile_size']} px, {description}. "
                     f"Медиана обработки {item['sahi']['median_ms']:.1f} мс.")
    write_text(OUTPUT / '05_selection.md', '\n'.join(lines) + '\n')


def evaluate_test():
    selection = load_json(OUTPUT / '05_selection.json')
    results = {}
    for profile in PROFILES:
        selection_row = selection[profile]
        best = selection_row['selected']
        reference = selection_row['reference']
        cohorts = (
            ('v1', baseline_choice(profile, 'v1'), 'operating'),
            ('yolo11_reference', reference, 'operating'),
            ('selected', best, 'chosen_operating'),
        )
        cached = {}
        models = {}
        for label, model, operating_key in cohorts:
            weights = model['weights']
            digest = sha256(weights)
            if digest != model['weight_sha256']:
                raise ValueError(f'Weights changed after validation selection: {weights}')
            key = (digest, model['imgsz'])
            if key not in cached:
                cached[key] = run_evaluation(profile, f'final_{label}', weights, model['imgsz'], split='test')
            report = cached[key]
            if report['weight_sha256'] != digest or report['manifest_sha256'] != model['manifest_sha256']:
                raise ValueError(f'Model or dataset changed after validation: {weights}')
            threshold = model[operating_key]['confidence']
            counts = count_at(report['records'], threshold)
            models[label] = {'name': model['name'], 'arch': model['arch'], 'ap': report['ap'],
                             'metrics': counts, 'imgsz': model['imgsz'], 'weight_sha256': digest,
                             'file_list_sha256': report['file_list_sha256']}
            if label == 'selected':
                error_gallery(report, threshold, OUTPUT / 'test_examples' / profile / 'errors')
                successes = []
                for record in report['records']:
                    point = count_at([record], threshold)
                    if point['tp'] and not (point['fp'] or point['fn']):
                        successes.append(record)
                    if len(successes) == 3:
                        break
                if successes:
                    error_gallery({'records': successes}, threshold, OUTPUT / 'test_examples' / profile / 'successes')
        if len({model['file_list_sha256'] for model in models.values()}) != 1:
            raise ValueError(f'Test cohort changed between evaluations: {profile}')
        results[profile] = models
    write_json(OUTPUT / '06_test.json', results)
    lines = ['# Шаг 6. Итоговая оценка', '',
             'Веса и пороги зафиксированы до оценки. Для каждого профиля все модели обработали одну тестовую выборку. '
             'Историческая выборка уже использовалась в отчёте v1.', '',
             '| Профиль | Модель | mAP50–95 | Precision | Recall | Recall мелких | Ложные рамки на пустых кадрах |',
             '|---|---|---:|---:|---:|---:|---:|']
    for profile, models in results.items():
        for label, row in models.items():
            point = row['metrics']
            lines.append(f"| {profile} | {label} ({row['arch']}, {row['imgsz']} px) | {row['ap']['map50_95']:.4f} | "
                         f"{point['precision']:.3f} | {point['recall']:.3f} | {point['small_recall']:.3f} | "
                         f"{point['negative_false_positives']} / {point['negative_images']} |")
    write_text(OUTPUT / '06_test.md', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('5', '6'))
    args = parser.parse_args()
    (select_models if args.step == '5' else evaluate_test)()
