"""Run the YOLO11 comparison and the YOLO26 architecture screen."""
import argparse
import csv
from importlib.metadata import version
import json
import sys

from litter_detector.artifacts import write_json, write_text
from litter_detector.detector import ROOT
from quality_evaluation import BASELINES, OUTPUT, PROFILES, compact_result, isolated_video_benchmark, run_evaluation
from v2_pipeline import EXPERIMENT, PYTHON, run


def epochs_completed(profile, name):
    with (EXPERIMENT / profile / name / 'results.csv').open(newline='') as stream:
        return max(int(row['epoch']) for row in csv.DictReader(stream))


def evaluate_experiment(summary):
    name = summary['name']
    profile = summary['profile']
    result = run_evaluation(profile, name, summary['weights'], summary['imgsz'])
    balanced = max(result['thresholds'], key=lambda row: row['f1'])
    return {
        'name': name, 'profile': profile, 'arch': summary['arch'],
        'epochs_completed': epochs_completed(profile, name),
        'prediction_file': str(OUTPUT / 'predictions' / f'{profile}_{name}_val.json'),
        'run': summary,
        **compact_result(result, balanced['confidence']),
    }


def baseline_choice(profile, key):
    row = json.loads((OUTPUT / '01_baseline.json').read_text(encoding='utf-8'))[profile][key]
    row['prediction_file'] = str(OUTPUT / 'predictions' / f'{profile}_{key}_val.json')
    if key == 'v1':
        return {'name': 'v1', 'profile': profile, 'arch': 'yolo11n', 'epochs_completed': 30, **row}
    name = BASELINES[profile][0]
    summary = json.loads((EXPERIMENT / profile / name / 'summary.json').read_text(encoding='utf-8'))
    return {'name': name, 'profile': profile, 'arch': summary['arch'],
            'epochs_completed': epochs_completed(profile, name), 'run': summary, **row}


def format_row(row):
    point = row['operating']
    return (f"| {row['profile']} | {row['arch']} ({row['name']}) | {row['epochs_completed']} | "
            f"{row['ap']['map50_95']:.4f} | {point['precision']:.3f} | {point['small_recall']:.3f} |")


TABLE = [
    '| Профиль | Серия | Эпох | mAP50–95 | Precision | Recall мелких |',
    '|---|---|---:|---:|---:|---:|',
]


def compare_yolo11():
    summary = run('underwater', 's_compare', arch='yolo11s', imgsz=960, batch=8, python=PYTHON)
    selected_land = baseline_choice('land', 'YOLO11s')
    underwater = [baseline_choice('underwater', 'v1'), baseline_choice('underwater', 'YOLO11n')]
    small = evaluate_experiment(summary)
    small['video'] = isolated_video_benchmark('underwater', small['name'], small['weights'], 960,
                                              small['operating']['confidence'])
    underwater.append(small)
    selected_underwater = max(underwater, key=lambda row: row['ap']['map50_95'])
    report = {'comparison': {'land': [baseline_choice('land', 'v1'), selected_land], 'underwater': underwater},
              'selected': {'land': selected_land, 'underwater': selected_underwater}}
    write_json(OUTPUT / '02_yolo11.json', report)
    lines = ['# Шаг 2. Сравнение YOLO11', '', *TABLE]
    for rows in report['comparison'].values():
        lines.extend(format_row(row) for row in rows)
    lines += ['', f"По mAP50–95 под водой лидирует серия `{selected_underwater['name']}`. "
              'Порог v1 взят из приложения, для новых серий выбран по F1 на validation.', '']
    write_text(OUTPUT / '02_yolo11.md', '\n'.join(lines))


def compare_yolo26():
    ultralytics_version = version('ultralytics')
    if not (OUTPUT / '02_yolo11.json').is_file():
        raise FileNotFoundError('Complete the YOLO11 comparison first')
    if tuple(map(int, ultralytics_version.split('.')[:2])) < (8, 4):
        raise RuntimeError('Run the YOLO26 screen in its separate Conda environment')
    comparisons = {}
    for profile in PROFILES:
        pilots = []
        for arch in ('yolo26n', 'yolo26s'):
            name = f'quality_{arch}_screen'
            summary = run(profile, name, python=sys.executable, arch=arch, imgsz=960, batch=8,
                          epochs=12, patience=5)
            pilot = evaluate_experiment(summary)
            pilot['cpu_video'] = isolated_video_benchmark(profile, name, pilot['weights'], pilot['imgsz'],
                                                           pilot['operating']['confidence'], device='cpu')
            pilots.append(pilot)
        chosen = max(pilots, key=lambda row: row['ap']['map50_95'])
        arch = chosen['arch']
        name = f'quality_{arch}_full'
        summary = run(profile, name, python=sys.executable, arch=arch, imgsz=960, batch=8)
        comparisons[profile] = {'pilot': pilots, 'full': evaluate_experiment(summary)}
        write_json(OUTPUT / f'03_yolo26_{profile}.json', comparisons[profile])
    write_json(OUTPUT / '03_yolo26.json', {'ultralytics': ultralytics_version, 'profiles': comparisons})
    lines = ['# Шаг 3. Сравнение YOLO26', '',
             f'Ultralytics {ultralytics_version}. Короткие серии до 12 эпох, полные до 80. '
             'Вход 960 px, seed 42, общая validation-выборка внутри каждого профиля.', '', *TABLE]
    for profile in PROFILES:
        rows = comparisons[profile]
        lines.extend(format_row(row) for row in [*rows['pilot'], rows['full']])
    lines += ['', '| Профиль | Короткая серия | FPS PyTorch CPU |', '|---|---|---:|']
    for profile, rows in comparisons.items():
        lines.extend(f"| {profile} | {row['arch']} | {row['cpu_video']['fps']:.1f} |" for row in rows['pilot'])
    write_text(OUTPUT / '03_yolo26.md', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('2', '3'))
    args = parser.parse_args()
    (compare_yolo11 if args.step == '2' else compare_yolo26)()
