"""Render a concise source-backed local readout of the completed dataset run."""
import csv
import json

from litter_detector.detector import ROOT


def build_report():
    lines=['# Результаты выполнения','',
           'Обучение и оценка выполнены только на готовых TACO и TrashCan. Физические камеры не подключались.','',
           '## Данные','',
           '| Профиль | Train | Validation | Test | Объектов в test |','|---|---:|---:|---:|---:|']
    for profile in ('land','underwater'):
        manifest=json.loads((ROOT/'data/prepared'/profile/'manifest.json').read_text(encoding='utf8'))
        s=manifest['summary']
        lines.append(f"| {profile} | {s['train']['images']} | {s['val']['images']} | {s['test']['images']} | {s['test']['objects']} |")
    lines += ['', 'Разбиение по целым партиям TACO и исходным видео TrashCan; группы точных дубликатов объединены. Seed=42. Подводные природные ветки, животные, растения и ROV не относятся к классу trash.', '',
              '## Обучение и итоговая оценка','',
              '| Профиль | Эпох выполнено | Confidence | Precision | Recall | F1 | mAP50 | mAP50–95 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for profile in ('land','underwater'):
        report=json.loads((ROOT/'outputs/evaluation'/profile/'metrics.json').read_text(encoding='utf8'))
        rows=list(csv.DictReader((ROOT/'models'/profile/'results.csv').open(encoding='utf8')))
        point=report['test_operating_point'];ap=report['test_ap_metrics']
        lines.append(f"| {profile} | {int(float(rows[-1]['epoch']))} | {report['chosen_confidence']:.2f} | {point['precision']:.3f} | {point['recall']:.3f} | {point['f1']:.3f} | {ap['metrics/mAP50(B)']:.3f} | {ap['metrics/mAP50-95(B)']:.3f} |")
    lines += ['', '## Ошибки и мелкие объекты', '',
              '| Профиль | Пропущено объектов (FN) | Ложных рамок (FP) | Recall мелких объектов | Ложных рамок на отрицательных кадрах |',
              '|---|---:|---:|---:|---:|']
    for profile in ('land','underwater'):
        report=json.loads((ROOT/'outputs/evaluation'/profile/'metrics.json').read_text(encoding='utf8'))
        point=report['test_operating_point']
        small=f"{point['small_recall']:.3f}" if point['small_recall'] is not None else 'нет объектов'
        negatives=str(point.get('negative_false_positives',0)) if point.get('negative_images',0) else 'нет отрицательных кадров'
        lines.append(f"| {profile} | {point['fn']} | {point['fp']} | {small} | {negatives} |")
    lines += ['', 'Мелкие объекты: площадь исходной подготовленной рамки менее 32×32 пикселей. Это recall при рабочем пороге, не COCO AP-small. В TACO нет полностью отрицательных изображений в данном разбиении, поэтому отдельная оценка пустых сцен для land отсутствует.']
    lines += ['', 'Precision/recall/F1 — итоговый test, IoU=0.5, рабочий порог выбран только по validation. AP — отдельная оценка полной кривой при низком пороге. Для обоих профилей использована YOLO11n; это исходный воспроизводимый результат, не сравнение всех архитектур.', '',
              '## Ограничения','',
              '- Нет проверки физической обычной камеры или подводного видеоканала, полевых испытаний и оценки задержки камера → экран.',
              '- Качество зависит от публичного датасета; перенос на другие места, пресную воду, камеры и освещение не подтверждён.',
              '- Рамка означает предмет, похожий на размеченный мусор; модель не устанавливает, выброшен ли предмет или используется человеком.',
              '- Известные видеогруппы и точные дубликаты разделены, но неизвестные повторы сцен и неполная разметка всё ещё возможны.',
              '- Видео обрабатывается в CFR без переноса звука. Демо-последовательность собрана из готовых кадров с назначенными временными интервалами.',
              '- Чистая установка на другом компьютере и коммерческие права на исходные изображения не подтверждены.', '',
              '## Артефакты','',
              '- `models/land/weights/best.pt` и `models/underwater/weights/best.pt` — обученные веса.',
              '- `models/*/training_summary.json`, `args.yaml`, `results.csv` — сведения обучения.',
              '- `outputs/evaluation/*/metrics.json` — рабочие пороги, счётчики и AP.',
              '- `data/prepared/*/manifest.json` — разбиение, контрольные суммы и разметка.',
              '- `outputs/examples/` — результаты на изображениях из test.',
              '- `outputs/demo-video/` — размеченная демонстрационная последовательность.', '']
    (ROOT/'docs/RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines[:22]))


if __name__=='__main__':build_report()
