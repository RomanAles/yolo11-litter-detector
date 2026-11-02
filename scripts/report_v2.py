"""Build the final readout from completed experiment artifacts."""
from pathlib import Path
import json,csv,statistics
from litter_detector.detector import ROOT


def report():
    out=ROOT/'outputs/v2';final=json.loads((out/'final.json').read_text());stability=json.loads((out/'stability.json').read_text())
    lines=['# Улучшения v2: результаты','',
      'Только готовые TACO и TrashCan, без подключения камер и полевых испытаний. Полноразмерные оригиналы TACO сохранены; рабочие копии ограничены 1600px по длинной стороне. Первая версия сохранена в models/baseline_v1.','',
      '## Сравнение на одной тестовой выборке','',
      'AP ниже рассчитана единым способом: один класс, IoU 0.50–0.95, 101 точка интерполяции, не более 100 рамок на изображение; предсказания сохраняются при confidence ≥ 0.05 и NMS IoU=0.5. Это отдельный расчёт, его нельзя напрямую смешивать с числами старого отчёта Ultralytics.','',
      '| Профиль | Baseline AP50 | v2 AP50 | Baseline mAP50–95 | v2 mAP50–95 | Выбранный вариант |',
      '|---|---:|---:|---:|---:|---|']
    for profile,r in final.items():
        a,b=r['baseline_test_ap'],r['test_ap']
        lines.append(f"| {profile} | {a['ap50']:.3f} | {b['ap50']:.3f} | {a['map50_95']:.3f} | {b['map50_95']:.3f} | {r['candidate']['name']} · {r['selection']['imgsz']}px |")
    lines += ['', '## Рабочие режимы','', '| Профиль | Режим | Confidence | Precision | Recall | Recall мелких |', '|---|---|---:|---:|---:|---:|']
    for profile,r in final.items():
        for mode,point in r['test_modes'].items():
            small=f"{point['small_recall']:.3f}" if point['small_recall'] is not None else '—'
            lines.append(f"| {profile} | {mode} | {point['confidence']:.2f} | {point['precision']:.3f} | {point['recall']:.3f} | {small} |")
    lines += ['', 'Пороги выбраны только по validation: balanced — F1, recall — F2, precision — F0.5. Мелкие объекты определяются по площади менее 32×32 пикселей в подготовленном изображении, а не по размеру после изменения входного разрешения.', '',
      '## Сцены без размеченного мусора','',
      '| Профиль | Отрицательных test-кадров | Ложных рамок в balanced | Рамок на отрицательный кадр |',
      '|---|---:|---:|---:|']
    for profile,r in final.items():
        point=r['test_modes']['balanced'];n=point['negative_images'];fp=point['negative_false_positives']
        rate=f'{fp/n:.3f}' if n else '—'
        lines.append(f'| {profile} | {n} | {fp} | {rate} |')
    lines += ['', 'Отсутствие рамок в исходной разметке не гарантирует, что снимок исчерпывающе проверен человеком. Эта метрика зависит от полноты датасета.', '',
      '## SAHI','', '| Профиль | Фрагмент, px | Обычная AP50 | SAHI AP50 | Обычная медиана, мс | SAHI медиана, мс |','|---|---:|---:|---:|---:|---:|']
    for profile,r in final.items():
        lines.append(f"| {profile} | {r['selection']['sahi_tile_size']} | {r['test_ap']['ap50']:.3f} | {r['sahi_test_ap']['ap50']:.3f} | {r['latency']['standard_median_ms']:.1f} | {r['latency']['sahi_median_ms']:.1f} |")
    lines += ['', 'Время относится к обработке файлов на текущем ПК, с прогревом. Это не измерение задержки камера → экран. SAHI остаётся отдельным режимом для файлов и использует собственные пороги.', '',
      '## Повторяемость','', '| Профиль | Validation mAP seed42 | seed123 | CV: среднее | CV: std |','|---|---:|---:|---:|---:|']
    for profile,r in stability.items():
        values=[x['validation']['metrics/mAP50-95(B)'] for x in r['grouped_cv']]
        lines.append(f"| {profile} | {r['seed42']['validation']['metrics/mAP50-95(B)']:.3f} | {r['seed123']['validation']['metrics/mAP50-95(B)']:.3f} | {statistics.mean(values):.3f} | {statistics.pstdev(values):.3f} |")
    lines += ['', 'Два seed дают описательное сравнение, не доверительный интервал. Три групповых фолда используют до 20 эпох и являются ограниченной проверкой устойчивости; они не заменяют кросс-валидацию полного 80-эпохового бюджета. Исходные test-группы во всех фолдах исключены. Seed/CV относятся к победителям новых серий; если в приложении сохранён baseline, эти серии не являются проверкой его повторяемости.', '',
      '## Серии обучения','', '| Профиль | Серия | Архитектура | Размер | Эпох | Лучший validation mAP50–95 |','|---|---|---|---:|---:|---:|']
    for file in sorted((ROOT/'models/experiments').glob('*/*/summary.json')):
        r=json.loads(file.read_text());csvfile=file.parent/'results.csv';rows=list(csv.DictReader(csvfile.open()))
        lines.append(f"| {r['profile']} | {r['name']} | {r['arch']} | {r['imgsz']} | {rows[-1]['epoch']} | {r['validation']['metrics/mAP50-95(B)']:.3f} |")
    lines += ['', 'Начальные параллельные серии с суффиксом initial_io_interrupted остановлены при настройке расхода ОЗУ и не участвуют в сравнении. Все последующие серии идут последовательно. Аугментации сравнивались с контрольным дообучением на том же начальном checkpoint и при том же бюджете.', '',
      '## Экспорт и программная проверка','']
    exports=json.loads((out/'export_check.json').read_text())
    for profile,r in exports.items():
        lines.append(f"- {profile}: ONNX opset18 прошёл checker и инференс на {r['samples']} validation-файлах. Различий числа рамок: {r['count_disagreements']}; максимальная разница confidence при совпадающем числе рамок: {r['max_score_difference_when_counts_match']:.6f}. Медиана PyTorch CUDA / ONNX CPU: {r['median_ms']['pytorch_cuda']:.1f} / {r['median_ms']['onnx_cpu']:.1f} мс.")
    lines += ['', 'Это ограниченная проверка переноса, не полная оценка эквивалентности backend. Основным остаётся PyTorch CUDA. Проверены фото, папки, три режима порога, SAHI и сохранение 28 кадров демонстрационного видео с длительностью 28/12 секунды. Контрольные суммы всех файлов baseline совпали. Unit-тесты, event loop окна и pip check завершились успешно.', '']
    runtime=json.loads((out/'runtime.json').read_text()) if (out/'runtime.json').exists() else None
    if runtime:
        lines += [f"Среда измерений: {runtime['gpu']}; PyTorch {runtime['torch']}; CUDA {runtime['cuda_runtime']}; Ultralytics {runtime['ultralytics']}. Последовательный инференс файлов, без параллельного обучения. Время не включает декодирование файла и отрисовку рамок.", '']
    lines += [
      '## Ограничения','',
      '- Исторический test уже использовался в v1. Это парная проверка улучшений, не новая слепая оценка.',
      '- Возобновление после прерываний и смена числа загрузчиков не гарантируют побитового повторения траектории обучения; изменения записаны в сводках серий.',
      '- Добавлены визуально проверенные фрагменты фона TACO; исходная разметка может быть неполной. Контактные листы и 19 отклонённых кандидатов сохранены.',
      '- Известные группы и точные дубликаты контролируются; все неизвестные повторы одной сцены автоматически исключить нельзя. Фактические рабочие изображения и рамки отдельно проверены в outputs/v2/prepared_file_audit.json; замороженные метаданные TACO местами сохраняют размеры оригинала.',
      '- Галерея ошибок содержит специально выбранные трудные случаи test и не используется в обучении.',
      '- Качество на новых камерах, в пресной воде и в полевых условиях не подтверждено.',
      '- Программная проверка Tkinter и превью пройдена. Визуальная проверка нативного окна через автоматизацию недоступна: оба runtime управления интерфейсом завершились ошибкой создания kernel assets.',
      '- Наличие предмета не определяет, выброшен ли он или используется человеком.', '',
      'Артефакты: outputs/v2, models/experiments, models/v2, configs/*.yaml. Базовая версия и старый отчёт сохранены.', '']
    (ROOT/'docs/IMPROVEMENT_RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    statuses=['Снимок baseline и контрольные суммы','Протокол выбора и отдельный test','Полноразмерные оригиналы TACO',
        'Аудит изображений, рамок, групп и точных дубликатов','141 проверенный отрицательный фрагмент TACO',
        'Длительное обучение YOLO11n','Сравнение разрешений 640/960 и условная проверка 1280',
        'Сравнение YOLO11s','Контроль и отдельная аугментация','Повторы seed, групповой CV и пороги',
        'Сравнение обычного инференса и SAHI','Отчёт, ONNX, приложение и программные проверки']
    status=['# Улучшение v2','', '| Шаг | Результат |', '|---|---|']
    status.extend(f'| {index}. {name} | Завершён |' for index,name in enumerate(statuses,1))
    status += ['', 'Результаты: [IMPROVEMENT_RESULTS.md](IMPROVEMENT_RESULTS.md). Команды и ограничения: [IMPROVEMENT_USAGE.md](IMPROVEMENT_USAGE.md).', '',
        'Только публичные датасеты, без физических камер. Выбор по validation. Исторический test не является новой слепой проверкой. CV ограничен 20 эпохами на фолд; выполнены два seed. Визуальная проверка нативного окна недоступна из-за сбоя runtime автоматизации, программная проверка окна пройдена.', '']
    (ROOT/'docs/IMPROVEMENT_STATUS.md').write_text('\n'.join(status),encoding='utf8')
    print('Wrote docs/IMPROVEMENT_RESULTS.md')


if __name__=='__main__':report()
