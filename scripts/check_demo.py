"""Exercise the demo container through HTTP with both prepared profiles."""
import argparse
import json
from time import perf_counter

import cv2
import requests

from litter_detector.artifacts import write_json, write_text
from litter_detector.detector import ROOT, read_image


def check(base_url):
    root = ROOT / 'outputs' / 'quality'
    destination = root / 'container_examples'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'demo-assets' / 'manifest.json').read_text(encoding='utf-8'))
    session = requests.Session()

    def get(path):
        response = session.get(base_url + path, timeout=30)
        response.raise_for_status()
        return response

    health = get('/health').json()
    assert health['status'] == 'ok' and health['profiles'] == ['land', 'underwater']
    assert 'Детектор мусора' in get('/').content.decode('utf-8')
    report = {}
    for profile in health['profiles']:
        report[profile] = {}
        for example in ('photo', 'video'):
            started = perf_counter()
            response = session.post(base_url + '/api/detect', data={'profile': profile, 'example': example}, timeout=180)
            response.raise_for_status()
            elapsed = perf_counter() - started
            result = response.json()
            assert result['kind'] == example
            output = destination / (profile + ('.jpg' if example == 'photo' else '.mp4'))
            output.write_bytes(get(result['media']).content)
            log = get(result['detections'])
            if example == 'photo':
                record = log.json()
                assert len(record['detections']) == result['count']
                assert result['count'] > 0 and read_image(output).size > 0
                records = [record]
            else:
                records = [json.loads(line) for line in log.content.splitlines()]
                assert len(records) == result['frames'] == 72
                capture = cv2.VideoCapture(str(output))
                try:
                    assert capture.isOpened()
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    assert capture.read()[0] and fps == 12 and frames == 72
                finally:
                    capture.release()
                assert result['seconds'] == 6 and [record['frame_id'] for record in records] == list(range(frames))
                assert any(record['detections'] for record in records)
            assert all(record['profile'] == profile and record['model_sha256'] == manifest[profile]['model_sha256']
                       for record in records)
            output.with_suffix('.json' if example == 'photo' else '.jsonl').write_bytes(log.content)
            report[profile][example] = {'seconds': elapsed, 'frames': len(records),
                                        'detections': sum(len(record['detections']) for record in records)}
            print(profile, example, report[profile][example], flush=True)

    asset = ROOT / 'demo-assets' / manifest['land']['photo']
    with asset.open('rb') as stream:
        response = session.post(base_url + '/api/detect', data={'profile': 'land'},
                                files={'file': ('photo.jpg', stream, 'image/jpeg')}, timeout=60)
    response.raise_for_status()
    uploaded = response.json()
    assert uploaded['count'] == report['land']['photo']['detections']
    assert get(uploaded['media']).headers['Content-Type'].startswith('image/jpeg')
    bad_file = session.post(base_url + '/api/detect', data={'profile': 'land'},
                            files={'file': ('invalid.jpg', b'not an image', 'image/jpeg')}, timeout=20)
    assert bad_file.status_code == 400
    write_json(root / '08_docker.json', {'health': health, 'examples': report, 'upload': 'passed', 'invalid_file': 'passed'})
    lines = ['# Шаг 8. Проверка Docker-демо', '', f"Контейнер `{health['device']}` прошёл HTTP-проверку.", '',
             '| Профиль | Фото, объектов | Видео, кадров | Обработка видео, секунд |', '|---|---:|---:|---:|']
    for profile, values in report.items():
        lines.append(f"| {profile} | {values['photo']['detections']} | {values['video']['frames']} | "
                     f"{values['video']['seconds']:.1f} |")
    lines += ['', 'Загрузка файла, скачивание разметки и обработка ошибки повреждённого файла проверены.', '']
    write_text(root / '08_docker.md', '\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:7860')
    check(parser.parse_args().url.rstrip('/'))
