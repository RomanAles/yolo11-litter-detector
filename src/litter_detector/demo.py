"""Local browser demo for the exported land and underwater detectors."""
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4

import cv2
from flask import Flask, abort, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from .detector import ROOT, Detector
from .experiments import file_sha256
from .media import VideoFrameLimitExceeded, process_video


PROFILES = ('land', 'underwater')
PHOTO_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
RESULTS = {'result.jpg', 'result.mp4', 'detections.json', 'detections.jsonl'}
MAX_REQUEST_BYTES = 128 * 1024 * 1024
MAX_VIDEO_FRAMES = 3600


def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_REQUEST_BYTES
    app.config['TRUSTED_HOSTS'] = ['localhost', '127.0.0.1', '[::1]']
    assets = Path(os.environ.get('LITTER_DEMO_ASSETS', ROOT / 'demo-assets')).resolve()
    results = Path(os.environ.get('LITTER_DEMO_RESULTS', ROOT / 'outputs' / 'browser-demo')).resolve()
    results.mkdir(parents=True, exist_ok=True)
    device = os.environ.get('LITTER_DEMO_DEVICE', 'cpu')
    manifest = json.loads((assets / 'manifest.json').read_text(encoding='utf-8'))
    detectors = {}
    inference = Lock()

    def asset(path):
        candidate = (assets / path).resolve()
        if not candidate.is_relative_to(assets) or not candidate.is_file():
            raise FileNotFoundError(path)
        return candidate

    for profile in PROFILES:
        for item in ('model', 'photo', 'video'):
            asset(manifest[profile][item])
        model_path = asset(manifest[profile]['model'])
        if model_path.suffix.lower() != '.onnx' or file_sha256(model_path) != manifest[profile]['model_sha256']:
            raise ValueError(f'Invalid ONNX model or model checksum for {profile}: {model_path}')

    def detector_for(profile):
        if profile not in detectors:
            settings = manifest[profile]
            detector = Detector(profile, model_path=asset(settings['model']),
                                confidence=settings['confidence'], device=device)
            detector.config['imgsz'] = settings['imgsz']
            detectors[profile] = detector
        return detectors[profile]

    @app.get('/')
    def home():
        return send_file(Path(__file__).with_name('demo.html'))

    @app.get('/health')
    def health():
        return jsonify(status='ok', device=device, profiles=list(PROFILES))

    @app.get('/result/<uuid:job>/<filename>')
    def result(job, filename):
        path = results / job.hex / filename
        if filename not in RESULTS or not path.is_file():
            abort(404)
        return send_file(path)

    @app.post('/api/detect')
    def detect():
        origin = request.headers.get('Origin')
        if origin is not None and origin != request.host_url.rstrip('/'):
            return jsonify(error='Откройте страницу демонстрации для загрузки файла.'), 403
        if not inference.acquire(blocking=False):
            return jsonify(error='Обработка уже идёт. Повторите после её завершения.'), 503
        try:
            return detect_media()
        finally:
            inference.release()

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error='Запрос превышает 128 МиБ.'), 413

    def detect_media():
        profile = request.form.get('profile')
        example = request.form.get('example')
        upload = request.files.get('file')
        if profile not in PROFILES:
            return jsonify(error='Выберите профиль «Суша» или «Под водой».'), 400
        if example not in (None, 'photo', 'video') or (example is None) == (upload is None):
            return jsonify(error='Выберите один исходник.'), 400
        suffix = (Path(upload.filename or '').suffix.lower() if upload is not None
                  else Path(manifest[profile][example]).suffix.lower())
        if suffix in PHOTO_EXTENSIONS:
            kind = 'photo'
        elif suffix in VIDEO_EXTENSIONS:
            kind = 'video'
        else:
            return jsonify(error='Загрузите JPG, PNG, WebP или видео MP4, MOV, AVI, MKV, WebM.'), 400
        detector = detector_for(profile)
        job = uuid4()
        target = results / job.hex
        with TemporaryDirectory(prefix='job-', dir=results) as folder:
            temporary = Path(folder)
            if upload is not None:
                source = temporary / f'upload{suffix}'
                upload.save(source)
            else:
                source = asset(manifest[profile][example])
            try:
                if kind == 'photo':
                    image, record = detector.image(source, temporary)
                    media_name = 'result.jpg'
                    json_name = 'detections.json'
                    summary = {'count': len(record['detections'])}
                    data = image.with_suffix('.json')
                else:
                    annotated = temporary / 'annotated.mp4'
                    record = process_video(detector, source, annotated, max_frames=MAX_VIDEO_FRAMES)
                    image = temporary / 'browser.mp4'
                    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(annotated),
                                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                    str(image)], check=True, timeout=120)
                    media_name = 'result.mp4'
                    json_name = 'detections.jsonl'
                    summary = {'frames': record['frames'], 'seconds': record['duration_seconds']}
                    data = annotated.with_suffix('.jsonl')
            except VideoFrameLimitExceeded:
                return jsonify(error=f'В видео больше {MAX_VIDEO_FRAMES} кадров.'), 413
            except (ValueError, OSError, cv2.error) as error:
                app.logger.info('Could not process demo media: %s', error)
                return jsonify(error='Не удалось прочитать файл. Проверьте формат и содержимое.'), 400
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                app.logger.exception('Could not encode demo video')
                return jsonify(error='Не удалось подготовить видео для браузера.'), 500
            published = temporary / 'published'
            published.mkdir()
            image.replace(published / media_name)
            data.replace(published / json_name)
            published.replace(target)
        return jsonify(kind=kind, media=f'/result/{job}/{media_name}',
                       detections=f'/result/{job}/{json_name}', **summary)

    return app


if __name__ == '__main__':
    create_app().run(host='0.0.0.0', port=7860, threaded=True)
