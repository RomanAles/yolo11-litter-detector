from __future__ import annotations

import json
from pathlib import Path
import time

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def process_folder(detector, source, output, stop=None, callback=None):
    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise NotADirectoryError(source)
    count = 0
    errors = []
    # Snapshot inputs before creating outputs; preserve subfolders and avoid name collisions.
    files = sorted(f for f in source.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()
                   and not f.resolve().is_relative_to(output.resolve()))
    for file in files:
        if stop is not None and stop.is_set():
            break
        try:
            target, record = detector.image(file, output / file.relative_to(source).parent / file.suffix[1:])
            count += 1
            if callback:
                callback(target, record)
        except (ValueError, OSError) as error:
            errors.append({"source": str(file), "error": str(error)})
    output.mkdir(parents=True, exist_ok=True)
    (output / "batch_summary.json").write_text(json.dumps({"processed": count, "errors": errors},
                                                         ensure_ascii=False, indent=2), encoding="utf-8")
    if not files:
        raise ValueError("В папке нет поддерживаемых изображений")
    return count


def process_video(detector, source, output, stop=None, callback=None):
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise ValueError("Выходное видео не должно перезаписывать входное")
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    writer = None
    count = 0
    started = time.perf_counter()
    try:
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not 0 < fps <= 1000:
            raise ValueError("Не удалось определить FPS; требуется видео с корректными метаданными")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise OSError("Не удалось создать выходное видео")
        with output.with_suffix(".jsonl").open("w", encoding="utf-8") as log:
            while stop is None or not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, record = detector.predict(frame, frame_id=count, source_time=count/fps)
                writer.write(annotated)
                log.write(json.dumps(record, ensure_ascii=False)+"\n")
                if callback:
                    callback(annotated, record)
                count += 1
        if not count:
            raise ValueError("Видео не содержит читаемых кадров")
        summary = {"source": str(source), "output": str(output), "frames": count, "fps": fps,
                   "duration_seconds": count/fps, "wall_seconds": time.perf_counter()-started,
                   "audio_preserved": False, "timestamp_policy": "CFR using source FPS",
                   "cancelled": bool(stop is not None and stop.is_set())}
        output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        cap.release()
        if writer is not None:
            writer.release()
