from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Поиск мусора на фото и видео")
    parser.add_argument("mode", choices=["image", "folder", "video", "camera"])
    parser.add_argument("source", help="Файл, папка или индекс обычной веб-камеры")
    parser.add_argument("--profile", choices=["land", "underwater"], default="land")
    parser.add_argument("--output", default="outputs/predict")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--model")
    parser.add_argument("--device", help="cpu или индекс GPU, например 0")
    parser.add_argument("--process-fps", type=float, default=15)
    parser.add_argument('--operating-mode',choices=['balanced','recall','precision'],default='balanced')
    parser.add_argument('--sliced',action='store_true',help='SAHI: обработка фотографий по фрагментам')
    args = parser.parse_args()
    try:
        from .detector import Detector
        if args.sliced and args.mode=='camera':raise ValueError('SAHI предназначен для файлов, не для веб-камеры')
        detector = Detector(args.profile, args.conf, args.model, args.device,args.operating_mode,args.sliced)
        if args.mode == "image":
            target, record = detector.image(args.source, args.output)
            print(json.dumps({"output": str(target), "detections": len(record["detections"])}, ensure_ascii=False))
        elif args.mode == "folder":
            from .media import process_folder
            print("Обработано:", process_folder(detector, args.source, args.output))
        elif args.mode == "video":
            from .media import process_video
            print(json.dumps(process_video(detector, args.source, Path(args.output)/"detected.mp4"), ensure_ascii=False))
        else:
            from .camera import run_camera
            run_camera(detector, int(args.source), args.process_fps)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
