from __future__ import annotations

import json
import hashlib
from pathlib import Path
import threading
import time

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_image(path: str | Path):
    import numpy as np
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return image


def write_image(path: str | Path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise OSError(f"Не удалось закодировать изображение: {path}")
    encoded.tofile(path)


class Detector:
    def __init__(self, profile="land", confidence=None, model_path=None, device=None, operating_mode="balanced", sliced=False):
        from ultralytics import YOLO
        import torch
        config_path = ROOT / "configs" / f"{profile}.yaml"
        if profile not in ("land", "underwater"):
            raise ValueError("Неизвестный профиль")
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.profile = profile
        if operating_mode not in ('balanced','recall','precision'):
            raise ValueError('Неизвестный режим порога')
        self.operating_mode=operating_mode
        preset_key='sahi_operating_modes' if sliced else 'operating_modes'
        preset=self.config.get(preset_key,{}).get(operating_mode,self.config['confidence'])
        self.confidence = float(confidence if confidence is not None else preset)
        if not 0 < self.confidence <= 1:
            raise ValueError("Confidence должен быть в диапазоне (0, 1]")
        self.weights = Path(model_path) if model_path else ROOT / self.config["model"]
        if not self.weights.is_file():
            raise FileNotFoundError(f"Нет весов профиля {profile}: {self.weights}. Сначала выполните обучение.")
        self.weight_sha256=hashlib.sha256(self.weights.read_bytes()).hexdigest()
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(str(self.weights))
        self.sliced=sliced
        if sliced:
            from sahi import AutoDetectionModel
            sahi_device='cpu' if self.device=='cpu' else (f'cuda:{self.device}' if str(self.device).isdigit() else str(self.device))
            self.sahi_model=AutoDetectionModel.from_pretrained(model_type='ultralytics',model_path=str(self.weights),
                confidence_threshold=self.confidence,device=sahi_device,image_size=self.config['imgsz'])
        self._lock = threading.Lock()

    def predict(self, image, frame_id=0, source_time=None):
        if image is None or not image.size:
            raise ValueError("Пустой кадр")
        started = time.perf_counter()
        if self.sliced:
            return self._predict_sliced(image,frame_id,source_time,started)
        with self._lock:
            result = self.model.predict(image, conf=self.confidence, iou=self.config["iou"],
                                        imgsz=self.config["imgsz"], device=self.device,
                                        verbose=False)[0]
        objects = []
        if result.boxes is not None:
            for xyxy, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                        result.boxes.cls.cpu().tolist()):
                objects.append({"class_id": int(cls), "class_name": result.names[int(cls)],
                                "confidence": round(score, 6), "xyxy": [round(v, 2) for v in xyxy]})
        record = {"profile": self.profile, "model": self.weights.name, "frame_id": frame_id,
                  "operating_mode":self.operating_mode,"sliced":False,
                  "model_sha256":self.weight_sha256,
                  "source_time_seconds": source_time, "width": image.shape[1], "height": image.shape[0],
                  "inference_ms": round((time.perf_counter()-started)*1000, 3), "detections": objects}
        return result.plot(), record

    def _predict_sliced(self,image,frame_id,source_time,started):
        from sahi.predict import get_sliced_prediction
        with self._lock:
            tile=self.config.get('sahi_tile_size',640)
            result=get_sliced_prediction(image[:,:,::-1],self.sahi_model,slice_height=tile,slice_width=tile,
                overlap_height_ratio=.2,overlap_width_ratio=.2,perform_standard_pred=True,
                postprocess_type='NMS',postprocess_match_metric='IOU',postprocess_match_threshold=.5,verbose=0)
        objects=[]
        for obj in result.object_prediction_list:
            objects.append({'class_id':int(obj.category.id),'class_name':str(obj.category.name),'confidence':round(float(obj.score.value),6),
                            'xyxy':[round(float(v),2) for v in obj.bbox.to_xyxy()]})
        record={'profile':self.profile,'model':self.weights.name,'frame_id':frame_id,'source_time_seconds':source_time,
                'model_sha256':self.weight_sha256,
                'width':image.shape[1],'height':image.shape[0],'inference_ms':round((time.perf_counter()-started)*1000,3),
                'detections':objects,'sliced':True,'operating_mode':self.operating_mode}
        draw=image.copy()
        for obj in objects:
            x1,y1,x2,y2=map(round,obj['xyxy']);cv2.rectangle(draw,(x1,y1),(x2,y2),(255,120,0),2)
            cv2.putText(draw,f"trash {obj['confidence']:.2f}",(x1,max(16,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,120,0),2)
        return draw,record

    def image(self, source, output_dir):
        source, output_dir = Path(source), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated, record = self.predict(read_image(source))
        record["source"] = str(source.resolve())
        target = output_dir / f"{source.stem}_detected.jpg"
        write_image(target, annotated)
        target.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return target, record
