"""Reproducible full-dataset YOLO11n training; resume uses the saved optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from litter_detector.detector import ROOT


def train(profile, epochs=30, resume=False):
    from ultralytics import YOLO
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна; обучение не запускается незаметно на CPU")
    dataset = ROOT / "data/prepared" / profile / "data.yaml"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    base = ROOT / "models/pretrained/yolo11n.pt"
    if not base.is_file():
        raise FileNotFoundError(base)
    target = ROOT / "models" / profile
    manifest = ROOT / "data/prepared" / profile / "manifest.json"
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    checkpoint = target / "weights/last.pt" if resume else base
    if not resume and (target / "weights/best.pt").exists():
        raise FileExistsError(f"Model already trained: {target}; use --resume or move it explicitly")
    model = YOLO(str(checkpoint))
    result = model.train(data=str(dataset), epochs=epochs, imgsz=640, batch=16, device=0,
                         workers=2, seed=42, deterministic=True, patience=8, amp=True,
                         project=str(ROOT/"models"), name=profile, exist_ok=True,
                         plots=True, cache=False, resume=resume, close_mosaic=5,
                         save=True, verbose=False)
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_sha:
        raise RuntimeError("Dataset manifest changed during training")
    shutil.copy2(manifest,target/'dataset_manifest.json')
    report = {"profile": profile, "architecture": "YOLO11n", "requested_epochs": epochs,
              "dataset_manifest_sha256": manifest_sha,
              "pretrained_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
              "seed": 42, "device": torch.cuda.get_device_name(0),
              "best_weights": str(target/"weights/best.pt"),
              "validation_metrics": {k: float(v) for k,v in result.results_dict.items()}}
    (target/"training_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("TRAINING_FINISHED", profile, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=["land", "underwater", "all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for profile in ("land", "underwater") if args.profile == "all" else (args.profile,):
        train(profile, args.epochs, args.resume)
