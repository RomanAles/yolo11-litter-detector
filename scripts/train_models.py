"""Reproducible full-dataset YOLO11n training; resume uses the saved optimizer."""
import argparse
from pathlib import Path
import shutil

from filelock import FileLock

from litter_detector.artifacts import write_json
from litter_detector.detector import ROOT
from litter_detector.experiments import experiment_folder, file_sha256, validate_training_settings


def train(profile, epochs=30, resume=False, *, seed=42, run_name=None):
    if profile not in ('land', 'underwater'):
        raise ValueError(f'Unsupported profile: {profile}')
    target = ROOT / "models" / profile if run_name is None else experiment_folder(ROOT, profile, run_name)
    target.mkdir(parents=True, exist_ok=True)
    with FileLock(target / '.training.lock', timeout=0):
        return _train(profile, epochs, resume, seed, target)


def _train(profile, epochs, resume, seed, target):
    from ultralytics import YOLO
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна; обучение не запускается незаметно на CPU")
    dataset = ROOT / "data/prepared" / profile / "data.yaml"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    base = ROOT / "models/pretrained/yolo11n.pt"
    manifest = ROOT / "data/prepared" / profile / "manifest.json"
    snapshot = target / 'dataset_manifest.json'
    manifest_sha = file_sha256(manifest)
    dataset_sha = file_sha256(dataset)
    base_sha = file_sha256(base)
    checkpoint = target / "weights/last.pt" if resume else base
    best = target / 'weights/best.pt'
    if not resume and (best.exists() or (target / 'weights/last.pt').exists()):
        raise FileExistsError(f"Model already trained: {target}; use --resume or move it explicitly")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    settings = dict(data=str(dataset), epochs=epochs, imgsz=640, batch=16, device=0,
                    workers=2, seed=seed, deterministic=True, patience=8, amp=True,
                    project=str(target.parent), name=target.name, close_mosaic=5)
    recover_completed = False
    if resume:
        if snapshot.is_file() and file_sha256(snapshot) != manifest_sha:
            raise RuntimeError('Dataset manifest changed since this run started')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        validate_training_settings(saved['train_args'], settings)
        recover_completed = saved.get('optimizer') is None
        if recover_completed and (not best.is_file() or not (target / 'results.csv').is_file()):
            raise RuntimeError(f'Checkpoint optimizer absent and completed artifacts missing: {checkpoint}')
        del saved
    shutil.copy2(manifest, snapshot)
    model = YOLO(str(best if recover_completed else checkpoint))
    if recover_completed:
        result = model.val(data=str(dataset), split='val', imgsz=640, batch=16, device=0,
                           workers=2, project=str(target.parent), name=target.name,
                           exist_ok=True, plots=False, verbose=False)
    else:
        result = model.train(**settings, exist_ok=True, plots=True, cache=False,
                             resume=resume, save=True, verbose=False)
    if file_sha256(manifest) != manifest_sha or file_sha256(dataset) != dataset_sha or file_sha256(base) != base_sha:
        raise RuntimeError("Dataset or pretrained weights changed during training")
    if not best.is_file():
        raise FileNotFoundError(best)
    report = {"profile": profile, "architecture": "YOLO11n", "requested_epochs": epochs,
              "dataset_manifest_sha256": manifest_sha,
              "pretrained_sha256": base_sha,
              "seed": seed, "device": torch.cuda.get_device_name(0),
              "best_weights": str(target/"weights/best.pt"),
              "validation_metrics": {k: float(v) for k,v in result.results_dict.items()}}
    write_json(target / 'training_summary.json', report)
    print("TRAINING_FINISHED", profile, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=["land", "underwater", "all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    for profile in ("land", "underwater") if args.profile == "all" else (args.profile,):
        train(profile, args.epochs, args.resume, seed=args.seed, run_name=args.run_name)
