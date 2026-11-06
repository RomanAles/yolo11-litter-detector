from argparse import Namespace
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from litter_detector.experiments import normalize_arguments, validate_existing_run, validate_training_settings
from scripts import experiment, v2_pipeline


@pytest.fixture
def recorded_run(tmp_path):
    arguments = {'profile': 'land', 'name': 'review'}
    normalized = normalize_arguments(arguments, tmp_path)
    arguments = {**normalized, 'augment': None}
    data = tmp_path/'data/v2/land'
    data.mkdir(parents=True)
    for name in ('data.yaml', 'train.txt', 'val.txt', 'manifest.json'):
        (data/name).write_text('fixture')
    digests = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in data.iterdir()}
    out = tmp_path/'models/experiments/land/review'
    (out/'weights').mkdir(parents=True)
    (out/'weights/last.pt').write_bytes(b'checkpoint')
    (out/'run_recipe.json').write_text(json.dumps({'arguments': arguments, 'input_sha256': digests}))
    return tmp_path, out, arguments, digests


@pytest.mark.parametrize('changed', [
    {'arch': 'yolo11s'}, {'augment': '{"hsv_v": 0.1}'}, {'patience': 8},
    {'weights': 'different.pt'}, {'batch': 8}, {'epochs': 15}, {'seed': 123},
])
def test_resume_rejects_changed_recipe_before_loading_gpu_or_checkpoint(recorded_run, monkeypatch, changed):
    root, _, arguments, _ = recorded_run
    monkeypatch.setattr(experiment, 'ROOT', root)
    with pytest.raises(RuntimeError, match='recipe mismatch'):
        experiment.run(Namespace(**{**arguments, **changed}))


def test_semantically_equal_augmentation_and_worker_tuning_are_allowed(recorded_run):
    root, out, arguments, digests = recorded_run
    recipe = json.loads((out/'run_recipe.json').read_text())
    recipe['arguments']['augment'] = '{"hsv_v": 0.1, "hsv_s": 0.2}'
    (out/'run_recipe.json').write_text(json.dumps(recipe))
    requested = {**arguments, 'workers': 0, 'augment': '{"hsv_s": 0.2, "hsv_v": 0.1}'}
    assert validate_existing_run(out, requested, root, digests) is None


@pytest.mark.parametrize('entrypoint', ['experiment', 'pipeline'])
def test_completed_run_cannot_bypass_recipe_validation(recorded_run, monkeypatch, entrypoint):
    root, out, arguments, _ = recorded_run
    (out/'summary.json').write_text('{}')
    if entrypoint == 'experiment':
        monkeypatch.setattr(experiment, 'ROOT', root)
        call = lambda: experiment.run(Namespace(**{**arguments, 'arch': 'yolo11s'}))
    else:
        monkeypatch.setattr(v2_pipeline, 'ROOT', root)
        monkeypatch.setattr(v2_pipeline, 'EXPERIMENT', root/'models/experiments')
        call = lambda: v2_pipeline.run('land', 'review', arch='yolo11s')
    with pytest.raises(RuntimeError, match='recipe mismatch for arch'):
        call()


@pytest.mark.parametrize('has_batch', [True, False])
def test_legacy_completed_run_uses_saved_training_arguments(recorded_run, has_batch):
    root, out, arguments, digests = recorded_run
    (out/'run_recipe.json').unlink()
    summary = {'profile': 'land', 'name': 'review', 'arch': 'yolo11n', 'imgsz': 640,
               'epochs_requested': 80, 'batch': 16, 'seed': 42, 'data': arguments['data'],
               'initial_weights': arguments['weights'], 'augmentation': {}, 'input_sha256': digests}
    if not has_batch:summary.pop('batch')
    (out/'summary.json').write_text(json.dumps(summary))
    (out/'args.yaml').write_text(yaml.safe_dump({'patience': 15, 'batch': 16}))
    assert validate_existing_run(out, arguments, root) == summary
    with pytest.raises(RuntimeError, match='recipe mismatch for patience'):
        validate_existing_run(out, {**arguments, 'patience': 8}, root)


def test_changed_inputs_are_rejected_even_for_completed_run(recorded_run):
    root, out, arguments, _ = recorded_run
    (out/'summary.json').write_text('{}')
    (root/'data/v2/land/train.txt').write_text('different cohort')
    with pytest.raises(RuntimeError, match='inputs differ'):
        validate_existing_run(out, arguments, root)


def test_checkpoint_settings_are_verified_beyond_seed_and_resolution():
    with pytest.raises(RuntimeError, match='Resume recipe mismatch for hsv_v'):
        validate_training_settings({'imgsz': 640, 'hsv_v': 0.4}, {'imgsz': 640, 'hsv_v': 0.1})


def test_matching_resume_records_effective_worker_settings(recorded_run, monkeypatch):
    root, out, arguments, _ = recorded_run
    saved = {'data': arguments['data'], 'epochs': 80, 'patience': 15, 'imgsz': 640,
             'batch': 16, 'device': '0', 'workers': 2, 'amp': True, 'optimizer': 'AdamW',
             'lr0': .001, 'seed': 42, 'deterministic': True, 'close_mosaic': 10}

    class Model:
        def __init__(self, path):
            assert path == str(out/'weights/last.pt')

        def add_callback(self, event, callback):
            self.callback = callback

        def train(self, **kwargs):
            assert kwargs['resume'] is True
            trainer = SimpleNamespace(args=Namespace(**saved), save_dir=out)
            self.callback(trainer)
            assert trainer.args.workers == 0
            return SimpleNamespace(results_dict={'metrics/mAP50-95(B)': .3})

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda: None,
                             max_memory_allocated=lambda: 0),
        load=lambda *a, **kw: {'epoch': 2, 'optimizer': {}, 'train_args': saved},
    )
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=Model))
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', SimpleNamespace(YAML=SimpleNamespace(save=lambda *a: None)))
    monkeypatch.setattr(experiment, 'ROOT', root)
    experiment.run(Namespace(**{**arguments, 'workers': 0}))
    report = json.loads((out/'summary.json').read_text())
    assert report['workers'] == report['training_arguments']['workers'] == 0
    assert report['training_arguments']['patience'] == 15
    assert report['validation']['metrics/mAP50-95(B)'] == .3
