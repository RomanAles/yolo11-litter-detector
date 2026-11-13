"""Validate experiment identity before resuming or reusing completed artifacts."""
import hashlib
import json
from pathlib import Path
import re

import yaml


DEFAULT_ARGUMENTS = {
    'arch': 'yolo11n', 'imgsz': 640, 'epochs': 80, 'patience': 15,
    'batch': None, 'seed': 42, 'data': None, 'weights': None,
    'augment': None, 'workers': 1, 'lr0': .001,
}


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def experiment_folder(root, profile, name):
    if profile not in ('land', 'underwater') or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Use a supported profile and a run name containing only letters, digits, "_" or "-"')
    folder = root / 'models' / 'experiments' / profile / name
    if not folder.resolve().is_relative_to((root / 'models' / 'experiments').resolve()):
        raise ValueError(f'Experiment folder escapes models/experiments: {folder}')
    return folder


def normalize_arguments(arguments, root):
    values = {**DEFAULT_ARGUMENTS, **arguments}
    if values['batch'] is None:
        values['batch'] = 32 if (values['profile'] == 'underwater'
                                and values['imgsz'] == 640 and values['arch'] == 'yolo11n') else 16
    values['data'] = str(Path(values['data'] or root/'data/v2'/values['profile']/'data.yaml').resolve())
    values['weights'] = str(Path(values['weights'] or root/'models/pretrained'/f"{values['arch']}.pt").resolve())
    changes = values['augment'] or {}
    if isinstance(changes, str):
        changes = json.loads(changes)
    if not isinstance(changes, dict):
        raise ValueError('Augment must be a JSON object')
    values['augment'] = changes
    return values


def validate_existing_run(folder, arguments, root, input_sha256=None):
    """Allow worker tuning; changes to the training recipe require a new run name."""
    folder = Path(folder)
    summary_file = folder/'summary.json'
    recipe_file = folder/'run_recipe.json'
    summary = json.loads(summary_file.read_text(encoding='utf-8')) if summary_file.exists() else None
    initial_weights_sha256 = None
    if recipe_file.exists():
        recipe = json.loads(recipe_file.read_text(encoding='utf-8'))
        recorded = recipe['arguments']
        checks = recipe['input_sha256']
        initial_weights_sha256 = recipe.get('initial_weights_sha256')
    elif summary is not None:
        # Older completed runs predate run_recipe.json. Recover their actual settings.
        config_file = folder/'args.yaml'
        if not config_file.is_file():
            raise RuntimeError(f'Cannot verify completed experiment without args.yaml: {folder}')
        config = yaml.safe_load(config_file.read_text(encoding='utf-8'))
        recorded = {
            'profile': summary['profile'], 'name': summary['name'], 'arch': summary['arch'],
            'imgsz': summary['imgsz'], 'epochs': summary['epochs_requested'],
            'patience': config['patience'], 'batch': summary.get('batch', config.get('batch')), 'seed': summary['seed'],
            'data': summary['data'], 'weights': summary['initial_weights'],
            'augment': summary['augmentation'], 'lr0': config.get('lr0', DEFAULT_ARGUMENTS['lr0']),
        }
        if recorded['batch'] is None:
            raise RuntimeError(f'Cannot verify completed experiment without its batch size: {folder}')
        checks = summary.get('input_sha256')
        if checks is None:
            checks = {summary['data']: summary['data_yaml_sha256']}
    else:
        if (folder/'weights/last.pt').exists():
            raise RuntimeError(f'Cannot resume an experiment without its recorded recipe: {folder}')
        return None
    requested = normalize_arguments(arguments, root)
    previous = normalize_arguments(recorded, root)
    for key, value in requested.items():
        # Workers affect loader resource use and are explicitly overridden on resume.
        if key != 'workers' and previous.get(key) != value:
            raise RuntimeError(f'Experiment recipe mismatch for {key}: {previous.get(key)!r} != {value!r}; use a new run name')
    for path, digest in checks.items():
        file = Path(path)
        if not file.is_file() or file_sha256(file) != digest:
            raise RuntimeError(f'Experiment inputs differ from its recorded recipe: {path}')
    if initial_weights_sha256 is not None:
        weights = Path(requested['weights'])
        if not weights.is_file() or file_sha256(weights) != initial_weights_sha256:
            raise RuntimeError(f'Initial weights differ from the recorded recipe: {weights}')
    if input_sha256 is not None and checks != input_sha256:
        raise RuntimeError('Experiment inputs differ from its recorded recipe')
    if summary is not None and 'weights' in summary:
        best = Path(summary['weights'])
        digest = summary.get('weight_sha256')
        if not best.is_file() or (digest is not None and file_sha256(best) != digest):
            raise RuntimeError(f'Best checkpoint missing or changed for completed run: {best}')
    return summary


def validate_training_settings(actual, expected):
    for key, value in expected.items():
        if key not in ('workers', 'device') and actual.get(key) != value:
            raise RuntimeError(f'Resume recipe mismatch for {key}: {actual.get(key)!r} != {value!r}')
