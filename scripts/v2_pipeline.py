"""Long-running experiment stages; results and checkpoints are persisted per run.

This runner does not touch application configs or evaluate the historical test set.
"""
import subprocess,json,time,sys
import yaml
from litter_detector.detector import ROOT
from litter_detector.artifacts import write_json
from litter_detector.experiments import experiment_folder, validate_existing_run

PROJECT_PYTHON = ROOT / '.venv/python.exe'
PYTHON = str(PROJECT_PYTHON) if PROJECT_PYTHON.is_file() else sys.executable
EXPERIMENT=ROOT/'models/experiments'


def completed(profile,name):return EXPERIMENT/profile/name/'summary.json'


def run(profile,name,*,python=PYTHON,**kwargs):
    folder = experiment_folder(ROOT, profile, name)
    arguments = {'profile': profile, 'name': name, **kwargs}
    summary = validate_existing_run(folder, arguments, ROOT)
    if summary is not None:
        return summary
    command=[str(python),str(ROOT/'scripts/experiment.py'),profile,name]
    for key, value in kwargs.items():
        if value is not None:
            command += ['--' + key.replace('_', '-'), json.dumps(value) if isinstance(value, dict) else str(value)]
    log=ROOT/'.cache'/f'v2-{profile}-{name}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w',encoding='utf8') as output:
        subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
    summary = validate_existing_run(folder, arguments, ROOT)
    if summary is None:
        raise RuntimeError(f'Training did not produce a summary: {folder}')
    return summary


def score(report):return report['validation']['metrics/mAP50-95(B)']


def pipeline():
    state=ROOT/'outputs/v2';state.mkdir(parents=True,exist_ok=True)
    # Initial long runs are launched sequentially to keep host RAM usage bounded.
    for profile in ('land','underwater'):
        run(profile,'n640_long')
    selected={}
    for profile in ('land','underwater'):
        n640=json.loads(completed(profile,'n640_long').read_text())
        n960=run(profile,'n960_long',imgsz=960,batch=16 if profile=='underwater' else 8)
        minimum_gain=yaml.safe_load((ROOT/'configs/experiment_protocol.yaml').read_text())['selection']['minimum_gain_for_larger_resolution']
        best_n=n960 if score(n960)>=score(n640)+minimum_gain else n640
        s=run(profile,'s_compare',arch='yolo11s',imgsz=best_n['imgsz'],batch=8)
        best=max([best_n,s],key=score)
        # Matched 15-epoch fine-tuning control and single changed augmentation.
        control=run(profile,'aug_control',arch=best['arch'],imgsz=best['imgsz'],batch=8,
                    epochs=15,patience=8,weights=best['weights'])
        aug={'hsv_v':0.25} if profile=='land' else {'hsv_s':0.35}
        variant=run(profile,'aug_variant',arch=best['arch'],imgsz=best['imgsz'],batch=8,
                    epochs=15,patience=8,weights=best['weights'],augment=json.dumps(aug))
        best=max([best,control,variant],key=score)
        selected[profile]=best
        write_json(state/f'{profile}_selected.json',best)
        print('SELECTED',profile,best['name'],score(best),flush=True)
    write_json(state/'selected.json',selected)
    print('MAIN_EXPERIMENTS_COMPLETE',flush=True)


if __name__=='__main__':pipeline()
