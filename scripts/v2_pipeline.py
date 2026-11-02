"""Long-running experiment stages; results and checkpoints are persisted per run.

This runner does not touch application configs or evaluate the historical test set.
"""
from pathlib import Path
import subprocess,json,time,sys,hashlib
import yaml
from litter_detector.detector import ROOT
from litter_detector.artifacts import write_json

PYTHON=str(ROOT/'.venv/python.exe')
EXPERIMENT=ROOT/'models/experiments'


def completed(profile,name):return EXPERIMENT/profile/name/'summary.json'


def run(profile,name,**kwargs):
    if completed(profile,name).exists():
        result=json.loads(completed(profile,name).read_text())
        checks=result.get('input_sha256',{result['data']:result['data_yaml_sha256']})
        if any(not Path(path).exists() or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in checks.items()):
            raise RuntimeError(f'Inputs of completed experiment {profile}/{name} have changed; use a new versioned run')
        return result
    command=[PYTHON,str(ROOT/'scripts/experiment.py'),profile,name]
    for key,value in kwargs.items():command += ['--'+key.replace('_','-'),str(value)]
    log=ROOT/'.cache'/f'v2-{profile}-{name}.log'
    with log.open('w',encoding='utf8') as output:
        subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
    return json.loads(completed(profile,name).read_text())


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
