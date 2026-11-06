"""Sequential final checks after training; resumable through cached predictions.

Run with --wait-for-training when the separate training runners are active.
No camera source is opened by this script or its checks.
"""
import argparse,json,subprocess,sys,time
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

from litter_detector.artifacts import write_json,write_text
from litter_detector.detector import ROOT


def portable_requirements(frozen):
    lock=[]
    for line in frozen.splitlines():
        if line.startswith('# Editable install'):continue
        name,separator,url=line.partition(' @ ')
        if separator and urlsplit(url).scheme=='file':
            line=f'{name}=={version(name)}'
        lock.append('-e .' if line.startswith('-e ') else line)
    if '-e .' not in lock:raise RuntimeError('Editable project installation is missing')
    return '\n'.join(lock)+'\n'


def complete(wait=False):
    output=ROOT/'outputs/v2';output.mkdir(parents=True,exist_ok=True)
    required=[output/'selected.json',output/'stability.json']
    while not all(path.is_file() for path in required):
        if not wait:raise RuntimeError('Main experiments and stability must finish before final evaluation')
        time.sleep(10)
    selected=json.loads(required[0].read_text());stability=json.loads(required[1].read_text())
    assert set(selected)==set(stability)=={'land','underwater'}
    for profile in selected:
        assert stability[profile]['seed42']['weights']==selected[profile]['weights']
        assert len(stability[profile]['grouped_cv'])==3
    state={'completed':[],'physical_camera_opened':False}
    stages=[
        ('finalize',['scripts/finalize_v2.py']),
        ('export',['scripts/export_check_v2.py']),
        ('file_verification',['scripts/verify_v2.py']),
        ('software_tests',['-m','pytest','-q','--basetemp','.cache/pytest-v2-final']),
        ('ui_smoke',['-m','litter_detector.ui','--smoke-test']),
        ('dependencies',['-m','pip','check']),
    ]
    for name,arguments in stages:
        log=ROOT/'.cache'/f'v2-final-{name}.log'
        print('FINAL_STAGE_START',name,flush=True)
        with log.open('w',encoding='utf8') as handle:
            subprocess.run([sys.executable,*arguments],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(name)
        write_json(output/'completion.json',state)
        print('FINAL_STAGE_COMPLETE',name,flush=True)
    frozen=subprocess.check_output([sys.executable,'-m','pip','freeze'],cwd=ROOT,text=True,encoding='utf8')
    write_text(ROOT/'requirements-lock.txt',portable_requirements(frozen))
    import platform,torch,ultralytics
    write_json(output/'runtime.json',{'python':sys.version,'platform':platform.platform(),
        'torch':torch.__version__,'ultralytics':ultralytics.__version__,
        'gpu':torch.cuda.get_device_name(0),'cuda_runtime':torch.version.cuda,
        'timing_scope':'Sequential offline file inference; no concurrent training; excludes image decoding and drawing.'})
    subprocess.run([sys.executable,'scripts/report_v2.py'],cwd=ROOT,check=True)
    state['completed'].extend(['requirements_lock','report'])
    state['finished_utc']=datetime.now(timezone.utc).isoformat()
    write_json(output/'completion.json',state)
    print('ALL_V2_CHECKS_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-training',action='store_true')
    complete(parser.parse_args().wait_for_training)
