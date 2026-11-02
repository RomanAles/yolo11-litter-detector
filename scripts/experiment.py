"""One isolated experiment. Never evaluates or selects on test."""
import argparse,json,hashlib,time
from pathlib import Path
import yaml
from litter_detector.detector import ROOT
from litter_detector.artifacts import write_json


def run(args):
    from ultralytics import YOLO
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    name=args.name;out=ROOT/'models/experiments'/args.profile/name
    if args.batch is None:args.batch=32 if args.profile=='underwater' and args.imgsz==640 and args.arch=='yolo11n' else 16
    if (out/'summary.json').exists():print('Already complete:',out);return
    data=Path(args.data).resolve() if args.data else ROOT/'data/v2'/args.profile/'data.yaml'
    weights=Path(args.weights) if args.weights else ROOT/'models/pretrained'/f'{args.arch}.pt'
    inputs=[data]+[data.parent/f'{split}.txt' for split in ('train','val')]
    if (data.parent/'manifest.json').exists():inputs.append(data.parent/'manifest.json')
    digests={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    changes=json.loads(args.augment) if args.augment else {}
    checkpoint=out/'weights/last.pt'
    resume=checkpoint.exists()
    resumed_epoch=None
    recover_completed=False
    out.mkdir(parents=True,exist_ok=True)
    recipe_file=out/'run_recipe.json'
    if recipe_file.exists():
        recorded=json.loads(recipe_file.read_text())
        if recorded['input_sha256']!=digests:raise RuntimeError('Experiment inputs differ from its recorded recipe')
    else:recipe_file.write_text(json.dumps({'input_sha256':digests,'arguments':vars(args)},indent=2))
    if resume:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        resumed_epoch=int(saved.get('epoch',-1))
        if saved.get('optimizer') is None:
            if (out/'weights/best.pt').exists() and (out/'results.csv').exists():recover_completed=True
            else:raise RuntimeError(f'Checkpoint optimizer absent: {checkpoint}; completed artifacts missing')
        previous=saved.get('train_args',{})
        for key,expected in [('imgsz',args.imgsz),('seed',args.seed),('epochs',args.epochs)]:
            if previous.get(key)!=expected:raise RuntimeError(f'Resume recipe mismatch for {key}: {previous.get(key)} != {expected}')
        del saved
    model=YOLO(str(out/'weights/best.pt' if recover_completed else checkpoint if resume else weights))
    def runtime_loader_settings(trainer):
        # Ultralytics resume restores workers from its checkpoint; override before loaders exist.
        trainer.args.workers=args.workers
        from ultralytics.utils import YAML
        YAML.save(trainer.save_dir/'args.yaml',vars(trainer.args))
    model.add_callback('on_pretrain_routine_start',runtime_loader_settings)
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    if recover_completed:
        result=model.val(data=str(data),split='val',imgsz=args.imgsz,batch=args.batch,device=0,workers=args.workers,
            project=str(out.parent),name=name,exist_ok=True,plots=False,verbose=False)
    else:
        result=model.train(data=str(data),epochs=args.epochs,patience=args.patience,imgsz=args.imgsz,batch=args.batch,
            device=0,workers=args.workers,amp=True,optimizer='AdamW',lr0=.001,seed=args.seed,deterministic=True,
            close_mosaic=min(10,args.epochs//4),project=str(out.parent),name=name,exist_ok=True,
            verbose=False,plots=True,cache=False,resume=resume,**changes)
    report={'name':name,'profile':args.profile,'arch':args.arch,'imgsz':args.imgsz,'seed':args.seed,
            'epochs_requested':args.epochs,'batch':args.batch,'workers':args.workers,'augmentation':changes,'data':str(data),'initial_weights':str(weights),
            'weights':str(out/'weights/best.pt'),'wall_seconds':time.perf_counter()-started,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
            'resumed_from_epoch':resumed_epoch,'wall_seconds_this_invocation':time.perf_counter()-started,
            'recovered_completed_run':recover_completed,
            'validation':{k:float(v) for k,v in result.results_dict.items()},
            'data_yaml_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'input_sha256':digests}
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in digests.items()):
        raise RuntimeError('Experiment inputs changed during training')
    snapshots=out/'input_snapshot';snapshots.mkdir(exist_ok=True)
    for path in inputs:__import__('shutil').copy2(path,snapshots/path.name)
    write_json(out/'summary.json',report)
    print('EXPERIMENT_FINISHED',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('profile',choices=['land','underwater']);p.add_argument('name')
    p.add_argument('--arch',default='yolo11n');p.add_argument('--imgsz',type=int,default=640);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--patience',type=int,default=15);p.add_argument('--batch',type=int);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--data');p.add_argument('--weights');p.add_argument('--augment');p.add_argument('--workers',type=int,default=1);run(p.parse_args())
