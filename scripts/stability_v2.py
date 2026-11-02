"""Independent seed replication and grouped, development-only CV screen."""
from collections import defaultdict,Counter
from pathlib import Path
import json,random,time,sys
import yaml
from v2_pipeline import run,score,EXPERIMENT
from litter_detector.detector import ROOT
from litter_detector.artifacts import write_json


def stability():
    output=ROOT/'outputs/v2'
    while not (output/'selected.json').exists():time.sleep(10)
    selected=json.loads((output/'selected.json').read_text())
    reports={}
    for profile,best in selected.items():
        base=best
        if best['name'].startswith('aug_'):
            base=json.loads((Path(best['initial_weights']).parents[1]/'summary.json').read_text())
        base_batch=base.get('batch',8)
        base_workers=base.get('workers',yaml.safe_load((Path(base['weights']).parents[1]/'args.yaml').read_text()).get('workers',0))
        replica=run(profile,'seed123_main',arch=base['arch'],imgsz=base['imgsz'],batch=base_batch,workers=base_workers,seed=123)
        if best['name'].startswith('aug_'):
            replica=run(profile,'seed123_post',arch=base['arch'],imgsz=base['imgsz'],batch=best.get('batch',8),seed=123,
                        epochs=15,patience=8,weights=replica['weights'],augment=json.dumps(best['augmentation']),workers=best.get('workers',0))
        manifest=json.loads((ROOT/'data/v2'/profile/'manifest.json').read_text())
        rows=[r for r in manifest['images'] if r['split']!='test']
        groups=defaultdict(list)
        for row in rows:groups[row['group']].append(row)
        keys=list(groups);random.Random(81).shuffle(keys);keys.sort(key=lambda k:len(groups[k]),reverse=True)
        folds=[[],[],[]];sizes=[0,0,0]
        for key in keys:
            index=min(range(3),key=lambda i:sizes[i]);folds[index].append(key);sizes[index]+=len(groups[key])
        summaries=[]
        for fold in range(3):
            folder=ROOT/'data/v2/folds'/profile/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
            validation=set(folds[fold]);train=[r for r in rows if r['group'] not in validation];val=[r for r in rows if r['group'] in validation]
            assert not ({r['group'] for r in train}&{r['group'] for r in val})
            assert all(r['split']!='test' for r in train+val)
            assert train and val and any(r['labels'] for r in train) and any(r['labels'] for r in val)
            for split,subset in [('train',train),('val',val)]:
                (folder/f'{split}.txt').write_text('\n'.join(r['image_path'] for r in subset)+'\n')
            config={'path':str(folder),'train':'train.txt','val':'val.txt','names':{0:'trash'}}
            (folder/'data.yaml').write_text(yaml.safe_dump(config))
            (folder/'groups.json').write_text(json.dumps({'validation_groups':sorted(validation),'train_images':len(train),'val_images':len(val),'test_excluded':True},indent=2))
            summaries.append(run(profile,f'groupcv_{fold}',arch=base['arch'],imgsz=base['imgsz'],batch=base_batch,
                seed=42,epochs=20,patience=8,data=str(folder/'data.yaml'),augment=json.dumps(base.get('augmentation',{}))))
        reports[profile]={'seed42':best,'seed123':replica,'grouped_cv':summaries,
            'limitation':'Two full-training seeds are descriptive, not a confidence interval. Three grouped folds use only 20 epochs and are a robustness screen, not full-budget CV.'}
        write_json(output/f'{profile}_stability.json',reports[profile])
        print('STABILITY_COMPLETE',profile,flush=True)
    write_json(output/'stability.json',reports)


if __name__=='__main__':stability()
