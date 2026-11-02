"""Versioned data, full TACO originals, unchanged held-out source groups."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict,Counter
from pathlib import Path
import hashlib,json,shutil
import cv2
import numpy as np
import yaml
from litter_detector.detector import ROOT,write_image
from litter_detector.datasets import load_coco_pixels


def prepare(profile):
    old=ROOT/'data/prepared'/profile
    manifest=json.loads((old/'manifest.json').read_text(encoding='utf8'))
    output=ROOT/'data/v2'/profile;output.mkdir(parents=True,exist_ok=True)
    def convert(row):
        row=dict(row)
        if profile=='land':
            original=row['source'].replace('data\\raw\\taco\\','data\\raw\\taco_full\\').replace('data/raw/taco/','data/raw/taco_full/')
            source=ROOT/original
            # Windows manifest paths are platform-native.
            if not source.exists():source=ROOT/'data/raw/taco_full'/row['source'].split('taco\\')[-1]
            pixels=load_coco_pixels(source,{'width':row.get('annotation_width',row['width']),'height':row.get('annotation_height',row['height'])})
            image=output/'images'/row['split']/row['name']
            row['full_source_pixel_sha256']=hashlib.sha256(pixels.tobytes()).hexdigest()
            if max(pixels.shape[:2])>1600:
                scale=1600/max(pixels.shape[:2]);pixels=cv2.resize(pixels,(round(pixels.shape[1]*scale),round(pixels.shape[0]*scale)),interpolation=cv2.INTER_AREA)
            if not image.exists():
                write_image(image,pixels)
            # Audit the encoded file also when an earlier invocation already created it.
            from litter_detector.detector import read_image
            pixels=read_image(image)
            row['full_source']=str(source.relative_to(ROOT));row['width']=pixels.shape[1];row['height']=pixels.shape[0]
            row['pixel_sha256']=hashlib.sha256(pixels.tobytes()).hexdigest()
            row['file_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
        else:
            image=old/'images'/row['split']/row['name']
        row['image_path']=str(image.resolve())
        label=output/'labels'/row['split']/Path(row['name']).with_suffix('.txt')
        if profile=='underwater':
            # Image paths must have sibling labels; use hard links in the versioned tree.
            new_image=output/'images'/row['split']/row['name'];new_image.parent.mkdir(parents=True,exist_ok=True)
            if not new_image.exists():
                try:__import__('os').link(image,new_image)
                except OSError:shutil.copy2(image,new_image)
            row['image_path']=str(new_image.resolve())
        label.parent.mkdir(parents=True,exist_ok=True);label.write_text('\n'.join(row['labels'])+('\n' if row['labels'] else ''),encoding='utf8')
        return row
    with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(convert,manifest['images']))
    # Protect held-out images: drop train/val duplicates of later held-out splits, never move test.
    by_hash=defaultdict(list)
    for row in rows:by_hash[row['pixel_sha256']].append(row)
    dropped=[];priority={'test':2,'val':1,'train':0}
    for group in by_hash.values():
        best=max(priority[r['split']] for r in group)
        for row in group:
            if priority[row['split']]<best:row['excluded']='duplicate of held-out image';dropped.append(row['name'])
    rows=[r for r in rows if not r.get('excluded')]
    for split in ('train','val','test'):
        files=[r['image_path'] for r in rows if r['split']==split]
        (output/f'{split}.txt').write_text('\n'.join(files)+'\n',encoding='utf8')
    config={'path':str(output.resolve()),'train':'train.txt','val':'val.txt','test':'test.txt','names':{0:'trash'}}
    (output/'data.yaml').write_text(yaml.safe_dump(config),encoding='utf8')
    summary={s:{'images':sum(r['split']==s for r in rows),'objects':sum(len(r['labels']) for r in rows if r['split']==s),'negatives':sum(not r['labels'] for r in rows if r['split']==s)} for s in ('train','val','test')}
    audit={'profile':profile,'images':rows,'summary':summary,'excluded_cross_split_duplicates':dropped,'split_policy':'v1 source groups preserved; full decoded pixel duplicate exclusion; no test-to-train transfer'}
    (output/'manifest.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    # A full decode audit, not merely the existence of a file.
    for group in by_hash.values():
        assert len({r['split'] for r in group if not r.get('excluded')})<=1
    for row in rows:
        for line in row['labels']:
            cls,x,y,w,h=map(float,line.split());assert cls==0 and 0<w<=1 and 0<h<=1 and x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1+1e-6 and y+h/2<=1+1e-6
    print(profile,json.dumps(summary),'excluded duplicates',len(dropped),flush=True)


if __name__=='__main__':
    for profile in ('land','underwater'):prepare(profile)
