"""Admit visually reviewed TACO background crops, preserving source splits."""
from pathlib import Path
from collections import Counter
import json,shutil
from litter_detector.detector import ROOT

REJECTED={11,17,24,32,34,44,58,63,74,76,96,98,107,116,118,119,128,141,159}


def admit():
    folder=ROOT/'data/v2/negative_candidates';data=json.loads((folder/'candidates.json').read_text())
    base=ROOT/'data/v2/land';manifest=json.loads((base/'manifest.json').read_text())
    rows=[r for r in manifest['images'] if not r.get('derived_negative')];counts=Counter()
    for row in data:
        row['reviewed']=True;row['accepted']=row['id'] not in REJECTED
        row['review_note']='Rejected: possible unlabelled debris or ambiguous man-made/organic object' if not row['accepted'] else 'No target debris visible in contact-sheet review; zero original bbox intersection'
        if not row['accepted']:continue
        image=base/'images'/row['split']/row['name'];image.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(row['image_path'],image)
        label=base/'labels'/row['split']/Path(row['name']).with_suffix('.txt');label.parent.mkdir(parents=True,exist_ok=True);label.write_text('')
        rows.append({**row,'image_path':str(image.resolve()),'labels':[],'derived_negative':True})
        counts[row['split']]+=1
    for split in ('train','val','test'):
        (base/f'{split}.txt').write_text('\n'.join(r['image_path'] for r in rows if r['split']==split)+'\n')
    manifest['images']=rows;manifest['negative_review']={'source':'TACO background crops; no bbox overlap; visual review','accepted':dict(counts),'rejected':sorted(REJECTED),'limitation':'Contact-sheet review is not independent exhaustive reannotation.'}
    manifest['summary']={s:{'images':sum(r['split']==s for r in rows),'objects':sum(len(r['labels']) for r in rows if r['split']==s),'negatives':sum(not r['labels'] for r in rows if r['split']==s)} for s in ('train','val','test')}
    (base/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    (folder/'review.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    print('Admitted',dict(counts),'Rejected',len(REJECTED),flush=True)


if __name__=='__main__':admit()
