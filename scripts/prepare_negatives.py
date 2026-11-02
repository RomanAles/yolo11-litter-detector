"""Candidate negative crops derived exclusively from the annotated public TACO set.

Every crop stays in the split of its parent image. Zero bbox overlap is necessary,
but not sufficient: contact sheets must be reviewed before admission.
"""
from collections import Counter
import json,hashlib,random
from pathlib import Path
import cv2,numpy as np
from litter_detector.detector import ROOT,write_image
from litter_detector.datasets import load_coco_pixels


def candidates():
    old=json.loads((ROOT/'data/prepared/land/manifest.json').read_text(encoding='utf8'))
    rows=list(old['images']);random.Random(171).shuffle(rows)
    target=ROOT/'data/v2/negative_candidates';target.mkdir(parents=True,exist_ok=True)
    counts=Counter();records=[];limits={'train':100,'val':30,'test':30}
    for row in rows:
        split=row['split']
        if counts[split]>=limits[split]:continue
        suffix=row['source'].split('taco\\')[-1].split('taco/')[-1]
        source=ROOT/'data/raw/taco_full'/suffix
        pixels=load_coco_pixels(source,{'width':row.get('annotation_width',row['width']),'height':row.get('annotation_height',row['height'])})
        h,w=pixels.shape[:2];side=min(h,w)//3
        if side<160:continue
        boxes=[]
        for line in row['labels']:
            _,x,y,bw,bh=map(float,line.split());boxes.append(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h))
        found=None
        locations=[(x,y) for y in (0,(h-side)//2,h-side) for x in (0,(w-side)//2,w-side)]
        random.Random(row['name']).shuffle(locations)
        for x,y in locations:
            if any(min(x+side,x2)>max(x,x1) and min(y+side,y2)>max(y,y1) for x1,y1,x2,y2 in boxes):continue
            found=(x,y);break
        if found is None:continue
        x,y=found;crop=pixels[y:y+side,x:x+side]
        name=f'negative_{len(records):03d}.jpg';path=target/name;write_image(path,crop)
        records.append({'id':len(records),'name':name,'split':split,'parent':row['name'],'group':row['group'],
                        'crop_xyxy':[x,y,x+side,y+side],'pixel_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),
                        'source':str(source.relative_to(ROOT)),'image_path':str(path),'reviewed':False})
        counts[split]+=1
    assert all(counts[s]==limits[s] for s in limits),counts
    (target/'candidates.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    for offset in range(0,len(records),40):
        tiles=[]
        for row in records[offset:offset+40]:
            image=cv2.imread(row['image_path']);image=cv2.resize(image,(180,150))
            tile=np.full((175,180,3),255,dtype='uint8');tile[25:]=image
            cv2.putText(tile,f"{row['id']} {row['split']}",(5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1);tiles.append(tile)
        while len(tiles)<40:tiles.append(np.full((175,180,3),255,dtype='uint8'))
        sheet=np.concatenate([np.concatenate(tiles[i:i+8],axis=1) for i in range(0,40,8)],axis=0)
        write_image(target/f'review_{offset//40}.jpg',sheet)
    print('Candidates',dict(counts),'review contact sheets before admission',flush=True)


if __name__=='__main__':candidates()
