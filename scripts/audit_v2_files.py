"""Audit actual prepared pixels without rewriting frozen training manifests."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json

from litter_detector.detector import ROOT,read_image
from litter_detector.artifacts import write_json


def audit():
    reports={}
    for profile in ('land','underwater'):
        folder=ROOT/'data/v2'/profile;manifest=folder/'manifest.json'
        source=json.loads(manifest.read_text());rows=source['images']
        def inspect(row):
            image=Path(row['image_path']);pixels=read_image(image);height,width=pixels.shape[:2]
            label=image.parent.parent.parent/'labels'/image.parent.name/image.with_suffix('.txt').name
            lines=label.read_text().splitlines()
            assert lines==row['labels'],label
            if 'width' in row and 'height' in row:
                assert abs(width/height-row['width']/row['height'])<.002,image
            for line in lines:
                cls,x,y,w,h=map(float,line.split())
                assert cls==0 and 0<w<=1 and 0<h<=1
                assert x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1+1e-6 and y+h/2<=1+1e-6
            return {'image':str(image),'split':row['split'],'group':row['group'],
                'width':width,'height':height,'manifest_width':row.get('width',width),'manifest_height':row.get('height',height),
                'file_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                'pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
                'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),'objects':len(lines)}
        with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(inspect,rows))
        for split in ('train','val','test'):
            assert set((folder/f'{split}.txt').read_text().splitlines())=={r['image'] for r in records if r['split']==split}
        checks={}
        for field in ('group','file_sha256','pixel_sha256'):
            groups=defaultdict(set)
            for row in records:groups[row[field]].add(row['split'])
            conflicts={key:sorted(value) for key,value in groups.items() if len(value)>1}
            assert not conflicts,(profile,field,conflicts)
            checks[field+'_cross_split_conflicts']=len(conflicts)
        reports[profile]={'images':len(records),'checks':checks,
            'source_dimensions_differ_from_prepared':sum((r['width'],r['height'])!=(r['manifest_width'],r['manifest_height']) for r in records),
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'records':records}
        print(profile,{key:value for key,value in reports[profile].items() if key!='records'},flush=True)
    write_json(ROOT/'outputs/v2/prepared_file_audit.json',{'profiles':reports,
        'note':'Frozen manifests retain original/source dimension provenance on previously materialized TACO copies. Actual prepared dimensions and checksums are recorded here; normalized labels are unchanged.'})


if __name__=='__main__':audit()
