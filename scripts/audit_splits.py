"""Validate existing manifests without modifying the data or opening cameras."""
from collections import defaultdict
import json

from litter_detector.detector import ROOT


def audit(profile):
    base=ROOT/'data/prepared'/profile
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf8'))
    by_group=defaultdict(set);by_hash=defaultdict(set)
    for row in manifest['images']:
        by_group[row['group']].add(row['split'])
        by_hash[row['pixel_sha256']].add(row['split'])
        labels=base/'labels'/row['split']/__import__('pathlib').Path(row['name']).with_suffix('.txt')
        assert labels.exists(),str(labels)
        for line in labels.read_text().splitlines():
            cls,x,y,w,h=map(float,line.split())
            assert cls==0 and 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1,(labels,line)
            assert x-w/2 >= -1e-6 and y-h/2 >= -1e-6 and x+w/2<=1+1e-6 and y+h/2<=1+1e-6
    assert all(len(s)==1 for s in by_group.values()),'source group leakage'
    assert all(len(s)==1 for s in by_hash.values()),'exact pixel duplicate leakage'
    print(profile,'PASS',manifest['summary'])


if __name__=='__main__':
    for profile in ('land','underwater'):audit(profile)
