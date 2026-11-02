"""Examples selected by annotation size, never by model confidence."""
import json
import argparse
from pathlib import Path
import shutil

from litter_detector.detector import ROOT,Detector
from litter_detector.media import process_folder,process_video


def examples(profile):
    base=ROOT/'data/prepared'/profile
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf8'))
    positives=[r for r in manifest['images'] if r['split']=='test' and r['labels']]
    def area(row):
        return max(float(l.split()[3])*float(l.split()[4]) for l in row['labels'])
    positives.sort(key=area,reverse=True)
    chosen=positives[:3]
    negatives=[r for r in manifest['images'] if r['split']=='test' and not r['labels']]
    chosen+=negatives[:1]
    source=ROOT/'data/samples'/profile;source.mkdir(parents=True,exist_ok=True)
    for row in chosen:shutil.copy2(base/'images/test'/row['name'],source/row['name'])
    (source/'provenance.json').write_text(json.dumps({'selection':'three largest annotated objects plus first negative if available',
        'dataset':'TACO' if profile=='land' else 'TrashCan', 'images':[r['source'] for r in chosen]},indent=2),encoding='utf8')
    detector=Detector(profile)
    count=process_folder(detector,source,ROOT/'outputs/examples'/profile)
    print('EXAMPLES_FINISHED',profile,count,flush=True)
    return detector


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('profile',nargs='?',choices=['land','underwater','all'],default='all');args=parser.parse_args()
    for profile in ('land','underwater') if args.profile=='all' else (args.profile,):
        detector=examples(profile)
        if profile=='underwater':
            summary=process_video(detector,ROOT/'data/samples/underwater_dataset_sequence.mp4',ROOT/'outputs/demo-video/detected.mp4')
            print('VIDEO_FINISHED',summary,flush=True)
