"""Build a clearly labelled demonstration clip from held-out public dataset frames."""
from collections import defaultdict
import json

import cv2

from litter_detector.detector import ROOT,read_image


def make_demo():
    base=ROOT/'data/prepared/underwater'
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf8'))
    groups=defaultdict(list)
    for row in manifest['images']:
        if row['split']=='test':groups[row['group']].append(row)
    rows=max(groups.values(),key=len)
    rows=sorted(rows,key=lambda row:row['name'])[:48]
    out=ROOT/'data/samples';out.mkdir(parents=True,exist_ok=True)
    video=out/'underwater_dataset_sequence.mp4'
    first=read_image(base/'images/test'/rows[0]['name']);h,w=first.shape[:2]
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),12,(w,h))
    if not writer.isOpened():raise OSError(video)
    try:
        for row in rows:
            frame=read_image(base/'images/test'/row['name'])
            if frame.shape[:2]!=(h,w):frame=cv2.resize(frame,(w,h))
            writer.write(frame)
    finally:writer.release()
    (out/'sequence_provenance.json').write_text(json.dumps({'kind':'frame sequence assembled from existing dataset',
        'not_original_camera_video':True,'fps':12,'frames':[row['source'] for row in rows],
        'sampling_note':'Timing assigned for demonstration; original capture cadence is unknown'},indent=2),encoding='utf8')
    print(video,len(rows))


if __name__=='__main__':make_demo()
