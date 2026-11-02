"""Choose operating thresholds on validation; evaluate test exactly afterwards."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from litter_detector.detector import ROOT


def ious(box, others):
    if not len(others):
        return np.array([])
    lt = np.maximum(box[:2], others[:, :2])
    rb = np.minimum(box[2:], others[:, 2:])
    intersection = np.prod(np.maximum(0, rb-lt), axis=1)
    return intersection / (np.prod(box[2:]-box[:2]) + np.prod(others[:,2:]-others[:,:2],axis=1) - intersection + 1e-9)


def operating_points(model, folder):
    threshold_counts = {round(float(t),2): Counter() for t in np.arange(.1,.81,.05)}
    for result in model.predict(str(folder), stream=True, conf=.1, iou=.5, imgsz=640, device=0, verbose=False):
        label = folder.parent.parent / 'labels' / folder.name / Path(result.path).with_suffix('.txt').name
        truth = []
        if label.exists():
            for row in label.read_text().splitlines():
                _,x,y,w,h = map(float,row.split())
                truth.append([x-w/2,y-h/2,x+w/2,y+h/2])
        truth = np.array(truth).reshape(-1,4)
        small = set(np.flatnonzero(np.prod(truth[:,2:]-truth[:,:2],axis=1)*result.orig_shape[0]*result.orig_shape[1] < 32**2))
        predictions = result.boxes.xyxyn.cpu().numpy() if result.boxes is not None else np.empty((0,4))
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
        for threshold, counts in threshold_counts.items():
            matched = set()
            selected = np.flatnonzero(scores >= threshold)
            for index in selected[np.argsort(-scores[selected])]:
                overlap = ious(predictions[index], truth)
                eligible = [j for j in np.argsort(-overlap) if j not in matched and overlap[j]>=.5]
                if eligible:
                    matched.add(eligible[0]); counts['tp']+=1
                else:
                    counts['fp']+=1
            counts['fn'] += len(truth)-len(matched)
            counts['small_gt'] += len(small)
            counts['small_tp'] += len(small & matched)
            counts['images'] += 1
            if not len(truth):
                counts['negative_images']+=1
                counts['negative_false_positives']+=len(selected)
    table=[]
    for threshold, counts in threshold_counts.items():
        p=counts['tp']/max(1,counts['tp']+counts['fp']);r=counts['tp']/max(1,counts['tp']+counts['fn'])
        table.append({'confidence':threshold, **counts, 'precision':p,'recall':r,'f1':2*p*r/max(1e-9,p+r),
                      'small_recall':counts['small_tp']/counts['small_gt'] if counts['small_gt'] else None})
    return table


def evaluate(profile):
    path=ROOT/'models'/profile/'weights/best.pt'
    model=YOLO(str(path))
    data=ROOT/'data/prepared'/profile
    table=operating_points(model,data/'images/val')
    chosen=max(table,key=lambda row:row['f1'])
    # This decision uses validation only, before opening the test set.
    config_file=ROOT/'configs'/f'{profile}.yaml'
    config=yaml.safe_load(config_file.read_text(encoding='utf8'));config['confidence']=chosen['confidence']
    config_file.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf8')
    test_points=operating_points(model,data/'images/test')
    point=next(row for row in test_points if row['confidence']==chosen['confidence'])
    metrics=model.val(data=str(data/'data.yaml'),split='test',imgsz=640,batch=16,device=0,
                      workers=2,conf=.001,iou=.7,plots=True,project=str(ROOT/'outputs/evaluation'),name=profile,exist_ok=True)
    report={'profile':profile,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'selection':'maximum F1 at IoU=0.5 on validation only; one-to-one greedy matching',
            'validation_thresholds':table,'chosen_confidence':chosen['confidence'],'test_operating_point':point,
            'test_ap_metrics':{k:float(v) for k,v in metrics.results_dict.items()},
            'limitations':['Evaluation on public datasets only; no physical cameras tested.',
                           'No independent annotation completeness audit; AP-by-size is not computed by this script.']}
    out=ROOT/'outputs/evaluation'/profile;out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('EVALUATION_FINISHED',profile,json.dumps(point),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('profile',choices=['land','underwater','all']);args=parser.parse_args()
    for profile in ('land','underwater') if args.profile=='all' else (args.profile,):evaluate(profile)
