"""Single-class, one-to-one detection matching and 101-point interpolated AP."""
from collections import Counter
import numpy as np


def iou_one(box, others):
    others=np.asarray(others,dtype=float).reshape(-1,4)
    if not len(others):return np.empty(0)
    lt=np.maximum(box[:2],others[:,:2]);rb=np.minimum(box[2:],others[:,2:])
    intersection=np.maximum(0,rb-lt).prod(1)
    return intersection/(np.maximum(0,box[2:]-box[:2]).prod()+np.maximum(0,others[:,2:]-others[:,:2]).prod(1)-intersection+1e-9)


def count_at(records,threshold,iou=.5):
    counts=Counter({key:0 for key in ('tp','fp','fn','small_gt','small_tp','images','negative_images','negative_false_positives')})
    for row in records:
        truth=np.asarray(row['truth']).reshape(-1,4);pred=np.asarray(row['predictions']).reshape(-1,5)
        matched=set();selected=pred[pred[:,4]>=threshold];selected=selected[np.argsort(-selected[:,4])]
        small={i for i,box in enumerate(truth) if (box[2]-box[0])*row['width']*(box[3]-box[1])*row['height']<1024}
        for box in selected:
            overlap=iou_one(box[:4],truth)
            eligible=[j for j in np.argsort(-overlap) if j not in matched and overlap[j]>=iou]
            if eligible:matched.add(eligible[0]);counts['tp']+=1
            else:counts['fp']+=1
        counts['fn']+=len(truth)-len(matched);counts['small_gt']+=len(small);counts['small_tp']+=len(small&matched)
        counts['images']+=1
        if not len(truth):counts['negative_images']+=1;counts['negative_false_positives']+=len(selected)
    p=counts['tp']/max(1,counts['tp']+counts['fp']);r=counts['tp']/max(1,counts['tp']+counts['fn'])
    return {'confidence':threshold,**counts,'precision':p,'recall':r,'f1':2*p*r/max(1e-12,p+r),
            'f2':5*p*r/max(1e-12,4*p+r),'f0_5':1.25*p*r/max(1e-12,.25*p+r),
            'small_recall':counts['small_tp']/counts['small_gt'] if counts['small_gt'] else None}


def average_precision(records):
    truth={i:np.asarray(row['truth']).reshape(-1,4) for i,row in enumerate(records)}
    predictions=[]
    for index,row in enumerate(records):
        for box in sorted(row['predictions'],key=lambda b:-b[4])[:100]:predictions.append((box[4],index,np.asarray(box[:4])))
    predictions.sort(key=lambda row:-row[0]);total=sum(len(v) for v in truth.values());aps=[]
    for threshold in np.arange(.5,.951,.05):
        matched={i:set() for i in truth};hits=[]
        for score,index,box in predictions:
            overlap=iou_one(box,truth[index]);eligible=[j for j in np.argsort(-overlap) if j not in matched[index] and overlap[j]>=threshold]
            hits.append(bool(eligible))
            if eligible:matched[index].add(eligible[0])
        tp=np.cumsum(hits);recall=tp/max(1,total);precision=tp/np.maximum(1,np.arange(1,len(hits)+1))
        envelope=np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else np.array([])
        interpolated=[float(envelope[recall>=r].max()) if np.any(recall>=r) else 0.0 for r in np.linspace(0,1,101)]
        aps.append(float(np.mean(interpolated)))
    return {'ap50':aps[0],'map50_95':float(np.mean(aps)), 'method':'101-point single-class AP, max100 predictions/image, IoU .50:.95; no crowd/ignore regions'}
