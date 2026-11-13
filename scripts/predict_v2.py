"""Cache offline predictions for validation/test; no hyperparameter decisions here."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from litter_detector.detector import ROOT,read_image
from litter_detector.metrics import average_precision,count_at
from litter_detector.artifacts import write_json
from litter_detector.experiments import file_sha256


def predict(weights,profile,split='val',imgsz=640,sliced=False,output=None,tile_size=640):
    from ultralytics import YOLO
    import torch
    folder = ROOT / 'data' / 'v2' / profile
    file_list = folder / f'{split}.txt'
    file_list_bytes = file_list.read_bytes()
    files = file_list_bytes.decode('utf-8').splitlines()
    if not files:
        raise ValueError(f'Empty evaluation cohort: {file_list}')
    inputs = {
        Path(weights): file_sha256(weights),
        folder / 'manifest.json': file_sha256(folder / 'manifest.json'),
        file_list: hashlib.sha256(file_list_bytes).hexdigest(),
    }
    output=Path(output) if output else ROOT/'outputs/v2/predictions'/f'{profile}_{Path(weights).stem}_{imgsz}_{split}_{sliced}.json'
    model=YOLO(str(weights))
    if sliced:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        sahi_model=AutoDetectionModel.from_pretrained(model_type='ultralytics',model_path=str(weights),confidence_threshold=.05,device='cuda:0',image_size=imgsz)
    records=[];elapsed=[]
    for index,file in enumerate(files):
        if index==3:torch.cuda.reset_peak_memory_stats()
        path=Path(file);image=read_image(path);h,w=image.shape[:2]
        label=path.parent.parent.parent/'labels'/path.parent.name/path.with_suffix('.txt').name
        if not label.exists():raise FileNotFoundError(label)
        truth=[]
        for line in label.read_text().splitlines():
            _,x,y,bw,bh=map(float,line.split());truth.append([x-bw/2,y-bh/2,x+bw/2,y+bh/2])
        started=time.perf_counter()
        if sliced:
            result=get_sliced_prediction(image[:,:,::-1],sahi_model,slice_height=tile_size,slice_width=tile_size,overlap_height_ratio=.2,overlap_width_ratio=.2,
                perform_standard_pred=True,postprocess_type='NMS',postprocess_match_metric='IOU',postprocess_match_threshold=.5,verbose=0)
            predictions=[[float(obj.bbox.minx/w),float(obj.bbox.miny/h),float(obj.bbox.maxx/w),float(obj.bbox.maxy/h),float(obj.score.value)] for obj in result.object_prediction_list]
        else:
            result=model.predict(image,conf=.05,iou=.5,imgsz=imgsz,device=0,verbose=False)[0]
            predictions=[[*box,float(score)] for box,score in zip(result.boxes.xyxyn.cpu().tolist(),result.boxes.conf.cpu().tolist())]
        elapsed.append((time.perf_counter()-started)*1000)
        records.append({'file':file,'width':w,'height':h,'truth':truth,'predictions':predictions})
    table=[count_at(records,round(float(t),2)) for t in np.arange(.05,.901,.05)]
    report={'weights':str(weights),'profile':profile,'split':split,'imgsz':imgsz,'sliced':sliced,'tile_size':tile_size,'records':records,
            'thresholds':table,'ap':average_precision(records),'median_ms':float(np.median(elapsed[3:] or elapsed)),
            'p95_ms':float(np.quantile(elapsed[3:] or elapsed,.95)), 'weight_sha256':inputs[Path(weights)]}
    report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
    report['manifest_sha256'] = inputs[folder / 'manifest.json']
    report['file_list_sha256'] = inputs[file_list]
    if any(file_sha256(path) != digest for path, digest in inputs.items()):
        raise RuntimeError('Model weights or dataset changed during evaluation')
    write_json(output,report)
    print(profile,split,imgsz,'SAHI' if sliced else 'full',report['ap'],report['median_ms'],flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('weights');p.add_argument('profile');p.add_argument('--split',default='val');p.add_argument('--imgsz',type=int,default=640);p.add_argument('--sliced',action='store_true');p.add_argument('--output');p.add_argument('--tile-size',type=int,default=640);a=p.parse_args()
    predict(a.weights,a.profile,a.split,a.imgsz,a.sliced,a.output,a.tile_size)
