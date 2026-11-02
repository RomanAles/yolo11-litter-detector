"""Optional ONNX export with file-based output parity and measured latency."""
from pathlib import Path
import json,time,shutil
import numpy as np
from ultralytics import YOLO
from litter_detector.detector import ROOT,read_image


def check():
    reports={}
    for profile in ('land','underwater'):
        config=__import__('yaml').safe_load((ROOT/'configs'/f'{profile}.yaml').read_text(encoding='utf8'))
        source=ROOT/config['model'];export_folder=ROOT/'models/exports_v2'/profile;export_folder.mkdir(parents=True,exist_ok=True)
        weights=export_folder/'model.pt';shutil.copy2(source,weights);model=YOLO(str(weights));size=config['imgsz']
        exported=Path(model.export(format='onnx',imgsz=size,opset=18,dynamic=False,simplify=False,batch=1))
        __import__('onnx').checker.check_model(__import__('onnx').load(str(exported)))
        onnx=YOLO(str(exported),task='detect')
        files=(ROOT/'data/v2'/profile/'val.txt').read_text().splitlines()[:20]
        times={'pytorch_cuda':[],'onnx_cpu':[]};max_score_error=0.0;max_box_delta=0.0;count_disagreements=0
        for file in files:
            image=read_image(file);outputs=[]
            for name,engine,device in [('pytorch_cuda',model,0),('onnx_cpu',onnx,'cpu')]:
                started=time.perf_counter();result=engine.predict(image,imgsz=size,conf=.25,iou=.5,device=device,verbose=False,rect=False)[0]
                times[name].append((time.perf_counter()-started)*1000);outputs.append(result.boxes.data.cpu().numpy())
            a,b=outputs
            if len(a)!=len(b):count_disagreements+=1
            elif len(a):
                # Stable score ordering avoids class/box list order differences.
                a=a[np.argsort(-a[:,4])];b=b[np.argsort(-b[:,4])]
                max_score_error=max(max_score_error,float(np.abs(a[:,4]-b[:,4]).max()))
                max_box_delta=max(max_box_delta,float(np.abs(a[:,:4]-b[:,:4]).max()))
        reports[profile]={'onnx':str(exported),'samples':len(files),'count_disagreements':count_disagreements,
            'max_score_difference_when_counts_match':max_score_error,
            'max_box_delta_pixels_when_score_sorted_counts_match':max_box_delta,
            'median_ms':{k:float(np.median(v[3:])) for k,v in times.items()},
            'deployment':'PyTorch CUDA retained; ONNX CPU is an optional portability artifact, not an assumed GPU speedup',
            'limitation':'20 validation files, not a complete backend accuracy evaluation; timings compare available backends on this host.'}
    (ROOT/'outputs/v2/export_check.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))


if __name__=='__main__':check()
