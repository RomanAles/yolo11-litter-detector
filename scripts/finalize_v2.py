"""Frozen validation selection, one final paired test evaluation, staged app configs."""
from pathlib import Path
import json,hashlib,shutil,time
import cv2,numpy as np,yaml
from litter_detector.detector import ROOT,read_image,write_image
from litter_detector.metrics import count_at
from litter_detector.artifacts import write_json,write_text
from predict_v2 import predict


def cached(weights,profile,split,size,sliced,name,tile_size=640):
    path=ROOT/'outputs/v2/predictions'/f'{profile}_{name}_{split}.json'
    if path.exists():
        try:previous=json.loads(path.read_text())
        except json.JSONDecodeError:previous=None
        if previous is None:return predict(weights,profile,split,size,sliced,path,tile_size)
        if (previous['weight_sha256']==hashlib.sha256(Path(weights).read_bytes()).hexdigest()
            and previous['imgsz']==size and previous['sliced']==sliced and previous.get('tile_size',640)==tile_size
            and previous.get('manifest_sha256')==hashlib.sha256((ROOT/'data/v2'/profile/'manifest.json').read_bytes()).hexdigest()
            and previous.get('file_list_sha256')==hashlib.sha256((ROOT/'data/v2'/profile/f'{split}.txt').read_bytes()).hexdigest()):return previous
    return predict(weights,profile,split,size,sliced,path,tile_size)


def error_gallery(report,threshold,folder):
    ranked=[]
    for row in report['records']:
        counts=count_at([row],threshold);ranked.append((counts['fp']+counts['fn'],row))
    ranked.sort(key=lambda item:-item[0]);folder.mkdir(parents=True,exist_ok=True)
    for index,(errors,row) in enumerate(ranked[:20]):
        image=read_image(row['file']);h,w=image.shape[:2]
        for box in row['truth']:
            x1,y1,x2,y2=map(round,[box[0]*w,box[1]*h,box[2]*w,box[3]*h]);cv2.rectangle(image,(x1,y1),(x2,y2),(0,180,0),2)
        for box in row['predictions']:
            if box[4]<threshold:continue
            x1,y1,x2,y2=map(round,[box[0]*w,box[1]*h,box[2]*w,box[3]*h]);cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(image,'GT: green | prediction: red',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        write_image(folder/f'{index:02d}_errors{errors}.jpg',image)
    (folder/'selection.json').write_text(json.dumps({'selection':'20 largest FP+FN counts; illustrative worst cases, not a random sample'},indent=2))


def finalize():
    output=ROOT/'outputs/v2';selected=json.loads((output/'selected.json').read_text());final={}
    for profile,best in selected.items():
        baseline_weights=ROOT/'models/baseline_v1'/profile/'weights/best.pt'
        baseline=cached(baseline_weights,profile,'val',640,False,'baseline')
        candidate=cached(best['weights'],profile,'val',best['imgsz'],False,'candidate')
        # Retain v1 if the experiment fails to improve the paired validation cohort.
        if candidate['ap']['map50_95'] < baseline['ap']['map50_95']:
            best={**best,'weights':str(baseline_weights),'imgsz':640,'name':'baseline_retained'};candidate=baseline
        if best['imgsz']==960:
            high=cached(best['weights'],profile,'val',1280,False,'resolution1280')
            if high['ap']['map50_95']>candidate['ap']['map50_95']+.005:best['imgsz']=1280;candidate=high
        sliced=cached(best['weights'],profile,'val',best['imgsz'],True,'sahi')
        if any(max(row['width'],row['height'])>640 for row in sliced['records']):
            alternate=cached(best['weights'],profile,'val',best['imgsz'],True,'sahi960',tile_size=960)
            if alternate['ap']['map50_95']>sliced['ap']['map50_95']:sliced=alternate
        tile_size=sliced.get('tile_size',640)
        modes={'balanced':max(candidate['thresholds'],key=lambda r:r['f1'])['confidence'],
               'recall':max(candidate['thresholds'],key=lambda r:r['f2'])['confidence'],
               'precision':max(candidate['thresholds'],key=lambda r:r['f0_5'])['confidence']}
        sliced_modes={mode:max(sliced['thresholds'],key=lambda r:r[metric])['confidence'] for mode,metric in [('balanced','f1'),('recall','f2'),('precision','f0_5')]}
        frozen={'weights':best['weights'],'imgsz':best['imgsz'],'operating_modes':modes,'sahi_operating_modes':sliced_modes,'sahi_tile_size':tile_size,
                'selected_by':'paired validation AP; no test selection','sahi_validation':sliced['ap'],
                'standard_validation':candidate['ap'],'baseline_validation':baseline['ap']}
        write_json(output/f'{profile}_frozen_selection.json',frozen)
        # Final evaluation follows all selection; historical v1 test is explicitly not blind.
        test=cached(best['weights'],profile,'test',best['imgsz'],False,'final')
        baseline_test=cached(baseline_weights,profile,'test',640,False,'baseline')
        sliced_test=cached(best['weights'],profile,'test',best['imgsz'],True,'sahi',tile_size=tile_size)
        report={'selection':frozen,'candidate':best,'baseline_test_ap':baseline_test['ap'],'test_ap':test['ap'],
                'test_modes':{mode:count_at(test['records'],threshold) for mode,threshold in modes.items()},
                'sahi_test_ap':sliced_test['ap'],'sahi_test_modes':{mode:count_at(sliced_test['records'],threshold) for mode,threshold in sliced_modes.items()},
                'latency':{'standard_median_ms':test['median_ms'],'sahi_median_ms':sliced_test['median_ms']},
                'limitations':['Historical test was evaluated in v1 and is not a new blind benchmark.','Public datasets only; no physical cameras.','SAHI costs extra processing; no real-time claim.']}
        error_gallery(test,modes['balanced'],output/'errors'/profile)
        weights_dir=ROOT/'models/v2'/profile;weights_dir.mkdir(parents=True,exist_ok=True)
        temporary=weights_dir/'best.next.pt';shutil.copy2(best['weights'],temporary);temporary.replace(weights_dir/'best.pt')
        config=yaml.safe_load((ROOT/'configs'/f'{profile}.yaml').read_text(encoding='utf8'))
        config.update(model=str((weights_dir/'best.pt').relative_to(ROOT)).replace('\\','/'),imgsz=best['imgsz'],confidence=modes['balanced'],operating_modes=modes,sahi_operating_modes=sliced_modes,sahi_tile_size=tile_size)
        write_text(ROOT/'configs'/f'{profile}.yaml',yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
        write_json(output/f'{profile}_final.json',report);final[profile]=report
        print('FINALIZED',profile,best['name'],report['test_ap'],flush=True)
    write_json(output/'final.json',final)


if __name__=='__main__':finalize()
