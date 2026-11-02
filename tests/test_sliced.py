import json,threading,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from litter_detector.detector import Detector


def test_sahi_numpy_values_are_json_serializable(monkeypatch):
    import sahi.predict
    obj=SimpleNamespace(category=SimpleNamespace(id=np.int64(0),name='trash'),
        score=SimpleNamespace(value=np.float32(.8)),bbox=SimpleNamespace(to_xyxy=lambda:[np.float32(v) for v in [1,2,7,8]]))
    monkeypatch.setattr(sahi.predict,'get_sliced_prediction',lambda *args,**kwargs:SimpleNamespace(object_prediction_list=[obj]))
    detector=Detector.__new__(Detector);detector._lock=threading.Lock();detector.sahi_model=None
    detector.profile='land';detector.weights=Path('fake.pt');detector.operating_mode='balanced'
    detector.weight_sha256='mock'
    detector.config={}
    _,record=detector._predict_sliced(np.zeros((10,10,3),dtype='uint8'),0,None,time.perf_counter())
    assert json.loads(json.dumps(record))['detections'][0]['class_id']==0
