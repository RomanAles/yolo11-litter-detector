from pathlib import Path
import json

import numpy as np
import pytest

from litter_detector.detector import read_image, write_image
from litter_detector.media import process_folder


class StubImageDetector:
    def image(self,source,output):
        image=read_image(source)
        target=Path(output)/(Path(source).stem+'_detected.jpg')
        write_image(target,image)
        return target,{'detections':[]}


def test_folder_preserves_paths_and_same_stem_extensions(tmp_path):
    source=tmp_path/'input';out=source/'results';image=np.zeros((8,12,3),dtype='uint8')
    write_image(source/'same.jpg',image);write_image(source/'same.png',image)
    write_image(source/'nested'/'same.jpg',image)
    (source/'corrupt.jpg').write_text('broken')
    assert process_folder(StubImageDetector(),source,out)==3
    assert len(list(out.rglob('*_detected.jpg')))==3
    summary=json.loads((out/'batch_summary.json').read_text())
    assert len(summary['errors'])==1


def test_folder_refuses_empty_source(tmp_path):
    with pytest.raises(ValueError):process_folder(StubImageDetector(),tmp_path,tmp_path/'out')
