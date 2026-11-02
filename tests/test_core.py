from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

from litter_detector.camera import Frame, LatestFrame,run_camera
from litter_detector.datasets import coco_box, split_groups,load_coco_pixels
from litter_detector.detector import read_image, write_image
from litter_detector.media import process_folder, process_video


def test_clipped_box_remains_in_image():
    assert coco_box([-10,20,50,100],100,100)==(.2,.6,.4,.8)


@pytest.mark.parametrize("box",[[1,1,0,2],[1,1,-1,2],[200,1,2,2],[float('nan'),1,2,2]])
def test_invalid_box_rejected(box):
    assert coco_box(box,100,100) is None


def test_split_keeps_entire_videos_and_duplicate_groups_together():
    rows=[{'group':str(i//3),'pixel_sha256':str(i)} for i in range(30)]
    rows[9]['pixel_sha256']=rows[0]['pixel_sha256']
    prepared=split_groups(rows)
    for group in {r['group'] for r in prepared}:
        assert len({r['split'] for r in prepared if r['group']==group})==1
    assert prepared[0]['split']==prepared[9]['split']
    assert {r['split'] for r in prepared}=={'train','val','test'}


def test_latest_frame_overwrites_old_frames():
    slot=LatestFrame()
    for i in range(1000):slot.publish(Frame(i,float(i),None))
    assert slot.get().frame_id==999
    assert slot.get(999,timeout=.001) is None
    slot.close();assert slot.get(999,timeout=.001) is None


def test_unicode_image_roundtrip_and_corrupt_input(tmp_path):
    path=tmp_path/'мусор.png';image=np.zeros((12,20,3),dtype='uint8');write_image(path,image)
    assert read_image(path).shape==image.shape
    path.write_text('not an image')
    with pytest.raises(ValueError):read_image(path)


def test_exif_orientation_matches_annotation_coordinates(tmp_path):
    from PIL import Image
    image=Image.new('RGB',(40,20));exif=Image.Exif();exif[274]=6
    path=tmp_path/'rotated.jpg';image.save(path,exif=exif)
    pixels=load_coco_pixels(path,{'width':20,'height':40})
    assert pixels.shape[:2]==(40,20)


class StubDetector:
    def predict(self,image,frame_id=0,source_time=None):
        return image,{'frame_id':frame_id,'source_time_seconds':source_time,'detections':[]}


def test_video_keeps_frame_count_and_duration(tmp_path):
    source=tmp_path/'source.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),12,(64,48))
    assert writer.isOpened()
    for i in range(24):writer.write(np.full((48,64,3),i*5,dtype='uint8'))
    writer.release()
    output=tmp_path/'out.mp4';summary=process_video(StubDetector(),source,output)
    assert summary['frames']==24
    assert summary['duration_seconds']==pytest.approx(2)
    cap=cv2.VideoCapture(str(output));assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==24;cap.release()
    with pytest.raises(ValueError):process_video(StubDetector(),source,source)


def test_missing_video_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):process_video(StubDetector(),tmp_path/'missing.mp4',tmp_path/'out.mp4')


def test_camera_pipeline_with_mock_capture_only():
    class FakeCapture:
        released=False
        def __init__(self):self.count=0
        def isOpened(self):return True
        def read(self):
            self.count+=1
            if self.count>5:return False,None
            return True,np.full((8,8,3),self.count,dtype='uint8')
        def release(self):self.released=True
    class FakeDetector(StubDetector):profile='land'
    capture=FakeCapture();seen=[]
    run_camera(FakeDetector(),capture_factory=lambda _:capture,callback=lambda image,record:seen.append((int(image[0,0,0]),record['frame_id'])))
    assert capture.released and seen
    assert all(pixel==frame_id+1 for pixel,frame_id in seen)


def test_underwater_camera_is_rejected_without_opening_device():
    class FakeDetector(StubDetector):profile='underwater'
    def forbidden(_):raise AssertionError('A physical device must not be opened')
    with pytest.raises(ValueError):run_camera(FakeDetector(),capture_factory=forbidden)
