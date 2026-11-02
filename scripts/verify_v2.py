"""Final verification on local public dataset files; never opens cameras."""
from pathlib import Path
import hashlib,json
import cv2,yaml
from litter_detector.detector import ROOT,Detector
from litter_detector.media import process_folder,process_video
from litter_detector.artifacts import write_json


def verify():
    baseline=ROOT/'models/baseline_v1';snapshot=json.loads((baseline/'snapshot.json').read_text())
    for relative,digest in snapshot['files'].items():
        path=baseline/relative
        assert path.resolve().is_relative_to(baseline.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,relative
    audit=json.loads((ROOT/'outputs/v2/prepared_file_audit.json').read_text())
    for profile,report in audit['profiles'].items():
        assert hashlib.sha256((ROOT/'data/v2'/profile/'manifest.json').read_bytes()).hexdigest()==report['manifest_sha256']
        for row in report['records']:
            image=Path(row['image']);label=image.parent.parent.parent/'labels'/image.parent.name/image.with_suffix('.txt').name
            assert hashlib.sha256(image.read_bytes()).hexdigest()==row['file_sha256'],image
            assert hashlib.sha256(label.read_bytes()).hexdigest()==row['label_sha256'],label
    reports={}
    for profile in ('land','underwater'):
        config=yaml.safe_load((ROOT/'configs'/f'{profile}.yaml').read_text(encoding='utf8'))
        frozen=json.loads((ROOT/'outputs/v2'/f'{profile}_frozen_selection.json').read_text())
        assert config['imgsz']==frozen['imgsz']
        assert config['operating_modes']==frozen['operating_modes']
        assert config['sahi_operating_modes']==frozen['sahi_operating_modes']
        expected=hashlib.sha256(Path(frozen['weights']).read_bytes()).hexdigest()
        assert hashlib.sha256((ROOT/config['model']).read_bytes()).hexdigest()==expected
        detector=Detector(profile)
        count=process_folder(detector,ROOT/'data/samples'/profile,ROOT/'outputs/v2/examples'/profile)
        source=next(p for p in (ROOT/'data/samples'/profile).iterdir() if p.suffix.lower() in ('.jpg','.png'))
        records={}
        for mode in ('balanced','recall','precision'):
            engine=Detector(profile,operating_mode=mode)
            path,record=engine.image(source,ROOT/'outputs/v2/mode_checks'/profile/mode)
            assert record['model_sha256']==expected and cv2.imread(str(path)) is not None
            records[mode]={'detections':len(record['detections']),'confidence':engine.confidence}
            del engine
        sliced=Detector(profile,sliced=True)
        path,record=sliced.image(source,ROOT/'outputs/v2/mode_checks'/profile/'sahi')
        assert record['sliced'] and record['model_sha256']==expected and cv2.imread(str(path)) is not None
        reports[profile]={'file_batch':count,'operating_modes':records,'sahi_file':'passed','model_sha256':expected}
        del sliced
        if profile=='underwater':
            target=ROOT/'outputs/v2/demo-video/detected.mp4'
            summary=process_video(detector,ROOT/'data/samples/underwater_dataset_sequence.mp4',target)
            cap=cv2.VideoCapture(str(target));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
            assert frames==28==summary['frames'] and fps==12
            assert abs(summary['duration_seconds']-frames/fps)<1e-6
            rows=target.with_suffix('.jsonl').read_text().splitlines();assert len(rows)==28
            assert [json.loads(row)['frame_id'] for row in rows]==list(range(28))
            reports[profile]['video']={'frames':frames,'fps':fps,'duration_seconds':frames/fps}
        del detector
    write_json(ROOT/'outputs/v2/verification.json',{'baseline_snapshot_hashes':'passed','prepared_image_and_label_hashes':'passed','profiles':reports,'physical_camera_opened':False})
    print('V2 baseline integrity, mode presets, images and video passed')


if __name__=='__main__':verify()
