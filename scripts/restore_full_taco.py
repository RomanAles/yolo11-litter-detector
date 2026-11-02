"""Restore all original TACO photographs from the cached public mirror archive."""
from pathlib import Path
import hashlib,json,shutil,tarfile
from litter_detector.detector import ROOT


def restore():
    archive=ROOT/'.cache/taco-mirror.tar';source=ROOT/'data/raw/taco/annotations.json'
    if not archive.exists():raise FileNotFoundError('Run scripts/download_data.py taco --mirror first')
    target=ROOT/'data/raw/taco_full';target.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target/'annotations.json')
    data=json.loads(source.read_text(encoding='utf8'));lookup={i['file_name'].replace('/','_'):i['file_name'] for i in data['images']}
    with tarfile.open(archive,'r:') as tar:
        for entry in tar:
            key=Path(entry.name).name
            if not entry.isfile() or '/img/' not in entry.name or key not in lookup:continue
            path=target/lookup[key]
            if not path.resolve().is_relative_to(target.resolve()):raise ValueError('Unsafe source filename')
            if path.exists():continue
            path.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(entry) as contents,path.open('wb') as output:shutil.copyfileobj(contents,output)
    missing=[i['file_name'] for i in data['images'] if not (target/i['file_name']).is_file()]
    if missing:raise RuntimeError(f'Missing {len(missing)} original images')
    with archive.open('rb') as contents:digest=hashlib.file_digest(contents,'sha256').hexdigest()
    (target/'provenance.json').write_text(json.dumps({'archive':str(archive),'sha256':digest,'images':len(data['images']),
        'source':'Public Dataset Ninja TACO mirror; annotations from original author repository'},indent=2),encoding='utf8')
    print('Full-resolution TACO restored:',len(data['images']))


if __name__=='__main__':restore()
