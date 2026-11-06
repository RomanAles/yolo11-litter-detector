import json

import numpy as np
import pytest

from litter_detector import datasets
from litter_detector.detector import read_image, write_image


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, 'ROOT', tmp_path)
    folder = tmp_path/'data/raw/taco'
    images, annotations = [], []
    for index in range(10):
        name = f'batch_{index}/image.jpg'
        write_image(folder/name, np.full((32, 32, 3), index*20, dtype=np.uint8))
        images.append({'id': index, 'file_name': name, 'width': 32, 'height': 32})
        annotations.append({'image_id': index, 'category_id': 1, 'bbox': [4, 4, 12, 12]})
    (folder/'annotations.json').write_text(json.dumps({
        'images': images, 'annotations': annotations, 'categories': [{'id': 1, 'name': 'trash'}],
    }))
    return tmp_path, datasets.prepare('land')


@pytest.mark.parametrize('changed', ['image', 'annotations'])
def test_changed_sources_are_rejected_before_prepared_files_are_modified(tiny_dataset, changed):
    root, manifest = tiny_dataset
    output = root/'data/prepared/land'
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob('*') if path.is_file()}
    if changed == 'image':
        write_image(root/manifest['images'][0]['source'], np.full((32, 32, 3), 255, dtype=np.uint8))
    else:
        path = root/'data/raw/taco/annotations.json'
        data = json.loads(path.read_text())
        data['annotations'][0]['bbox'] = [8, 8, 12, 12]
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Source .*changed'):
        datasets.prepare('land')
    assert {path.relative_to(output): path.read_bytes() for path in output.rglob('*') if path.is_file()} == before


def test_interrupted_preparation_rebuilds_copies_without_provenance(tiny_dataset):
    root, manifest = tiny_dataset
    output = root/'data/prepared/land'
    (output/'manifest.json').unlink()
    row = manifest['images'][0]
    write_image(root/row['source'], np.full((32, 32, 3), 255, dtype=np.uint8))
    regenerated = datasets.prepare('land')
    row = regenerated['images'][0]
    assert np.all(read_image(output/'images'/row['split']/row['name']) == 255)


def test_unchanged_sources_can_be_prepared_again(tiny_dataset):
    _, manifest = tiny_dataset
    assert datasets.prepare('land') == manifest
