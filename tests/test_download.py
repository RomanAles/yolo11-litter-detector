import json
import threading

from scripts import download_data


def range_server(monkeypatch, payload):
    requested = []

    class Response:
        status_code = 206

        def __init__(self, start, end):
            self.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}', 'ETag': '"v1"'}
            self.content = payload[start:end+1]

        def iter_content(self, chunk_size):
            yield self.content

        def raise_for_status(self):pass
        def __enter__(self):return self
        def __exit__(self, *args):pass

    def get(url, headers, timeout, stream):
        assert stream is True
        start, end = map(int, headers['Range'].split('=')[1].split('-'))
        if (start, end) != (0, 0):
            assert headers['If-Range'] == '"v1"'
        requested.append((start, end))
        return Response(start, end)

    monkeypatch.setattr(download_data.requests, 'get', get)
    return requested


def test_range_download_bounds_live_chunks_and_preserves_all_bytes(tmp_path, monkeypatch):
    payload = bytes(range(256))*128
    range_server(monkeypatch, payload)
    lock = threading.Lock()
    active = peak = 0

    class TrackedBuffer(bytearray):
        def __init__(self):
            super().__init__()
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)

        def __del__(self):
            nonlocal active
            with lock:
                active -= 1

    monkeypatch.setattr(download_data, 'bytearray', TrackedBuffer, raising=False)
    output = download_data.range_download('mock://archive', tmp_path/'archive.tar', chunk_size=1024)
    assert output.read_bytes() == payload
    assert 1 <= peak <= 6


def test_range_download_resumes_without_fetching_completed_chunks(tmp_path, monkeypatch):
    payload = bytes(range(32))
    requested = range_server(monkeypatch, payload)
    output = tmp_path/'archive.tar'
    partial = bytearray(len(payload))
    partial[:8] = payload[:8]
    partial[16:24] = payload[16:24]
    output.with_suffix('.part').write_bytes(partial)
    state = {'archive': {'url': 'mock://archive', 'size': len(payload), 'chunk_size': 8, 'validator': '"v1"'},
             'completed': [0, 2]}
    output.with_suffix('.chunks.json').write_text(json.dumps(state))
    download_data.range_download('mock://archive', output, chunk_size=8)
    assert output.read_bytes() == payload
    assert set(requested) == {(0, 0), (8, 15), (24, 31)}


def test_range_download_restarts_partial_archive_from_a_different_version(tmp_path, monkeypatch):
    payload = bytes(range(16))
    requested = range_server(monkeypatch, payload)
    output = tmp_path / 'archive.tar'
    output.with_suffix('.part').write_bytes(b'x' * len(payload))
    state = {'archive': {'url': 'mock://archive', 'size': len(payload), 'chunk_size': 8, 'validator': '"old"'},
             'completed': [0, 1]}
    output.with_suffix('.chunks.json').write_text(json.dumps(state))
    assert download_data.range_download('mock://archive', output, chunk_size=8).read_bytes() == payload
    assert set(requested) == {(0, 0), (0, 7), (8, 15)}
