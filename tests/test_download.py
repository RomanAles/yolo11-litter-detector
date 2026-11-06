import json
import threading

from scripts import download_data


def range_server(monkeypatch, payload):
    requested = []
    lock = threading.Lock()

    class Chunk(bytes):
        active = 0
        peak = 0

        def __new__(cls, value):
            result = super().__new__(cls, value)
            with lock:
                cls.active += 1
                cls.peak = max(cls.peak, cls.active)
            return result

        def __del__(self):
            with lock:
                type(self).active -= 1

    class Response:
        status_code = 206

        def __init__(self, start, end):
            self.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}'}
            self.content = Chunk(payload[start:end+1])

        def raise_for_status(self):pass
        def __enter__(self):return self
        def __exit__(self, *args):pass

    def get(url, headers, timeout):
        start, end = map(int, headers['Range'].split('=')[1].split('-'))
        requested.append((start, end))
        return Response(start, end)

    monkeypatch.setattr(download_data.requests, 'get', get)
    return requested, Chunk


def test_range_download_bounds_live_chunks_and_preserves_all_bytes(tmp_path, monkeypatch):
    payload = bytes(range(256))*128
    _, chunks = range_server(monkeypatch, payload)
    output = download_data.range_download('mock://archive', tmp_path/'archive.tar', chunk_size=1024)
    assert output.read_bytes() == payload
    # Six in-flight chunks plus the one-byte size probe; independent of archive size.
    assert chunks.peak <= 8


def test_range_download_resumes_without_fetching_completed_chunks(tmp_path, monkeypatch):
    payload = bytes(range(32))
    requested, _ = range_server(monkeypatch, payload)
    output = tmp_path/'archive.tar'
    partial = bytearray(len(payload))
    partial[:8] = payload[:8]
    partial[16:24] = payload[16:24]
    output.with_suffix('.part').write_bytes(partial)
    output.with_suffix('.chunks.json').write_text(json.dumps([0, 2]))
    download_data.range_download('mock://archive', output, chunk_size=8)
    assert output.read_bytes() == payload
    assert set(requested) == {(0, 0), (8, 15), (24, 31)}
