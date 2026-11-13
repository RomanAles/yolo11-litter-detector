"""Download the original public datasets, keeping provenance and resumable files."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import time
import zipfile
import re
import tarfile

import requests
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
API = "https://conservancy.umn.edu/server/api"
TACO_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"


def download(url: str, path: Path) -> Path:
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(10, 25)) as response:
                response.raise_for_status()
                expected = int(response.headers.get("content-length", 0))
                with temp.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                if expected and temp.stat().st_size != expected:
                    raise OSError("Incomplete download")
            temp.replace(path)
            return path
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def taco() -> None:
    folder = RAW / "taco"
    annotations = download(TACO_URL, folder / "annotations.json")
    data = json.loads(annotations.read_text(encoding="utf-8"))
    failures = []
    downloaded_sources = {}
    def image(item):
        path = folder / item["file_name"]
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Unsafe dataset filename")
        if path.is_file():
            return item["file_name"], "existing download from original flickr_url"
        # The official metadata also provides 640px copies; boxes use normalized coordinates.
        urls = list(dict.fromkeys(u for u in (item.get("flickr_640_url"), item.get("flickr_url"), item.get("coco_url")) if u))
        for url in urls:
            try:
                download(url, path)
                return item["file_name"], url
            except (requests.RequestException, OSError):
                if url == urls[-1]:
                    raise
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(image, item): item for item in data["images"]}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                name, url = future.result()
                downloaded_sources[name] = url
            except Exception as error:
                failures.append({"file": futures[future]["file_name"], "error": str(error)})
                print("TACO unavailable:", futures[future]["file_name"], str(error)[:120], flush=True)
            if count % 100 == 0:
                print(f"TACO {count}/{len(futures)}, failures={len(failures)}", flush=True)
    (folder / "download_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    (folder / "download_sources.json").write_text(json.dumps(downloaded_sources, indent=2), encoding="utf-8")
    print(f"TACO complete: {len(data['images'])-len(failures)} images; failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError("Some TACO images failed; rerun the downloader to retry")


def taco_mirror() -> None:
    """Recover original images from the public Dataset Ninja archive, stream-only."""
    folder = RAW / "taco"
    data = json.loads(download(TACO_URL, folder/"annotations.json").read_text(encoding="utf-8"))
    lookup = {i["file_name"].replace("/", "_"): i["file_name"] for i in data["images"]}
    page = requests.get("https://raw.githubusercontent.com/dataset-ninja/taco/main/DOWNLOAD.md", timeout=30)
    page.raise_for_status()
    url = re.search(r"\[Download\]\(([^)]+)\)", page.text).group(1)
    count = 0
    archive_path = range_download(url, ROOT/'.cache/taco-mirror.tar')
    with archive_path.open('rb') as contents:
        with tarfile.open(fileobj=contents, mode="r|") as archive:
            for member in archive:
                name = Path(member.name).name
                original = lookup.get(name)
                if not member.isfile() or "/img/" not in member.name or not original:
                    continue
                target = folder / original
                if not target.resolve().is_relative_to(folder.resolve()):
                    raise ValueError('Unsafe image path in source metadata')
                if target.exists():
                    continue
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary = target.with_suffix(".mirror.part")
                with archive.extractfile(member) as contents:
                    temporary.write_bytes(contents.read())
                temporary.replace(target)
                count += 1
                if count % 100 == 0:
                    print("TACO mirror recovered", count, flush=True)
    missing = [i["file_name"] for i in data["images"] if not (folder/i["file_name"]).is_file()]
    provenance = {"source":"https://github.com/dataset-ninja/taco", "archive_url":url,
                  "images_recovered":count, "missing":missing,
                  "filename_policy":"batch_N_filename maps to original batch_N/filename"}
    (folder/"mirror_provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    if missing:
        raise RuntimeError(f"Mirror missing {len(missing)} images")
    print("TACO complete:",len(data["images"]),flush=True)


def range_download(url, destination, chunk_size=16*1024*1024):
    """Restart-safe range downloads, bounded memory, no discarded multi-GB transfer."""
    if chunk_size < 1:
        raise ValueError('Chunk size must be positive')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(destination.with_suffix(destination.suffix + '.lock'), timeout=0):
        return _range_download(url, destination, chunk_size)


def _range_download(url, destination, chunk_size):
    if destination.is_file():
        return destination
    headers = {'Accept-Encoding': 'identity'}
    with requests.get(url, headers={**headers, 'Range': 'bytes=0-0'}, stream=True, timeout=30) as response:
        response.raise_for_status()
        content_range = re.fullmatch(r'bytes 0-0/([1-9][0-9]*)', response.headers.get('Content-Range', ''))
        if response.status_code != 206 or content_range is None:
            raise RuntimeError('Server does not support partial downloads')
        size = int(content_range[1])
        etag = response.headers.get('ETag')
        validator = etag if etag and not etag.startswith('W/') else response.headers.get('Last-Modified')
        if validator:
            headers['If-Range'] = validator
    temporary=destination.with_suffix('.part')
    ledger=destination.with_suffix('.chunks.json')
    identity = {'url': url, 'size': size, 'chunk_size': chunk_size, 'validator': validator}
    total = (size + chunk_size - 1) // chunk_size
    try:
        state = json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else None
    except json.JSONDecodeError:
        state = None
    completed = set()
    if (validator and isinstance(state, dict) and state.get('archive') == identity
            and temporary.is_file() and temporary.stat().st_size == size):
        completed = {index for index in state.get('completed', []) if type(index) is int and 0 <= index < total}
    if not completed:
        ledger.unlink(missing_ok=True)
        with temporary.open('wb') as output:
            output.truncate(size)

    def fetch(index):
        start=index*chunk_size;end=min(size,start+chunk_size)-1
        for attempt in range(5):
            try:
                with requests.get(url, headers={**headers, 'Range': f'bytes={start}-{end}'},
                                  stream=True, timeout=(20,90)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise OSError('Incorrect range response')
                    if r.headers.get('Content-Encoding', 'identity') != 'identity':
                        raise OSError('Unexpected content encoding for byte range')
                    expected = end - start + 1
                    data = bytearray()
                    for chunk in r.iter_content(min(expected, 1024 * 1024)):
                        if len(data) + len(chunk) > expected:
                            raise OSError('Incorrect range response size')
                        data.extend(chunk)
                    if len(data) != expected:
                        raise OSError('Incorrect range response size')
                    return index, data
            except (requests.RequestException,OSError):
                if attempt==4:raise
                time.sleep(2**attempt)
    pending=(i for i in range(total) if i not in completed)
    workers=6
    with temporary.open('r+b') as output, ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fetch,i) for i in islice(pending,workers)}
        while futures:
            finished,futures=wait(futures,return_when=FIRST_COMPLETED)
            while finished:
                future=finished.pop()
                index,data=future.result()
                del future
                output.seek(index*chunk_size);output.write(data);output.flush()
                os.fsync(output.fileno())
                del data
                completed.add(index)
                ledger_tmp=ledger.with_suffix('.tmp')
                ledger_tmp.write_text(json.dumps({'archive': identity, 'completed': sorted(completed)}),
                                      encoding='utf-8')
                ledger_tmp.replace(ledger)
                if len(completed)%10==0:print(f'Archive chunks {len(completed)}/{total}',flush=True)
            futures.update(pool.submit(fetch,i) for i in islice(pending,workers-len(futures)))
    temporary.replace(destination)
    ledger.unlink(missing_ok=True)
    return destination


def trashcan() -> None:
    folder = RAW / "trashcan"
    folder.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"{API}/pid/find", params={"id": "11299/214865"}, timeout=40)
    response.raise_for_status()
    item = response.json()
    (folder / "repository_metadata.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
    response = requests.get(item["_links"]["bundles"]["href"], params={"size": 100}, timeout=40)
    response.raise_for_status()
    bundles = response.json()["_embedded"]["bundles"]
    archive = None
    for bundle in bundles:
        if bundle["name"] != "ORIGINAL":
            continue
        response = requests.get(bundle["_links"]["bitstreams"]["href"], params={"size": 100}, timeout=40)
        response.raise_for_status()
        for bitstream in response.json()["_embedded"]["bitstreams"]:
            if bitstream["name"] == "dataset.zip":
                print("TrashCan: downloading original dataset.zip", bitstream.get("sizeBytes"), flush=True)
                archive = download(bitstream["_links"]["content"]["href"], folder / "dataset.zip")
                expected = bitstream.get("checkSum", {})
                if expected.get("checkSumAlgorithm", "").upper() == "MD5":
                    with archive.open("rb") as contents:
                        actual = hashlib.file_digest(contents, "md5").hexdigest()
                    if actual != expected["value"]:
                        raise ValueError("TrashCan checksum mismatch")
                (folder / "bitstream_metadata.json").write_text(json.dumps(bitstream, indent=2), encoding="utf-8")
    if archive is None:
        raise RuntimeError("Original dataset.zip not found in university repository")
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            target = (folder / entry.filename).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError("Unsafe archive path")
        zipped.extractall(folder)
    print("TrashCan extracted", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["taco", "trashcan", "all"])
    parser.add_argument("--mirror", action="store_true", help="Use the public Dataset Ninja TACO archive")
    args = parser.parse_args()
    if args.dataset in ("taco", "all"):
        taco_mirror() if args.mirror else taco()
    if args.dataset in ("trashcan", "all"):
        trashcan()
