"""COCO -> single-class YOLO, grouped splits and reproducible provenance."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random

import cv2
import numpy as np
import yaml

from .detector import ROOT, write_image


def load_coco_pixels(path, image):
    """Choose the orientation matching COCO, then strip EXIF on re-encoding."""
    encoded=np.fromfile(path,dtype=np.uint8)
    for flag in (cv2.IMREAD_COLOR,cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION):
        pixels=cv2.imdecode(encoded,flag)
        if pixels is not None:
            h,w=pixels.shape[:2]
            if abs((w/h)/(image['width']/image['height'])-1)<=.01:
                return pixels
    raise ValueError(f"Image orientation/dimensions disagree with COCO: {path}")


def coco_box(bbox, width, height):
    x, y, w, h = map(float, bbox)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x+w), min(height, y+h)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)


def split_groups(records, seed=42):
    # Exact decoded-pixel duplicates join their entire source groups.
    parent = {r["group"]: r["group"] for r in records}
    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    seen = {}
    for record in records:
        group = record["group"]
        digest = record["pixel_sha256"]
        if digest in seen:
            parent[find(group)] = find(seen[digest])
        else:
            seen[digest] = group
    groups = defaultdict(list)
    for record in records:
        record["group"] = find(record["group"])
        groups[record["group"]].append(record)
    if len(groups) < 3:
        raise ValueError("Не хватает независимых групп для train/val/test")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)
    totals = {"train": len(records)*0.7, "val": len(records)*0.15, "test": len(records)*0.15}
    counts = Counter()
    for key in keys:
        split = max(totals, key=lambda s: totals[s]-counts[s])
        for record in groups[key]:
            record["split"] = split
        counts[split] += len(groups[key])
    return records


def prepare(profile, seed=42):
    raw = ROOT / "data/raw"
    if profile == "land":
        folder = raw / "taco"
        sources = [(folder/"annotations.json", folder)]
    elif profile == "underwater":
        folder = raw / "trashcan/dataset"
        images = folder / "original_data/images"
        sources = [(folder/"instance_version"/f"instances_{s}_trashcan.json", images) for s in ("train", "val")]
    else:
        raise ValueError(profile)
    records = []
    category_mapping = {}
    invalid = 0
    missing = []
    annotations_hashes = {}
    for annotation_path, images_folder in sources:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        annotations_hashes[str(annotation_path.relative_to(ROOT))] = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
        mapping = {c["id"]: (0 if profile == "land" or (c["name"].startswith("trash_") and c["name"] != "trash_branch") else None)
                   for c in data["categories"]}
        category_mapping.update({c["name"]: mapping[c["id"]] for c in data["categories"]})
        by_image = defaultdict(list)
        for annotation in data["annotations"]:
            if mapping[annotation["category_id"]] == 0:
                by_image[annotation["image_id"]].append(annotation)
        for image in data["images"]:
            path = images_folder / image["file_name"]
            if not path.resolve().is_relative_to(images_folder.resolve()):
                raise ValueError("Unsafe image path")
            if not path.is_file():
                missing.append(str(path))
                continue
            pixels = load_coco_pixels(path,image)
            h, w = pixels.shape[:2]
            if abs((w/h)/(image["width"]/image["height"])-1) > .01:
                raise ValueError(f"Annotation dimensions disagree: {path}: {(w,h)} vs {image['width'],image['height']}")
            labels = []
            for annotation in by_image[image["id"]]:
                box = coco_box(annotation["bbox"], image["width"], image["height"])
                if box is None:
                    invalid += 1
                else:
                    labels.append("0 " + " ".join(f"{v:.8f}" for v in box))
            group = Path(image["file_name"]).parent.as_posix() if profile == "land" else path.stem.split("_frame")[0]
            name = image["file_name"].replace("/", "_").replace("\\", "_")
            records.append({"source": str(path.relative_to(ROOT)), "name": name, "group": group,
                            "source_annotation": annotation_path.name, "labels": sorted(set(labels)),
                            "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "width": w, "height": h,
                            "annotation_width": image["width"], "annotation_height": image["height"]})
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} source images, e.g. {missing[:3]}")
    records = split_groups(records, seed)
    output = ROOT / "data/prepared" / profile
    # Refuse to mix a previous split and a new one in the same tree.
    if (output / "manifest.json").exists():
        previous = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        if [(r["source"],r["split"]) for r in previous["images"]] != [(r["source"],r["split"]) for r in records]:
            raise ValueError("Prepared dataset already exists with a different split; choose a fresh destination")
        if previous.get("annotation_sha256") != annotations_hashes:
            raise ValueError("Source annotations changed; use a fresh versioned dataset")
        for old, record in zip(previous["images"], records):
            if old.get("file_sha256") != record["file_sha256"]:
                raise ValueError(f"Source image changed: {record['source']}; use a fresh versioned dataset")
        reusable = True
    else:
        # An interrupted first preparation has no provenance for existing copies.
        reusable = False
    for record in records:
        image_dir = output / "images" / record["split"]
        label_dir = output / "labels" / record["split"]
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        source = ROOT / record["source"]
        target = image_dir / record["name"]
        if not reusable or not target.exists():
            # Re-encode to strip EXIF orientation and cap very large TACO images.
            pixels = load_coco_pixels(source,{'width':record['annotation_width'],'height':record['annotation_height']})
            if max(pixels.shape[:2]) > 1600:
                scale = 1600/max(pixels.shape[:2])
                pixels = cv2.resize(pixels, (round(pixels.shape[1]*scale), round(pixels.shape[0]*scale)), interpolation=cv2.INTER_AREA)
            write_image(target, pixels)
        (label_dir / Path(record["name"]).with_suffix(".txt")).write_text("\n".join(record["labels"])+("\n" if record["labels"] else ""), encoding="utf-8")
    config = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "trash"}}
    (output/"data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    summary = {}
    for split in ("train", "val", "test"):
        subset = [r for r in records if r["split"] == split]
        summary[split] = {"images": len(subset), "groups": len({r["group"] for r in subset}),
                          "objects": sum(len(r["labels"]) for r in subset), "negative_images": sum(not r["labels"] for r in subset)}
        if not subset or not summary[split]["objects"]:
            raise ValueError(f"Empty split or missing positive objects: {split}")
    manifest = {"profile": profile, "seed": seed, "split_policy": "whole TACO batches / whole TrashCan video IDs; exact pixel duplicate groups joined",
                "category_mapping": category_mapping, "annotation_sha256": annotations_hashes,
                "discarded_invalid_boxes": invalid, "summary": summary, "images": records,
                "limitations": "Video IDs/batches and exact duplicates grouped; unlabelled cross-video scene overlap cannot be ruled out."}
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output/"class_mapping.json").write_text(json.dumps(category_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(profile, json.dumps(summary), flush=True)
    return manifest
