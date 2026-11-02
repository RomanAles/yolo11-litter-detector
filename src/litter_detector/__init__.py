"""Offline-first litter detection with separate land and underwater profiles."""

import os
from pathlib import Path

# Keep Ultralytics configuration in the project, including launches from the UI.
_cache = Path(__file__).resolve().parents[2] / ".cache"
_cache.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_cache))

__version__ = "0.1.0"
