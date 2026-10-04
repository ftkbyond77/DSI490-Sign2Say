"""Logging, timers, seeding, config and path helpers."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("THAISLM_ARTIFACTS", ROOT / "artifacts"))   # model, pretrained weights, run outputs, reports (in git)
DATA = Path(os.environ.get("THAISLM_DATA", ROOT / "cloud_s3"))           # local mirror of s3://dsi490-lake-signdata (not in git)
RAW = DATA / "raw_data"                                                  # raw videos / landmarks as published
PREP = DATA / "data_prep"                                                # standard-schema outputs (this repo builds it; pushed to S3)
CACHE = ROOT / "cache"                                                   # local scratch (not in git, not on S3)
MODEL = Path(os.environ.get("THAISLM_MODELS", ROOT / "models"))         # the deployable model: everything inference loads (in git, LFS)
PRETRAINED = ART / "pretrained"                                          # Uni-Sign base weights


def get_logger(name: str = "thaislm", level: str | None = None) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(level or os.environ.get("LOG_LEVEL", "INFO"))
    return log


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def device(pref: str | None = None) -> str:
    pref = pref or os.environ.get("DEVICE", "auto")
    if pref != "auto":
        return pref
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store.setdefault(key, []).append((time.perf_counter() - t0) * 1000.0)


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def safe_id(clip_id: str) -> str:
    """clip ids contain ':' which is not allowed in Windows filenames."""
    return clip_id.replace(":", "_")
