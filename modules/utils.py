"""Logging, timers, seeding, config and path helpers."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _strip_env_comments() -> None:
    """VS Code (Jupyter kernels, terminals) loads .env into the process as written: `DEVICE=auto   # auto | cuda | cpu` arrives as
    "auto   # auto | cuda | cpu". For every key .env / .env.example define, a trailing ` # …` is cut from the value (never printed)."""
    for name in (".env", ".env.example"):
        p = ROOT / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            k = line.split("=", 1)[0].strip()
            v = os.environ.get(k)
            if v and re.search(r"\s#", v):
                os.environ[k] = re.split(r"\s+#", v, maxsplit=1)[0].strip()


_strip_env_comments()
ART =Path(os.environ.get("THAISLM_ARTIFACTS", ROOT / "artifacts"))   # model, pretrained weights, run outputs, reports (in git)
DATA = Path(os.environ.get("THAISLM_DATA", ROOT / "cloud_s3"))           # local mirror of s3://dsi490-lake-signdata (not in git)
RAW = DATA / "raw_data"                                                  # raw videos / landmarks as published
PREP = DATA / "data_prep"                                                # standard-schema outputs (this repo builds it; pushed to S3)
CACHE = ROOT / "cache"                                                   # local scratch (not in git, not on S3)
MODEL = Path(os.environ.get("THAISLM_MODELS", ROOT / "models" / "model-v10"))  # the deployable model (main forward = model-v10; v9 = model-v9 rollback; in git, LFS)
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
    """Choose local hardware only; explicit unavailable devices fail clearly."""
    import torch
    pref = str(pref or os.environ.get("DEVICE", "auto")).lower()
    mps = getattr(torch.backends, "mps", None)
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if mps is not None and mps.is_available() else "cpu"
    target = torch.device(pref)
    if target.type == "cuda":
        if not torch.cuda.is_available() or (target.index is not None and target.index >= torch.cuda.device_count()):
            raise ValueError(f"Device {pref} is unavailable; use --device auto or cpu")
    elif target.type == "mps":
        if mps is None or not mps.is_available():
            raise ValueError("MPS is unavailable; use --device cpu")
    elif target.type != "cpu":
        raise ValueError(f"Unsupported device: {pref}; choose auto, cpu, mps or cuda")
    return str(target)


def inference_precision(target):
    """Use BF16 only on CUDA hardware that supports it; Mac/CPU stay FP32."""
    import torch
    if torch.device(target).type == "cuda":
        with torch.cuda.device(target):
            if torch.cuda.is_bf16_supported():
                return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


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
