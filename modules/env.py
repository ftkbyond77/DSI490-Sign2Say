"""Load secrets from .env (preferred) or .env.example into os.environ without printing them."""
from __future__ import annotations

import os

from .utils import ROOT


def load_env(override=False):
    for name in (".env", ".env.example"):
        p = ROOT / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.split(" #")[0].strip().strip('"').strip("'")
            if v and (override or not os.environ.get(k.strip())):
                os.environ[k.strip()] = v
    return bool(os.environ.get("OPENAI_API_KEY"))
