"""Wire format of a motion timeline (API → browser). Compact and lossless enough for display:
quaternions as int16 (q · 32767, error < 0.003°·… ≈ 1e-4 rad), morph weights as uint8. A 10 s sentence ≈ 250 frames × 45 bones →
~90 kB raw, ~45 kB gzip."""
from __future__ import annotations

import base64

import numpy as np

from .retarget import MORPHS
from .rig import BONES


def encode(q, morph, fps=25.0, segments=None, extra=None):
    q = np.asarray(q, np.float32)
    T = q.shape[0]
    qi = np.clip(np.round(q * 32767.0), -32767, 32767).astype("<i2")
    mi = np.clip(np.round(np.asarray(morph, np.float32) * 255.0), 0, 255).astype(np.uint8)
    out = dict(fps=float(fps), frames=int(T), bones=BONES, morphs=MORPHS,
               q=base64.b64encode(qi.tobytes()).decode("ascii"), m=base64.b64encode(mi.tobytes()).decode("ascii"),
               segments=segments or [])
    if extra:
        out.update(extra)
    return out


def decode(d):
    T, B, M = d["frames"], len(d["bones"]), len(d["morphs"])
    q = np.frombuffer(base64.b64decode(d["q"]), "<i2").astype(np.float32).reshape(T, B, 4) / 32767.0
    m = np.frombuffer(base64.b64decode(d["m"]), np.uint8).astype(np.float32).reshape(T, M) / 255.0
    return q, m
