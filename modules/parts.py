"""Encoder input preparation (NumPy only — shared by training, the ONNX production runtime and the web API).

Standard-Schema keypoints [T,133,2] + scores [T,133] → the four part streams the Uni-Sign pose encoder reads:
  body 9 · left hand 21 · right hand 21 · face 18, each [T,V,3] = (x, y, score), normalised exactly like Uni-Sign's
  `datasets.load_part_kp` (body box → [-1, 1]; hands / face relative to their root, divided by the body scale).
"""
from __future__ import annotations

import copy

import numpy as np

MODES = ("body", "left", "right", "face_all")
BODY_IDX = [0] + list(range(3, 11))
FACE_IDX = list(range(23, 23 + 17))[::2] + list(range(83, 83 + 8)) + [53]


def _crop_scale(motion, thr):
    result = copy.deepcopy(motion)
    valid = motion[motion[..., 2] > thr][:, :2]
    if len(valid) < 4:
        return np.zeros(motion.shape), 0
    xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
    ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape), 0
    xs, ys = (xmin + xmax - scale) / 2, (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result, scale


def prepare_parts(kp: np.ndarray, sc: np.ndarray, thr=0.3):
    """kp [T,133,2] (isotropic units), sc [T,133] → {part: float32 [T,V,3]} (official load_part_kp)."""
    kp = kp.astype(np.float64); sc = sc.astype(np.float64)
    out = {}
    body = np.concatenate([kp[:, BODY_IDX], sc[:, BODY_IDX, None]], -1)
    out["body"], scale = _crop_scale(body, thr)
    for part, sl, root in (("left", slice(91, 112), 0), ("right", slice(112, 133), 0), ("face_all", FACE_IDX, -1)):
        k = kp[:, sl]
        k = k - k[:, [root] if root >= 0 else [-1]]
        r = np.concatenate([k, sc[:, sl, None]], -1)
        if scale == 0:
            r = np.zeros(r.shape)
        else:
            r[..., :2] = r[..., :2] / scale
            r = np.clip(r, -1, 1)
            r[r[..., 2] <= thr] = 0
        out[part] = r
    return {p: out[p].astype(np.float32) for p in MODES}


def resample_len(kp, sc, n_out):
    n = len(kp)
    if n_out == n or n < 2:
        return kp, sc
    t = np.linspace(0, n - 1, max(2, n_out))
    i0 = np.floor(t).astype(int); i1 = np.minimum(i0 + 1, n - 1); w = (t - i0)[:, None, None]
    return (kp[i0] * (1 - w) + kp[i1] * w).astype(np.float32), np.minimum(sc[i0], sc[i1])


MAX_LEN = 96                   # frames at 25 fps (≈3.8 s); longer spans are uniformly sub-sampled


def prep(kp_iso, sc, max_len=MAX_LEN):
    if len(kp_iso) > max_len:
        kp_iso, sc = resample_len(kp_iso, sc, max_len)
    return prepare_parts(kp_iso, sc)


def to_iso(kp, hw):
    H, W = hw
    return kp * np.array([W / H, 1.0], np.float32)


def collate_np(parts_list):
    """→ dict part → [B,L,V,3] float32 (padded by repeating the last frame, as the official collate) + mask [B,L] float32."""
    L = max(len(p["body"]) for p in parts_list)
    B = len(parts_list)
    out = {m: np.zeros((B, L, parts_list[0][m].shape[1], 3), np.float32) for m in MODES}
    mask = np.zeros((B, L), np.float32)
    for b, p in enumerate(parts_list):
        n = len(p["body"])
        for m in MODES:
            out[m][b, :n] = p[m]
            if n < L:
                out[m][b, n:] = p[m][-1]
        mask[b, :n] = 1.0
    return out, mask
