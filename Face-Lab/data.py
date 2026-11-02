"""FER2013 data for Face-Lab: load once into uint8 arrays (cached), profile it, split it, augment it on the GPU.

FER2013-Data/{train,test}/<class>/*.jpg|png — 48x48 grayscale faces. The official `test` folder (7178) is never used to choose
anything; the validation set is cut from `train` (stratified, duplicate-safe: identical images stay on the same side).
"""
from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "FER2013-Data"
CACHE = ROOT / "cache"
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]      # FER2013 order = class ids 0..6 (label_map.json)
IMG = 48


def _read(p):
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
    return im


def load_raw(force=False):
    """-> dict(split -> (X uint8 [N,48,48], y int64 [N], names [N])) from FER2013-Data (cached in Face-Lab/cache/fer2013.npz)."""
    f = CACHE / "fer2013.npz"
    if f.exists() and not force:
        z = np.load(f, allow_pickle=False)
        return {s: (z[f"{s}_x"], z[f"{s}_y"], z[f"{s}_n"]) for s in ("train", "test")}
    out, bad = {}, []
    for s in ("train", "test"):
        paths = [(p, i) for i, c in enumerate(CLASSES) for p in sorted((RAW / s / c).iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        with ThreadPoolExecutor(8) as ex:
            ims = list(ex.map(lambda t: _read(t[0]), paths))
        X, y, n = [], [], []
        for (p, i), im in zip(paths, ims):
            if im is None or im.ndim != 2 or im.shape != (IMG, IMG):
                bad.append((str(p.relative_to(RAW)), None if im is None else list(im.shape)))
                continue
            X.append(im); y.append(i); n.append(f"{s}/{CLASSES[i]}/{p.name}")
        out[s] = (np.stack(X), np.array(y, np.int64), np.array(n))
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(f, **{f"{s}_{k}": v for s, t in out.items() for k, v in zip("xyn", t)})
    (CACHE / "load_issues.json").write_text(json.dumps(bad))
    return out


def dhash(im):
    """64-bit difference hash: near-duplicate detector (same picture re-encoded / lightly changed)."""
    s = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    return int("".join("1" if v else "0" for v in (s[:, 1:] > s[:, :-1]).ravel()), 2)


def profile(d):
    """Numbers for the dataset report: counts, classes, pixel range, exact / near duplicates, cross-split leakage."""
    P = {"classes": CLASSES, "image": {"size": [IMG, IMG], "channels": 1, "dtype": "uint8"}}
    ex = {}
    for s, (X, y, n) in d.items():
        h = [hashlib.md5(x.tobytes()).hexdigest() for x in X]
        P[s] = {"n": int(len(X)), "per_class": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
                "pixel_min": int(X.min()), "pixel_max": int(X.max()), "mean": round(float(X.mean()) / 255, 4), "std": round(float(X.std()) / 255, 4),
                "blank_or_flat": int((X.reshape(len(X), -1).std(1) < 2).sum())}
        ex[s] = h
        by = {}
        for i, k in enumerate(h):
            by.setdefault(k, []).append(i)
        dup = [v for v in by.values() if len(v) > 1]
        conflict = [v for v in dup if len({int(y[i]) for i in v}) > 1]
        P[s].update(exact_dup_groups=len(dup), exact_dup_images=int(sum(len(v) for v in dup)), exact_dup_label_conflicts=len(conflict))
    tr, te = set(ex["train"]), set(ex["test"])
    P["leak_exact_test_in_train"] = int(sum(k in tr for k in ex["test"]))
    # near duplicates (dHash equal) across train -> test
    ht = {}
    for i, x in enumerate(d["train"][0]):
        ht.setdefault(dhash(x), []).append(i)
    near = [i for i, x in enumerate(d["test"][0]) if dhash(x) in ht]
    P["leak_near_test_in_train"] = len(near)
    cnt = np.array([P["train"]["per_class"][c] for c in CLASSES])
    P["imbalance_ratio_max_min"] = round(float(cnt.max() / cnt.min()), 2)
    return P


def split_train_val(X, y, names, val_frac=0.1, seed=42):
    """Stratified validation split from `train`; identical images (same md5) always land on the same side (no duplicate leakage)."""
    h = np.array([hashlib.md5(x.tobytes()).hexdigest() for x in X])
    rng = np.random.RandomState(seed)
    val = np.zeros(len(X), bool)
    for c in range(len(CLASSES)):
        groups = {}
        for i in np.where(y == c)[0]:
            groups.setdefault(h[i], []).append(i)
        keys = list(groups); rng.shuffle(keys)
        need, got = int(round(val_frac * (y == c).sum())), 0
        for k in keys:
            if got >= need:
                break
            val[groups[k]] = True; got += len(groups[k])
    return np.where(~val)[0], np.where(val)[0]


def prepare(seed=42, val_frac=0.1):
    """-> dict(train, val, test = (X, y), clean_test = bool mask [N_test], excluded = record).
    Excluded from TRAINING (recorded, nothing else is dropped): flat / blank images and every image of an exact-duplicate group whose
    copies carry different labels (ambiguous). The official test set is untouched; `clean_test` marks the test images that have no exact
    or near (dHash) twin anywhere in the training folder — the leak-free subset (FER2013 test has ~8% images that also sit in train)."""
    d = load_raw()
    X, y, names = d["train"]
    h = np.array([hashlib.md5(x.tobytes()).hexdigest() for x in X])
    flat = X.reshape(len(X), -1).std(1) < 2
    lab = {}
    for k, c in zip(h, y):
        lab.setdefault(k, set()).add(int(c))
    conflict = np.array([len(lab[k]) > 1 for k in h])
    keep = ~(flat | conflict)
    tr_i, va_i = split_train_val(X[keep], y[keep], names[keep], val_frac, seed)
    idx = np.where(keep)[0]
    tr_i, va_i = idx[tr_i], idx[va_i]
    Xt, yt, _ = d["test"]
    seen_md5 = set(h)
    seen_dh = {dhash(x) for x in X}
    clean = np.array([hashlib.md5(x.tobytes()).hexdigest() not in seen_md5 and dhash(x) not in seen_dh for x in Xt])
    return dict(train=(X[tr_i], y[tr_i]), val=(X[va_i], y[va_i]), test=(Xt, yt), clean_test=clean,
                excluded=dict(flat_images=int(flat.sum()), label_conflict_duplicates=int(conflict.sum()), total=int((~keep).sum()),
                              test_not_clean=int((~clean).sum())))


def class_weights(y, power=0.5):
    """Inverse-frequency^power weights, mean 1 (power 0.5 = sqrt: disgust is 11x rarer than happy — full inverse would let a few hundred
    images dominate the loss)."""
    c = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    w = (c.sum() / c) ** power
    return (w / (w * c / c.sum()).sum()).astype(np.float32)


# ---------------------------------------------------------------- GPU augmentation / preprocessing
def to_res(x, res):
    """uint8-valued [N,1,48,48] -> [N,1,res,res] in 0..1 (bilinear): the evaluation / deployment preprocessing."""
    import torch.nn.functional as F
    return F.interpolate(x / 255.0, size=(res, res), mode="bilinear", align_corners=False)


def augment(x, res, strength=1.0, gen=None):
    """Training augmentation on the GPU: x uint8-valued float [N,1,48,48] -> [N,1,res,res] in 0..1. One resampling does flip +
    rotation + scale + shift + the 48 -> res enlargement. Then (each sample, random) low-resolution/blur, brightness / contrast / gamma,
    noise, a small occluding patch. Mild by design: an expression must stay the same expression."""
    import torch
    import torch.nn.functional as F
    n, dev = x.shape[0], x.device
    r = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi, generator=gen)       # noqa: E731
    ang = r(-12, 12) * strength * math.pi / 180
    sc = 1.0 + r(-0.10, 0.10) * strength
    flip = (torch.rand(n, device=dev, generator=gen) < 0.5).float() * 2 - 1
    tx, ty = r(-0.08, 0.08) * strength, r(-0.08, 0.08) * strength
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    g = F.affine_grid(theta, (n, 1, res, res), align_corners=False)
    y = F.grid_sample(x / 255.0, g, mode="bilinear", padding_mode="border", align_corners=False)
    # low-resolution webcam / compression look: some samples are shrunk to 24-40 px and enlarged again
    low = torch.rand(n, device=dev, generator=gen) < 0.25
    if low.any():
        s = int(np.random.randint(24, 41))
        y[low] = F.interpolate(F.interpolate(y[low], size=(s, s), mode="area"), size=(res, res), mode="bilinear", align_corners=False)
    c, b, gm = r(0.75, 1.3), r(-0.12, 0.12), torch.exp(r(-0.25, 0.25))
    y = ((y.clamp(1e-4, 1) ** gm.view(-1, 1, 1, 1) - 0.5) * c.view(-1, 1, 1, 1) + 0.5 + b.view(-1, 1, 1, 1))
    y = y + torch.randn(y.shape, device=dev, generator=gen) * r(0, 0.03).view(-1, 1, 1, 1)
    # occluding patch (hand / hair / mic edge) on 20% of the samples, up to 1/4 of the side
    occ = torch.rand(n, device=dev, generator=gen) < 0.2
    if occ.any():
        idx = occ.nonzero().squeeze(1)
        for i in idx.tolist():
            w, h = int(res * np.random.uniform(0.12, 0.3)), int(res * np.random.uniform(0.12, 0.3))
            x0, y0 = np.random.randint(0, res - w), np.random.randint(0, res - h)
            y[i, :, y0:y0 + h, x0:x0 + w] = float(np.random.uniform(0.1, 0.9))
    return y.clamp(0, 1)
