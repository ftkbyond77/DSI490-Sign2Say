"""Fingerspelling: letters spelled with the hands (English A–Z = the one-handed American manual alphabet, as Thai schools for the
deaf teach it; Thai consonants / vowels = Thai fingerspelling, mostly two-handed).

Letters differ by handshape and orientation, not by large movements, so they get their own small recogniser on top of the same
Standard Schema landmarks (no extra model input): per short window (8 frames = 0.32 s at 25 fps)
  * the dominant (right, after canonical_hands) hand's shape: 21 points relative to the wrist, scaled by the palm length,
    orientation kept (G/H/P/Q differ only by orientation)
  * finger bend angles and fingertip distances
  * the other hand's shape and where the dominant fingertip is relative to it (Thai letters and vowels use both hands)
  * a short fingertip trajectory (J and Z are drawn in the air)
  * where the hand is relative to the shoulders
→ an MLP (NumPy forward pass; weights in models/fingerspell.npz) over the letters of both alphabets + "not a letter".

`spell_runs` turns window probabilities into letter events: windows of a held, confident letter, merged, at least 0.2 s long.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import LH, RH

SH_L, SH_R, WR_L, WR_R = 5, 6, 9, 10           # COCO-WholeBody body slots
WIN, STRIDE = 8, 2
TIPS = [4, 8, 12, 16, 20]
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
EN_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
TH_LETTERS = list("กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ") + ["สระอิ", "สระอี", "สระอุ", "สระอู", "สระโอ", "สระใอ", "สระไอ"]
NONE = "_none"


def _hand(kpi, sc, idx, thr=0.3):
    """[T,21,2] points (iso coords) and a [T] presence mask."""
    pts = kpi[:, idx, :]
    ok = sc[:, idx].mean(1) > thr
    return pts, ok


def _shape(pts):
    """[T,21,2] → [T,42] wrist-relative points scaled by the palm length (wrist → middle-finger base), orientation kept."""
    rel = pts - pts[:, :1]
    palm = np.linalg.norm(rel[:, 9], axis=1, keepdims=True) + 1e-6
    return (rel / palm[:, :, None]).reshape(len(pts), -1), palm[:, 0]


def _upright(pts):
    """[T,21,2] → ([T,42] shape rotated so the palm axis (wrist → middle-finger base) points up, [T,2] that axis' cos/sin):
    finger configuration independent of how the hand / camera is tilted; the orientation itself stays a separate feature."""
    rel = pts - pts[:, :1]
    ax = rel[:, 9]
    L = np.linalg.norm(ax, axis=1, keepdims=True) + 1e-6
    c, s = ax[:, 1:2] / L, ax[:, 0:1] / L                      # rotate the axis onto (0, -1) (image y grows downwards)
    rot = np.stack([np.concatenate([-c, s], 1), np.concatenate([-s, -c], 1)], 1)    # [T,2,2]
    up = np.einsum("tij,tkj->tki", rot, rel) / L[:, :, None]
    return up.reshape(len(pts), -1), np.concatenate([c, s], 1)


def _detail(up):
    """Fine distinctions in the upright hand ([T,42] → [T,F]):
    * thumb tip relative to each finger's base and middle joint (A / S / T / M / N / E differ only in where the thumb sits)
    * index–middle relation: signed horizontal offset of the tips (crossed = R), their spread and height difference (U / V / R / K)
    * every fingertip's height above its base (bent vs straight)."""
    P = up.reshape(len(up), 21, 2)
    th = P[:, 4]
    f = [th - P[:, j] for j in (5, 6, 9, 10, 13, 14, 17, 18)]                       # thumb tip → base / middle joint of each finger
    f.append((P[:, 8, :1] - P[:, 12, :1]))                                          # signed: index left/right of middle (crossing)
    f.append(np.linalg.norm(P[:, 8] - P[:, 12], axis=1, keepdims=True))             # spread
    f.append(P[:, 8, 1:] - P[:, 12, 1:])
    f += [P[:, t, 1:] - P[:, b, 1:] for t, b in ((4, 2), (8, 5), (12, 9), (16, 13), (20, 17))]
    return np.concatenate([x.reshape(len(up), -1) for x in f], 1)


def _contacts(dom, non):
    """Where the dominant thumb / index tip is on the other hand (Thai vowels touch different points of it): offsets to the other
    hand's 5 fingertips and 5 finger bases, scaled by its palm length → [T,40]."""
    palm = np.linalg.norm(non[:, 9] - non[:, 0], axis=1, keepdims=True)[:, :, None] + 1e-6
    out = []
    for tip in (4, 8):
        out.append(((dom[:, tip:tip + 1] - non[:, [4, 8, 12, 16, 20, 1, 5, 9, 13, 17]]) / palm).reshape(len(dom), -1))
    return np.concatenate(out, 1)


def _angles(pts):
    """[T,21,2] → [T,15] bend angle at each finger joint (cos of the angle between consecutive bones)."""
    out = []
    for f in FINGERS:
        chain = [0] + f
        for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:]):
            u, v = pts[:, b] - pts[:, a], pts[:, c] - pts[:, b]
            out.append((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-6))
    return np.stack(out, 1)


def _tipdist(pts, palm):
    d = [np.linalg.norm(pts[:, TIPS[i]] - pts[:, TIPS[j]], axis=1) for i in range(5) for j in range(i + 1, 5)]
    return np.stack(d, 1) / palm[:, None]


def frame_features(kpi, sc):
    """Per-frame features [T,F] and the dominant-hand presence [T]."""
    T = len(kpi)
    dom, dok = _hand(kpi, sc, RH)
    non, nok = _hand(kpi, sc, LH)
    dshape, dpalm = _shape(dom)
    nshape, npalm = _shape(non)
    sh_mid = (kpi[:, SH_L] + kpi[:, SH_R]) / 2
    sw = np.linalg.norm(kpi[:, SH_L] - kpi[:, SH_R], axis=1) + 1e-6
    dup, dax = _upright(dom)
    feats = [
        dshape, dup, dax, _detail(dup), _angles(dom), _tipdist(dom, dpalm),
        np.where(nok[:, None], np.clip(_contacts(dom, non), -6, 6), 0.0),
        np.where(nok[:, None], nshape, 0.0), np.where(nok[:, None], _angles(non), 0.0), nok[:, None].astype(np.float32),
        np.where(nok[:, None], (dom[:, 8] - non[:, 9]) / sw[:, None], 0.0),           # dominant index tip → other palm centre
        np.where(nok[:, None], (dom[:, 0] - non[:, 0]) / sw[:, None], 0.0),
        (dom[:, 0] - sh_mid) / sw[:, None],                                            # hand position relative to the shoulders
        (dpalm / sw)[:, None],
    ]
    F = np.concatenate([np.asarray(f, np.float32).reshape(T, -1) for f in feats], 1)
    F[~dok] = 0.0
    return np.nan_to_num(F), dok


def window_features(kpi, sc, starts=None):
    """Window features [N,D] for windows of WIN frames starting at `starts` (default: every STRIDE frames) and those starts."""
    T = len(kpi)
    F, dok = frame_features(kpi, sc)
    if starts is None:
        starts = np.arange(0, max(T - WIN + 1, 1), STRIDE)
    dom, _ = _hand(kpi, sc, RH)
    _, dpalm = _shape(dom)
    out, keep = [], []
    for s in starts:
        e = min(T, s + WIN)
        m = dok[s:e]
        if m.sum() < max(3, (e - s) // 2):
            keep.append(False); out.append(None); continue
        f = F[s:e][m]
        palm = np.median(dpalm[s:e][m]) + 1e-6
        traj = []
        for tip in (8, 20, 0):                      # index tip, little-finger tip, wrist: path relative to the window start
            p = dom[s:e][m, tip]
            q = (p[np.linspace(0, len(p) - 1, 4).round().astype(int)] - p[0]) / palm
            traj.append(q.reshape(-1))
        out.append(np.concatenate([f.mean(0), f.std(0)[:42], np.concatenate(traj)]).astype(np.float32))
        keep.append(True)
    keep = np.array(keep, bool)
    D = next((x.shape[0] for x in out if x is not None), 0)
    X = np.stack([x if x is not None else np.zeros(D, np.float32) for x in out]) if out and D else np.zeros((0, D), np.float32)
    return X, np.asarray(starts), keep


class Speller:
    """NumPy MLPs (an average of `n_models` independently trained nets): standardise → (Linear → ReLU)×k → Linear → softmax."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.classes = [str(c) for c in z["classes"]]
        self.thr = float(z["thr"]) if "thr" in z.files else 0.6
        n = int(z["n_models"]) if "n_models" in z.files else 1
        pre = [f"m{i}_" for i in range(n)] if "n_models" in z.files else [""]
        self.nets = [dict(mu=z[p + "mu"], sd=z[p + "sd"], W=[z[f"{p}W{i}"] for i in range(int(z[p + "n_layers"]))],
                          b=[z[f"{p}b{i}"] for i in range(int(z[p + "n_layers"]))]) for p in pre]

    def proba(self, X):
        out = 0.0
        for net in self.nets:
            h = (X - net["mu"]) / net["sd"]
            for i, (W, b) in enumerate(zip(net["W"], net["b"])):
                h = h @ W + b
                if i < len(net["W"]) - 1:
                    h = np.maximum(h, 0.0)
            h = h - h.max(1, keepdims=True)
            e = np.exp(h)
            out = out + e / e.sum(1, keepdims=True)
        return out / len(self.nets)


def load_speller(model_dir) -> Speller | None:
    f = Path(model_dir) / "fingerspell.npz"
    return Speller(f) if f.exists() else None


def spell_runs(P, starts, keep, classes, allowed, thr=0.6, min_frames=5, max_gap=4):
    """Window probabilities → letter events [dict(letter, f0, f1, p)]: windows whose best allowed class is a letter with
    probability ≥ thr, consecutive windows of the same letter merged (gaps ≤ max_gap windows), events shorter than min_frames
    dropped. `allowed` = the class names that may be output (the chosen alphabet)."""
    idx = [i for i, c in enumerate(classes) if c in allowed]
    none = classes.index(NONE) if NONE in classes else None
    ev = []
    for w, (s, k) in enumerate(zip(starts, keep)):
        if not k:
            continue
        p = P[w]
        j = idx[int(np.argmax(p[idx]))]
        if p[j] < thr or (none is not None and p[none] >= p[j]):
            continue
        if ev and ev[-1]["letter"] == classes[j] and w - ev[-1]["w1"] <= max_gap + 1:
            ev[-1].update(f1=int(s) + WIN, w1=w, ps=ev[-1]["ps"] + [float(p[j])])
        else:
            ev.append(dict(letter=classes[j], f0=int(s), f1=int(s) + WIN, w1=w, ps=[float(p[j])]))
    out = []
    for e in ev:
        if e["f1"] - e["f0"] >= min_frames + WIN - 1 or len(e["ps"]) >= 2:
            out.append(dict(letter=e["letter"], f0=e["f0"], f1=e["f1"], p=float(np.mean(e["ps"]))))
    return out
