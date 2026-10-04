"""Sequence utilities shared by training and the production runtime (NumPy only): kinematic features, temporal resampling,
spotting windows, tagger decoding and boundary metrics."""
from __future__ import annotations

import numpy as np

from .schema import LH, RH, SH_L, SH_R, WR_L, WR_R

N_KIN = 13
FPS = 25.0


def kin_features(kp, sc, thr=0.3):
    """kp iso [T,133,2], sc [T,133] → [T,13]: body seen, per hand (seen, raise, speed, dist-to-nose, shape change), max raise/speed."""
    T = len(kp)
    ok_s = (sc[:, SH_L] > thr) & (sc[:, SH_R] > thr)
    sw = np.linalg.norm(kp[:, SH_L] - kp[:, SH_R], axis=1)
    swm = float(np.median(sw[ok_s])) if ok_s.any() else 0.2
    mid = (kp[:, SH_L] + kp[:, SH_R]) / 2
    if ok_s.any():
        for d in range(2):
            mid[:, d] = np.interp(np.arange(T), np.flatnonzero(ok_s), mid[ok_s, d])
    nose = kp[:, 0]
    feats = [ok_s.astype(np.float32)]
    raises, speeds = [], []
    for hand, wr in ((LH, WR_L), (RH, WR_R)):
        hok = sc[:, hand].mean(1) > thr
        w = np.where(hok[:, None], kp[:, hand].mean(1), kp[:, wr])
        wok = hok | (sc[:, wr] > thr)
        raise_ = np.where(wok, (mid[:, 1] - w[:, 1]) / swm, -3.0)
        v = np.r_[0.0, np.linalg.norm(np.diff(w, axis=0), axis=1)] / swm * FPS
        v = np.where(wok & np.r_[False, wok[:-1]], v, 0.0).clip(0, 20)
        dn = np.where(wok, np.linalg.norm(w - nose, axis=1) / swm, 4.0).clip(0, 6)
        h = kp[:, hand] - kp[:, hand][:, :1]
        dshape = np.r_[0.0, np.abs(np.diff(h, axis=0)).mean((1, 2))] / swm * FPS
        dshape = np.where(hok & np.r_[False, hok[:-1]], dshape, 0.0).clip(0, 20)
        feats += [hok.astype(np.float32), raise_, v / 5, dn / 3, dshape / 5]
        raises.append(raise_); speeds.append(v)
    feats += [np.maximum(*raises), np.maximum(*speeds) / 5]
    return np.nan_to_num(np.stack(feats, 1)).astype(np.float32)


def resample(kp, sc, factor):
    n = len(kp)
    m = max(2, int(round(n / factor)))
    t = np.linspace(0, n - 1, m)
    i0 = np.floor(t).astype(int); i1 = np.minimum(i0 + 1, n - 1); w = (t - i0)[:, None, None]
    return (kp[i0] * (1 - w) + kp[i1] * w).astype(np.float32), np.minimum(sc[i0], sc[i1])


def window_list(T, lens, stride, act=None, min_active=0.5):
    out = []
    for a in range(0, T, stride):
        for L in lens:
            b = a + L
            if b > T:
                continue
            if act is None or act[a:b].mean() >= min_active:
                out.append((a, b))
    return out


def gt_segments(y):
    segs, t = [], 0
    while t < len(y):
        if y[t] == 0:
            t += 1; continue
        s = t; t += 1
        while t < len(y) and y[t] == 2:
            t += 1
        segs.append((s, t))
    return segs


def decode_tagger(prob, thr_in=0.5, thr_begin=0.3, min_len=6):
    p_in = np.convolve(prob[:, 1] + prob[:, 2], np.ones(3) / 3, "same")
    inside = p_in > thr_in
    segs, t, T = [], 0, len(prob)
    while t < T:
        if not inside[t]:
            t += 1; continue
        s = t
        while t < T and inside[t]:
            t += 1
        cuts = [s]
        pb = prob[s:t, 1]
        for i in range(1, len(pb) - 1):
            if pb[i] > thr_begin and pb[i] >= pb[i - 1] and pb[i] >= pb[i + 1] and s + i - cuts[-1] >= min_len:
                cuts.append(s + i)
        cuts.append(t)
        segs += [(x, y) for x, y in zip(cuts[:-1], cuts[1:]) if y - x >= min_len]
    return segs


def boundary_f1(pred, gt, tol=3):
    used, tp = set(), 0
    for a, b in pred:
        for j, (c, d) in enumerate(gt):
            if j not in used and abs(a - c) <= tol and abs(b - d) <= tol:
                used.add(j); tp += 1; break
    p = tp / max(len(pred), 1); r = tp / max(len(gt), 1)
    return 2 * p * r / max(p + r, 1e-9)
