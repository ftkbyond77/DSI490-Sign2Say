"""Landmark quality without ground truth (v10): the Pose Quality Gate proxies (prompt_14 §4.4 G-P3) and the pose-quality evidence
of the decision layer (a sign whose hands were barely seen should not be confidently named).

Input = Standard Schema in isotropic units (kp [T,133,2] = x·W/H, y; sc [T,133]), usually canonical (dominant hand = right slots).
Everything is measured over ONE SPAN (a sign's frames), never the whole clip — the hands are legitimately out of view while resting.

    q = segment_quality(kp, sc, f0, f1)          # proxies of one span, per hand ("R" = dominant after canonicalisation, "L")
    a = agreement(kpA, scA, kpB, scB, f0, f1)    # the same frames seen by two extractors: hand distance in hand-size units

Proxies (each one is a way a hand model fails that a sign reader feels):
  seen       share of the span's frames with the hand detected (mean point score > thr) — MediaPipe drops a hand, RTMW lowers
             its scores, so for RTMW this is a score-aware presence
  gap_s      longest run of missing frames (s) — a short gap is interpolated upstream, a long one loses the movement
  jitter     median distance between the raw hand points and a zero-phase One-Euro copy (the filter run forward and backward,
             averaged — no lag on a smooth movement) in hand-size units: frame-to-frame noise with the real motion removed
  bone_cv    coefficient of variation of the 20 finger-bone lengths ÷ palm size over the span — anatomically constant, so its spread
             is estimation noise plus foreshortening (the same for two extractors on the same frames)
  wrist_gap  distance hand-model wrist ↔ body-model wrist (shoulder widths) — two models that disagree about where the hand is
  swap       share of frames where the hand's wrist lies nearer the OTHER body wrist — left / right confusion
"""
from __future__ import annotations

import numpy as np

from .schema import LH, RH, SH_L, SH_R, WR_L, WR_R

FPS = 25.0
HANDS = {"R": (RH, WR_R, WR_L), "L": (LH, WR_L, WR_R)}
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14),
         (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def hand_seen(sc, hand, thr=0.3):
    return np.asarray(sc)[:, hand].mean(1) > thr


def longest_run(mask):
    """Longest run of True in a boolean vector."""
    best = cur = 0
    for m in np.asarray(mask, bool):
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def one_euro(x, fps=FPS, min_cutoff=1.0, beta=0.5, d_cutoff=1.0):
    """One-Euro filter (Casiez et al. 2012) along axis 0 of x [T, ...] — adaptive low-pass: smooth when slow, follows fast motion.
    Units of beta: per (unit of x)/s; positions here are hand-size units."""
    x = np.asarray(x, np.float64)
    if len(x) < 2:
        return x.copy()
    alpha = lambda fc: 1.0 / (1.0 + fps / (2 * np.pi * fc))          # noqa: E731
    out = np.empty_like(x); out[0] = x[0]
    dx_prev = np.zeros_like(x[0])
    for t in range(1, len(x)):
        dx = (x[t] - out[t - 1]) * fps
        dx_hat = dx_prev + alpha(d_cutoff) * (dx - dx_prev)
        fc = min_cutoff + beta * np.abs(dx_hat)
        a = alpha(fc)
        out[t] = out[t - 1] + a * (x[t] - out[t - 1])
        dx_prev = dx_hat
    return out


def _shoulder_width(kp, sc, thr=0.3):
    ok = (sc[:, SH_L] > thr) & (sc[:, SH_R] > thr)
    return float(np.median(np.linalg.norm(kp[ok, SH_L] - kp[ok, SH_R], axis=1))) if ok.any() else np.nan


def _hand_size(h):
    """wrist → middle-finger MCP, per frame [T]."""
    return np.linalg.norm(h[:, 9] - h[:, 0], axis=1)


def hand_quality(kp, sc, hand="R", thr=0.3, fps=FPS):
    """Proxies of one hand over all frames of kp / sc (slice the span before calling)."""
    idx, wr_same, wr_other = HANDS[hand]
    seen = hand_seen(sc, idx, thr)
    T = len(seen)
    q = dict(seen=float(seen.mean()) if T else 0.0, gap_s=longest_run(~seen) / fps if T else 0.0, n=int(seen.sum()))
    if seen.sum() < 3:
        return dict(q, jitter=np.nan, bone_cv=np.nan, wrist_gap=np.nan, swap=np.nan)
    h = kp[:, idx][seen].astype(np.float64)
    size = float(np.median(_hand_size(h))) or 1e-3
    hn = h / size
    # jitter: raw vs One-Euro filtered, contiguous runs only (a gap would look like a jump)
    res = []
    runs = np.split(np.flatnonzero(seen), np.flatnonzero(np.diff(np.flatnonzero(seen)) > 1) + 1)
    k = 0
    for r in runs:
        seg = hn[k:k + len(r)]; k += len(r)
        if len(seg) >= 3:
            smooth = 0.5 * (one_euro(seg, fps) + one_euro(seg[::-1], fps)[::-1])      # zero-phase: a causal filter lags motion
            res.append(np.linalg.norm(seg - smooth, axis=2).mean(1)[1:-1])
    jitter = float(np.median(np.concatenate(res))) if res else np.nan
    L = np.stack([np.linalg.norm(h[:, b] - h[:, a], axis=1) for a, b in BONES], 1) / (_hand_size(h)[:, None] + 1e-6)
    bone_cv = float(np.mean(L.std(0) / (L.mean(0) + 1e-6)))
    sw = _shoulder_width(kp, sc, thr)
    bw = (sc[:, wr_same] > thr) & seen
    wrist_gap = float(np.median(np.linalg.norm(kp[bw, idx[0]] - kp[bw, wr_same], axis=1)) / sw) if bw.any() and np.isfinite(sw) else np.nan
    both = (sc[:, wr_same] > thr) & (sc[:, wr_other] > thr) & seen
    swap = float(np.mean(np.linalg.norm(kp[both, idx[0]] - kp[both, wr_other], axis=1)
                         < np.linalg.norm(kp[both, idx[0]] - kp[both, wr_same], axis=1))) if both.any() else np.nan
    return dict(q, jitter=jitter, bone_cv=bone_cv, wrist_gap=wrist_gap, swap=swap)


def segment_quality(kp, sc, f0=None, f1=None, thr=0.3, fps=FPS):
    """Both hands' proxies over frames [f0, f1) + a single summary `dom_seen` (dominant = right after canonicalisation)."""
    f0 = 0 if f0 is None else max(0, int(f0)); f1 = len(kp) if f1 is None else min(len(kp), int(f1))
    k, s = np.asarray(kp[f0:f1], np.float32), np.asarray(sc[f0:f1], np.float32)
    out = {h: hand_quality(k, s, h, thr, fps) for h in ("R", "L")}
    out["dom_seen"] = out["R"]["seen"]
    out["any_seen"] = float((hand_seen(s, RH, thr) | hand_seen(s, LH, thr)).mean()) if len(s) else 0.0
    return out


def agreement(kpA, scA, kpB, scB, f0=None, f1=None, thr=0.3):
    """Two extractors on the same frames (same time axis, same canonical side): per hand, the median point distance in hand-size
    units over frames both see, the share of frames both see, and the share only one of them sees."""
    T = min(len(kpA), len(kpB))
    f0 = 0 if f0 is None else max(0, int(f0)); f1 = T if f1 is None else min(T, int(f1))
    out = {}
    for name, (idx, _, _) in HANDS.items():
        a, b = hand_seen(scA[f0:f1], idx, thr), hand_seen(scB[f0:f1], idx, thr)
        both = a & b
        d = np.nan
        if both.any():
            ha, hb = kpA[f0:f1][both][:, idx], kpB[f0:f1][both][:, idx]
            size = (_hand_size(ha) + _hand_size(hb)) / 2 + 1e-6
            d = float(np.median(np.linalg.norm(ha - hb, axis=2).mean(1) / size))
        n = max(f1 - f0, 1)
        out[name] = dict(dist=d, both=float(both.sum() / n), only_a=float((a & ~b).sum() / n), only_b=float((b & ~a).sum() / n))
    return out
