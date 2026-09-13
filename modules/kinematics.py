"""v4 kinematic signals from whole-body keypoints — used by the segmentation model and by weak-label mining.

All quantities are normalised by shoulder width (scale invariant) and computed per frame:
  hand speed (wrist + fingertips), hand height above the rest level, distance hand→face/chest,
  handshape change rate (finger-joint angles), hand confidence."""
from __future__ import annotations

import numpy as np

LSHO, RSHO, LELB, RELB, LWRI, RWRI = 5, 6, 7, 8, 9, 10
LH, RH = slice(91, 112), slice(112, 133)
TIPS = [4, 8, 12, 16, 20]
FINGER_CHAINS = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (0, 5, 6), (5, 6, 7), (6, 7, 8), (0, 9, 10), (9, 10, 11), (10, 11, 12),
                 (0, 13, 14), (13, 14, 15), (14, 15, 16), (0, 17, 18), (17, 18, 19), (18, 19, 20)]
N_FEAT = 18


def to_px(kp, hw):
    H, W = hw
    return kp * np.array([W, H], np.float32)


def _interp_missing(x, ok):
    """Linear interpolation over frames where ok is False (per column)."""
    x = x.copy()
    T = len(x)
    idx = np.arange(T)
    if ok.sum() == 0:
        return np.zeros_like(x)
    for d in range(x.shape[1]):
        x[:, d] = np.interp(idx, idx[ok], x[ok, d])
    return x


def smooth(x, k=3):
    if len(x) < k:
        return x
    ker = np.ones(k) / k
    pad = np.pad(x, [(k // 2, k // 2)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.stack([np.convolve(pad[:, d], ker, "valid") for d in range(x.shape[1])], 1) if x.ndim == 2 else np.convolve(pad, ker, "valid")


def joint_angles(h):
    """h [T,21,2] → [T,15] cosine of finger-joint angles."""
    a = h[:, [c[0] for c in FINGER_CHAINS]] - h[:, [c[1] for c in FINGER_CHAINS]]
    b = h[:, [c[2] for c in FINGER_CHAINS]] - h[:, [c[1] for c in FINGER_CHAINS]]
    na = np.linalg.norm(a, axis=-1) + 1e-6; nb = np.linalg.norm(b, axis=-1) + 1e-6
    return (a * b).sum(-1) / na / nb


def frame_features(kp, sc, hw, rest_ref=None, thr=0.3):
    """kp [T,133,2] (0..1) sc [T,133] → feats [T,18] (float32), plus dict of named signals."""
    p = to_px(kp.astype(np.float32), hw)
    T = len(p)
    ok_s = (sc[:, LSHO] > thr) & (sc[:, RSHO] > thr)
    sw = np.linalg.norm(p[:, LSHO] - p[:, RSHO], axis=1)
    sw_med = np.median(sw[ok_s]) if ok_s.any() else max(hw) * 0.25
    sw = np.where(ok_s, sw, sw_med); sw = smooth(np.maximum(sw, 1.0), 5)
    mid = np.where(ok_s[:, None], (p[:, LSHO] + p[:, RSHO]) / 2, np.nan)
    mid = _interp_missing(np.nan_to_num(mid, nan=0.0), ok_s) if ok_s.any() else np.tile(np.array(hw[::-1]) / 2, (T, 1))
    nose = p[:, 0]
    feats, sig = [], {}
    for name, hs, wr in (("L", LH, LWRI), ("R", RH, RWRI)):
        conf = sc[:, hs].mean(1)
        ok = conf > thr
        h = p[:, hs]
        wrist = _interp_missing(h[:, 0], ok) if ok.any() else _interp_missing(p[:, wr], sc[:, wr] > thr) if (sc[:, wr] > thr).any() else np.tile(mid.mean(0), (T, 1))
        rel = (wrist - mid) / sw[:, None]                           # hand position relative to shoulders
        height = -rel[:, 1]                                          # up = positive
        vel = np.r_[np.zeros((1, 2)), np.diff(wrist, axis=0)] / sw[:, None]
        tips = h[:, TIPS]
        tips = np.stack([_interp_missing(tips[:, k], ok) if ok.any() else tips[:, k] for k in range(5)], 1)
        tip_vel = np.r_[np.zeros((1,)), np.linalg.norm(np.diff(tips, axis=0), axis=-1).mean(-1)] / sw
        ang = joint_angles(h)
        ang = _interp_missing(ang, ok) if ok.any() else ang
        dang = np.r_[np.zeros((1,)), np.abs(np.diff(ang, axis=0)).mean(-1)]
        d_face = np.linalg.norm(wrist - nose, axis=1) / sw
        speed = np.linalg.norm(vel, axis=1)
        feats += [rel[:, 0], height, speed, tip_vel, dang, d_face, conf, ok.astype(np.float32)]
        sig[name] = dict(height=height, speed=speed, tip_vel=tip_vel, dang=dang, conf=conf)
    # rest level: typical (low) hand height of this session
    hL, hR = sig["L"]["height"], sig["R"]["height"]
    if rest_ref is None:
        rest_ref = (np.percentile(hL, 15), np.percentile(hR, 15))
    raise_l, raise_r = hL - rest_ref[0], hR - rest_ref[1]
    feats += [raise_l, raise_r]
    X = np.stack(feats, 1).astype(np.float32)
    act = np.maximum(raise_l, raise_r)
    motion = smooth(sig["L"]["speed"] + sig["R"]["speed"] + 0.5 * (sig["L"]["tip_vel"] + sig["R"]["tip_vel"]), 3)
    sig.update(activity=act, motion=motion, rest_ref=rest_ref, sw=sw)
    return np.nan_to_num(X), sig


def heuristic_segments(sig, fps=12.5, raise_thr=0.25, min_len_s=0.3, max_len_s=2.5, split_min_s=0.25):
    """Unsupervised baseline: frames with a raised hand are 'signing'; long runs are split at motion minima.
    Returns [(f0, f1)] half-open frame ranges."""
    act = smooth(sig["activity"], 3) > raise_thr
    motion = sig["motion"]
    segs, T, t = [], len(act), 0
    while t < T:
        if not act[t]:
            t += 1; continue
        s = t
        while t < T and act[t]:
            t += 1
        segs.append((s, t))
    out = []
    max_len, min_len, split_min = int(max_len_s * fps), int(min_len_s * fps), int(split_min_s * fps)
    for s, e in segs:
        stack = [(s, e)]
        while stack:
            a, b = stack.pop()
            if b - a <= max_len:
                if b - a >= min_len:
                    out.append((a, b))
                continue
            inner = motion[a + split_min:b - split_min]
            cut = a + split_min + int(np.argmin(inner)) if len(inner) else (a + b) // 2
            stack += [(a, cut), (cut, b)]
    return sorted(out)
