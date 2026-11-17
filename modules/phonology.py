"""Phonology pseudo-labels per clip (prompt-opt-v11 §1B.4) for the factorised embedding of encoder run R6.

    python main.py v11 phonology            # → cloud_s3/data_prep/manifest/phonology_labels.npz (+ audit sheets)

Computed on canonical (right-dominant) isotropic landmarks of the sign span; the HOLD = the 5 frames around the minimum
dominant-wrist speed inside the middle 60 % of the span.

| sub-space   | label |
|-------------|-------|
| handshape   | k-means cluster (k = 50, fitted on TRAIN-signer clips) of the dominant hand's shape at the hold (modules/handshape.hand_shape: 15 bend cosines, 9 tip distances, palm direction) + one- / two-handed (×2 → 100 classes) |
| orientation | palm-normal proxy = signed area of the wrist / index-MCP / pinky-MCP triangle (6 bins) × finger direction wrist → middle MCP (6 × 60° bins) → 36 classes |
| location    | body region of the dominant hand centre at the hold: forehead · eye/cheek · nose · mouth/chin · neck · chest · belly · shoulder side · neutral space · other hand (centres closer than 0.25 hand sizes … 1.0 hand size, see code) → 10 classes |
| movement    | dominant-wrist path over the span: hold · straight in 8 body-frame directions · arc · circle · repeated (≥ 2 reversals) (12) × internal movement none / hand-shape change / wrist twist (3) → 36 classes |

-1 = not measurable (dominant hand unseen at the hold). These are pseudo-labels: they shape sub-spaces, they are never shown.
"""
from __future__ import annotations

import numpy as np

from .handshape import hand_shape
from .schema import LH, RH, SH_L, SH_R

LOC_NAMES = ["forehead", "eye_cheek", "nose", "mouth_chin", "neck", "chest", "belly", "shoulder_side", "neutral", "other_hand"]
NOSE, CHIN = 0, 31


def hold_index(kp, sc, f0, f1, thr=0.3):
    a, b = f0 + int(0.2 * (f1 - f0)), max(f0 + int(0.8 * (f1 - f0)), f0 + int(0.2 * (f1 - f0)) + 1)
    w = kp[a:b, RH[0]]
    v = np.r_[np.inf, np.linalg.norm(np.diff(w, axis=0), axis=1)]
    v[sc[a:b, RH].mean(1) <= thr] = np.inf
    return a + int(np.argmin(v)) if np.isfinite(v).any() else None


def raw_features(kp, sc, f0, f1, thr=0.3):
    """→ dict of the measurements behind the four labels (None when the dominant hand is not seen at the hold)"""
    T = len(kp)
    f0, f1 = max(0, int(f0)), min(T, int(f1))
    if f1 - f0 < 4:
        f0, f1 = 0, T
    t = hold_index(kp, sc, f0, f1, thr)
    if t is None:
        return None
    hold = slice(max(f0, t - 2), min(f1, t + 3))
    ok = (sc[f0:f1, SH_L] > thr) & (sc[f0:f1, SH_R] > thr)
    sw = float(np.median(np.linalg.norm(kp[f0:f1, SH_L] - kp[f0:f1, SH_R], axis=1)[ok])) if ok.any() else 0.2
    mid = np.median(((kp[f0:f1, SH_L] + kp[f0:f1, SH_R]) / 2)[ok], 0) if ok.any() else np.array([0.5, 0.5])
    R = kp[:, RH]; L = kp[:, LH]
    rs = sc[:, RH].mean(1) > thr; ls = sc[:, LH].mean(1) > thr
    hs = hand_shape(R[hold]).mean(0)
    # one / two-handed: the other hand seen, raised above the belly line and moving for ≥ 40 % of the span
    span_l = ls[f0:f1]
    raised = span_l & (L[f0:f1, 0, 1] < mid[1] + 1.4 * sw)
    lmov = np.r_[0, np.linalg.norm(np.diff(L[f0:f1, 0], axis=0), axis=1)].sum() / max(sw, 1e-3) if span_l.sum() > 2 else 0.0
    two = bool(raised.mean() >= 0.4 and lmov > 0.15)
    # orientation: signed area of wrist / index MCP / pinky MCP (hand-size units) and the finger direction
    h = R[hold].mean(0); size = np.linalg.norm(h[9] - h[0]) + 1e-6
    v1, v2 = h[5] - h[0], h[17] - h[0]
    palm = float((v1[0] * v2[1] - v1[1] * v2[0]) / size ** 2)
    d = h[9] - h[0]; ang = float(np.arctan2(d[1], d[0]))
    # location of the dominant hand centre at the hold
    c = h.mean(0)
    nose = np.median(kp[f0:f1, NOSE][sc[f0:f1, NOSE] > thr], 0) if (sc[f0:f1, NOSE] > thr).any() else mid - [0, 0.8 * sw]
    chin = np.median(kp[f0:f1, CHIN][sc[f0:f1, CHIN] > thr], 0) if (sc[f0:f1, CHIN] > thr).any() else nose + [0, 0.25 * sw]
    lo = L[hold].mean(0).mean(0) if ls[hold].mean() > 0.5 else None
    # movement of the dominant wrist over the CORE of the sign (middle 70 %: the rise from / return to rest is not the movement),
    # path smoothed over 5 frames; reversals counted along the main axis with hysteresis (0.08 shoulder widths)
    a_, b_ = f0 + int(0.15 * (f1 - f0)), max(f0 + int(0.85 * (f1 - f0)), f0 + int(0.15 * (f1 - f0)) + 3)
    W = R[a_:b_, 0][rs[a_:b_]]
    if len(W) >= 5:
        k_ = np.ones(5) / 5
        W = np.stack([np.convolve(np.pad(W[:, d], 2, mode="edge"), k_, "valid") for d in range(2)], 1)
    path = float(np.linalg.norm(np.diff(W, axis=0), axis=1).sum() / max(sw, 1e-3)) if len(W) > 2 else 0.0
    net_v = (W[-1] - W[0]) / max(sw, 1e-3) if len(W) > 2 else np.zeros(2)
    net = float(np.linalg.norm(net_v))
    rev = 0
    if len(W) > 4:
        C = W - W.mean(0)
        u = np.linalg.svd(C, full_matrices=False)[2][0]
        x = C @ u / max(sw, 1e-3)
        ext, direction = x[0], 0
        for v_ in x[1:]:
            if direction >= 0 and v_ < ext - 0.08:
                rev += direction > 0; direction = -1; ext = v_
            elif direction <= 0 and v_ > ext + 0.08:
                rev += direction < 0; direction = 1; ext = v_
            else:
                ext = max(ext, v_) if direction >= 0 else min(ext, v_)
    core = rs[a_:b_]
    shp = hand_shape(R[a_:b_][core]) if core.sum() > 4 else None
    if shp is not None:
        n3 = max(1, len(shp) // 3)
        hs_change = float(np.linalg.norm(shp[:n3].mean(0) - shp[-n3:].mean(0)))
    else:
        hs_change = 0.0
    palms = []
    for k in np.flatnonzero(core):
        hh = R[a_ + k]; sz = np.linalg.norm(hh[9] - hh[0]) + 1e-6
        a1, a2 = hh[5] - hh[0], hh[17] - hh[0]
        palms.append((a1[0] * a2[1] - a1[1] * a2[0]) / sz ** 2)
    twist = float(np.mean(palms[-max(1, len(palms) // 3):]) - np.mean(palms[:max(1, len(palms) // 3)])) if len(palms) > 3 else 0.0
    twist = abs(twist)
    hips = (sc[f0:f1, 11] > thr) & (sc[f0:f1, 12] > thr)
    torso = float(np.median(((kp[f0:f1, 11, 1] + kp[f0:f1, 12, 1]) / 2 - (kp[f0:f1, SH_L, 1] + kp[f0:f1, SH_R, 1]) / 2)[hips])) if hips.sum() > 2 else 1.4 * sw
    torso = float(np.clip(torso, 0.5 * sw, 2.0 * sw))
    return dict(torso=torso, hs=hs, two=two, palm=palm, ang=ang, c=c, nose=nose, chin=chin, mid=mid, sw=sw, other=lo, hand_size=size,
                path=path, net=net, net_v=net_v, rev=rev, hs_change=hs_change, twist=twist)


def location(r):
    """2-D body regions (audited on contact sheets, 2026-11-13: v1 called hip-height hands 'chest' and wide head-height hands
    'eye/cheek' → face regions narrowed to ±0.45 shoulder widths around the nose, chest / belly split by height)"""
    c, sw = r["c"], r["sw"]
    if r["other"] is not None and np.linalg.norm(c - r["other"]) < 1.0 * r["hand_size"]:
        return 9                                                    # contact with the other hand
    nose, chin, mid = r["nose"], r["chin"], r["mid"]
    dx = abs(c[0] - nose[0]) / sw
    lat = abs(c[0] - mid[0]) / sw
    y = (c[1] - mid[1]) / sw                                        # + = below the shoulder line
    if c[1] < nose[1] - 0.35 * sw and dx < 0.45:
        return 0                                                    # forehead
    if c[1] < nose[1] + 0.05 * sw and dx < 0.45:
        return 2 if dx < 0.15 else 1                                # nose / eye-cheek
    if c[1] < chin[1] + 0.12 * sw and dx < 0.4:
        return 3                                                    # mouth / chin
    if y < -0.05 and dx < 0.35:
        return 4                                                    # neck
    if lat > 0.6 and y < 0.7:
        return 7                                                    # shoulder side (also beside the head)
    yt = (c[1] - mid[1]) / r.get("torso", 1.4 * sw)                # torso units (shoulder line 0 → hip line 1)
    if lat < 0.6 and -0.05 <= y and yt < 0.45:
        return 5                                                    # chest
    if lat < 0.7 and 0.45 <= yt < 1.05:
        return 6                                                    # belly
    return 8                                                        # neutral space (in front / low / wide)


def orientation(r):
    pb = int(np.clip(np.floor((r["palm"] + 1.2) / 2.4 * 6), 0, 5))
    ab = int(np.floor(((r["ang"] + np.pi) % (2 * np.pi)) / (np.pi / 3))) % 6
    return pb * 6 + ab


def movement(r, hs_thr=0.12, twist_thr=0.5, hold_thr=0.25):
    if r["path"] < hold_thr:
        base = 0
    elif r["rev"] >= 2:
        base = 11
    elif r["net"] / max(r["path"], 1e-6) > 0.7:
        a = np.arctan2(r["net_v"][1], r["net_v"][0])
        base = 1 + int(np.floor(((a + np.pi / 8) % (2 * np.pi)) / (np.pi / 4))) % 8
    elif r["net"] / max(r["path"], 1e-6) > 0.35:
        base = 9
    else:
        base = 10
    internal = 1 if r["hs_change"] > hs_thr else 2 if r["twist"] > twist_thr else 0
    return base * 3 + internal


def label_all(items, train_mask, k=50, seed=0):
    """items: list of raw_features dicts (or None) → [N, 4] int labels (hs, or, loc, mov), -1 = unmeasurable; + the fitted k-means"""
    from sklearn.cluster import KMeans
    ok = np.array([r is not None for r in items])
    H = np.stack([r["hs"] if r is not None else np.zeros(26) for r in items]).astype(np.float32)
    mu, sd = H[ok & train_mask].mean(0), H[ok & train_mask].std(0) + 1e-6
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit((H[ok & train_mask] - mu) / sd)
    lab = np.full((len(items), 4), -1, np.int64)
    cl = km.predict((H[ok] - mu) / sd)
    tr = [items[i] for i in np.flatnonzero(ok & train_mask)]
    # movement thresholds from TRAIN clips: hold = lowest 15 % path; hand-shape change / twist = top 20 %
    th = dict(hold_thr=float(np.percentile([r["path"] for r in tr], 15)), hs_thr=float(np.percentile([r["hs_change"] for r in tr], 80)),
              twist_thr=float(np.percentile([r["twist"] for r in tr], 80)))
    for j, (i, r) in enumerate(zip(np.flatnonzero(ok), [items[i] for i in np.flatnonzero(ok)])):
        lab[i] = [cl[j] + k * int(r["two"]), orientation(r), location(r), movement(r, **th)]
    return lab, dict(mu=mu, sd=sd, centers=km.cluster_centers_, **{k_: np.array(v) for k_, v in th.items()})
