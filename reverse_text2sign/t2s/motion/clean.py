"""Landmark cleaning before retargeting — repairs the MediaPipe failure modes found in the Phase 1 audit (data/audit/swaps.jpg,
finger_noise.jpg): one hand drawn twice on two-hand contact signs, a hand attached to the wrong arm, single-frame finger flicker,
short tracking gaps. Nothing is invented beyond short gaps (≤ MAX_GAP frames are interpolated; longer gaps stay missing and are
reported, the motion builder then refuses or down-grades the clip)."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ..data.quality import L_SH, R_SH, L_WR, R_WR, iso3, present

MAX_GAP = 8                       # 0.32 s at 25 fps
POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24),
              (25, 26), (27, 28), (29, 30), (31, 32)]
BS_NAMES = None                   # filled lazily from the forward extractor's list


def _bs_names():
    global BS_NAMES
    if BS_NAMES is None:
        BS_NAMES = ["_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
                    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
                    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
                    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
                    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
                    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
                    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
                    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
                    "noseSneerLeft", "noseSneerRight"]
    return BS_NAMES


def mirror(raw):
    """Left-dominant signer → right-dominant (mirror the image): x → 1 − x, left/right landmarks and blendshapes swapped."""
    out = dict(raw)
    pose = raw["pose"].copy()
    pose[..., 0] = np.where(pose[..., 3] > 0, 1.0 - pose[..., 0], 0.0) if pose.shape[-1] > 3 else 1.0 - pose[..., 0]
    for a, b in POSE_PAIRS:
        pose[:, [a, b]] = pose[:, [b, a]]
    out["pose"] = pose
    lh, rh = raw["lh"].copy(), raw["rh"].copy()
    for h in (lh, rh):
        m = present(h)
        h[m, :, 0] = 1.0 - h[m, :, 0]
    out["lh"], out["rh"] = rh, lh
    names = _bs_names()
    idx = list(range(len(names)))
    for i, n in enumerate(names):
        for a, b in (("Left", "Right"), ("Right", "Left")):
            if n.endswith(a) and n[:-len(a)] + b in names:
                idx[i] = names.index(n[:-len(a)] + b)
    out["bs"] = raw["bs"][:, idx]
    out["mirrored"] = True
    return out


def _chirality(h):
    """sign of the 2D palm cross product (wrist→index MCP × wrist→pinky MCP): mirror-image hands have opposite signs."""
    a, b = h[5, :2] - h[0, :2], h[17, :2] - h[0, :2]
    return np.sign(a[0] * b[1] - a[1] * b[0])


def _same_hand(a, b, sw):
    """one physical hand detected twice: the 21 points coincide AND the two skeletons have the same chirality (palm-to-palm
    contact signs — wai / สวัสดี, ขอบคุณ — also overlap, but as mirror images)."""
    close = float(np.linalg.norm(a[:, :2] - b[:, :2], axis=1).mean()) < 0.15 * sw
    return close and _chirality(a) == _chirality(b)


def _runs(mask):
    i, T = 0, len(mask)
    while i < T:
        if not mask[i]:
            i += 1; continue
        j = i
        while j < T and mask[j]:
            j += 1
        yield i, j
        i = j


def clean(raw):
    """raw MediaPipe dict (normalised image coords) → cleaned copy + report. Works in isotropic units internally."""
    hw = tuple(int(v) for v in raw["hw"])
    T = len(raw["t"])
    pose = raw["pose"].copy(); lh = raw["lh"].copy(); rh = raw["rh"].copy()
    P = iso3(pose, hw)
    vis = pose[:, :, 3]
    sh = (vis[:, L_SH] > 0.5) & (vis[:, R_SH] > 0.5)
    sw = float(np.median(np.linalg.norm(P[sh, L_SH, :2] - P[sh, R_SH, :2], axis=1))) if sh.any() else 0.3
    H, W = hw
    rep = dict(dup=0, swapped=0, detached=0, flicker=0, filled=0, gaps_left=0, sw=sw)
    hands = [lh, rh]
    wr = (L_WR, R_WR)
    for t in range(T):
        hp = [np.abs(h[t]).sum() > 0 for h in hands]
        iso_h = [iso3(h[t][None], hw)[0] for h in hands]
        if hp[0] and hp[1] and _same_hand(iso_h[0], iso_h[1], sw):
            d = [np.linalg.norm(iso_h[k][0, :2] - P[t, wr[k], :2]) for k in (0, 1)]
            drop = int(np.argmax(d))
            hands[drop][t] = 0; hp[drop] = False; rep["dup"] += 1
        for k in (0, 1):
            if not hp[k]:
                continue
            root = iso3(hands[k][t][None], hw)[0, 0, :2]
            d_own = np.linalg.norm(root - P[t, wr[k], :2]); d_oth = np.linalg.norm(root - P[t, wr[1 - k], :2])
            if d_own > 0.45 * sw and d_oth < 0.3 * sw:
                if not hp[1 - k]:
                    hands[1 - k][t] = hands[k][t]; hp[1 - k] = True; rep["swapped"] += 1
                hands[k][t] = 0; hp[k] = False
            elif d_own > 0.6 * sw:
                hands[k][t] = 0; hp[k] = False; rep["detached"] += 1
    # single-frame finger flicker: residual of the wrist-relative shape vs a smooth fit
    for h in hands:
        m = present(h)
        rel = h[:, :, :2] - h[:, :1, :2]
        rel_iso = rel * np.array([W / H, 1.0])
        size = np.median(np.linalg.norm(rel_iso[m, 9], axis=1)) if m.any() else 1.0
        for i, j in _runs(m):
            if j - i < 9:
                continue
            seg = rel_iso[i:j].reshape(j - i, -1)
            res = np.linalg.norm((seg - savgol_filter(seg, 7, 2, axis=0)).reshape(j - i, 21, 2), axis=-1).mean(1)
            bad = np.flatnonzero(res > 0.25 * max(size, 1e-3))
            for b in bad:
                h[i + b] = 0; rep["flicker"] += 1
    # fill short gaps: hand shape relative to its root linearly, root relative to the pose wrist linearly
    for k, h in enumerate(hands):
        m = present(h)
        idx = np.flatnonzero(m)
        if len(idx) < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            g = b - a - 1
            if g <= 0:
                continue
            if g > MAX_GAP:
                rep["gaps_left"] += 1
                continue
            ra, rb = h[a] - h[a, :1], h[b] - h[b, :1]
            oa = h[a, 0, :2] - pose[a, wr[k], :2]; ob = h[b, 0, :2] - pose[b, wr[k], :2]
            for n in range(1, g + 1):
                u = n / (g + 1)
                rel = (1 - u) * ra + u * rb
                root = pose[a + n, wr[k], :2] + (1 - u) * oa + u * ob
                h[a + n] = rel; h[a + n, :, :2] += root; h[a + n, 0, 2] = 0
                rep["filled"] += 1
    # light temporal smoothing (contiguous runs only)
    for h in hands:
        m = present(h)
        for i, j in _runs(m):
            if j - i >= 7:
                h[i:j] = savgol_filter(h[i:j], 5, 2, axis=0)
    ok = vis > 0.3
    P2 = pose.copy()
    if T >= 7:
        P2[:, :, :3] = savgol_filter(pose[:, :, :3], 5, 2, axis=0)
    pose = np.where(ok[..., None], P2, pose)
    out = dict(raw); out.update(pose=pose, lh=hands[0], rh=hands[1])
    return out, rep
