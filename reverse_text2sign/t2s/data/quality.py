"""Per-clip motion quality for the REVERSE direction (a clip is played back as THE sign, so the bar is higher than for recognition).

Everything is measured on the raw MediaPipe Holistic arrays the clip will be retargeted from (data_prep/pose_mp_raw), in isotropic
image units (x·W/H, y) normalised by the signer's shoulder width (sw), so numbers are comparable across 480p studio dictionaries,
600×600 crops and 640×360 YouTube lessons.
"""
from __future__ import annotations

import numpy as np

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, NOSE = 11, 12, 13, 14, 15, 16, 0


def load_raw(path):
    z = np.load(path)
    try:
        return dict(t=z["t"].astype(np.float64), hw=tuple(int(v) for v in z["hw"]), pose=z["pose"].astype(np.float32),
                    lh=z["lh"].astype(np.float32), rh=z["rh"].astype(np.float32), face=z["face"].astype(np.float32),
                    bs=z["bs"].astype(np.float32))
    finally:
        z.close()


def iso3(a, hw):
    """normalised (x/W, y/H, z/W) → isotropic (x·W/H, y, z·W/H) in units of frame height."""
    H, W = hw
    out = a[..., :3].copy()
    out[..., 0] *= W / H
    if out.shape[-1] > 2:
        out[..., 2] *= W / H
    return out


def present(h):
    return np.abs(h).reshape(len(h), -1).sum(1) > 0


def rest_position(w, vis):
    """a wrist's rest position = median of its lowest 15 % visible frames (hands hanging, or clasped at the belly in TTRS)."""
    ok = vis > 0.5
    if ok.sum() < 3:
        return None
    y = w[ok, 1]
    low = y >= np.percentile(y, 85)
    return np.median(w[ok][low, :2], axis=0)


def active_frames(pose_iso, vis, sw, mid_y):
    """A wrist is 'active' (signing) when visible and away from its rest position by > 0.35 shoulder widths, or above the chest line."""
    act = []
    for wi in (L_WR, R_WR):
        w = pose_iso[:, wi, :2]
        r = rest_position(w, vis[:, wi])
        a = (vis[:, wi] > 0.5) & (w[:, 1] < mid_y + 0.6 * sw)
        if r is not None:
            a |= (vis[:, wi] > 0.5) & (np.linalg.norm(w - r, axis=1) > 0.35 * sw)
        act.append(a)
    return np.stack(act, 1)                          # [T, 2] left, right


def run_noise(x, mask, win=7):
    """noise() over contiguous runs of tracked frames only (a lost frame must not count as a jump); median over frames."""
    from scipy.signal import savgol_filter
    res = []
    i, T = 0, len(mask)
    while i < T:
        if not mask[i]:
            i += 1; continue
        j = i
        while j < T and mask[j]:
            j += 1
        if j - i >= win + 2:
            seg = x[i:j]
            r = seg - savgol_filter(seg, win, 3, axis=0)
            res.append(np.linalg.norm(r[2:-2], axis=-1))          # run edges = hand entering / leaving the picture
        i = j
    return float(np.median(np.concatenate(res))) if res else None


def noise(x, win=7):
    """tracking noise of a trajectory [T, D]: median residual after a Savitzky-Golay fit (cubic, 0.28 s) — real signing motion, even
    fast, is smooth at that scale; detector flicker is not."""
    from scipy.signal import savgol_filter
    if len(x) < win + 2:
        return 0.0
    r = x - savgol_filter(x, win, 3, axis=0)
    return float(np.median(np.linalg.norm(r, axis=-1)))


def _accel(x):
    """median |2nd difference| of a trajectory [T, D] (per frame at 25 fps)."""
    if len(x) < 3:
        return 0.0
    return float(np.median(np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1)))


def _runs(mask):
    """longest run of True."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def clip_quality(raw, fps=25.0):
    """raw MediaPipe arrays → dict of quality metrics (see module docstring)."""
    T = len(raw["t"])
    hw = raw["hw"]
    out = dict(T=T, dur_s=T / fps)
    if T < 5:
        out["fatal"] = "too_short"
        return out
    pose = iso3(raw["pose"], hw); vis = raw["pose"][:, :, 3]
    lh, rh = iso3(raw["lh"], hw), iso3(raw["rh"], hw)
    hp = np.stack([present(raw["lh"]), present(raw["rh"])], 1)
    sh_ok = (vis[:, L_SH] > 0.5) & (vis[:, R_SH] > 0.5)
    out["shoulder_frac"] = float(sh_ok.mean())
    if sh_ok.sum() < 3:
        out["fatal"] = "no_body"
        return out
    sw = float(np.median(np.linalg.norm(pose[sh_ok, L_SH, :2] - pose[sh_ok, R_SH, :2], axis=1)))
    mid_y = float(np.median((pose[sh_ok, L_SH, 1] + pose[sh_ok, R_SH, 1]) / 2))
    out["sw"] = sw
    out["room_below"] = float((1.0 - mid_y) / max(sw, 1e-3))              # signing space below the shoulders (sw)
    out["head_room"] = float(np.median(pose[sh_ok, NOSE, 1]) / max(sw, 1e-3))  # space above the nose (sw)
    act = active_frames(pose, vis, sw, mid_y)
    any_act = act.any(1)
    out["active_frac"] = float(any_act.mean())
    if any_act.sum() < 3:
        out["fatal"] = "no_signing"
        return out
    idx = np.flatnonzero(any_act)
    f0, f1 = int(idx[0]), int(idx[-1]) + 1
    out["span_f0"], out["span_f1"] = f0, f1
    out["span_s"] = (f1 - f0) / fps
    out["lead_s"], out["tail_s"] = f0 / fps, (T - f1) / fps              # rest before / after (needed for clean trimming)
    s = slice(f0, f1)
    # dominance: the hand that is raised for longer and travels further leads one-handed signs
    travel = []
    for k, wi in enumerate((L_WR, R_WR)):
        w = pose[s, wi, :2]
        travel.append(float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum() / sw))
    act_n = act[s].sum(0)
    out["act_left"], out["act_right"] = int(act_n[0]), int(act_n[1])
    out["travel_left"], out["travel_right"] = travel
    dom = 1 if (act_n[1] + 3 * travel[1]) >= (act_n[0] + 3 * travel[0]) else 0
    out["dominant"] = "right" if dom == 1 else "left"
    out["two_handed"] = float((act[s].all(1)).mean())
    # hand tracking where the hand is up (the frames the avatar must show)
    for k, name in ((0, "left"), (1, "right")):
        a = act[s, k]
        out[f"hand_rate_{name}"] = float(hp[s, k][a].mean()) if a.any() else None
        out[f"hand_gap_{name}"] = int(_runs(a & ~hp[s, k]))
    dk = "right" if dom == 1 else "left"
    out["hand_rate_dom"] = out[f"hand_rate_{dk}"]
    out["hand_gap_dom"] = out[f"hand_gap_{dk}"]
    # handedness swaps: a detected hand whose root sits on the other arm's wrist
    swaps = 0; n_chk = 0
    for k, (h, wi, wo) in enumerate(((lh, L_WR, R_WR), (rh, R_WR, L_WR))):
        m = hp[s, k] & (vis[s, wi] > 0.5) & (vis[s, wo] > 0.5)
        if m.any():
            d_own = np.linalg.norm(h[s][m, 0, :2] - pose[s][m, wi, :2], axis=1)
            d_oth = np.linalg.norm(h[s][m, 0, :2] - pose[s][m, wo, :2], axis=1)
            swaps += int((d_oth < d_own).sum()); n_chk += int(m.sum())
    out["swap_frac"] = swaps / n_chk if n_chk else 0.0
    # tracking noise: wrist in shoulder widths; finger tips relative to the wrist in palm lengths (only frames with the hand tracked)
    wj = []
    for wi in (L_WR, R_WR):
        w = pose[s, wi, :2]
        wj.append(noise(w) / sw)
    out["wrist_jitter"] = float(max(wj)) if wj else None
    fj = []
    for k, h in enumerate((lh, rh)):
        m = hp[s, k]
        if m.sum() >= 9:
            hh = h[s]
            rel = (hh[:, [4, 8, 12, 16, 20], :2] - hh[:, :1, :2])
            size = float(np.median(np.linalg.norm(hh[m, 9, :2] - hh[m, 0, :2], axis=1))) or 1e-3   # wrist → middle knuckle
            v = run_noise(rel.reshape(len(hh), -1), m)
            if v is not None:
                fj.append(v / size)
    out["finger_jitter"] = float(max(fj)) if fj else None
    # framing: active wrist outside the picture (the signer's hand leaves the frame)
    H, W = hw
    outf = 0; n_a = 0
    for k, wi in enumerate((L_WR, R_WR)):
        a = act[s, k] | (vis[s, wi] > 0.5)
        x, y = raw["pose"][s, wi, 0], raw["pose"][s, wi, 1]
        o = (x < 0.01) | (x > 0.99) | (y < 0.01)
        outf += int((o & a).sum()); n_a += int(a.sum())
    out["hand_out_frac"] = outf / n_a if n_a else 0.0
    face_ok = np.abs(raw["face"][s]).reshape(f1 - f0, -1).sum(1) > 0
    out["face_frac"] = float(face_ok.mean())
    out["bs_frac"] = float((np.abs(raw["bs"][s]).sum(1) > 0).mean())
    # body stability: shoulders should not drift (camera zoom / pan / person walking)
    shc = (pose[s, L_SH, :2] + pose[s, R_SH, :2]) / 2
    out["torso_drift"] = float(np.linalg.norm(shc.max(0) - shc.min(0)) / sw)
    swr = np.linalg.norm(pose[s, L_SH, :2] - pose[s, R_SH, :2], axis=1)
    out["zoom_ratio"] = float(np.percentile(swr, 95) / max(np.percentile(swr, 5), 1e-3))
    # pose depth sanity: MediaPipe z of the wrists relative to shoulders (in sw) — extreme values mean a broken depth estimate
    zrel = pose[s][:, [L_WR, R_WR], 2] - pose[s][:, [L_SH, R_SH], 2].mean(1, keepdims=True)
    out["wrist_z_p95"] = float(np.percentile(np.abs(zrel), 95) / sw)
    return out


# Hard thresholds for "this clip may be played as a sign". Chosen from the distributions in data/audit/ (≈ the 1st/99th percentile
# of the clean studio dictionaries) and checked by eye on contact sheets (result_v1.md §2).
RULES = dict(
    shoulder_frac=(">=", 0.9),        # the whole upper body must be in the picture
    hand_rate_dom=(">=", 0.80),       # the dominant hand is tracked (after cleaning) in ≥ 80 % of the frames where it is up
    hand_gap_dom=("<=", 10),          # no lost-hand gap longer than 0.4 s after short-gap filling (would need invented fingers)
    swap_frac=("<=", 0.08),           # hands not swapped between arms
    wrist_jitter=("<=", 0.03),        # median wrist tracking residual (shoulder widths) — 99th pct of the studio dictionaries
    finger_jitter=("<=", 0.25),       # median finger residual (palm lengths) — ≈ 99th pct of the studio dictionaries
    hand_out_frac=("<=", 0.05),       # hands stay inside the picture
    zoom_ratio=("<=", 1.3),           # no zoom during the sign
    span_s=("<=", 8.0),               # one dictionary sign, not a lesson / sentence
    face_frac=(">=", 0.8),            # the face is visible (non-manual features)
)


# Tier B: a clip that misses only these "soft" rules, by at most this much, may still be played when its concept has no tier-A
# clip (the IR then reports a lower motion confidence). Everything else is a hard rejection.
SOFT = dict(hand_rate_dom=(">=", 0.6), hand_gap_dom=("<=", 20), finger_jitter=("<=", 0.35), swap_frac=("<=", 0.25),
            span_s=("<=", 12.0), face_frac=(">=", 0.3))
# face_frac / swap_frac are soft since the audit: signs made ON the face hide it (หน้ากาก), crossing / touching hands raise the swap
# score (ป้องกัน, มารยาท, ผีเสื้อ) — after cleaning they are usable as tier B, checked by eye in the renders.


def tier(q):
    """'A' (all rules pass) · 'B' (only soft rules fail, within SOFT limits) · None (rejected)."""
    bad = failures(q)
    if not bad:
        return "A"
    if all(b in SOFT for b in bad):
        for b in bad:
            op, thr = SOFT[b]
            v = q.get(b)
            if v is None or (isinstance(v, float) and np.isnan(v)) or (op == ">=" and v < thr) or (op == "<=" and v > thr):
                return None
        return "B"
    return None


def failures(q):
    """list of rule names a clip fails (empty = usable)."""
    if isinstance(q.get("fatal"), str):
        return [q["fatal"]]
    bad = []
    for k, (op, thr) in RULES.items():
        v = q.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if k in ("hand_rate_dom", "hand_gap_dom", "finger_jitter"):
                bad.append(k)
            continue
        if (op == ">=" and v < thr) or (op == "<=" and v > thr):
            bad.append(k)
    return bad
