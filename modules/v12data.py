"""v12 data layers (prompt-opt-v12 §1D / §1E): movement phases, handedness, quality, log-replay corruption.

All functions take Standard Schema arrays (kp [T,133,2] as x/W, y/H; sc [T,133]; 25 fps) — the same arrays training and inference read.
Units: shoulder width (sw) on isotropic coordinates (x·W/H), so the rules do not depend on the camera.
"""
from __future__ import annotations

import numpy as np

FPS = 25.0
SH_L, SH_R, WR_L, WR_R = 5, 6, 9, 10           # COCO-WholeBody body slots (left / right shoulder, left / right wrist)
LH = slice(91, 112)
RH = slice(112, 133)


def _iso(kp, hw):
    H, W = (int(hw[0]), int(hw[1])) if hw is not None else (1, 1)
    out = np.asarray(kp, np.float32).copy()
    out[..., 0] *= W / max(H, 1)
    return out


def _smooth(x, k=5):
    if len(x) < k:
        return x
    w = np.ones(k) / k
    pad = k // 2
    xp = np.pad(x, ((pad, pad),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    if x.ndim == 1:
        return np.convolve(xp, w, mode="valid")
    return np.stack([np.convolve(xp[:, j], w, mode="valid") for j in range(x.shape[1])], 1)


def hand_present(sc):
    """[T,2] bool: left / right hand found (any of its 21 points with confidence > 0)"""
    sc = np.asarray(sc)
    return np.stack([(sc[:, LH] > 0).any(1), (sc[:, RH] > 0).any(1)], 1)


def shoulder_geometry(kp, sc, hw):
    """(mid-shoulder [T,2], shoulder width median, ok mask) on isotropic coordinates"""
    k = _iso(kp, hw)
    ok = (np.asarray(sc)[:, SH_L] > 0.3) & (np.asarray(sc)[:, SH_R] > 0.3)
    mid = (k[:, SH_L] + k[:, SH_R]) / 2
    sw = float(np.median(np.linalg.norm(k[ok, SH_L] - k[ok, SH_R], axis=1))) if ok.sum() >= 3 else np.nan
    return mid, sw, ok


def runs(mask):
    """lengths of True runs"""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def hand_loss_share(sc, min_s=0.3):
    """share of frames inside runs of > min_s with NO hand, counted only between the first and last frame with a hand"""
    hp = hand_present(sc).any(1)
    if not hp.any():
        return 1.0
    i0, i1 = np.flatnonzero(hp)[[0, -1]]
    seg = ~hp[i0:i1 + 1]
    r = runs(seg)
    return float(r[r > min_s * FPS].sum() / max(len(seg), 1))


def phases(kp, sc, hw, dominant="right", rest_drop=1.3, rest_speed=0.15):
    """Movement phases of one isolated clip (prompt 1D): returns dict(f0, f1, f2, f3, found).
    f0 preparation start · f1 stroke start · f2 stroke end · f3 retraction end (frame indices, 25 fps).
    Rest = wrist more than `rest_drop` sw below the shoulder line (or missing) and slower than `rest_speed` sw/s."""
    T = len(kp)
    out = dict(f0=0, f1=0, f2=max(T - 1, 0), f3=max(T - 1, 0), found=False)
    if T < 8:
        return out
    k = _iso(kp, hw)
    mid, sw, ok = shoulder_geometry(kp, sc, hw)
    if not np.isfinite(sw) or sw <= 0:
        return out
    wr = WR_R if dominant == "right" else WR_L
    hand = RH if dominant == "right" else LH
    scs = np.asarray(sc)
    # wrist: the hand root (point 0 of the 21) when the hand is found, else the body-pose wrist
    hroot = k[:, hand.start]
    have_h = scs[:, hand.start] > 0
    have_w = scs[:, wr] > 0.3
    w = np.where(have_h[:, None], hroot, k[:, wr])
    seen = have_h | have_w
    if seen.sum() < 4:
        return out
    idx = np.arange(T)
    for j in range(2):                      # fill unseen frames by interpolation for the speed profile only
        w[:, j] = np.interp(idx, idx[seen], w[seen, j])
    sy = np.interp(idx, idx[ok], mid[ok, 1]) if ok.sum() >= 2 else np.full(T, np.nanmedian(mid[:, 1]))
    height = (sy - w[:, 1]) / sw            # > 0 = wrist above the shoulder line (y grows downwards)
    ws = _smooth(w, 5)
    speed = np.r_[0.0, np.linalg.norm(np.diff(ws, axis=0), axis=1)] * FPS / sw
    rest = ((height < -rest_drop) | ~seen) & (speed < max(rest_speed, 0.0)) | ~seen
    rest |= (height < -rest_drop) & (speed < 1.0)          # hanging hand drifting slowly also counts as rest
    # resting posture of THIS clip (e.g. the e-TSL signer holds her hands clasped at the waist between signs, never below 1.3 sw):
    # frames whose BOTH wrists are within 0.35 sw of the clip's start / end posture and that move slowly are rest too
    other = LH if dominant == "right" else RH
    ow = np.where((scs[:, other.start] > 0)[:, None], k[:, other.start], k[:, WR_L if dominant == "right" else WR_R])
    oseen = (scs[:, other.start] > 0) | (scs[:, WR_L if dominant == "right" else WR_R] > 0.3)
    if oseen.sum() >= 4:
        for j in range(2):
            ow[:, j] = np.interp(idx, idx[oseen], ow[oseen, j])
    n0 = max(3, min(10, T // 10))
    pose2 = np.concatenate([w, ow], 1)
    dist = np.minimum(np.linalg.norm(pose2 - np.median(pose2[:n0], 0), axis=1),
                      np.linalg.norm(pose2 - np.median(pose2[-n0:], 0), axis=1)) / sw
    rest |= (dist < 0.35) & (speed < 1.0)
    act = np.flatnonzero(~rest)
    if len(act) < 4:
        return out
    f0, f3 = int(act[0]), int(act[-1])
    hs = _smooth(height, 5)
    seg = hs[f0:f3 + 1]
    if len(seg) < 6:
        return dict(out, f0=f0, f3=f3, f1=f0, f2=f3)
    lo, hi = float(seg.min()), float(seg.max())
    up = np.flatnonzero(seg >= lo + 0.5 * (hi - lo))
    a_up, b_up = f0 + int(up[0]), f0 + int(up[-1])          # the rise ends / the fall begins around these frames
    sp = _smooth(speed, 3)
    mins = [i for i in range(max(f0 + 1, 1), min(f3, T - 2) + 1) if sp[i] <= sp[i - 1] and sp[i] <= sp[i + 1]]
    after = [i for i in mins if i >= a_up]
    before = [i for i in mins if i <= b_up]
    f1 = int(after[0]) if after else a_up
    f2 = int(before[-1]) if before else b_up
    if f2 - f1 < 3:                                          # one-movement sign: the stroke is the raised part
        f1, f2 = a_up, max(b_up, a_up + 3)
    f1, f2 = int(np.clip(f1, f0, f3)), int(np.clip(f2, f0, f3))
    return dict(f0=f0, f1=f1, f2=max(f2, f1), f3=f3, found=bool(f3 - f0 >= 6 and f2 > f1))


def handedness(sc, kp, hw):
    """(dominant 'left'/'right'/None, n_hands 1/2, symmetric bool) of one clip from hand presence and wrist travel"""
    hp = hand_present(sc)
    k = _iso(kp, hw)
    travel = []
    for sl in (LH, RH):
        r = k[:, sl.start]
        m = np.asarray(sc)[:, sl.start] > 0
        travel.append(float(np.linalg.norm(np.diff(r[m], axis=0), axis=1).sum()) if m.sum() > 2 else 0.0)
    frac = hp.mean(0)
    if frac.max() < 0.05:
        return None, 0, False
    both = (hp.all(1).mean() > 0.5) and min(travel) > 0.4 * max(travel)
    dom = "right" if travel[1] >= travel[0] else "left"
    sym = bool(both and min(travel) > 0.75 * max(travel))
    return dom, 2 if both else 1, sym


def quality(sc, kp, hw, lock_ok=True):
    """0–1 score + flags: hand_frac × (1 − hand-loss share) × framing_ok × lock_ok (prompt 1D)"""
    scs = np.asarray(sc)
    hp = hand_present(scs).any(1)
    hand_frac = float(hp.mean()) if len(hp) else 0.0
    loss = hand_loss_share(scs)
    mid, sw, ok = shoulder_geometry(kp, scs, hw)
    flags = []
    k = np.asarray(kp)
    hands_y = np.concatenate([k[:, LH, 1][scs[:, LH] > 0], k[:, RH, 1][scs[:, RH] > 0]]) if hp.any() else np.zeros(0)
    cut_bottom = bool(len(hands_y) and np.mean(hands_y > 0.97) > 0.1)
    W = float(hw[1]) / max(float(hw[0]), 1.0) if hw is not None else 1.0
    small = bool(np.isfinite(sw) and sw / max(W, 1e-6) < 0.12)
    framing_ok = 1.0
    if not ok.mean() > 0.6:
        flags.append("no_shoulders"); framing_ok *= 0.5
    if cut_bottom:
        flags.append("cut_bottom"); framing_ok *= 0.8
    if small:
        flags.append("small"); framing_ok *= 0.8
    if loss > 0.1:
        flags.append("hand_loss")
    if not lock_ok:
        flags.append("two_person")
    q = hand_frac * (1 - loss) * framing_ok * (1.0 if lock_ok else 0.5)
    return float(np.clip(q, 0, 1)), flags, dict(hand_frac=hand_frac, hand_loss=loss, sw_frac=float(sw / W) if np.isfinite(sw) else None)


# ----------------------------------------------------------------------------- log-replay corruption (prompt 1E.5)
class Replay:
    """Nuisance statistics of the saved live requests (cache/live_requests, read-only): frame-time sequences, hand-loss runs,
    hand-size ratio. apply(kp, sc, rng) returns a corrupted copy at 25 fps (labels unchanged). Never trains on LOG content itself."""

    DATA_HAND_RATIO = 0.245          # dataset median of wrist -> middle-MCP / shoulder width (artifacts/reports/v12_eda.json, isolated clips)

    def __init__(self, traces):
        self.traces = [t for t in traces if len(t.get("dt", ())) >= 8]
        hs = np.array([t["hand_ratio"] for t in self.traces if t.get("hand_ratio")], np.float32)
        self.hand_iqr = (float(np.percentile(hs, 25)), float(np.percentile(hs, 75))) if len(hs) >= 4 else (0.9, 1.1)

    @staticmethod
    def from_requests(paths, max_n=400):
        import gzip
        import json
        traces = []
        for p in list(paths)[:max_n]:
            try:
                op = gzip.open if str(p).endswith(".gz") else open
                with op(p, "rt", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            fr = d.get("frames") or d.get("landmarks") or []
            ts = [x.get("t") for x in fr if isinstance(x, dict) and x.get("t") is not None]
            if len(ts) < 8:
                continue
            dt = np.diff(np.asarray(ts, np.float64))
            dt = dt[(dt > 0) & (dt < 2000)]
            hands = np.array([bool(x.get("lh") or x.get("rh") or x.get("left_hand") or x.get("right_hand")) for x in fr])
            W, H = float(d.get("width") or 640), float(d.get("height") or 480)
            rs = []                                    # hand size (wrist -> middle-finger MCP) / shoulder width, isotropic pixels
            for x in fr:
                po = x.get("pose") or []
                if len(po) < 13:
                    continue
                sw = np.hypot((po[11][0] - po[12][0]) * W, (po[11][1] - po[12][1]) * H)
                for h in (x.get("lh") or [], x.get("rh") or []):
                    if len(h) == 21 and sw > 1:
                        rs.append(np.hypot((h[9][0] - h[0][0]) * W, (h[9][1] - h[0][1]) * H) / sw)
            traces.append(dict(dt=(dt / 1000.0 if np.median(dt) > 1 else dt).tolist(), hand=hands.tolist(),
                               hand_ratio=float(np.median(rs)) if len(rs) >= 5 else None))
        return Replay(traces)

    def apply(self, kp, sc, rng):
        if not self.traces:
            return kp, sc
        tr = self.traces[rng.randint(len(self.traces))]
        T = len(kp)
        dt = np.asarray(tr["dt"], np.float64)
        # 1) frame times: sample the clip at the trace's real frame times, hold the last sample until the next one (as 25 fps resampling does)
        t_src = np.arange(T) / FPS
        o = rng.randint(len(dt))
        ts = [0.0]
        while ts[-1] < t_src[-1]:
            ts.append(ts[-1] + dt[o % len(dt)]); o += 1
        ts = np.asarray(ts)
        pick = np.clip(np.searchsorted(ts, t_src, side="right") - 1, 0, len(ts) - 1)
        src = np.clip(np.round(ts[pick] * FPS).astype(int), 0, T - 1)
        kp2, sc2 = kp[src].copy(), sc[src].copy()
        # 2) hand-loss runs of the trace (zero both hands where the trace had none, between its first and last hand)
        h = np.asarray(tr["hand"], bool)
        if h.any():
            i0, i1 = np.flatnonzero(h)[[0, -1]]
            hseg = h[i0:i1 + 1]
            idx = np.clip((np.arange(T) * len(hseg) / max(T, 1)).astype(int), 0, len(hseg) - 1)
            lost = ~hseg[idx]
            for sl in (LH, RH):
                sc2[lost, sl] = 0; kp2[lost, sl] = 0
        # 3) hand size: scale both hands about their wrist within the inter-quartile range of the LOG
        s = rng.uniform(*self.hand_iqr) / self.DATA_HAND_RATIO     # LOG hand size relative to the dataset median (v12 EDA, same definition)
        for sl in (LH, RH):
            m = sc2[:, sl.start] > 0
            root = kp2[m, sl.start][:, None]
            kp2[m, sl] = root + (kp2[m, sl] - root) * s
        return kp2, sc2


def caption_words(caption, vocab, concept_of=None):
    """Thai caption → candidate vocabulary concepts (a BAG, order kept for readability; prompt 1C): newmm tokens, speaker markers and
    unsigned particles dropped, tokens mapped through the lexicon (concept_of) and kept when they are vocabulary concepts.
    A longest-match pass over the raw caption catches compounds the tokenizer splits (e.g. ประชุม → ประ|ชุม)."""
    from pythainlp.tokenize import word_tokenize
    drop = {"ถาม", "ตอบ", ":", "ครับ", "ค่ะ", "คะ", "นะ", "ๆ", "จ้ะ", "จ้า", "ฮะ", "?", "!", ",", "-", "เออ"}
    text = str(caption)
    for m in ("ถาม :", "ตอบ :", "ถาม:", "ตอบ:"):
        text = text.replace(m, " ")
    cmap = concept_of or {}
    out = []
    for t in word_tokenize(text, engine="newmm", keep_whitespace=False):
        t = t.strip()
        if not t or t in drop:
            continue
        c = cmap.get(t, t)
        if c in vocab and c not in out:
            out.append(c)
    rest = text
    for w in sorted(vocab, key=len, reverse=True):          # compounds / words the tokenizer cut differently
        if len(w) >= 3 and w in rest:
            c = cmap.get(w, w)
            if c not in out:
                out.append(c)
            rest = rest.replace(w, " ")
    return out
