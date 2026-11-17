"""The browser's live-mode cutting rule (webapp/web/lib/useSignEngine.ts + holistic.ts handsUp / handsRaised) in Python, for
`live-replay` (prompt-opt-v12 1I) and the live-rule experiments of TPH 3E. Parity with the TypeScript is checked by
tests/test_live_rule.py (node evaluates the TS functions on saved requests).

Defaults = the shipped v10 behaviour: pre-roll 500 ms, start after 250 ms of a raised real hand, end after `end_ms` (1,500) without
hands up, frames kept up to 400 ms after the last hands-up, cap 20 s. v12 options (all off by default): wrist_grace_ms (LM1: the body-pose
wrist keeps an utterance open only this long after the last real 21-point hand), continue_ms (LM3: hands back within this time after an
end → the same utterance continues), cap_cut (LM4: at the cap, cut at the longest hands-down gap of the last 5 s).
"""
from __future__ import annotations

import numpy as np

RAISED_SW = 0.8


def frames_to_arrays(frames):
    """list of browser frames (t ms, pose 33×4, lh / rh 21×3 or [], face, bs) → dict of arrays (t in seconds)"""
    T = len(frames)
    pose = np.zeros((T, 33, 4), np.float32); lh = np.zeros((T, 21, 3), np.float32); rh = np.zeros((T, 21, 3), np.float32)
    t = np.zeros(T, np.float64)
    for i, f in enumerate(frames):
        t[i] = float(f["t"]) / 1000.0
        p = f.get("pose") or []
        if len(p) >= 33:
            a = np.asarray(p, np.float32)
            pose[i, :, :a.shape[1]] = a[:, :4]
        for dst, k in ((lh, "lh"), (rh, "rh")):
            h = f.get(k) or []
            if len(h) == 21:
                a = np.asarray(h, np.float32); dst[i, :, :a.shape[1]] = a[:, :3]
    return dict(t=t, pose=pose, lh=lh, rh=rh)


def flags(raw):
    """per frame: handsUp (end rule: a real hand's mean y < mid + 1.5 sw, or a body wrist with visibility > 0.5 < mid + 1.3 sw),
    handsRaised (start rule: a real hand < mid + 0.8 sw), real hand present, real-hand-up (no wrist guess)"""
    p = raw["pose"]
    ls, rs = p[:, 11], p[:, 12]
    sw = np.hypot(ls[:, 0] - rs[:, 0], ls[:, 1] - rs[:, 1]); sw[sw == 0] = 0.2
    mid = (ls[:, 1] + rs[:, 1]) / 2
    has_l = np.abs(raw["lh"]).sum((1, 2)) > 0
    has_r = np.abs(raw["rh"]).sum((1, 2)) > 0
    my_l, my_r = raw["lh"][:, :, 1].mean(1), raw["rh"][:, :, 1].mean(1)
    upk = lambda k: (has_l & (my_l < mid + k * sw)) | (has_r & (my_r < mid + k * sw))      # noqa: E731
    wrist = lambda i: (p[:, i, 3] > 0.5) & (p[:, i, 1] < mid + 1.3 * sw)                  # noqa: E731
    pose_ok = np.abs(p[:, :17, :2]).sum((1, 2)) > 0
    real_up = upk(1.5) & pose_ok
    up = (real_up | wrist(15) | wrist(16)) & pose_ok
    raised = upk(RAISED_SW) & pose_ok
    return up, raised, has_l | has_r, real_up


def live_utts(raw, end_ms=1500.0, pre=500.0, tail=400.0, up_ms=250.0, max_ms=20000.0, wrist_grace_ms=None, continue_ms=None, cap_cut=False):
    """the live state machine over a continuous stream → (list of frame-index arrays, number dropped as too short)"""
    up, raised, hands, real_up = flags(raw)
    t = raw["t"] * 1000.0
    T = len(t)
    utts, dropped = [], 0
    state, up_since, start, last_up, last_real, u0 = "w", None, 0.0, 0.0, 0.0, 0
    prev_end_t, prev_kept = None, False            # LM3 bookkeeping: when the last utterance ended (by the end rule) and was kept
    for i in range(T):
        if state == "w":
            if raised[i]:
                if up_since is None:
                    up_since = t[i]
            else:
                up_since = None
            if up_since is not None and t[i] - up_since >= up_ms:
                state, last_up, last_real = "s", t[i], t[i]
                if continue_ms and prev_kept and prev_end_t is not None and t[i] - prev_end_t <= continue_ms:
                    prev = utts.pop()                                   # the same utterance continues after a short pause
                    u0, start = int(prev[0]), t[int(prev[0])]
                else:
                    u0, start = int(np.searchsorted(t, t[i] - pre)), t[i]
        else:
            if real_up[i]:
                last_real = t[i]
            ok_up = up[i] if wrist_grace_ms is None else (real_up[i] or (up[i] and t[i] - last_real <= wrist_grace_ms))
            if ok_up:
                last_up = t[i]
            capped = t[i] - start >= max_ms
            if t[i] - last_up >= end_ms or capped or i == T - 1:
                idx = np.arange(u0, i + 1)
                if capped and cap_cut:                                  # cut at the longest hands-down gap of the last 5 s
                    w = idx[t[idx] >= t[i] - 5000]
                    best, cur, best_end = 0, 0, None
                    for k, d in enumerate(~up[w]):
                        cur = cur + 1 if d else 0
                        if cur > best:
                            best, best_end = cur, w[k]
                    if best_end is not None:
                        idx = idx[idx <= best_end]
                else:
                    idx = idx[t[idx] <= last_up + tail]
                dur = t[idx[-1]] - t[idx[0]] if len(idx) > 1 else 0
                upms = float(np.sum(np.diff(t[idx])[up[idx][1:]])) if len(idx) > 1 else 0
                if dur < 1200 or upms < 600 or hands[idx].sum() < 3:
                    dropped += 1; prev_kept = False
                else:
                    utts.append(idx); prev_kept = not capped
                prev_end_t = t[i]
                state, up_since = "w", None
    return utts, dropped
