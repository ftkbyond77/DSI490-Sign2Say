"""IR sign units → one continuous motion timeline (continuous signing, prompt point 6).

A dictionary clip is an isolated citation sign: rest → (lift) → STROKE → (drop) → rest. In a sentence the hands must not drop to
rest between signs, so each clip is cut to its stroke (between the first and last moment the hands are at signing distance from
rest, snapped to the nearest speed minimum = a hold) and consecutive strokes are joined by a synthesised transition:
  * quaternion slerp per bone with a minimum-jerk time profile (zero velocity at both ends, like the holds it joins),
  * duration from the distance the hands travel, D = c0 + c1·d (fitted on the lift phases of thousands of dictionary clips —
    how fast real signers reach a sign; timing.json), scaled by the sentence tempo factor measured on TSL51 (the same signer
    signing words in isolation vs in sentences),
  * face morphs cross-faded over the same window.
Sentence start / end: from / to the rest pose. Clause boundaries: a short hold. Text units (no recorded sign): a pause with the
hands in neutral space while the word is captioned. Fingerspelling: the hold of each letter clip, fast letter-to-letter transitions.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation as R

from .. import paths
from .retarget import MORPHS
from .rig import BONE_INDEX, BONES, get_rig, identity_pose

FPS = 25.0
TIMING_PATH = paths.MODELS / "timing.json"
DEFAULT_TIMING = dict(c0=0.45, c1=0.003, tempo=1.0, min_s=0.3, max_s=0.6, rest_c0=0.55, rest_c1=0.003, clause_hold_s=0.25,
                      hold_cap_s=0.24, text_pause_s=0.9, letter_hold_s=0.30, letter_trans_s=0.12)


@lru_cache(maxsize=1)
def timing():
    t = dict(DEFAULT_TIMING)
    if TIMING_PATH.exists():
        t.update(json.loads(TIMING_PATH.read_text(encoding="utf-8")))
    return t


@lru_cache(maxsize=1)
def rest_pose():
    """relaxed half-body rest: upper arms down along the body, elbows slightly bent, hands in front of the thighs, fingers softly
    curled (model frame: left arm points −X at rest; rotating about +Z lowers it)."""
    q = identity_pose(1)[0].astype(np.float64)
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        q[BONE_INDEX[f"{side}UpperArm"]] = R.from_euler("zx", [sgn * 72, 4], degrees=True).as_quat()
        q[BONE_INDEX[f"{side}LowerArm"]] = R.from_euler("y", [sgn * -18], degrees=True).as_quat()
        q[BONE_INDEX[f"{side}Hand"]] = R.from_euler("z", [sgn * 6], degrees=True).as_quat()
        for f in ("Index", "Middle", "Ring", "Little"):
            for g, a in (("Proximal", 14), ("Intermediate", 18), ("Distal", 10)):
                q[BONE_INDEX[f"{side}{f}{g}"]] = R.from_euler("z", [sgn * a], degrees=True).as_quat()
        q[BONE_INDEX[f"{side}ThumbProximal"]] = R.from_euler("y", [sgn * -10], degrees=True).as_quat()
    return q.astype(np.float32)


def wrists(q, rig=None):
    """avatar wrist positions [T, 2, 3] (left, right) by forward kinematics."""
    rig = rig or get_rig()
    fk = rig.fk(q, names=["leftHand", "rightHand"])
    return np.stack([fk["leftHand"][0], fk["rightHand"][0]], 1)


def stroke(q, core, rig=None):
    """[a, b) frames of the stroke inside a library clip (see module docstring)."""
    a0, b0 = core
    W = wrists(q, rig)
    rest_w = wrists(rest_pose()[None], rig)[0]
    d = np.linalg.norm(W - rest_w[None], axis=-1).max(1)              # the hand furthest from rest
    n = b0 - a0
    if n < 6:
        return a0, b0
    mid = d[a0 + n // 4: b0 - n // 4] if n >= 8 else d[a0:b0]
    lvl = 0.8 * float(np.median(mid))
    above = np.flatnonzero(d[a0:b0] >= lvl) + a0
    if len(above) < 2:
        return a0, b0
    s, e = int(above[0]), int(above[-1]) + 1
    sp = np.r_[0, np.linalg.norm(np.diff(W, axis=0), axis=-1).max(1)]
    def snap(i, lo, hi):
        lo, hi = max(lo, a0), min(hi, b0 - 1)
        if hi <= lo:
            return i
        return int(lo + np.argmin(sp[lo:hi + 1]))
    s = snap(s, s - 2, s + 3)
    e = snap(e - 1, e - 4, e + 1) + 1
    if e - s < 4:
        return a0, b0
    return s, e


def cap_holds(q, m, cap, head=True, tail=True):
    """connected signing reduces the long holds of citation forms: a still start / end longer than `cap` frames is shortened."""
    T = len(q)
    if T < 2 * cap + 4:
        return q, m
    d = np.linalg.norm(np.diff(q.reshape(T, -1), axis=0), axis=1)
    still = d < 0.02
    a, b = 0, T
    if head:
        k = 0
        while k < T - 1 and still[k]:
            k += 1
        a = max(0, k - cap)
    if tail:
        k = T - 2
        while k > a and still[k]:
            k -= 1
        b = min(T, k + 2 + cap)
    return q[a:b], m[a:b]


def min_jerk(n):
    u = np.linspace(0, 1, n + 2)[1:-1]
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


FINGER_BONES = np.array([i for i, b in enumerate(BONES) if any(f in b for f in ("Thumb", "Index", "Middle", "Ring", "Little"))])
HANDSHAPE_LEAD = 0.75          # fingers reach the next handshape at 75 % of a transition (anticipatory coarticulation)


def transition(qa, qb, ma, mb, n, lead=HANDSHAPE_LEAD):
    """n in-between frames from pose a to pose b: per-bone slerp with a minimum-jerk time profile; the fingers run on a
    compressed clock so the next handshape is formed before the hand arrives (fluent signers pre-shape the hand during the
    movement — anticipatory coarticulation; the stroke itself is never changed). Morphs lerp."""
    if n <= 0:
        return np.zeros((0,) + qa.shape, np.float32), np.zeros((0,) + ma.shape, np.float32)
    w = min_jerk(n)
    u = np.linspace(0, 1, n + 2)[1:-1]
    uf = np.clip(u / lead, 0, 1)
    wf = 10 * uf ** 3 - 15 * uf ** 4 + 6 * uf ** 5
    ra, rb = R.from_quat(qa), R.from_quat(qb)
    rel = (ra.inv() * rb).as_rotvec()                                  # [B,3]
    W = np.repeat(w[:, None], len(qa), 1)
    W[:, FINGER_BONES] = wf[:, None]
    q = np.stack([(ra * R.from_rotvec(rel * W[k][:, None])).as_quat() for k in range(n)]).astype(np.float32)
    m = (ma[None] * (1 - w[:, None]) + mb[None] * w[:, None]).astype(np.float32)
    return q, m


def trans_frames(qa, qb, rig, kind="sign", tempo=1.0, wa=None, wb=None):
    """duration of a transition from the hands' travel distance (timing.json model), in frames.
    wa / wb: precomputed wrist positions [2, 3] of the two poses (library index) — FK only when missing."""
    t = timing()
    if wa is None or wb is None:
        W = wrists(np.stack([qa, qb]), rig)
        wa, wb = (W[0] if wa is None else wa), (W[1] if wb is None else wb)
    d = float(np.linalg.norm(np.asarray(wb) - np.asarray(wa), axis=-1).max()) / rig.sw          # shoulder widths
    if kind == "rest":
        sec = t["rest_c0"] + t["rest_c1"] * d
    else:
        sec = (t["c0"] + t["c1"] * d) * t["tempo"]
    sec = float(np.clip(sec / tempo, t["min_s"], t["max_s"] * (1.6 if kind == "rest" else 1.0)))
    return max(2, int(round(sec * FPS)))


def retime(q, m, factor):
    """play a stroke faster (factor > 1) by resampling with slerp."""
    if abs(factor - 1.0) < 1e-3 or len(q) < 3:
        return q, m
    T = len(q)
    n = max(2, int(round((T - 1) / factor)) + 1)
    src = np.linspace(0, T - 1, n)
    i0 = np.floor(src).astype(int); i1 = np.minimum(i0 + 1, T - 1); u = src - i0
    out = np.empty((n,) + q.shape[1:], np.float32)
    for b in range(q.shape[1]):
        ra, rb = R.from_quat(q[i0, b]), R.from_quat(q[i1, b])
        out[:, b] = (ra * R.from_rotvec((ra.inv() * rb).as_rotvec() * u[:, None])).as_quat()
    mm = m[i0] * (1 - u[:, None]) + m[i1] * u[:, None]
    return out, mm.astype(np.float32)


def letter_hold(q, core, rig):
    """the steadiest window of a fingerspelling clip (the letter's handshape hold)."""
    a, b = stroke(q, core, rig)
    W = wrists(q, rig)
    sp = np.r_[0, np.linalg.norm(np.diff(W, axis=0), axis=-1).max(1)]
    k = max(3, int(round(timing()["letter_hold_s"] * FPS)))
    if b - a <= k:
        return a, b
    cs = np.convolve(sp[a:b], np.ones(k), "valid")
    s = a + int(np.argmin(cs))
    return s, s + k


def neutral_pose():
    """hands slightly raised in neutral signing space — used while a text card is shown (no sign is invented)."""
    q = rest_pose().astype(np.float64).copy()
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        q[BONE_INDEX[f"{side}UpperArm"]] = R.from_euler("zx", [sgn * 66, -12], degrees=True).as_quat()
        q[BONE_INDEX[f"{side}LowerArm"]] = R.from_euler("y", [sgn * -70], degrees=True).as_quat()
    return q.astype(np.float32)


def compose(units, lib, speed=1.0, clause_ends=(), overlays=None, rig=None):
    """units: SignUnit list (IR order). lib: motion Library. Returns q, m, segments (frame ranges per unit; unit timings are
    written into the units) and the sentence-level non-manual marks applied (nmm.py).
    clause_ends: unit indices after which a clause ends (short hold). overlays: [{type, units, neg_units}] per clause."""
    rig = rig or get_rig()
    t = timing()
    Q, Mo, segs = [], [], []
    M0 = np.zeros(len(MORPHS), np.float32)
    rest = rest_pose()
    cur_q, cur_m, cur_w = rest, M0, None
    Q.append(rest[None]); Mo.append(M0[None])
    total = 1
    first = True
    for ui, u in enumerate(units):
        pieces = []                                                    # (q, m, wrist at start, wrist at end) of this unit
        if u.kind in ("sign", "phrase") and u.concept in lib:
            q, m, info = lib.get(u.concept)
            a, b = info.get("stroke") or stroke(q, info["core"], rig)
            sq, sm = cap_holds(q[a:b], m[a:b], int(round(t.get("hold_cap_s", 0.24) * FPS)),
                               head=ui > 0, tail=ui < len(units) - 1)
            sq, sm = retime(sq, sm, speed * t.get("stroke_speed", 1.0))
            pieces.append((sq, sm, info.get("w_stroke", [None, None])[0], info.get("w_stroke", [None, None])[1]))
        elif u.kind == "fingerspell" and u.letters:
            for L in u.letters:
                if L not in lib:
                    continue
                q, m, info = lib.get(L)
                a, b = info.get("hold") or letter_hold(q, info["core"], rig)
                pieces.append((q[a:b], m[a:b], None, None))
        else:                                                          # text card: neutral hold
            n = int(round(t["text_pause_s"] * FPS / speed))
            pieces.append((np.repeat(neutral_pose()[None], n, 0), np.repeat(M0[None], n, 0), None, None))
        start = None
        for pi, (sq, sm, w0, w1) in enumerate(pieces):
            kind = "rest" if first else ("letter" if (u.kind == "fingerspell" and pi > 0) else "sign")
            if kind == "letter":
                nt = max(2, int(round(t["letter_trans_s"] * FPS / speed)))
            else:
                nt = trans_frames(cur_q, sq[0], rig, "rest" if first else "sign", tempo=speed, wa=cur_w, wb=w0)
            tq, tm = transition(cur_q, sq[0], cur_m, sm[0], nt)
            if len(tq):
                segs.append(dict(unit=ui, kind="transition", start=total, end=total + len(tq), label=""))
                Q.append(tq); Mo.append(tm); total += len(tq)
            if start is None:
                start = total
            Q.append(sq); Mo.append(sm); total += len(sq)
            cur_q, cur_m, cur_w = sq[-1], sm[-1], w1
            first = False
        segs.append(dict(unit=ui, kind=u.kind, start=start if start is not None else total, end=total, label=u.gloss))
        u.timing = dict(start_s=round((start or total) / FPS, 3), end_s=round(total / FPS, 3))
        if ui in clause_ends and ui < len(units) - 1:
            n = int(round(t["clause_hold_s"] * FPS / speed))
            Q.append(np.repeat(cur_q[None], n, 0)); Mo.append(np.repeat(cur_m[None], n, 0)); total += n
    # back to rest
    if not first:
        nt = trans_frames(cur_q, rest, rig, "rest", tempo=speed, wa=cur_w)
        tq, tm = transition(cur_q, rest, cur_m, M0, nt)
        Q.append(tq); Mo.append(tm); total += len(tq)
        segs.append(dict(unit=-1, kind="transition", start=total - len(tq), end=total, label=""))
        Q.append(rest[None]); Mo.append(M0[None]); total += 1
    q = np.concatenate(Q).astype(np.float32)
    m = np.concatenate(Mo).astype(np.float32)
    marks = []
    if overlays:
        from .nmm import apply_overlays
        q, m, marks = apply_overlays(q, m, overlays, segs)
    return q, m, segs, marks


def annotate(lib_dir=None, only_missing=False):
    """precompute per clip (once, offline): stroke [a, b), letter hold [a, b), avatar wrist positions at the stroke ends —
    so a request never runs forward kinematics over whole clips. only_missing: clips added since (library.add)."""
    from .library import LIB, Library
    lib_dir = lib_dir or LIB
    lib = Library(lib_dir)
    rig = get_rig()
    idx = lib.index
    for k, info in idx["clips"].items():
        if only_missing and "stroke" in info:
            continue
        q, m, _ = lib.get(k)
        a, b = stroke(q, info["core"], rig)
        W = wrists(q[[a, b - 1]], rig)
        info["stroke"] = [int(a), int(b)]
        info["w_stroke"] = [W[0].round(4).tolist(), W[1].round(4).tolist()]
        ha, hb = letter_hold(q, info["core"], rig)
        info["hold"] = [int(ha), int(hb)]
    (lib_dir / "index.json").write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    return len(idx["clips"])
