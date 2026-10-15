"""MediaPipe landmarks of a real signer → local bone rotations + face morphs of the VRM avatar (offline, per dictionary clip).

Design (why not plain "copy the joint angles"): sign meaning lives in WHERE the hand is relative to the body (chin, chest, temple),
in the hand's orientation and in the handshape. Avatar proportions differ from the signer's (VRoid head is large, shoulders narrow),
so the wrist target is placed relative to the nearest body anchor (own shoulder · mouth/nose · chest) scaled by shoulder width,
then a two-bone IK with the signer's own elbow as pole puts the arm there; the palm orientation comes from the hand's 3D landmarks;
fingers are hinge-constrained (no sideways-bent fingers from depth noise). Depth: MediaPipe's relative z is rescaled per clip so
that the signer's upper-arm and forearm keep a constant 3D length (the scale that makes bones rigid), and the hands are kept in
front of the avatar's torso surface.
All in the model frame of rig.py (+Y up, avatar faces −Z, its left = −X). Image → model: X = −x·W/H, Y = −y, Z = +z·W/H·α.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from ..data.quality import L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, NOSE, present, rest_position
from .rig import BONES, BONE_INDEX, FINGERS, SEGS, SIDES, get_rig, identity_pose

# avatar surface anchors measured from the VRM mesh (rig frame, metres) — see result_v1.md §3.2
AV_NOSE = np.array([0.0, 1.574, -0.084])
AV_MOUTH = np.array([0.0, 1.541, -0.066])
AV_CHEST_FRONT_Z = -0.118                      # torso front surface (y 1.20–1.40)
CONTACT_GAP = 0.05                             # m between the wrists of touching hands (≈ two hand thicknesses of the avatar)
FINGER_LM = {"Thumb": [1, 2, 3, 4], "Index": [5, 6, 7, 8], "Middle": [9, 10, 11, 12], "Ring": [13, 14, 15, 16], "Little": [17, 18, 19, 20]}
MORPHS = ["Fcl_BRW_Surprised", "Fcl_BRW_Angry", "Fcl_BRW_Sorrow", "Fcl_EYE_Close_L", "Fcl_EYE_Close_R", "Fcl_EYE_Spread",
          "Fcl_MTH_A", "Fcl_MTH_I", "Fcl_MTH_U", "Fcl_MTH_O", "Fcl_MTH_Fun", "Fcl_MTH_Angry"]
BS = {n: i for i, n in enumerate(["_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
      "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
      "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
      "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft",
      "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
      "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
      "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
      "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"])}


# ----------------------------------------------------------------------------- small vector helpers (vectorised over frames)
def _n(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def frame(p, s):
    """orthonormal frame [.., 3, 3] (columns e1 ∥ p, e2 = s ⟂ p, e3 = e1 × e2)."""
    e1 = _n(p)
    e3 = _n(np.cross(e1, s))
    e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3], -1)


def align(p, s, p0, s0):
    """rotation taking the rest frame (p0, s0) to the target frame (p, s); p, s [T,3], p0, s0 [3]. Degenerate targets
    (missing landmarks, parallel axes) fall back to the rest frame (identity) for that frame."""
    p = np.where(np.linalg.norm(p, axis=-1, keepdims=True) < 1e-9, p0, p)
    bad = np.linalg.norm(np.cross(_n(p), s), axis=-1) < 1e-6
    p = np.where(bad[:, None], p0, p); s = np.where(bad[:, None], s0, s)
    Ft = frame(p, s)
    F0 = frame(p0[None], s0[None])[0]
    return R.from_matrix(Ft @ F0.T)


def min_rot(a, b):
    """shortest rotation taking unit vectors a → b (both [T,3] or a [3])."""
    a = np.broadcast_to(a, b.shape)
    ax = np.cross(a, b)
    s = np.linalg.norm(ax, axis=-1)
    c = np.clip((a * b).sum(-1), -1, 1)
    ang = np.arctan2(s, c)
    axn = np.where(s[:, None] > 1e-8, ax / np.maximum(s[:, None], 1e-12), np.array([0, 0, 1.0]))
    return R.from_rotvec(axn * ang[:, None])


def signed_angle(a, b, axis):
    """signed angle a → b around axis (a, b projected on the plane ⟂ axis)."""
    a = a - (a * axis).sum(-1, keepdims=True) * axis
    b = b - (b * axis).sum(-1, keepdims=True) * axis
    return np.arctan2((np.cross(a, b) * axis).sum(-1), (a * b).sum(-1))


def flexion(a, b, axis, lo_deg, hi_deg):
    """finger hinge angle a → b, unwrapped: a curl that measures past 180° (wraps to a large negative angle) is still a curl, not a
    hyper-extension — without this a tightly curled finger flips between fully bent and straight from one frame to the next."""
    x = signed_angle(a, b, axis)
    x = np.where(x < np.deg2rad(-60), x + 2 * np.pi, x)
    from scipy.ndimage import median_filter
    if len(x) >= 5:
        x = median_filter(x, size=5, mode="nearest")          # one-frame landmark flicker
    return np.clip(x, np.deg2rad(lo_deg), np.deg2rad(hi_deg))


def frac(rot, k):
    """rotation scaled by k (slerp from identity)."""
    return R.from_rotvec(rot.as_rotvec() * k)


def swing_twist(rot, axis):
    """split rot = swing ∘ twist around unit axis (local)."""
    q = rot.as_quat()
    proj = (q[:, :3] * axis).sum(-1, keepdims=True) * axis
    tw = np.concatenate([proj, q[:, 3:]], -1)
    nrm = np.linalg.norm(tw, axis=-1, keepdims=True)
    tw = np.where(nrm > 1e-9, tw / np.maximum(nrm, 1e-12), np.array([0, 0, 0, 1.0]))
    twist = R.from_quat(tw)
    return rot * twist.inv(), twist


def clamp_angle(rot, max_deg):
    rv = rot.as_rotvec()
    a = np.linalg.norm(rv, axis=-1, keepdims=True)
    lim = np.deg2rad(max_deg)
    return R.from_rotvec(np.where(a > lim, rv / np.maximum(a, 1e-9) * lim, rv))


# ----------------------------------------------------------------------------- coordinates
def to_model(a, hw, alpha=1.0):
    """normalised MediaPipe [.., 3] → model frame (isotropic units of frame height, depth scaled by alpha)."""
    H, W = hw
    out = np.empty(a.shape[:-1] + (3,), np.float64)
    out[..., 0] = -a[..., 0] * W / H
    out[..., 1] = -a[..., 1]
    out[..., 2] = a[..., 2] * W / H * alpha
    return out


def fit_alpha(pose, vis, hw):
    """depth scale that makes the arm bones most rigid (min coefficient of variation of 3D upper-arm + forearm length)."""
    best, ba = 1e9, 0.4
    for alpha in np.linspace(0.05, 1.0, 20):
        P = to_model(pose[:, :, :3], hw, alpha)
        cv = []
        for a, b in ((L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR)):
            m = (vis[:, a] > 0.6) & (vis[:, b] > 0.6)
            if m.sum() < 5:
                continue
            ln = np.linalg.norm(P[m, a] - P[m, b], axis=1)
            cv.append(ln.std() / max(ln.mean(), 1e-6))
        if cv and np.mean(cv) < best:
            best, ba = float(np.mean(cv)), float(alpha)
    return ba, best


def fit_hand_alpha(hands, hw):
    """the same rigidity criterion for the hand landmarks (finger segment lengths)."""
    segs = [(0, 5), (5, 6), (0, 9), (9, 10), (0, 17), (5, 17)]
    best, ba = 1e9, 1.0
    for alpha in np.linspace(0.5, 5.0, 19):
        cv = []
        for h in hands:
            m = present(h)
            if m.sum() < 5:
                continue
            P = to_model(h[m], hw, alpha)
            for a, b in segs:
                ln = np.linalg.norm(P[:, a] - P[:, b], axis=1)
                cv.append(ln.std() / max(ln.mean(), 1e-6))
        if cv and np.mean(cv) < best:
            best, ba = float(np.mean(cv)), float(alpha)
    return ba


def _rest_frames(pose, vis, hw):
    """frames where both wrists rest (neutral calibration of head / torso / face)."""
    P = to_model(pose[:, :, :3], hw)
    sw = np.median(np.linalg.norm(P[:, L_SH, :2] - P[:, R_SH, :2], axis=1))
    rest = np.ones(len(P), bool)
    for wi in (L_WR, R_WR):
        r = rest_position(P[:, wi, :2] * [-1, -1], vis[:, wi])
        if r is None:
            continue
        rest &= np.linalg.norm(P[:, wi, :2] * [-1, -1] - r, axis=1) < 0.25 * sw
    if rest.sum() < 3:
        rest = np.zeros(len(P), bool); rest[:3] = True; rest[-3:] = True
    return rest


# ----------------------------------------------------------------------------- main
def retarget(raw, rig=None, face_depth=True):
    """cleaned raw MediaPipe dict → dict(q [T,B,4] local quats (x,y,z,w), morph [T,M], meta).
    face_depth: second pass that corrects the hands' depth where they overlap the face (face_depth_shift)."""
    rig = rig or get_rig()
    out = _retarget(raw, rig)
    if face_depth:
        shift, info = face_depth_shift(out["q"], rig)
        out["meta"].update(info)
        if shift is not None:
            q2 = _retarget(raw, rig, wrist_shift=shift)["q"]
            out["q"] = q2
    return out


FACE_BOX = dict(x=0.07, y0=-0.05, y1=0.13)     # head-local region in front of the face (chin … forehead, cheek to cheek), m
FACE_GAP = 0.03                                # closest hand point → nose plane during a face-located run, m
FACE_SHIFT_MAX = 0.15                          # m: the systematic depth bias measured on face runs (median 14 cm) — never more
HAND_POINTS = ["MiddleProximal"] + [f"{f}Tip" for f in FINGERS]


def _runs_of(mask, min_len=3, bridge=3):
    m = mask.copy()
    idx = np.flatnonzero(m)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= bridge + 1:
            m[a:b] = True
    out, i, T = [], 0, len(m)
    while i < T:
        if not m[i]:
            i += 1; continue
        j = i
        while j < T and m[j]:
            j += 1
        if j - i >= min_len:
            out.append((i, j))
        i = j
    return out


def face_depth_shift(q, rig, ramp=5):
    """Depth of hands at the face (v2). MediaPipe's pose depth is unreliable; in v1 a hand that pinched the nose (ห้องน้ำ) or
    covered the face (หน้ากาก) stayed ~20 cm IN FRONT of the face — right from the front camera, wrong in 3D. Signers do not
    hold a hand in front of their own face unless the sign is made there, so for every run of frames in which a hand overlaps the
    face (head-local box, chin to forehead) the whole run is moved in depth by ONE constant, so that its closest approach is
    FACE_GAP in front of the nose plane (or out of the face if it penetrated) — but never by more than FACE_SHIFT_MAX: face runs
    sat 7–22 cm in front (median 14 cm) across the dictionary, a systematic bias of the depth scaling, while a few signs are
    really made in front of the face (a pane of glass, a book) and must not be pulled onto it; monocular landmarks cannot tell
    those apart (apparent hand size was tested and does not separate them, result_v2.md §5.3). The recording's own relative depth
    motion inside the run is kept (a sign that starts at the eyes and moves away still moves away).
    Returns ({side: [T,3] wrist shift} | None, meta)."""
    T = len(q)
    names = ["head"] + [f"{s}{p}" for s in SIDES for p in HAND_POINTS]
    fk = rig.fk(np.asarray(q, np.float32), names=names)
    hp, hr = fk["head"]
    z_face = float((AV_NOSE - rig.rest["head"])[2])
    shift = {s: np.zeros((T, 3)) for s in SIDES}
    meta = dict(face_runs=0, face_gap_before=None, face_shift_max=0.0)
    gaps = []
    for s in SIDES:
        L = np.stack([hr.inv().apply(fk[f"{s}{p}"][0] - hp) for p in HAND_POINTS], 1)            # [T, P, 3] head-local
        inside = (np.abs(L[..., 0]) < FACE_BOX["x"]) & (L[..., 1] > FACE_BOX["y0"]) & (L[..., 1] < FACE_BOX["y1"]) & \
                 (L[..., 2] < z_face + 0.04)                                                       # in front of (or at) the face
        gap = np.where(inside, z_face - L[..., 2], np.inf).min(1)                                  # m in front of the nose plane
        dz = np.zeros(T)
        for a, b in _runs_of(np.isfinite(gap)):
            g = float(gap[a:b].min())
            d = float(np.clip(g - FACE_GAP, -0.05, FACE_SHIFT_MAX))
            if abs(d) < 0.01:
                continue
            gaps.append(g)
            meta["face_runs"] += 1
            env = np.zeros(T)
            env[a:b] = 1.0
            for r in range(1, ramp + 1):
                w = 0.5 + 0.5 * np.cos(np.pi * r / (ramp + 1))
                if a - r >= 0:
                    env[a - r] = max(env[a - r], w)
                if b - 1 + r < T:
                    env[b - 1 + r] = max(env[b - 1 + r], w)
            dz = np.where(np.abs(env * d) > np.abs(dz), env * d, dz)
        if np.any(dz):
            # head-local +z (towards the back of the head) in the world frame, per frame
            shift[s] = hr.apply(np.stack([np.zeros(T), np.zeros(T), dz], 1))
            meta["face_shift_max"] = max(meta["face_shift_max"], float(np.abs(dz).max()))
    if not gaps:
        return None, meta
    meta["face_gap_before"] = round(float(np.median(gaps)), 3)
    return shift, meta


def _retarget(raw, rig, wrist_shift=None):
    rig = rig or get_rig()
    hw = tuple(int(v) for v in raw["hw"])
    pose, vis = raw["pose"], raw["pose"][:, :, 3]
    T = len(pose)
    alpha, rig_cv = fit_alpha(pose, vis, hw)
    P = to_model(pose[:, :, :3], hw, alpha)
    sw_s = float(np.median(np.linalg.norm(P[:, L_SH, :2] - P[:, R_SH, :2], axis=1)))
    k = rig.sw / max(sw_s, 1e-6)                         # signer units → avatar metres
    rest = _rest_frames(pose, vis, hw)
    q = identity_pose(T).astype(np.float64)
    meta = dict(alpha=alpha, rigidity_cv=rig_cv, k=k, rest_frames=int(rest.sum()))

    # ---- torso: shoulder line relative to its neutral (camera offset removed), damped, split over 3 spine bones
    v = P[:, L_SH] - P[:, R_SH]
    up = np.repeat(np.array([[0, 1.0, 0]]), T, 0)
    tor = align(v, up, np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]))
    neutral = R.from_quat(tor[rest].as_quat()).mean()
    tor_rel = clamp_angle(frac(tor * neutral.inv(), 0.6), 20)
    part = frac(tor_rel, 1 / 3)
    for b in ("spine", "chest", "upperChest"):
        q[:, BONE_INDEX[b]] = part.as_quat()

    # ---- head: ear axis + nose direction, relative to neutral; 40 % neck / 60 % head
    ears_ok = (vis[:, 7] > 0.3) & (vis[:, 8] > 0.3)
    ear_axis = P[:, 7] - P[:, 8]
    fwd = P[:, NOSE] - (P[:, 7] + P[:, 8]) / 2
    hd = align(ear_axis, fwd, np.array([-1.0, 0, 0]), np.array([0, 0, -1.0]))
    hrest = rest & ears_ok
    hneutral = R.from_quat(hd[hrest if hrest.any() else rest].as_quat()).mean()
    head_w = clamp_angle(hd * hneutral.inv(), 40)
    chest_w = tor_rel
    head_rel = chest_w.inv() * head_w
    q[:, BONE_INDEX["neck"]] = frac(head_rel, 0.4).as_quat()
    q[:, BONE_INDEX["head"]] = frac(head_rel, 0.6).as_quat()

    # ---- avatar positions after the torso / head pose (FK of the chain so far)
    fk = rig.fk(q.astype(np.float32), names=["leftUpperArm", "rightUpperArm", "head", "upperChest"])
    head_p, head_r = fk["head"]
    av_nose = head_p + head_r.apply(np.repeat((AV_NOSE - rig.rest["head"])[None], T, 0))
    av_mouth = head_p + head_r.apply(np.repeat((AV_MOUTH - rig.rest["head"])[None], T, 0))
    s_mouth = (P[:, 9] + P[:, 10]) / 2
    s_nose = P[:, NOSE]

    hands = [raw["lh"], raw["rh"]]
    alpha_h = fit_hand_alpha(hands, hw)
    meta["alpha_hand"] = alpha_h
    hand_ok = []
    SIDES_ = (("left", L_SH, L_EL, L_WR), ("right", R_SH, R_EL, R_WR))
    tg = {}
    for si, (side, sh, el, wr) in enumerate(SIDES_):
        S_av = fk[f"{side}UpperArm"][0]
        Sg, Wg = P[:, sh], P[:, wr]
        # hand landmarks (better wrist than the pose model when tracked)
        hm = to_model(hands[si], hw, alpha_h)
        hp = present(hands[si])
        if hp.any():
            hm[hp] = hm[hp] - hm[hp, :1] + np.concatenate([hm[hp, :1, :2], Wg[hp, None, 2:]], -1)   # hand depth anchored at pose wrist
        Wg = np.where(hp[:, None], hm[:, 0], Wg)
        # wrist target from body anchors (own shoulder · face) scaled by shoulder width
        tgt_sh = S_av + k * (Wg - Sg)
        tgt_face = av_mouth + k * (Wg - s_mouth)
        d_face = np.linalg.norm((Wg - s_mouth)[:, :2], axis=1) / sw_s
        w_face = np.exp(-(d_face / 0.55) ** 2)[:, None]
        W_t = (1 - w_face) * tgt_sh + w_face * tgt_face
        if wrist_shift is not None:
            W_t = W_t + wrist_shift[side]
        tg[side] = dict(W_t=W_t, Wg=Wg, hm=hm, hp=hp)
    # two-hand contact (wai, clapping, hands touching): keep the hands' RELATIVE position of the signer — each wrist placed
    # around the avatar's midpoint of the two targets by the signer's own wrist-to-wrist vector, blended by closeness
    rel = tg["left"]["Wg"] - tg["right"]["Wg"]
    dh = np.linalg.norm(rel, axis=1) / sw_s
    wc = np.clip(1.0 - dh / 0.7, 0.0, 1.0)[:, None] ** 1.5
    mid = (tg["left"]["W_t"] + tg["right"]["W_t"]) / 2
    rel_av = k * rel
    # hands that touch keep a minimum lateral gap (two hand thicknesses): the avatar's palms meet, they do not pass through
    lat = rel_av[:, 0]
    short = (wc[:, 0] > 0.3) & (np.abs(lat) < CONTACT_GAP)
    rel_av[short, 0] = np.where(lat[short] > 0, 1.0, -1.0) * CONTACT_GAP      # ties: the left hand on the avatar's left (−X)
    tg["left"]["W_t"] = (1 - wc) * tg["left"]["W_t"] + wc * (mid + rel_av / 2)
    tg["right"]["W_t"] = (1 - wc) * tg["right"]["W_t"] + wc * (mid - rel_av / 2)
    meta["contact_frames"] = int((wc[:, 0] > 0.3).sum())
    # palm-to-palm frames reconstructed by the cleaner (one merged detection, edge-on): the palms face each other — the across-palm
    # axis loses its lateral component, so each palm normal points at the other hand (contact constraint, retarget only there)
    cm = np.asarray(raw.get("contact", np.zeros(T, bool)), bool)
    w_cm = np.convolve(cm.astype(float), np.ones(5) / 5, "same") if cm.any() else np.zeros(T)
    meta["contact_reconstructed"] = int(cm.sum())
    for si, (side, sh, el, wr) in enumerate(SIDES_):
        S_av = fk[f"{side}UpperArm"][0]
        L1 = np.linalg.norm(rig.rest[f"{side}LowerArm"] - rig.rest[f"{side}UpperArm"])
        L2 = np.linalg.norm(rig.rest[f"{side}Hand"] - rig.rest[f"{side}LowerArm"])
        Sg, Eg = P[:, sh], P[:, el]
        W_t, Wg, hm, hp = tg[side]["W_t"], tg[side]["Wg"], tg[side]["hm"], tg[side]["hp"]
        # keep hands in front of the torso (contact signs touch the chest, they do not enter it)
        in_torso_y = (W_t[:, 1] > 1.12) & (W_t[:, 1] < 1.45) & (np.abs(W_t[:, 0]) < 0.12)
        W_t[:, 2] = np.where(in_torso_y, np.minimum(W_t[:, 2], AV_CHEST_FRONT_Z - 0.035), W_t[:, 2])
        # two-bone IK, elbow pole = the signer's elbow
        d = W_t - S_av
        dl = np.clip(np.linalg.norm(d, axis=1), abs(L1 - L2) + 1e-3, (L1 + L2) * 0.999)
        dn = _n(d)
        W_t = S_av + dn * dl[:, None]
        E_hint = S_av + k * (Eg - Sg)
        pole = E_hint - S_av
        pole = pole - (pole * dn).sum(1, keepdims=True) * dn
        bad = np.linalg.norm(pole, axis=1) < 1e-4
        pole[bad] = np.array([0, -1.0, 0.2])
        pole = _n(pole - (pole * dn).sum(1, keepdims=True) * dn)
        a = (L1 ** 2 + dl ** 2 - L2 ** 2) / (2 * dl)
        h = np.sqrt(np.maximum(L1 ** 2 - a ** 2, 0))
        E = S_av + dn * a[:, None] + pole * h[:, None]
        u = _n(E - S_av); f = _n(W_t - E)
        d0 = _n(rig.rest[f"{side}LowerArm"] - rig.rest[f"{side}UpperArm"])
        d0f = _n(rig.rest[f"{side}Hand"] - rig.rest[f"{side}LowerArm"])
        n0 = _n(np.cross(d0, np.array([0, 0, -1.0])))
        n = np.cross(u, f)
        straight = np.linalg.norm(n, axis=1) < 0.05
        n = np.where(straight[:, None], np.cross(u, pole), n)
        n = _n(n)
        up_w = align(u, n, d0, n0)
        lo_w = align(f, n, d0f, n0)
        chest_world = rig.fk(q.astype(np.float32), names=[f"{side}Shoulder"])[f"{side}Shoulder"][1]
        up_l = chest_world.inv() * up_w
        lo_l = up_w.inv() * lo_w
        # palm orientation from the hand landmarks
        p0 = _n(rig.rest[f"{side}MiddleProximal"] - rig.rest[f"{side}Hand"])
        s0 = _n(rig.rest[f"{side}IndexProximal"] - rig.rest[f"{side}LittleProximal"])
        if hp.any():
            # palm axes smoothed over tracked runs: an edge-on palm makes index→pinky unreliable for single frames and the
            # orientation flips (the source of most wrist pops in the first library build)
            pv, sv = hm[:, 9] - hm[:, 0], hm[:, 5] - hm[:, 17]
            if w_cm.any():
                sv = sv.copy()
                sv[:, 0] *= 1.0 - np.clip(w_cm, 0, 1)
            pv, sv = _smooth_runs(_n(pv), hp), _smooth_runs(_n(sv), hp)
            hand_w = align(pv, sv, p0, s0)
            rel = lo_w.inv() * hand_w
            sw_, tw = swing_twist(rel, d0f)
            lo_l = lo_l * frac(tw, 0.5)
            lo_w2 = up_w * lo_l
            hand_l = clamp_angle(lo_w2.inv() * hand_w, 95)
        else:
            lo_w2 = lo_w
            hand_l = R.identity(T)
            hand_w = lo_w2
        hq = hand_l.as_quat()
        hq[~hp] = np.nan                                   # filled below (hold / ease to neutral)
        q[:, BONE_INDEX[f"{side}UpperArm"]] = up_l.as_quat()
        q[:, BONE_INDEX[f"{side}LowerArm"]] = lo_l.as_quat()
        q[:, BONE_INDEX[f"{side}Hand"]] = hq
        # fingers, in the hand's frame
        fq = np.full((T, 15, 4), np.nan)
        if hp.any():
            hw_rot = up_w * lo_l * R.from_quat(np.nan_to_num(hq, nan=0.0) + np.where(np.isnan(hq), [0, 0, 0, 1.0], 0))
            Hl = np.stack([hw_rot.inv().apply(hm[:, j] - hm[:, 0]) for j in range(21)], 1)    # hand landmarks in hand-local frame
            for fi, fname in enumerate(FINGERS):
                lm = FINGER_LM[fname]
                bn = [f"{side}{fname}{g}" for g in SEGS] + [f"{side}{fname}Tip"]
                r = [_n(rig.rest[bn[i + 1]] - rig.rest[bn[i]]) for i in range(3)]
                v = [_n(Hl[:, lm[i + 1]] - Hl[:, lm[i]]) for i in range(3)]
                if fname == "Thumb":
                    r1 = clamp_angle(min_rot(r[0], v[0]), 75)
                    w2 = r1.inv().apply(v[1]); r2 = clamp_angle(min_rot(r[1], w2), 70)
                    w3 = (r1 * r2).inv().apply(v[2]); r3 = clamp_angle(min_rot(r[2], w3), 80)
                else:
                    hinge = _n(np.cross(r[0], np.array([0, -1.0, 0])))
                    r1 = min_rot(r[0], v[0])
                    sw1, _ = swing_twist(r1, r[0])                     # no axial twist of a finger
                    r1 = clamp_angle(sw1, 100)
                    w2 = r1.inv().apply(v[1])
                    a2 = flexion(r[1], w2, hinge, -10, 110)
                    r2 = R.from_rotvec(hinge * a2[:, None])
                    w3 = (r1 * r2).inv().apply(v[2])
                    a3 = flexion(r[2], w3, hinge, -10, 95)
                    r3 = R.from_rotvec(hinge * a3[:, None])
                for gi, rr in enumerate((r1, r2, r3)):
                    qq = rr.as_quat(); qq[~hp] = np.nan
                    fq[:, fi * 3 + gi] = qq
        for fi, fname in enumerate(FINGERS):
            for gi, g in enumerate(SEGS):
                q[:, BONE_INDEX[f"{side}{fname}{g}"]] = fq[:, fi * 3 + gi]
        hand_ok.append(hp)

    q = fill_and_smooth(q)
    meta["hand_tracked"] = [float(h.mean()) for h in hand_ok]
    morph = face_morphs(raw["bs"], rest)
    return dict(q=q.astype(np.float32), morph=morph.astype(np.float32), meta=meta)


def _smooth_runs(x, mask, win=7):
    """Savitzky-Golay over each run of tracked frames (a lost frame is not smoothed across)."""
    x = x.copy()
    i, T = 0, len(mask)
    while i < T:
        if not mask[i]:
            i += 1; continue
        j = i
        while j < T and mask[j]:
            j += 1
        if j - i >= win:
            x[i:j] = savgol_filter(x[i:j], win, 2, axis=0)
        i = j
    return x


def clamp_velocity(q, bones, max_deg=30.0):
    """limit the per-frame rotation of the given bones (forward and backward pass, averaged → no lag): real arm / wrist motion
    stays below ~750°/s; anything faster is a tracking flip."""
    lim = np.deg2rad(max_deg)

    def one(seq):
        out = seq.copy()
        for t in range(1, len(seq)):
            ra, rb = R.from_quat(out[t - 1]), R.from_quat(seq[t])
            d = (ra.inv() * rb).as_rotvec()
            a = np.linalg.norm(d, axis=-1, keepdims=True)
            d = np.where(a > lim, d / np.maximum(a, 1e-9) * lim, d)
            out[t] = (ra * R.from_rotvec(d)).as_quat()
        return out
    for b in bones:
        f = one(q[:, b])
        bw = one(q[::-1, b])[::-1]
        for t in range(len(f)):
            if (f[t] * bw[t]).sum() < 0:
                bw[t] = -bw[t]
        m = f + bw
        q[:, b] = m / np.linalg.norm(m, axis=-1, keepdims=True)
    return q


def fill_and_smooth(q, hold=12):
    """NaN frames (hand not tracked beyond the cleaner's short-gap fill): hold the last pose for ≤ hold frames, then ease to the
    relaxed (identity) pose; quaternion sign continuity; light Savitzky–Golay smoothing."""
    T, B, _ = q.shape
    ident = np.array([0, 0, 0, 1.0])
    for b in range(B):
        x = q[:, b]
        bad = np.isnan(x).any(1)
        if not bad.any():
            continue
        good = np.flatnonzero(~bad)
        if not len(good):
            q[:, b] = ident; continue
        for t in np.flatnonzero(bad):
            prev = good[good < t]; nxt = good[good > t]
            if len(prev) and len(nxt) and nxt[0] - prev[-1] <= 2 * hold:
                u = (t - prev[-1]) / (nxt[0] - prev[-1])
                ra, rb = R.from_quat(x[prev[-1]]), R.from_quat(x[nxt[0]])
                x[t] = (ra * R.from_rotvec((ra.inv() * rb).as_rotvec() * u)).as_quat()
            else:
                ref = x[prev[-1]] if len(prev) else x[nxt[0]]
                dist = (t - prev[-1]) if len(prev) else (nxt[0] - t)
                u = float(np.clip((dist - hold) / hold, 0, 1))
                ra = R.from_quat(ref)
                x[t] = (ra * R.from_rotvec((ra.inv() * R.from_quat(ident)).as_rotvec() * u)).as_quat()
        q[:, b] = x
    # sign continuity + de-spike + smoothing
    for t in range(1, T):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t, flip] *= -1
    q = despike(q)
    q = clamp_velocity(q, [BONE_INDEX[b] for b in BONES if not any(f in b for f in FINGERS)])
    if T >= 7:
        q = savgol_filter(q, 5, 2, axis=0)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def _ang(a, b):
    return 2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1))


def despike(q, thr_deg=20.0, passes=3):
    """a bone that jumps away for one or two frames while its neighbours agree is a tracking spike: replace it by the slerp of the
    neighbours (frames t-1 and t+1 / t+2)."""
    q = q.copy()
    thr = np.deg2rad(thr_deg)
    T = len(q)
    for _ in range(passes):
        changed = False
        for gap in (1, 2):
            if T < gap + 2:
                continue
            a, b = q[:-gap - 1], q[gap + 1:]                     # neighbours around frames t .. t+gap-1
            agree = _ang(a, b) < thr * 0.75
            for k in range(gap):
                mid = q[1 + k:T - gap + k]
                spike = agree & (_ang(a, mid) > thr) & (_ang(b, mid) > thr)
                if spike.any():
                    u = (k + 1) / (gap + 1)
                    ia, ib = np.nonzero(spike)
                    ra, rb = R.from_quat(a[ia, ib]), R.from_quat(b[ia, ib])
                    q[1 + k + ia, ib] = (ra * R.from_rotvec((ra.inv() * rb).as_rotvec() * u)).as_quat()
                    changed = True
        if not changed:
            break
    return q


def face_morphs(bs, rest):
    """52 MediaPipe blendshapes → VRoid face morphs (MORPHS), relative to the signer's own neutral face in rest frames."""
    bs = np.nan_to_num(bs.astype(np.float64))
    base = np.median(bs[rest], axis=0) if rest.any() else np.percentile(bs, 20, axis=0)
    d = np.clip(bs - base, 0, 1)
    g = lambda *names: np.mean([d[:, BS[n]] for n in names], axis=0)
    raw_abs = lambda n: bs[:, BS[n]]
    out = np.stack([
        np.clip(2.2 * g("browInnerUp", "browOuterUpLeft", "browOuterUpRight"), 0, 1),       # BRW_Surprised
        np.clip(2.5 * g("browDownLeft", "browDownRight"), 0, 1),                              # BRW_Angry (furrow)
        np.clip(1.5 * np.maximum(g("browInnerUp") - g("browOuterUpLeft", "browOuterUpRight"), 0), 0, 1),  # BRW_Sorrow
        np.clip(1.2 * raw_abs("eyeBlinkLeft") - 0.15, 0, 1),                                  # EYE_Close_L (absolute: blinks)
        np.clip(1.2 * raw_abs("eyeBlinkRight") - 0.15, 0, 1),
        np.clip(2.0 * g("eyeWideLeft", "eyeWideRight"), 0, 1),                               # EYE_Spread
        np.clip(1.8 * g("jawOpen"), 0, 1),                                                    # MTH_A
        np.clip(1.5 * g("mouthStretchLeft", "mouthStretchRight"), 0, 1),                      # MTH_I
        np.clip(1.6 * g("mouthPucker"), 0, 1),                                                # MTH_U
        np.clip(1.6 * g("mouthFunnel"), 0, 1),                                                # MTH_O
        np.clip(1.6 * g("mouthSmileLeft", "mouthSmileRight"), 0, 1),                          # MTH_Fun
        np.clip(1.6 * g("mouthFrownLeft", "mouthFrownRight"), 0, 1),                          # MTH_Angry
    ], 1)
    if len(out) >= 7:
        out = np.clip(savgol_filter(out, 5, 2, axis=0), 0, 1)
    return out
