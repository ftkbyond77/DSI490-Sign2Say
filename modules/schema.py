"""L0: Standard Schema (introduced in v5 as CS5) — one pose format for every source, whatever extracted it.

Sources and their native topologies
  rtmw      RTMW-X whole-body 133 (COCO-WholeBody), score per point, 12.5 fps (v4 corpus) or any fps (inference)
  mp75      TSL-ONE-S .npy: MediaPipe Holistic pose 33 + left hand 21 + right hand 21, (x, y) only, 29.97 fps, zeros = missing.
            NOTE: the release does not say which frame the coordinates were normalised to. Assuming the paper's 1280×720 gives
            shoulder-width / mouth-drop ≈ 4.6 (anatomically ≈ 2.4 in every source with a known frame, incl. TSL51 MediaPipe
            experts at 1920×1080 / 600×600 / ~1180×950) → hands ~1.9× too wide. The aspect is therefore estimated per clip
            (shrunk 50 % toward the signer median) with `mp75_aspect()`.
  mp54      TSL51 .csv: MediaPipe Holistic shoulders/elbows/wrists 6 + brows 4 + mouth corners 2 + hands 21+21, (x, y, z),
            one row per video frame (fps from metadata: 30 primary, 50 th-sl, 25–50 Royal Society), NaN = missing.
            NOTE: the csv column `t_ms` is the extraction loop's wall-clock, not media time (a 13.0 s / 391-frame video
            spans 15.9 s of t_ms; expert clips up to 3×) — it is ignored.

CS5 = the 133-slot RTMW layout that the Uni-Sign encoder reads (body 9 · left hand 21 · right hand 21 · face 18), stored as
  kp [T,133,2] (x/W, y/H) · sc [T,133] ∈ {0} ∪ (0, 1] · fps 25 · hw (H, W) · src · extractor · imputed (names)
Only USED slots are filled.

Harmonisation decisions (each one removes a way the model could tell the datasets apart instead of the signs):
  * handedness     all sources verified non-mirrored with left = signer's left (hand wrist ↔ pose wrist distance), so hands map 1:1;
                   MediaPipe hand-21 and COCO-WholeBody hand-21 share the same joint order.
  * face           Uni-Sign's face part uses jaw 9 + inner-mouth 8 + nose tip. MediaPipe sources do not have them, so for EVERY
                   source (RTMW included) the 18 face points are a mean face template placed by a per-frame similarity fitted to
                   the anchors the source does have (nose / eyes / mouth corners / brows). Real mouth corners are kept.
  * missing body   TSL51 has no nose / eyes / ears → the same template fit (brows + mouth corners) imputes them (flagged).
  * confidence     RTMW scores clipped to [0, 1]; MediaPipe detections → 1.0; missing → 0.
  * hand dropouts  MediaPipe loses hands for a few frames where RTMW returns low-score points → gaps ≤ 0.2 s are interpolated
                   for all sources alike.
  * time           everything is resampled on its own timestamps to 25 fps (the rate Uni-Sign was trained on).
  * geometry       stored normalised by (W, H); `iso()` converts to isotropic units (x·W/H) so non-16:9 videos (TSL51 4:3,
                   portrait phone clips) keep true hand shapes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .parts import BODY_IDX, FACE_IDX
from .utils import MODEL

FPS = 25.0
LH = list(range(91, 112)); RH = list(range(112, 133))
USED = sorted(set(BODY_IDX) | set(LH) | set(RH) | set(FACE_IDX))
NOSE, EYE_L, EYE_R, EAR_L, EAR_R, SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
MOUTH_R, MOUTH_L = 23 + 48, 23 + 54                 # iBUG 48 = signer's right mouth corner (smaller x), 54 = left
BROW_R, BROW_L = list(range(23 + 17, 23 + 22)), list(range(23 + 22, 23 + 27))
# anchor vector used by the face template: nose, eyeL, eyeR, earL, earR, mouthL, mouthR, browL-centre, browR-centre
ANCHORS = ["nose", "eye_l", "eye_r", "ear_l", "ear_r", "mouth_l", "mouth_r", "brow_l", "brow_r"]
TEMPLATE_PATH = MODEL / "face_template.npz"
FIT_ANCHORS = {"rtmw": ["nose", "eye_l", "eye_r", "mouth_l", "mouth_r"], "mp75": ["nose", "eye_l", "eye_r", "mouth_l", "mouth_r"],
               "mp54": ["mouth_l", "mouth_r", "brow_l", "brow_r"]}


# ----------------------------------------------------------------------------- geometry helpers
def iso(kp, hw):
    """(x/W, y/H) → isotropic units of frame height."""
    H, W = hw
    return kp * np.array([W / H, 1.0], np.float32)


def uniso(kp, hw):
    H, W = hw
    return kp / np.array([W / H, 1.0], np.float32)


def _rtmw_anchors(kp, sc):
    """kp iso [T,133,2] → anchors [T,9,2], ok [T,9]."""
    A = np.stack([kp[:, NOSE], kp[:, EYE_L], kp[:, EYE_R], kp[:, EAR_L], kp[:, EAR_R], kp[:, MOUTH_L], kp[:, MOUTH_R],
                  kp[:, BROW_L].mean(1), kp[:, BROW_R].mean(1)], 1)
    ok = np.stack([sc[:, NOSE], sc[:, EYE_L], sc[:, EYE_R], sc[:, EAR_L], sc[:, EAR_R], sc[:, MOUTH_L], sc[:, MOUTH_R],
                   sc[:, BROW_L].min(1), sc[:, BROW_R].min(1)], 1) > 0.5
    return A, ok


def umeyama2d(src, dst):
    """Similarity (s, R, t) minimising ‖s·R·src + t − dst‖ (src, dst [N,2], N ≥ 2)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    D = np.diag([1.0, d])
    R = U @ D @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / max(var, 1e-9))
    t = mu_d - s * R @ mu_s
    return s, R, t


# ----------------------------------------------------------------------------- face template (fitted once on TTRS RTMW)
def fit_face_template(pose_files, max_frames=40000, seed=0):
    """Generalised Procrustes mean of [9 anchors + 18 Uni-Sign face points] over frontal, confident RTMW frames."""
    rng = np.random.RandomState(seed)
    shapes = []
    for f in pose_files:
        d = np.load(f)
        kp = iso(d["kp"].astype(np.float32), tuple(int(x) for x in d["hw"])); sc = d["sc"].astype(np.float32)
        A, ok = _rtmw_anchors(kp, sc)
        good = ok.all(1) & (sc[:, FACE_IDX].min(1) > 0.6)
        for t in np.flatnonzero(good)[::3]:
            shapes.append(np.concatenate([A[t], kp[t, FACE_IDX]], 0))
    X = np.stack(shapes).astype(np.float64)
    if len(X) > max_frames:
        X = X[rng.choice(len(X), max_frames, replace=False)]
    ref = X[0] - X[0].mean(0)
    for _ in range(5):
        aligned = []
        for x in X:
            s, R, t = umeyama2d(x, ref)
            aligned.append((s * (R @ x.T)).T + t)
        mean = np.mean(aligned, 0)
        mean = (mean - mean[0])                          # nose at origin
        mean /= np.linalg.norm(mean[5] - mean[6])        # unit mouth width
        ref = mean
    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TEMPLATE_PATH, anchors=ref[:9].astype(np.float32), face=ref[9:].astype(np.float32), n_frames=len(X))
    return ref


_TEMPLATE = None


def template():
    global _TEMPLATE
    if _TEMPLATE is None:
        d = np.load(TEMPLATE_PATH)
        _TEMPLATE = (d["anchors"].astype(np.float64), d["face"].astype(np.float64))
    return _TEMPLATE


def harmonise_face(kp_iso, sc, anchors, ok, names, fill_body=()):
    """Place template face (and optionally missing nose/eyes/ears) with a temporally smoothed per-frame similarity.
    kp_iso [T,133,2] (modified in place), anchors [T,9,2], ok [T,9] bool."""
    tA, tF = template()
    use = [ANCHORS.index(n) for n in names]
    T = len(kp_iso)
    params = np.full((T, 4), np.nan)                     # s, angle, tx, ty
    for t in range(T):
        u = [j for j in use if ok[t, j]]
        if len(u) < 2:
            continue
        s, R, tr = umeyama2d(tA[u], anchors[t, u])
        params[t] = [s, np.arctan2(R[1, 0], R[0, 0]), tr[0], tr[1]]
    good = ~np.isnan(params[:, 0])
    if good.sum() == 0:
        return np.zeros(T, bool)
    idx = np.arange(T)
    for k in range(4):                                   # fill gaps + 5-frame median smoothing of the pose of the face
        params[:, k] = np.interp(idx, idx[good], params[good, k])
        pad = np.pad(params[:, k], 2, mode="edge")
        params[:, k] = np.median(np.stack([pad[i:i + T] for i in range(5)]), 0)
    s, ang, tx, ty = params.T
    c, si = np.cos(ang), np.sin(ang)
    def place(P):  # [N,2] template points → [T,N,2]
        x = s[:, None] * (c[:, None] * P[None, :, 0] - si[:, None] * P[None, :, 1]) + tx[:, None]
        y = s[:, None] * (si[:, None] * P[None, :, 0] + c[:, None] * P[None, :, 1]) + ty[:, None]
        return np.stack([x, y], -1)
    F = place(tF)
    kp_iso[:, FACE_IDX] = F
    g = idx[good]                                         # distance to the nearest frame with a fitted face — O(T log T), not T×T
    j = np.clip(np.searchsorted(g, idx), 1, len(g) - 1) if len(g) > 1 else np.zeros(T, int)
    near = np.minimum(np.abs(idx - g[j - 1]), np.abs(idx - g[j])) if len(g) > 1 else np.abs(idx - g[0])
    live = good | (near <= 12)
    sc[:, FACE_IDX] = np.where(live[:, None], 1.0, 0.0)
    # real mouth corners are closest to template inner-mouth corners (face 60 → slot 83, face 64 → slot 87)
    for slot, a in ((83, ANCHORS.index("mouth_r")), (87, ANCHORS.index("mouth_l"))):
        m = ok[:, a]
        kp_iso[m, slot] = anchors[m, a]
    Aall = place(tA)
    for name in fill_body:
        j = ANCHORS.index(name); slot = {"nose": NOSE, "eye_l": EYE_L, "eye_r": EYE_R, "ear_l": EAR_L, "ear_r": EAR_R}[name]
        kp_iso[:, slot] = Aall[:, j]; sc[:, slot] = np.where(live, 1.0, 0.0)
    return live


# ----------------------------------------------------------------------------- temporal
def fill_hand_gaps(kp, sc, max_gap):
    """Linear interpolation of whole-hand dropouts no longer than max_gap frames (both sides valid)."""
    for hand in (LH, RH):
        ok = sc[:, hand].mean(1) > 0.3
        T = len(ok); t = 0
        while t < T:
            if ok[t]:
                t += 1; continue
            s = t
            while t < T and not ok[t]:
                t += 1
            if s > 0 and t < T and t - s <= max_gap:
                w = (np.arange(s, t) - (s - 1)) / (t - (s - 1))
                kp[s:t, hand] = kp[s - 1, hand][None] * (1 - w[:, None, None]) + kp[t, hand][None] * w[:, None, None]
                sc[s:t, hand] = np.minimum(sc[s - 1, hand], sc[t, hand])[None]
    return kp, sc


def resample(kp, sc, t, fps=FPS):
    """Irregular timestamps t [T] (s) → uniform grid. Points invalid at either neighbour stay invalid."""
    t = np.asarray(t, np.float64)
    order = np.argsort(t); t, kp, sc = t[order], kp[order], sc[order]
    keep = np.r_[True, np.diff(t) > 1e-6]
    t, kp, sc = t[keep], kp[keep], sc[keep]
    if len(t) < 2:
        return kp, sc
    grid = np.arange(t[0], t[-1] + 1e-6, 1.0 / fps)
    j = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    w = ((grid - t[j]) / np.maximum(t[j + 1] - t[j], 1e-6)).clip(0, 1)
    k = kp[j] * (1 - w)[:, None, None] + kp[j + 1] * w[:, None, None]
    s = np.minimum(sc[j], sc[j + 1])
    near = np.where((w < 0.5)[:, None], sc[j], sc[j + 1])            # single-sided valid point: keep the nearest sample if it is close
    s = np.where((s <= 0) & (near > 0) & (np.minimum(w, 1 - w)[:, None] < 0.25), near, s)
    return k.astype(np.float32), s.astype(np.float32)


def finish(kp_iso, sc, t, hw, src, extractor, imputed, gap_s=0.2):
    kp_iso, sc = fill_hand_gaps(kp_iso, sc, max_gap=max(1, int(round(gap_s / max(np.median(np.diff(t)) if len(t) > 1 else 0.04, 1e-3)))))
    kp_iso, sc = resample(kp_iso, sc, t)
    mask = np.zeros(133, bool); mask[USED] = True
    kp = uniso(kp_iso, hw); kp[:, ~mask] = 0; sc[:, ~mask] = 0
    src_fps = float(1.0 / np.median(np.diff(t))) if len(t) > 1 else FPS
    return dict(kp=kp.astype(np.float32), sc=np.clip(sc, 0, 1).astype(np.float32), fps=np.float32(FPS), hw=np.array(hw),
                src=src, extractor=extractor, src_fps=np.float32(src_fps), imputed=np.array(sorted(imputed)))


# ----------------------------------------------------------------------------- converters
def from_rtmw(kp, sc, hw, fps=12.5, t0=0.0, src="", real_face=False):
    kp_iso = iso(kp.astype(np.float32), hw).copy(); sc = sc.astype(np.float32).copy()
    T = len(kp_iso)
    A, ok = _rtmw_anchors(kp_iso, sc)
    sc = np.clip(sc, 0, 1)
    imputed = set()
    if not real_face:
        harmonise_face(kp_iso, sc, A, ok, FIT_ANCHORS["rtmw"]); imputed.add("face_template")
    return finish(kp_iso, sc, t0 + np.arange(T) / fps, hw, src, "rtmw", imputed)


MP_REF_RATIO = 2.40   # median shoulder-width / (shoulder-mid → mouth-mid drop) of MediaPipe clips with a known frame size


def mp75_ratio(arr):
    a = np.asarray(arr, np.float32).reshape(len(arr), 75, 2)
    sl, sr, ml, mr = a[:, 11], a[:, 12], a[:, 9], a[:, 10]
    sw = np.abs(sl[:, 0] - sr[:, 0]); dv = np.abs((sl[:, 1] + sr[:, 1]) / 2 - (ml[:, 1] + mr[:, 1]) / 2)
    ok = (dv > 1e-3) & (sw > 1e-3)
    return float(np.median(sw[ok] / dv[ok])) if ok.sum() >= 3 else np.nan


def mp75_aspect(clip_ratio, signer_ratio, w=0.5):
    """Frame aspect W/H that makes this clip's shoulder/mouth geometry anatomical (log-space shrinkage to the signer)."""
    r = signer_ratio if not np.isfinite(clip_ratio) else float(np.exp(w * np.log(clip_ratio) + (1 - w) * np.log(signer_ratio)))
    return float(np.clip(MP_REF_RATIO / r, 0.45, 1.9))


MP75_BODY = {NOSE: 0, EYE_L: 2, EYE_R: 5, EAR_L: 7, EAR_R: 8, SH_L: 11, SH_R: 12, EL_L: 13, EL_R: 14, WR_L: 15, WR_R: 16}


def from_mp75(arr, hw=(720, 1280), fps=29.97, src="", t=None, gap_s=0.2):
    """MediaPipe pose 33 + hands 21+21 (x/W, y/H; zeros = missing) → Standard Schema. t = per-frame timestamps (s) when the frame
    rate is irregular (webcam / browser, frame-skipping extraction); default = a fixed `fps`."""
    a = np.asarray(arr, np.float32).reshape(len(arr), 75, 2)
    T = len(a)
    kp = np.zeros((T, 133, 2), np.float32); sc = np.zeros((T, 133), np.float32)
    present = (np.abs(a).sum(-1) > 0)
    for slot, j in MP75_BODY.items():
        kp[:, slot] = a[:, j]; sc[:, slot] = present[:, j]
    kp[:, MOUTH_L] = a[:, 9]; sc[:, MOUTH_L] = present[:, 9]
    kp[:, MOUTH_R] = a[:, 10]; sc[:, MOUTH_R] = present[:, 10]
    for slots, j0 in ((LH, 33), (RH, 54)):
        kp[:, slots] = a[:, j0:j0 + 21]
        sc[:, slots] = present[:, j0:j0 + 21].all(1, keepdims=True) & present[:, j0:j0 + 21]
    kp_iso = iso(kp, hw)
    A = np.stack([kp_iso[:, NOSE], kp_iso[:, EYE_L], kp_iso[:, EYE_R], kp_iso[:, EAR_L], kp_iso[:, EAR_R], kp_iso[:, MOUTH_L], kp_iso[:, MOUTH_R],
                  np.zeros((T, 2)), np.zeros((T, 2))], 1)
    ok = np.stack([sc[:, NOSE], sc[:, EYE_L], sc[:, EYE_R], sc[:, EAR_L], sc[:, EAR_R], sc[:, MOUTH_L], sc[:, MOUTH_R], np.zeros(T), np.zeros(T)], 1) > 0
    harmonise_face(kp_iso, sc, A, ok, FIT_ANCHORS["mp75"])
    return finish(kp_iso, sc, np.arange(T) / fps if t is None else np.asarray(t, np.float64), hw, src, "mp75", {"face_template"}, gap_s=gap_s)


def from_mp54(df, hw, fps, src=""):
    """TSL51 landmark CSV (pandas DataFrame); time axis = row index / metadata fps (t_ms is processing time)."""
    T = len(df)
    kp = np.zeros((T, 133, 2), np.float32); sc = np.zeros((T, 133), np.float32)

    def xy(name):
        x, y = df[f"{name}_x"].to_numpy(np.float32), df[f"{name}_y"].to_numpy(np.float32)
        ok = ~(np.isnan(x) | np.isnan(y))
        if f"{name}_vis" in df:
            ok &= np.nan_to_num(df[f"{name}_vis"].to_numpy(np.float32), nan=0.0) > 0.3
        return np.nan_to_num(np.stack([x, y], 1)), ok

    for slot, name in ((SH_L, "l_shoulder"), (SH_R, "r_shoulder"), (EL_L, "l_elbow"), (EL_R, "r_elbow"), (WR_L, "l_wrist"), (WR_R, "r_wrist"),
                       (MOUTH_L, "mouth_left"), (MOUTH_R, "mouth_right")):
        kp[:, slot], sc[:, slot] = xy(name)
    for slots, pre in ((LH, "lh"), (RH, "rh")):
        for i, slot in enumerate(slots):
            kp[:, slot], sc[:, slot] = xy_(df, pre, i)
    kp_iso = iso(kp, hw)
    # CSV brow names are image-side: "lbrow_*" (mesh 105/70) is on the image left = signer's RIGHT brow
    br, okr = _mean_pts(df, ["lbrow_outer", "lbrow_inner"]); bl, okl = _mean_pts(df, ["rbrow_inner", "rbrow_outer"])
    br, bl = iso(br, hw), iso(bl, hw)
    A = np.stack([np.zeros((T, 2)), np.zeros((T, 2)), np.zeros((T, 2)), np.zeros((T, 2)), np.zeros((T, 2)),
                  kp_iso[:, MOUTH_L], kp_iso[:, MOUTH_R], bl, br], 1)
    ok = np.stack([np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), sc[:, MOUTH_L], sc[:, MOUTH_R], okl, okr], 1) > 0
    harmonise_face(kp_iso, sc, A, ok, FIT_ANCHORS["mp54"], fill_body=("nose", "eye_l", "eye_r", "ear_l", "ear_r"))
    t = (df["frame"].to_numpy(np.float64) if "frame" in df else np.arange(T)) / float(fps)
    return finish(kp_iso, sc, t, hw, src, "mp54", {"face_template", "nose", "eyes", "ears"})


def xy_(df, pre, i):
    x, y = df[f"{pre}_x{i}"].to_numpy(np.float32), df[f"{pre}_y{i}"].to_numpy(np.float32)
    ok = ~(np.isnan(x) | np.isnan(y))
    return np.nan_to_num(np.stack([x, y], 1)), ok


def _mean_pts(df, names):
    xs = np.stack([df[f"{n}_x"].to_numpy(np.float32) for n in names], 1); ys = np.stack([df[f"{n}_y"].to_numpy(np.float32) for n in names], 1)
    ok = ~(np.isnan(xs).any(1) | np.isnan(ys).any(1))
    return np.nan_to_num(np.stack([xs.mean(1), ys.mean(1)], 1)), ok


# ----------------------------------------------------------------------------- activity (signing vs rest) on CS5
def activity(kp, sc, hw, thr=0.3):
    """Per-frame hand activity in shoulder widths: raise of each wrist above its rest level + wrist speed.
    Works on every source (needs shoulders, wrists / hand roots only)."""
    p = iso(kp, hw)
    T = len(p)
    ok_s = (sc[:, SH_L] > thr) & (sc[:, SH_R] > thr)
    sw = np.linalg.norm(p[:, SH_L] - p[:, SH_R], axis=1)
    swm = float(np.median(sw[ok_s])) if ok_s.any() else 0.25
    mid_y = np.where(ok_s, (p[:, SH_L, 1] + p[:, SH_R, 1]) / 2, np.nan)
    mid_y = np.interp(np.arange(T), np.flatnonzero(ok_s), mid_y[ok_s]) if ok_s.any() else np.full(T, 0.4)
    out = {}
    for name, hand, wr in (("L", LH, WR_L), ("R", RH, WR_R)):
        hok = sc[:, hand].mean(1) > thr
        w = np.where(hok[:, None], p[:, hand].mean(1), p[:, wr])
        wok = hok | (sc[:, wr] > thr)
        y = (mid_y - w[:, 1]) / swm                      # up = positive, 0 at shoulder line
        y = np.where(wok, y, -3.0)                       # undetected hand ≈ lowered / out of frame
        v = np.r_[0.0, np.linalg.norm(np.diff(w, axis=0), axis=1)] / swm * FPS
        v = np.where(wok & np.r_[False, wok[:-1]], v, 0.0)
        out[name] = (y, v)
    height = np.maximum(out["L"][0], out["R"][0])
    speed = np.maximum(out["L"][1], out["R"][1])
    return dict(height=height, speed=speed, sw=swm, hands_seen=((sc[:, LH].mean(1) > thr) | (sc[:, RH].mean(1) > thr)),
                body_seen=ok_s)


def sign_span(kp, sc, hw, rest_h=None, pad=2):
    """Frames where a hand is raised clearly above that clip's lowest (rest) level, or moving fast."""
    a = activity(kp, sc, hw)
    h = a["height"]
    base = np.percentile(h[h > -2.5], 10) if (h > -2.5).any() else -2.0
    rest_h = base + 0.45 if rest_h is None else rest_h
    act = (h > max(rest_h, -1.6)) | ((a["speed"] > 1.2) & (h > base + 0.2))
    if act.sum() < 3:
        return None
    i0, i1 = np.flatnonzero(act)[[0, -1]]
    return max(0, int(i0) - pad), min(len(kp), int(i1) + 1 + pad)


def save(path: Path, d: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, kp=d["kp"].astype(np.float16), sc=d["sc"].astype(np.float16), fps=d["fps"], hw=d["hw"], src=d["src"],
                        extractor=d["extractor"], src_fps=d["src_fps"], imputed=d["imputed"])


def load(path):
    z = np.load(path, allow_pickle=False)
    return dict(kp=z["kp"].astype(np.float32), sc=z["sc"].astype(np.float32), hw=tuple(int(x) for x in z["hw"]), fps=float(z["fps"]),
                extractor=str(z["extractor"]), src=str(z["src"]))


# ----------------------------------------------------------------------------- handedness canonicalisation (v7)
_PAIRS = [(EYE_L, EYE_R), (EAR_L, EAR_R), (SH_L, SH_R), (EL_L, EL_R), (WR_L, WR_R), (11, 12), (13, 14), (15, 16)]
_PAIRS += [(23 + i, 23 + 16 - i) for i in range(8)] + [(23 + 17 + i, 23 + 26 - i) for i in range(5)]           # jaw, brows
_PAIRS += [(23 + 36 + i, 23 + 45 - i) for i in range(4)] + [(23 + 40, 23 + 47), (23 + 41, 23 + 46)]              # eyes
_PAIRS += [(23 + 31, 23 + 35), (23 + 32, 23 + 34), (23 + 48, 23 + 54), (23 + 49, 23 + 53), (23 + 50, 23 + 52), (23 + 55, 23 + 59),
           (23 + 56, 23 + 58), (23 + 60, 23 + 64), (23 + 61, 23 + 63), (23 + 65, 23 + 67)]                       # nose, mouth
_PAIRS += [(LH[i], RH[i]) for i in range(21)]
MIRROR_IDX = np.arange(133)
for _a, _b in _PAIRS:
    MIRROR_IDX[_a], MIRROR_IDX[_b] = _b, _a


def mirror(kp, sc):
    """Left-right mirror of a Standard-Schema sequence (x/W → 1 − x/W, left/right slots swapped)."""
    k = kp.copy()
    live = sc > 0
    k[..., 0] = np.where(live, 1.0 - k[..., 0], k[..., 0])
    return k[:, MIRROR_IDX], sc[:, MIRROR_IDX]


def dominant_hand(kp, sc, hw, thr=0.3):
    """'R' or 'L' + its share of the signing activity. Activity of a hand = Σ over frames where that HAND is detected of
    (wrist-centre speed in shoulder widths/s, clipped) + 2 × (raise above the hand's own rest height, clipped). A hand that is
    never detected (resting below the frame) contributes nothing — MediaPipe drops resting hands, RTMW reports them with jitter."""
    p = iso(kp, hw)
    ok_s = (sc[:, SH_L] > thr) & (sc[:, SH_R] > thr)
    if ok_s.sum() < 2:
        return "R", 0.5
    sw = max(float(np.median(np.linalg.norm(p[ok_s, SH_L] - p[ok_s, SH_R], axis=1))), 1e-3)
    mid_y = float(np.median((p[ok_s, SH_L, 1] + p[ok_s, SH_R, 1]) / 2))
    e = {}
    for name, hand in (("L", LH), ("R", RH)):
        hok = sc[:, hand].mean(1) > thr
        if hok.sum() < 2:
            e[name] = 0.0; continue
        w = p[:, hand].mean(1)
        y = (mid_y - w[:, 1]) / sw
        rest = min(np.percentile(y[hok], 10), -1.0)
        v = np.r_[0.0, np.linalg.norm(np.diff(w, axis=0), axis=1)] / sw * FPS
        moving = hok & np.r_[False, hok[:-1]]
        e[name] = float(np.sum(np.clip(v[moving], 0, 15)) + 2.0 * np.sum(np.clip(y[hok] - rest, 0, 3)))
    tot = e["L"] + e["R"]
    if tot <= 1e-6:
        return "R", 0.5
    return ("L", e["L"] / tot) if e["L"] > e["R"] else ("R", e["R"] / tot)


def canonical_hands(kp, sc, hw, margin=0.62):
    """Mirror a left-dominant signer into a right-dominant one (training clips, bank and every query go through this), so the
    model never has to learn both handednesses of a sign. Returns kp, sc, mirrored."""
    side, share = dominant_hand(kp, sc, hw)
    if side == "L" and share >= margin:
        k, s = mirror(kp, sc)
        return k, s, True
    return kp, sc, False
