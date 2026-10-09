"""The production pose extractor: MediaPipe Holistic (Tasks API) — one extractor for training data, server inference and the browser.

Why MediaPipe (v7): the web demo / real-time product extracts poses in the user's browser (MediaPipe Tasks Vision JS,
HolisticLandmarker = the same TFLite models as this Python module), so the server receives landmarks, not video. Training on
another extractor (RTMW-X in v6) left an "extractor gap" between the bank and real queries. v7 re-extracts every video source with
this module, so training clips, the bank and every query are produced by the same model family.

Raw output per video (`extract()` / `from_browser()`), one row per processed frame:
  t     [T]        seconds (media time)
  hw    (H, W)     frame size the coordinates are normalised by (the crop, if one was used)
  pose  [T,33,4]   x/W, y/H, z, visibility (MediaPipe pose landmarks; 0 = no person)
  lh/rh [T,21,3]   x/W, y/H, z of the signer's LEFT / RIGHT hand (MediaPipe naming = the person's own side); all 0 = not detected
  face  [T,68,2]   x/W, y/H of 68 iBUG-equivalent points taken from the 478-point face mesh (NMM features, head pose)
  bs    [T,52]     face blendshape scores (browInnerUp, browDownLeft, jawOpen, mouthSmileLeft, …) — the face-expression channel
`to_standard()` turns it into the Standard Schema (modules/schema.py) exactly like the TSL-ONE-S MediaPipe release (`mp75`):
pose 33 + hands 21+21 → 133 slots, face = template placed on nose / eyes / mouth corners, hand gaps ≤ 0.2 s filled, 25 fps.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
MODEL_PATH = Path(os.environ.get("MP_HOLISTIC_MODEL", Path.home() / ".cache" / "mediapipe_tasks" / "holistic_landmarker.task"))

# 478-point face mesh → 68 iBUG points (jaw 17 · brows 5+5 · nose 9 · eyes 6+6 · outer lip 12 · inner lip 8); image-left first
MESH68 = [127, 234, 93, 132, 58, 172, 136, 150, 152, 379, 365, 397, 288, 361, 323, 454, 356,
          70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
          168, 197, 5, 4, 75, 97, 2, 326, 305,
          33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
          61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
          78, 82, 13, 312, 308, 317, 14, 87]
BLENDSHAPES = ["_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
               "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
               "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
               "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
               "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
               "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
               "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
               "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"]


def ensure_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


HAND_MODEL_PATH = Path(os.environ.get("MP_HAND_MODEL", Path(__file__).resolve().parents[1] / "models" / "mediapipe" / "hand_landmarker.task"))


def wrist_raised(pose, i):
    """A pose wrist that is visible, inside the frame and at signing height (same rule as webapp/web/lib/holistic.ts wristRaised)."""
    ls, rs, w = pose[11], pose[12], pose[i]
    sw = float(np.hypot(*(ls[:2] - rs[:2]))) or 0.2
    mid_y = (ls[1] + rs[1]) / 2
    return bool(w[3] > 0.5 and 0 < w[0] < 1 and 0 < w[1] < 1 and w[1] < mid_y + 1.3 * sw)


def _same_hand(a, b, sw):
    """Two 21-point detections of the same physical hand: their points coincide (mean distance < 0.15 shoulder width). Touching hands
    (ชื่อ, สวัสดี, ขอบคุณ) have wrists close together but different fingers, so the wrist alone cannot tell them apart."""
    return float(np.linalg.norm(a[:, :2] - b[:, :2], axis=1).mean()) < 0.15 * sw


def fill_hands(pose, lh, rh, detected):
    """Hand fallback (same rule as the browser's fillHands): hands from a separate HandLandmarker fill the sides Holistic missed.
    v8.1: a detection is the hand Holistic already has only if its points coincide with it (not merely its wrist — that dropped the
    second hand of every two-hand contact sign); the rest go to the missing side's pose wrist if within ~1 shoulder width, the nearer
    wrist winning when both sides are missing. Returns lh, rh, (filled_left, filled_right)."""
    sw = float(np.hypot(*(pose[11, :2] - pose[12, :2]))) or 0.2
    out, fb = [lh, rh], [False, False]
    have = [np.abs(h).sum() > 0 for h in out]
    free = [d for d in detected if not any(have[k] and _same_hand(d, out[k], sw) for k in (0, 1))]
    free = [d for i, d in enumerate(free) if not any(_same_hand(d, e, sw) for e in free[:i])]        # one detection per hand
    pairs = []
    for k, wi in ((0, 15), (1, 16)):
        if have[k]:
            continue
        for j, d in enumerate(free):
            dw = float(np.hypot(*(d[0, :2] - pose[wi, :2])))
            if dw < 1.0 * sw:
                pairs.append((dw, k, j))
    used = set()
    for dw, k, j in sorted(pairs):
        if fb[k] or j in used:
            continue
        out[k] = free[j].astype(np.float32); fb[k] = True; used.add(j)
    return out[0], out[1], tuple(fb)


def fill_hands_v8(pose, lh, rh, detected):
    """The v8 rule (kept for scripts/hand_fill_test.py): v8 hand fallback (same rule as the browser's fillHands): hands from a separate HandLandmarker fill the sides Holistic missed;
    each goes to the nearer pose wrist, if within ~1 shoulder width. Returns lh, rh, (filled_left, filled_right)."""
    sw = float(np.hypot(*(pose[11, :2] - pose[12, :2]))) or 0.2
    used, fb = set(), [False, False]
    out = [lh, rh]
    for k, (wi, oi) in enumerate(((15, 16), (16, 15))):
        h = out[k]
        if np.abs(h).sum() > 0:
            for j, d in enumerate(detected):
                if np.hypot(*(d[0, :2] - h[0, :2])) < 0.5 * sw:
                    used.add(j)
            continue
        best, bd = -1, 1e9
        for j, d in enumerate(detected):
            if j in used:
                continue
            dw = np.hypot(*(d[0, :2] - pose[wi, :2])); do = np.hypot(*(d[0, :2] - pose[oi, :2]))
            if dw < do and dw < 1.0 * sw and dw < bd:
                best, bd = j, dw
        if best >= 0:
            out[k] = detected[best].astype(np.float32); used.add(best); fb[k] = True
    return out[0], out[1], tuple(fb)


def hand_report(raw, fb=None):
    """Per-utterance hand tracking (the v8 extractor metric): for each side, the share of frames with a raised, visible wrist whose 21 hand
    points were tracked, how many came from the fallback, the longest run of lost frames; plus the capture fps. v8.1: a hand whose wrist
    is at the bottom edge of the image is outside the picture, not lost — counted in out_below, not in hit; room_below = signing space
    below the shoulders in shoulder widths (training videos 1.2–1.8)."""
    pose, T = raw["pose"], len(raw["t"])
    rep = {}
    for name, key, wi, k in (("left", "lh", 15, 0), ("right", "rh", 16, 1)):
        n = got = nfb = run = mx = below = 0
        for i in range(T):
            if not wrist_raised(pose[i], wi):
                run = 0; continue
            if np.abs(raw[key][i]).sum() > 0:
                n += 1; got += 1; run = 0
                nfb += int(bool(fb is not None and fb[i][k]))
            elif pose[i, wi, 1] > 0.88:                  # v8.1: hand outside the picture at the bottom (resting, or a tight webcam frame):
                below += 1                               # not a tracking miss — reported apart, with room_below for the framing advice
            else:
                n += 1; run += 1; mx = max(mx, run)
        rep[name] = dict(n=n, hit=round(got / n, 3) if n else None, fallback=nfb, max_gap=mx, out_below=below)
    ok = (pose[:, 11, 3] > 0.5) & (pose[:, 12, 3] > 0.5)
    if ok.any():                                         # signing space below the shoulders, in shoulder widths (training videos 1.2–1.8)
        H, W = raw.get("hw", (1, 1))
        sw = np.hypot((pose[ok, 11, 0] - pose[ok, 12, 0]) * W / H, pose[ok, 11, 1] - pose[ok, 12, 1])
        rep["room_below"] = round(float(np.median((1 - (pose[ok, 11, 1] + pose[ok, 12, 1]) / 2) / np.maximum(sw, 1e-3))), 2)
    t = raw["t"]
    rep["fps"] = round(float((T - 1) / max(t[-1] - t[0], 1e-3)), 1) if T > 1 else 0.0
    return rep


class Holistic:
    """One MediaPipe HolisticLandmarker in VIDEO mode (tracking across frames). Create one per video (timestamps restart).
    hand_fallback (v8, inference): a HandLandmarker looks for hands Holistic missed while their wrist is raised (fill_hands)."""

    def __init__(self, model_path=None, min_pose=0.5, min_hand=0.5, hand_fallback=False):
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path or MODEL_PATH))),
            running_mode=vision.RunningMode.VIDEO, output_face_blendshapes=True,
            min_pose_detection_confidence=min_pose, min_pose_landmarks_confidence=min_pose, min_hand_landmarks_confidence=min_hand)
        self.lm = vision.HolisticLandmarker.create_from_options(opts)
        self.hands = None
        if hand_fallback and HAND_MODEL_PATH.exists():
            self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)), running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
        self.last_fb = (False, False)

    def close(self):
        self.lm.close()
        if self.hands is not None:
            self.hands.close()

    def frame(self, rgb, t_ms):
        import mediapipe as mp
        r = self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), int(t_ms))
        pose = np.zeros((33, 4), np.float32); lh = np.zeros((21, 3), np.float32); rh = np.zeros((21, 3), np.float32)
        face = np.zeros((68, 2), np.float32); bs = np.zeros(52, np.float32)
        if r.pose_landmarks:
            pose[:] = [(p.x, p.y, p.z, p.visibility if p.visibility is not None else 1.0) for p in r.pose_landmarks]
        if r.left_hand_landmarks:
            lh[:] = [(p.x, p.y, p.z) for p in r.left_hand_landmarks]
        if r.right_hand_landmarks:
            rh[:] = [(p.x, p.y, p.z) for p in r.right_hand_landmarks]
        if r.face_landmarks:
            fl = r.face_landmarks
            face[:] = [(fl[i].x, fl[i].y) for i in MESH68]
        if r.face_blendshapes:
            bs[:] = [c.score for c in r.face_blendshapes[:52]]
        self.last_fb = (False, False)
        if self.hands is not None and r.pose_landmarks and ((not r.left_hand_landmarks and wrist_raised(pose, 15)) or
                                                            (not r.right_hand_landmarks and wrist_raised(pose, 16))):
            h = self.hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), int(t_ms))
            det = [np.array([(p.x, p.y, p.z) for p in hl], np.float32) for hl in h.hand_landmarks]
            if det:
                lh, rh, self.last_fb = fill_hands(pose, lh, rh, det)
        return pose, lh, rh, face, bs


def exposure_lut(mean, spread):
    """(gamma, contrast, target) for a frame with luminance `mean` / `spread` (0..1) → a standard exposure: gamma brightens or
    darkens without clipping blacks/whites, then contrast is stretched around the image's own (gamma-corrected) mean, which is moved
    at most 0.2 toward 0.5. Same rule as webapp/web/lib/holistic.ts (Normalizer)."""
    m = float(np.clip(mean, 0.03, 0.97))
    g = float(np.clip(np.log(0.5) / np.log(m), 0.35, 2.0))
    mg = m ** g
    sg = max(spread * g * m ** (g - 1.0), 1e-3)
    c = float(np.clip(0.22 / sg, 0.85, 1.8))
    tgt = mg + float(np.clip(0.5 - mg, -0.2, 0.2))
    return g, c, mg, tgt


class ExposureNorm:
    """Inference-time image normalisation (same rule as the web client): dark rooms, back light and washed-out webcams reach
    MediaPipe looking alike. Measured on a thumbnail every 8 frames and smoothed. Training data was extracted without it
    (landmarks are positions only)."""

    def __init__(self):
        self.g, self.c, self.n, self.lut = 1.0, 1.0, 0, None

    def __call__(self, rgb):
        import cv2
        if self.n % 8 == 0:
            y = cv2.cvtColor(cv2.resize(rgb, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            g, c, mg, tgt = exposure_lut(float(y.mean()), float(y.std()))
            k = 0.35 if self.n else 1.0
            self.g = float(np.exp(np.log(self.g) + k * (np.log(g) - np.log(self.g)))); self.c += k * (c - self.c)
            if abs(self.g - 1) < 0.08 and abs(self.c - 1) < 0.06:
                self.lut = None
            else:
                x = (np.arange(256, dtype=np.float32) / 255.0) ** self.g
                self.lut = np.clip(((x - mg) * self.c + tgt) * 255.0, 0, 255).astype(np.uint8)
        self.n += 1
        return rgb if self.lut is None else cv2.LUT(rgb, self.lut)


def extract(video, fps=25.0, crop=None, max_side=1280, model_path=None, progress=None, normalize=False, hand_fallback=False):
    """Video file → raw MediaPipe arrays (module docstring). crop = (x0, y0, x1, y1) in pixels (static signer box) or None.
    Frames are sampled at ≤ `fps` on their own timestamps (a 50 fps source is not processed twice as often).
    progress(frac) is called about every 25 processed frames (web progress bar). normalize = ExposureNorm (inference only)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(src_fps) or src_fps <= 1 or src_fps > 240:
        src_fps = 25.0
    hol = Holistic(model_path, hand_fallback=hand_fallback)
    norm = ExposureNorm() if normalize else None
    T, P, L, R, F, B, FB = [], [], [], [], [], [], []
    i, next_t, hw = 0, 0.0, None
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            t = i / src_fps; i += 1
            if t + 1e-6 < next_t:
                continue
            next_t = t + 1.0 / fps - 0.25 / src_fps
            if crop is not None:
                x0, y0, x1, y1 = (int(round(v)) for v in crop)
                fr = fr[max(0, y0):y1, max(0, x0):x1]
            h, w = fr.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            hw = fr.shape[:2]
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            pose, lh, rh, face, bs = hol.frame(norm(rgb) if norm else rgb, t * 1000.0)
            T.append(t); P.append(pose); L.append(lh); R.append(rh); F.append(face); B.append(bs); FB.append(hol.last_fb)
            if progress and n_src and len(T) % 25 == 0:
                progress(min(i / n_src, 1.0))
    finally:
        cap.release(); hol.close()
    if not T:
        return None
    return dict(t=np.array(T, np.float64), hw=np.array(hw if hw else (0, 0), np.int32), pose=np.stack(P), lh=np.stack(L), rh=np.stack(R),
                face=np.stack(F), bs=np.stack(B), src_fps=np.float32(src_fps), fb=np.array(FB, bool))


def from_browser(frames, hw):
    """Landmarks sent by the web client (MediaPipe Tasks Vision JS HolisticLandmarker, same models) → the raw dict above.
    frames: list of {t (ms), pose [[x,y,z,vis]×33] | [], lh [[x,y,z]×21] | [], rh […], face [[x,y]×68] | [], bs [52] | []}."""
    n = len(frames)
    pose = np.zeros((n, 33, 4), np.float32); lh = np.zeros((n, 21, 3), np.float32); rh = np.zeros((n, 21, 3), np.float32)
    face = np.zeros((n, 68, 2), np.float32); bs = np.zeros((n, 52), np.float32); t = np.zeros(n, np.float64)
    fb = np.zeros((n, 2), bool)
    for k, f in enumerate(frames):
        t[k] = float(f.get("t", 0.0)) / 1000.0
        if f.get("fb"):
            fb[k] = [bool(x) for x in list(f["fb"])[:2]]
        for arr, key, d in ((pose, "pose", 4), (lh, "lh", 3), (rh, "rh", 3), (face, "face", 2)):
            v = f.get(key) or []
            if len(v):
                a = np.asarray(v, np.float32)
                if a.ndim == 2 and a.shape[0] == arr.shape[1]:
                    arr[k, :, :min(d, a.shape[1])] = a[:, :d]
        if f.get("bs"):
            b = np.asarray(f["bs"], np.float32)[:52]; bs[k, :len(b)] = b
    pose[pose[..., 3] == 0, 3] = np.where(np.abs(pose[pose[..., 3] == 0, :2]).sum(-1) > 0, 1.0, 0.0)
    t = t - t[0] if n else t
    return dict(t=t, hw=np.array(hw, np.int32), pose=pose, lh=lh, rh=rh, face=face, bs=bs, src_fps=np.float32(n / max(t[-1], 1e-3) if n > 1 else 25),
                fb=fb)


def to_mp75(raw, vis_thr=0.5):
    """raw → the TSL-ONE-S 75-point layout [T,75,2] (pose 33 + left hand 21 + right hand 21; zeros = missing)."""
    T = len(raw["t"])
    a = np.zeros((T, 75, 2), np.float32)
    vis = raw["pose"][:, :, 3] > vis_thr
    a[:, :33] = np.where(vis[..., None], raw["pose"][:, :, :2], 0.0)
    for j0, hand in ((33, raw["lh"]), (54, raw["rh"])):
        ok = np.abs(hand[:, :, :2]).sum((1, 2)) > 0
        a[ok, j0:j0 + 21] = hand[ok, :, :2]
    a[a == 0] = 0.0
    return a


def to_standard(raw, src="", gap_s=0.2):
    """raw → Standard Schema dict (kp, sc, fps 25, hw, extractor 'mp'); same harmonisation as the other MediaPipe sources.
    gap_s: hand drop-outs up to this long are interpolated (0.2 s for training data; live inference uses spotting.json "hand_gap_s")."""
    from .schema import from_mp75
    a = to_mp75(raw)
    a[np.abs(a).sum(-1) == 0] = 0.0
    hw = tuple(int(x) for x in raw["hw"])
    d = from_mp75(a, hw=hw, fps=25.0, src=src, t=raw["t"], gap_s=gap_s)
    d["extractor"] = "mp"
    return d


def handedness_check(raw, n=200):
    """Fraction of frames where the 'left' hand root is closer to the pose's left wrist than to the right one (should be ≈ 1)."""
    ok = (np.abs(raw["lh"][:, 0, :2]).sum(1) > 0) & (raw["pose"][:, 15, 3] > 0.5) & (raw["pose"][:, 16, 3] > 0.5)
    if not ok.any():
        return None
    lw, rw, root = raw["pose"][ok, 15, :2], raw["pose"][ok, 16, :2], raw["lh"][ok, 0, :2]
    return float((np.linalg.norm(root - lw, axis=1) < np.linalg.norm(root - rw, axis=1)).mean())


def save_raw(path, raw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=raw["t"].astype(np.float32), hw=raw["hw"], pose=raw["pose"].astype(np.float16), lh=raw["lh"].astype(np.float16),
                        rh=raw["rh"].astype(np.float16), face=raw["face"].astype(np.float16), bs=raw["bs"].astype(np.float16), src_fps=raw["src_fps"])


def load_raw(path):
    z = np.load(path)
    return dict(t=z["t"].astype(np.float64), hw=z["hw"], pose=z["pose"].astype(np.float32), lh=z["lh"].astype(np.float32),
                rh=z["rh"].astype(np.float32), face=z["face"].astype(np.float32), bs=z["bs"].astype(np.float32), src_fps=z["src_fps"])
