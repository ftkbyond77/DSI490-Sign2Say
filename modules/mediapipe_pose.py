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


class Holistic:
    """One MediaPipe HolisticLandmarker in VIDEO mode (tracking across frames). Create one per video (timestamps restart)."""

    def __init__(self, model_path=None, min_pose=0.5, min_hand=0.5):
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path or MODEL_PATH))),
            running_mode=vision.RunningMode.VIDEO, output_face_blendshapes=True,
            min_pose_detection_confidence=min_pose, min_pose_landmarks_confidence=min_pose, min_hand_landmarks_confidence=min_hand)
        self.lm = vision.HolisticLandmarker.create_from_options(opts)

    def close(self):
        self.lm.close()

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


def extract(video, fps=25.0, crop=None, max_side=1280, model_path=None, progress=None, normalize=False):
    """Video file → raw MediaPipe arrays (module docstring). crop = (x0, y0, x1, y1) in pixels (static signer box) or None.
    Frames are sampled at ≤ `fps` on their own timestamps (a 50 fps source is not processed twice as often).
    progress(frac) is called about every 25 processed frames (web progress bar). normalize = ExposureNorm (inference only)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(src_fps) or src_fps <= 1 or src_fps > 240:
        src_fps = 25.0
    hol = Holistic(model_path)
    norm = ExposureNorm() if normalize else None
    T, P, L, R, F, B = [], [], [], [], [], []
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
            T.append(t); P.append(pose); L.append(lh); R.append(rh); F.append(face); B.append(bs)
            if progress and n_src and len(T) % 25 == 0:
                progress(min(i / n_src, 1.0))
    finally:
        cap.release(); hol.close()
    if not T:
        return None
    return dict(t=np.array(T, np.float64), hw=np.array(hw if hw else (0, 0), np.int32), pose=np.stack(P), lh=np.stack(L), rh=np.stack(R),
                face=np.stack(F), bs=np.stack(B), src_fps=np.float32(src_fps))


def from_browser(frames, hw):
    """Landmarks sent by the web client (MediaPipe Tasks Vision JS HolisticLandmarker, same models) → the raw dict above.
    frames: list of {t (ms), pose [[x,y,z,vis]×33] | [], lh [[x,y,z]×21] | [], rh […], face [[x,y]×68] | [], bs [52] | []}."""
    n = len(frames)
    pose = np.zeros((n, 33, 4), np.float32); lh = np.zeros((n, 21, 3), np.float32); rh = np.zeros((n, 21, 3), np.float32)
    face = np.zeros((n, 68, 2), np.float32); bs = np.zeros((n, 52), np.float32); t = np.zeros(n, np.float64)
    for k, f in enumerate(frames):
        t[k] = float(f.get("t", 0.0)) / 1000.0
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
    return dict(t=t, hw=np.array(hw, np.int32), pose=pose, lh=lh, rh=rh, face=face, bs=bs, src_fps=np.float32(n / max(t[-1], 1e-3) if n > 1 else 25))


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
