"""Hand recovery layer for the production extractor (modules/mediapipe_pose.Holistic) — "method C" of lab4-handExtractor.ipynb.

Holistic stays the main source of pose / face / hands. When it loses a hand while the wrist is raised, a MediaPipe Hand Landmarker (the same
21-point model, IMAGE mode) is run on a ZOOMED, forearm-aligned crop around where the hand must be — the previous hand box (+ velocity) first,
then the pose wrist. A candidate is accepted only if it sits on the pose wrist (no background hands, no left/right swap). A failed recovery
is not retried for 2 frames, then less and less often (a hand that is not there costs almost nothing). Optionally the hands are smoothed with a One-Euro filter.

Why: Holistic finds a hand with a pose-guided ROI only; when the pose is off (fast motion, small / far hand, webcam) the hand is lost. Measured on
the 14 iPhone clips (lab4): missing-hand frames 3.5 % → 1.3 % (clean), 4.0 % → 1.8 % (browser drops 2 of 3 frames), PCK@0.1 +1.6 pts.

THAISLM_HAND_RECOVERY=0 turns it off (the extractor then behaves exactly as before); THAISLM_HAND_SMOOTH=1 turns the One-Euro filter on.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
SIDE = {"L": dict(wrist=15, elbow=13, pinky=17, index=19, thumb=21), "R": dict(wrist=16, elbow=14, pinky=18, index=20, thumb=22)}
HIP = {"L": 23, "R": 24}


def enabled():
    return os.environ.get("THAISLM_HAND_RECOVERY", "1").strip().lower() not in ("0", "false", "off", "no")


def smoothing():
    return os.environ.get("THAISLM_HAND_SMOOTH", "0").strip().lower() in ("1", "true", "on", "yes")


def hand_model_path(near=None):
    """env MP_HAND_MODEL → hand_landmarker.task next to the Holistic model → the MediaPipe cache (downloaded once if missing)"""
    if os.environ.get("MP_HAND_MODEL"):
        return Path(os.environ["MP_HAND_MODEL"])
    if near is not None and (Path(near).parent / "hand_landmarker.task").exists():
        return Path(near).parent / "hand_landmarker.task"
    p = Path.home() / ".cache" / "mediapipe_tasks" / "hand_landmarker.task"
    if not p.exists():
        import urllib.request
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, p)
    return p


# ------------------------------------------------------------------------------------------------------------ geometry (pixels)
def _px(lm, hw):
    H, W = hw
    return lm[..., :2] * np.array([W, H], np.float32)


def _size(p):
    return float(np.max(p.max(0) - p.min(0)))


def _pose_roi(pose, side, hw):
    """centre px, square side px of where the pose model thinks this hand is (None if the wrist is not seen)"""
    H, W = hw
    ix = SIDE[side]
    if pose[ix["wrist"], 3] < 0.3:
        return None
    P = pose[:, :2] * np.array([W, H], np.float32)
    w, e = P[ix["wrist"]], P[ix["elbow"]]
    tip_ix = [ix["pinky"], ix["index"], ix["thumb"]]
    tips = P[tip_ix].mean(0)
    c = 0.5 * (w + tips) if pose[tip_ix, 3].mean() > 0.3 else w
    fore = np.linalg.norm(w - e) if pose[ix["elbow"], 3] > 0.3 else 0.0
    sh = np.linalg.norm(P[11] - P[12]) if pose[11, 3] > 0.3 and pose[12, 3] > 0.3 else 0.0
    s = max(1.15 * fore, 0.55 * sh, 3.2 * np.linalg.norm(tips - w), 48.0)
    return c, float(min(s, 0.9 * max(H, W)))


def _forearm_angle(pose, side, hw):
    ix = SIDE[side]
    if pose[ix["wrist"], 3] < 0.3 or pose[ix["elbow"], 3] < 0.3:
        return 0.0
    H, W = hw
    d = (pose[ix["wrist"], :2] - pose[ix["elbow"], :2]) * np.array([W, H], np.float32)
    return float(math.degrees(math.atan2(d[0], -d[1])))


def _raised(pose, side, margin=0.05):
    hip = pose[HIP[side]]
    return bool(hip[3] < 0.3 or pose[SIDE[side]["wrist"], 1] < hip[1] + margin)


def _plausible(lm, pose, side, hw, tol=1.4):
    """the hand must sit on the pose wrist (≤ tol × forearm); False when the wrist is not seen (cannot be checked: the caller decides)"""
    H, W = hw
    ix = SIDE[side]
    if pose[ix["wrist"], 3] < 0.3:
        return False
    p = _px(lm, hw)
    w = pose[ix["wrist"], :2] * np.array([W, H], np.float32)
    fore = np.linalg.norm((pose[ix["wrist"], :2] - pose[ix["elbow"], :2]) * np.array([W, H], np.float32)) if pose[ix["elbow"], 3] > 0.3 else 0.15 * max(H, W)
    return bool(np.linalg.norm(p[0] - w) <= tol * max(fore, 30.0))


def _crop(rgb, c, s, out, ang):
    a = math.radians(ang)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32) * (s / out)
    t = np.asarray(c, np.float32) - R @ np.array([out / 2, out / 2], np.float32)
    M = np.concatenate([R, t[:, None]], 1)
    crop = cv2.warpAffine(rgb, M, (out, out), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    return crop, M


def _back(lm, M, out, hw):
    H, W = hw
    q = (lm[:, :2] * out) @ M[:, :2].T + M[:, 2]
    res = np.empty_like(lm)
    res[:, 0], res[:, 1] = q[:, 0] / W, q[:, 1] / H
    res[:, 2] = lm[:, 2] * out * float(np.linalg.norm(M[:, 0])) / W
    return res


class OneEuro:
    """One-Euro filter (Casiez 2012): low cut-off when the point is still, high when it moves fast"""

    def __init__(self, min_cutoff=1.0, beta=8.0, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _a(cut, dt):
        r = 2 * math.pi * cut * dt
        return r / (r + 1)

    def __call__(self, x, t):
        if self.x is None or t - self.t > 0.5:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.dx = self.dx + self._a(self.dc, dt) * ((x - self.x) / dt - self.dx)
        self.x = self.x + self._a(self.mc + self.beta * np.abs(self.dx), dt) * (x - self.x)
        self.t = t
        return self.x.copy()


class _HandSmoother:
    """One-Euro on pixel coordinates divided by a SLOW average of the hand size (a per-frame size would inject its own jitter)"""

    def __init__(self):
        self.f, self.size = OneEuro(), None

    def reset(self):
        self.f.x, self.size = None, None

    def __call__(self, hand, t, hw):
        H, W = hw
        sc = np.array([W, H, W], np.float32)
        sz = max(_size(hand[:, :2] * sc[:2]), 1.0)
        self.size = sz if self.size is None else 0.9 * self.size + 0.1 * sz
        return self.f(hand * sc / self.size, t) * self.size / sc


# ------------------------------------------------------------------------------------------------------------ the layer
class HandRecovery:
    """apply(rgb, pose, lh, rh, t_ms) → (lh, rh) with the hands Holistic lost recovered (and optionally smoothed). One instance per video."""

    def __init__(self, model_path=None, smooth=None, out=256, near=None):
        from mediapipe.tasks.python import BaseOptions, vision
        self.net = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path or hand_model_path(near))), running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4, min_tracking_confidence=0.4))
        self.out = out
        self.smooth = smoothing() if smooth is None else smooth
        self.prev = {"L": None, "R": None}                    # (landmarks, frame index) of the last accepted hand
        self.vel = {"L": np.zeros(2, np.float32), "R": np.zeros(2, np.float32)}
        self.cool = {"L": 0, "R": 0}
        self.fails = {"L": 0, "R": 0}
        self.filt = {"L": _HandSmoother(), "R": _HandSmoother()}
        self.i = 0
        self.stats = {"holistic": 0, "recovered": 0, "frames": 0}

    def close(self):
        self.net.close()

    def _detect(self, crop):
        import mediapipe as mp
        r = self.net.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
        return [(np.array([(p.x, p.y, p.z) for p in lm], np.float32), float(hd[0].score)) for lm, hd in zip(r.hand_landmarks, r.handedness)]

    def _size_ok(self, lm, pose, side, hw):
        """a recovered hand must have a believable size: within 0.5–2× the last hand of this side, and 0.33–1.5× the forearm (Holistic's own hands
        measured on 2,959 frames: 1st–96th percentile of hand size / forearm; a speck on the face or a lone finger is not a hand)"""
        H, W = hw
        size = _size(_px(lm, hw))
        pv = self.prev[side]
        if pv is not None and self.i - pv[1] <= 12:
            exp = _size(_px(pv[0], hw))
            if exp > 0 and not 0.5 * exp <= size <= 2.0 * exp:
                return False
        ix = SIDE[side]
        if pose[ix["wrist"], 3] >= 0.3 and pose[ix["elbow"], 3] >= 0.3:
            fore = float(np.linalg.norm((pose[ix["wrist"], :2] - pose[ix["elbow"], :2]) * np.array([W, H], np.float32)))
            if fore > 0 and not 0.33 * fore <= size <= 1.5 * fore:
                return False
        return True

    def _recover(self, rgb, pose, side, other):
        hw = rgb.shape[:2]
        if self.cool[side] > 0:
            self.cool[side] -= 1
            return None
        rois = []
        pv = self.prev[side]
        if pv is not None and self.i - pv[1] <= 12:                                    # 1) where the hand just was (+ velocity)
            p = _px(pv[0], hw)
            rois.append(((p.max(0) + p.min(0)) / 2 + self.vel[side] * (self.i - pv[1]), max(_size(p) * 2.2, 40.0), 0.0))
        pr = _pose_roi(pose, side, hw)
        if pr is not None and _raised(pose, side):                                      # 2) the pose wrist (a hanging hand is not worth the compute)
            rois.append((pr[0], pr[1], _forearm_angle(pose, side, hw)))
            if self.fails[side] % 3 == 2:
                rois.append((pr[0], pr[1] * 1.5, 0.0))
        for c, s, ang in rois:
            crop, M = _crop(rgb, c, s, self.out, ang)
            cont = pv is not None and self.i - pv[1] <= 2                                  # the hand was there a moment ago → a continuation
            for lm, score in self._detect(crop):
                if score < (0.6 if cont else 0.75):                                        # a hand that re-appears from nowhere must be convincing:
                    continue                                                               # on the wrist and sure of itself (lone frames flipped borderline words)
                lm = _back(lm, M, self.out, hw)
                if not _plausible(lm, pose, side, hw, 1.4 if cont else 0.7) and not (cont and pose[SIDE[side]["wrist"], 3] < 0.3):
                    continue                                                               # (only a continuation may be accepted without the body to check it)
                if not self._size_ok(lm, pose, side, hw):
                    continue
                if other is not None and np.linalg.norm(_px(lm, hw)[0] - _px(other, hw)[0]) < 0.25 * _size(_px(other, hw)):
                    continue                                                           # that is the other hand
                self.fails[side] = 0
                return lm
        self.fails[side] += 1
        self.cool[side] = 0 if not rois else 2 if self.fails[side] < 3 else min(2 * self.fails[side], 10)   # a hand that is really absent is tried less and less often
        return None

    def apply(self, rgb, pose, lh, rh, t_ms=0.0):
        hw = rgb.shape[:2]
        res = {"L": lh, "R": rh}
        for s in "LR":
            if res[s].any():
                self.stats["holistic"] += 1
                continue
            other = res["R" if s == "L" else "L"]
            lm = self._recover(rgb, pose, s, other if other.any() else None)
            if lm is not None:
                res[s] = lm
                self.stats["recovered"] += 1
        for s in "LR":
            if not res[s].any():
                self.filt[s].reset()
                continue
            if self.prev[s] is not None:
                d = _px(res[s], hw)[0] - _px(self.prev[s][0], hw)[0]
                self.vel[s] = 0.6 * self.vel[s] + 0.4 * d / max(self.i - self.prev[s][1], 1)
            self.prev[s] = (res[s].copy(), self.i)                                       # the RAW hand is what the next ROI follows
            if self.smooth:
                res[s] = self.filt[s](res[s], t_ms / 1000.0, hw).astype(np.float32)
        self.i += 1
        self.stats["frames"] += 1
        return res["L"], res["R"]
