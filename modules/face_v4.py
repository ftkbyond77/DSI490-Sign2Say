"""v4 Face model (separate from the Sign model; fused later by the language layer).

  emotion   ViT FER teacher (trpakov/vit-face-expression) on a tight face crop from the 68 RTMW face landmarks
  NMM       from landmarks, relative to the signer's own baseline in the same video:
              brow raise (yes/no question cue) · brow furrow (wh-question / negative affect) · mouth open
              head shake (negation) · head nod (affirmation) · head tilt/forward
"""
from __future__ import annotations

import numpy as np

FACE0 = 23          # COCO-WholeBody face landmarks 23..90 (68 points)
JAW = slice(FACE0 + 0, FACE0 + 17)
R_BROW, L_BROW = slice(FACE0 + 17, FACE0 + 22), slice(FACE0 + 22, FACE0 + 27)
NOSE_TIP = FACE0 + 30
R_EYE, L_EYE = slice(FACE0 + 36, FACE0 + 42), slice(FACE0 + 42, FACE0 + 48)
MOUTH_TOP, MOUTH_BOT = FACE0 + 62, FACE0 + 66
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMO_TH = {"angry": "โกรธ", "disgust": "รังเกียจ", "fear": "กลัว", "happy": "ดีใจ/มีความสุข", "neutral": "เป็นกลาง",
          "sad": "เศร้า", "surprise": "ประหลาดใจ"}


def face_box(kp_px, sc, pad=1.45, thr=0.3):
    f = kp_px[FACE0:FACE0 + 68]; ok = sc[FACE0:FACE0 + 68] > thr
    if ok.sum() < 20:
        return None
    x0, y0 = f[ok].min(0); x1, y1 = f[ok].max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.08 * (y1 - y0)
    s = max(x1 - x0, y1 - y0) * pad
    return cx, cy, s


def landmark_signals(kp, sc, hw, thr=0.3):
    H, W = hw
    p = kp * np.array([W, H], np.float32)
    eye_r, eye_l = p[:, R_EYE].mean(1), p[:, L_EYE].mean(1)
    iod = np.linalg.norm(eye_r - eye_l, axis=1) + 1e-3                      # inter-ocular distance
    brow = (eye_r[:, 1] - p[:, R_BROW, 1].mean(1) + eye_l[:, 1] - p[:, L_BROW, 1].mean(1)) / 2 / iod
    brow_gap = np.linalg.norm(p[:, FACE0 + 21] - p[:, FACE0 + 22], axis=1) / iod   # inner brow distance (furrow ↓)
    mouth = np.linalg.norm(p[:, MOUTH_TOP] - p[:, MOUTH_BOT], axis=1) / iod
    sw = np.linalg.norm(p[:, 5] - p[:, 6], axis=1) + 1e-3
    mid = (p[:, 5] + p[:, 6]) / 2
    nose = (p[:, NOSE_TIP] - mid) / sw[:, None]
    ok = sc[:, FACE0:FACE0 + 68].mean(1) > thr
    return dict(brow=brow, brow_gap=brow_gap, mouth=mouth, nose_x=nose[:, 0], nose_y=nose[:, 1], ok=ok)


def _z(x, ok):
    base = np.median(x[ok]) if ok.any() else 0.0
    mad = np.median(np.abs(x[ok] - base)) * 1.4826 + 1e-3 if ok.any() else 1.0
    return (x - base) / mad


def _cycles(sig, amp):
    s = sig - np.convolve(sig, np.ones(7) / 7, "same")
    big = np.abs(s) > amp
    sgn = np.sign(s[big])
    return int((np.diff(sgn) != 0).sum() // 2) if len(sgn) > 1 else 0


def nmm(kp, sc, hw, fps=12.5, spans=None):
    """Whole-utterance and per-span non-manual cues."""
    s = landmark_signals(kp, sc, hw)
    ok = s["ok"]
    zb, zg, zm = _z(s["brow"], ok), _z(s["brow_gap"], ok), _z(s["mouth"], ok)
    T = len(kp)
    tail = slice(int(T * 0.6), T)
    out = dict(
        brow_raise_tail=float(np.median(zb[tail])) if T else 0.0,
        brow_furrow=float(-np.percentile(zg, 10)),
        mouth_open_frac=float((zm > 2.0).mean()),
        head_shake_cycles=_cycles(s["nose_x"][ok], 0.03) if ok.sum() > fps else 0,
        head_nod_cycles=_cycles(s["nose_y"][ok], 0.025) if ok.sum() > fps else 0,
        head_forward_tail=float(np.median(s["nose_y"][tail]) - np.median(s["nose_y"][ok])) if ok.any() else 0.0,
    )
    out["question_yesno"] = bool(out["brow_raise_tail"] > 1.5)
    out["negation"] = bool(out["head_shake_cycles"] >= 2)
    out["affirmation"] = bool(out["head_nod_cycles"] >= 2 and not out["negation"])
    if spans:
        out["per_span"] = [dict(t0=round(a / fps, 2), t1=round(b / fps, 2), brow=float(np.median(zb[a:b])),
                                mouth=float(np.median(zm[a:b]))) for a, b in spans]
    return out


class FaceModel:
    def __init__(self, device="cuda"):
        from .heads import AffectTeacher
        self.teacher = AffectTeacher(device=device)

    def emotions(self, frames, kp, sc, every=2, size=224):
        import cv2
        H, W = frames.shape[1:3]
        crops, idx = [], []
        for i in range(0, len(frames), every):
            b = face_box(kp[i] * np.array([W, H], np.float32), sc[i])
            if b is None:
                continue
            cx, cy, s = b
            M = np.array([[size / s, 0, size / 2 - size / s * cx], [0, size / s, size / 2 - size / s * cy]], np.float32)
            crops.append(cv2.warpAffine(frames[i], M, (size, size), borderMode=cv2.BORDER_REPLICATE)); idx.append(i)
        if not crops:
            return dict(emotion="neutral", probs={}, per_frame=None)
        # the teacher expects the face crop centred inside a larger crop; give it the tight crop directly
        crops = np.stack(crops)
        pad = int(size * 0.22)
        crops = np.pad(crops, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
        probs = self.teacher(crops)
        mean = probs.mean(0)
        k = int(mean.argmax())
        neutral = mean[EMOTIONS.index("neutral")]
        non_neutral = [(EMOTIONS[j], float(mean[j])) for j in np.argsort(-mean) if EMOTIONS[j] != "neutral"]
        return dict(emotion=EMOTIONS[k], emotion_th=EMO_TH[EMOTIONS[k]], probs={e: round(float(v), 3) for e, v in zip(EMOTIONS, mean)},
                    strongest_non_neutral=dict(emotion=non_neutral[0][0], emotion_th=EMO_TH[non_neutral[0][0]], p=round(non_neutral[0][1], 3)),
                    intensity=round(float(max(0.0, 1 - neutral)), 3), frames=idx, per_frame=probs)
