"""Face emotion model (v9, models/face-model/): what the signer FEELS while signing — learned from labelled faces, explainable,
and fused with what the hands say.

Why a learned model: v7's face channel (modules/face.py) scores emotions with hand-written blendshape formulas; it only knows
"how far from a neutral dictionary face". Here a classifier is trained on RAVDESS (24 actors, 8 acted emotions × 2 intensities,
video) through the SAME extractor as the sign pipeline (MediaPipe Holistic → 52 blendshapes + 68 face points, browser and server
alike), and evaluated leave-actors-out (people the model never saw).

    fm = FaceEmotion.load("models/face-model")
    r = fm.analyse(raw, spans)        # raw = MediaPipe raw dict (bs [T,52], face [T,68,2], t, hw); spans = sign segments (25 fps)
    r → {emotion, emotion_th, probs{7}, valence, arousal, intensity, confidence, explanation_th, evidence[...], per_segment[...]}

Features of a stretch of face (a whole utterance, or one sign): per blendshape the mean, 90th percentile, rise above the stretch's
own 10th percentile (expression relative to this person's resting face) and spread; head-motion amplitude (yaw / pitch / roll).
Model: standardise → multinomial logistic regression (one weight per feature per emotion), so every decision decomposes into
per-feature contributions w·x → the explanation lists the strongest ones in Thai ("มุมปากยกขึ้น (ยิ้ม)", "คิ้วขมวด" …).

Fusion with the sign (`fuse_affect`): the face is the main evidence; words that carry emotion (รัก, ดีใจ, เสียใจ, โกรธ, ร้องไห้ …)
add a prior only when the hands were read with confidence; signing energy (hand speed / size vs the signer's own median) moves
arousal. Grammatical face signals (yes/no-question brows, negation head shake — modules/face.py) are reported apart and keep a
question's raised brows from being read as surprise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .mediapipe_pose import BLENDSHAPES

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
EMO_TH = {"neutral": "เป็นกลาง", "happy": "ดีใจ / มีความสุข", "sad": "เศร้า / เสียใจ", "angry": "โกรธ / ไม่พอใจ", "fear": "กลัว / กังวล",
          "disgust": "รังเกียจ / ไม่ชอบ", "surprise": "ประหลาดใจ / ตกใจ"}
# circumplex coordinates (valence, arousal) of each emotion: probabilities → a continuous affect point
VA = {"neutral": (0.0, 0.0), "happy": (0.8, 0.5), "sad": (-0.7, -0.4), "angry": (-0.6, 0.8), "fear": (-0.6, 0.7), "disgust": (-0.7, 0.3),
      "surprise": (0.2, 0.8)}
RAVDESS_MAP = {"neutral": "neutral", "calm": "neutral", "happy": "happy", "sad": "sad", "angry": "angry", "fear": "fear",
               "disgust": "disgust", "surprise": "surprise"}
BS = [b for b in BLENDSHAPES if b != "_neutral"]
BS_TH = {
    "browDownLeft": "คิ้วซ้ายขมวด/ลดต่ำ", "browDownRight": "คิ้วขวาขมวด/ลดต่ำ", "browInnerUp": "หัวคิ้วยกขึ้น", "browOuterUpLeft": "หางคิ้วซ้ายยก",
    "browOuterUpRight": "หางคิ้วขวายก", "cheekPuff": "แก้มป่อง", "cheekSquintLeft": "แก้มซ้ายยก", "cheekSquintRight": "แก้มขวายก",
    "eyeBlinkLeft": "หลับตาซ้าย", "eyeBlinkRight": "หลับตาขวา", "eyeLookDownLeft": "มองลง", "eyeLookDownRight": "มองลง",
    "eyeLookInLeft": "เหลือบตา", "eyeLookInRight": "เหลือบตา", "eyeLookOutLeft": "เหลือบตา", "eyeLookOutRight": "เหลือบตา",
    "eyeLookUpLeft": "มองขึ้น", "eyeLookUpRight": "มองขึ้น", "eyeSquintLeft": "หรี่ตาซ้าย", "eyeSquintRight": "หรี่ตาขวา",
    "eyeWideLeft": "ตาซ้ายเบิกกว้าง", "eyeWideRight": "ตาขวาเบิกกว้าง", "jawForward": "ยื่นคาง", "jawLeft": "ขยับคาง", "jawOpen": "อ้าปาก",
    "jawRight": "ขยับคาง", "mouthClose": "ปิดปาก", "mouthDimpleLeft": "มุมปากบุ๋ม", "mouthDimpleRight": "มุมปากบุ๋ม",
    "mouthFrownLeft": "มุมปากซ้ายตก", "mouthFrownRight": "มุมปากขวาตก", "mouthFunnel": "ห่อปาก", "mouthLeft": "เบ้ปาก",
    "mouthLowerDownLeft": "ริมฝีปากล่างลง", "mouthLowerDownRight": "ริมฝีปากล่างลง", "mouthPressLeft": "เม้มปาก", "mouthPressRight": "เม้มปาก",
    "mouthPucker": "จู๋ปาก", "mouthRight": "เบ้ปาก", "mouthRollLower": "เม้มริมฝีปากล่าง", "mouthRollUpper": "เม้มริมฝีปากบน",
    "mouthShrugLower": "ยู่ริมฝีปากล่าง", "mouthShrugUpper": "ยกริมฝีปากบน", "mouthSmileLeft": "มุมปากซ้ายยก (ยิ้ม)",
    "mouthSmileRight": "มุมปากขวายก (ยิ้ม)", "mouthStretchLeft": "ยืดมุมปาก", "mouthStretchRight": "ยืดมุมปาก",
    "mouthUpperUpLeft": "ยกริมฝีปากบน", "mouthUpperUpRight": "ยกริมฝีปากบน", "noseSneerLeft": "ย่นจมูก", "noseSneerRight": "ย่นจมูก",
}
STATS = ["mean", "p90", "rise", "std"]
HEAD = ["yaw_amp", "pitch_amp", "roll_amp", "pitch_mean"]
FEATURES = [f"{b}:{s}" for s in STATS for b in BS] + HEAD
# emotion carried by a recognised word (the sign's meaning), used only as a prior next to the face
EMO_LEX = {"รัก": "happy", "ดีใจ": "happy", "มีความสุข": "happy", "สนุก": "happy", "หัวเราะ": "happy", "ยิ้ม": "happy", "ขอบคุณ": "happy",
           "เสียใจ": "sad", "เศร้า": "sad", "ร้องไห้": "sad", "ผิดหวัง": "sad", "สงสาร": "sad", "เหนื่อย": "sad", "คิดถึง": "sad",
           "โกรธ": "angry", "โมโห": "angry", "หงุดหงิด": "angry", "เกลียด": "angry", "กลัว": "fear", "กังวล": "fear", "เครียด": "fear",
           "ตกใจ": "surprise", "ประหลาดใจ": "surprise", "แปลกใจ": "surprise", "รังเกียจ": "disgust", "ขยะแขยง": "disgust"}


def _head(face, ok):
    """Head yaw / pitch / roll proxies from the 68 face points (eyes, nose tip) → amplitudes over the stretch + mean pitch."""
    if ok.sum() < 3:
        return np.zeros(len(HEAD), np.float32)
    f = face[ok].astype(np.float32)
    er, el = f[:, 36:42].mean(1), f[:, 42:48].mean(1)
    iod = np.linalg.norm(el - er, axis=1) + 1e-4
    mid = (er + el) / 2
    yaw = (f[:, 30, 0] - mid[:, 0]) / iod; pitch = (f[:, 30, 1] - mid[:, 1]) / iod
    roll = np.arctan2(el[:, 1] - er[:, 1], el[:, 0] - er[:, 0])
    amp = lambda x: float(np.percentile(x, 90) - np.percentile(x, 10))     # noqa: E731
    return np.array([amp(yaw), amp(pitch), amp(roll), float(np.mean(pitch))], np.float32)


def features(bs, face, ok=None):
    """bs [T,52] (MediaPipe order, incl. _neutral), face [T,68,2] (x already scaled to y units) → [len(FEATURES)]."""
    bs = np.asarray(bs, np.float32)[:, 1:]
    if ok is None:
        ok = bs.sum(1) > 0
    if ok.sum() < 3:
        return np.zeros(len(FEATURES), np.float32)
    x = bs[ok]
    p10, p90 = np.percentile(x, 10, axis=0), np.percentile(x, 90, axis=0)
    return np.concatenate([x.mean(0), p90, p90 - p10, x.std(0), _head(np.asarray(face), ok)]).astype(np.float32)


class FaceEmotion:
    def __init__(self, mu, sd, W, b, meta=None):
        self.mu, self.sd, self.W, self.b = mu, sd, W, b          # W [F, 7]
        self.meta = meta or {}

    @classmethod
    def load(cls, model_dir):
        z = np.load(Path(model_dir) / "face_emotion.npz", allow_pickle=False)
        return cls(z["mu"], z["sd"], z["W"], z["b"], dict(temperature=float(z["temperature"]) if "temperature" in z.files else 1.0))

    def save(self, path, **extra):
        np.savez(path, mu=self.mu, sd=self.sd, W=self.W, b=self.b, features=np.array(FEATURES), emotions=np.array(EMOTIONS), **extra)

    def logits(self, X, domain=1.0):
        """v2 models carry one extra input after the 208 face features: the domain (0 = acted faces of the training corpus, 1 =
        signing — every real use), so in-domain data sets a signing-specific prior instead of bending the expression weights."""
        X = np.atleast_2d(X)
        if X.shape[1] == len(self.mu) - 1:
            X = np.c_[X, np.full(len(X), float(domain), X.dtype)]
        return ((X - self.mu) / self.sd) @ self.W + self.b

    def proba(self, X):
        L = self.logits(X) / self.meta.get("temperature", 1.0)
        e = np.exp(L - L.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def explain(self, x, k, top=4):
        """Strongest features pushing emotion k up relative to neutral → [(feature, contribution)]: w_k·x − w_neutral·x."""
        z = (x - self.mu[:len(x)]) / self.sd[:len(x)]
        c = z * (self.W[:len(x), k] - self.W[:len(x), 0])            # face features only (never the domain input)
        o = np.argsort(-c)[:top]
        return [(FEATURES[i], float(c[i])) for i in o if c[i] > 0.15]

    def predict_stretch(self, bs, face, ok=None):
        x = features(bs, face, ok)
        p = self.proba(x)[0]
        k = int(np.argmax(p))
        v = float(sum(p[i] * VA[e][0] for i, e in enumerate(EMOTIONS)))
        a = float(sum(p[i] * VA[e][1] for i, e in enumerate(EMOTIONS)))
        return dict(probs={e: round(float(p[i]), 3) for i, e in enumerate(EMOTIONS)}, emotion=EMOTIONS[k], p=float(p[k]), valence=v,
                    arousal=a, evidence=self.explain(x, k) if k else [])

    def analyse(self, raw, spans=None, fps=25.0):
        """Whole utterance + one reading per sign segment (spans in 25 fps frames mapped onto the raw timestamps)."""
        bs, face = np.asarray(raw["bs"], np.float32), np.asarray(raw["face"], np.float32).copy()
        H, W_ = (int(raw["hw"][0]), int(raw["hw"][1])) if len(raw["hw"]) == 2 else (1, 1)
        face[..., 0] *= W_ / max(H, 1)
        ok = (bs.sum(1) > 0) & (np.abs(face).sum((1, 2)) > 0)
        out = dict(face_frames=float(ok.mean()) if len(ok) else 0.0)
        if ok.sum() < 5:
            return dict(out, emotion="neutral", emotion_th=EMO_TH["neutral"], probs={}, valence=0.0, arousal=0.0, intensity=0.0,
                        confidence=0.0, evidence=[], explanation_th="ไม่เห็นใบหน้าชัดพอ", per_segment=[])
        r = self.predict_stretch(bs, face, ok)
        inten = float(np.clip(max(0.0, r["p"] - r["probs"]["neutral"]) * 1.5, 0, 1)) if r["emotion"] != "neutral" else 0.0
        out.update(r, emotion_th=EMO_TH[r["emotion"]], intensity=round(inten, 3), confidence=round(r["p"], 3))
        if spans:
            t = np.asarray(raw["t"], np.float64); tt = t - t[0]
            per = []
            for f0, f1 in spans:
                m = (tt >= f0 / fps) & (tt < max(f1, f0 + 1) / fps) & ok
                per.append(self.predict_stretch(bs[m], face[m], None) if m.sum() >= 3 else None)
            out["per_segment"] = [None if p is None else dict(emotion=p["emotion"], p=round(p["p"], 3)) for p in per]
        out["explanation_th"] = describe(out)
        return out


def describe(r, extra=()):
    """Thai explanation of a face reading: the emotion, how sure, and the visible evidence."""
    if r["emotion"] == "neutral":
        s = f"สีหน้าเป็นกลาง (ความมั่นใจ {r.get('confidence', r.get('p', 0)):.0%})"
    else:
        ev = []
        for f, _ in r.get("evidence", []):
            b = f.split(":")[0]
            d = BS_TH.get(b, {"yaw_amp": "ส่ายหน้า", "pitch_amp": "พยักหน้า/เงยหน้า", "roll_amp": "เอียงศีรษะ", "pitch_mean": "ก้ม/เงยหน้า"}.get(b, b))
            if d not in ev:
                ev.append(d)
        s = f"สีหน้า{EMO_TH[r['emotion']]} (ความมั่นใจ {r.get('confidence', r.get('p', 0)):.0%})" + (f" เพราะ{' · '.join(ev[:3])}" if ev else "")
    return " ; ".join([s, *extra])


def signing_energy(kpi, sc, segs):
    """Arousal from the hands: median wrist speed and hand-path extent of the sign segments relative to typical dictionary
    signing (speed 0.9 shoulder widths / s ≈ calm, ≥ 2.0 ≈ energetic). → value in [-1, 1] and a Thai phrase."""
    from .schema import SH_L, SH_R, WR_L, WR_R
    live = [s for s in segs if s["status"] != "transition"]
    if not live or len(kpi) < 4:
        return 0.0, ""
    ok = (sc[:, SH_L] > .3) & (sc[:, SH_R] > .3)
    sw = float(np.median(np.linalg.norm(kpi[ok, SH_L] - kpi[ok, SH_R], axis=1))) if ok.any() else 0.2
    sp = []
    for s in live:
        for wr in (WR_L, WR_R):
            v = np.linalg.norm(np.diff(kpi[s["f0"]:s["f1"], wr], axis=0), axis=1) * 25.0 / max(sw, 1e-3)
            m = (sc[s["f0"] + 1:s["f1"], wr] > .5)
            if m.sum() > 2:
                sp.append(np.median(v[m]))
    if not sp:
        return 0.0, ""
    speed = float(np.max(sp))
    e = float(np.clip((speed - 1.3) / 0.9, -1, 1))
    return e, ("ท่ามือเร็ว/แรง" if e > 0.4 else "ท่ามือช้า/เบา" if e < -0.4 else "")


def fuse_affect(face, segs, energy=0.0, energy_th="", w_lex=0.6, p_min=0.5, grammar=None):
    """Face reading + recognised words + signing energy → the utterance's affect with a Thai explanation.
    face = FaceEmotion.analyse output; segs = pipeline segments; grammar = modules/face.py cues (question / negation)."""
    probs = dict(face.get("probs") or {e: float(e == "neutral") for e in EMOTIONS})
    notes = []
    lex = [(s["nearest"], EMO_LEX[s["nearest"]]) for s in segs
           if s.get("status") in ("accepted", "uncertain") and s.get("p_correct", 0) >= p_min and s.get("nearest") in EMO_LEX]
    if lex:
        for w, e in lex:
            probs[e] = probs.get(e, 0) + w_lex / len(lex)
        notes.append("คำที่ทำ: " + ", ".join(f"'{w}' ({EMO_TH[e].split(' /')[0]})" for w, e in lex))
    if grammar and grammar.get("question_yesno") and probs.get("surprise", 0) > 0:
        probs["neutral"] += 0.5 * probs["surprise"]; probs["surprise"] *= 0.5     # raised brows of a question are grammar, not surprise
        notes.append("คิ้วยกเป็นสัญญาณคำถาม (ไวยากรณ์ ไม่ใช่อารมณ์)")
    if grammar and grammar.get("negation"):
        notes.append("ส่ายหน้า = ปฏิเสธ (ไวยากรณ์)")
    tot = sum(probs.values())
    probs = {k: v / tot for k, v in probs.items()}
    emo = max(probs, key=probs.get)
    v = float(sum(p * VA[e][0] for e, p in probs.items()))
    a = float(np.clip(sum(p * VA[e][1] for e, p in probs.items()) + 0.3 * energy, -1, 1))
    if energy_th:
        notes.append(energy_th)
    inten = float(np.clip((probs[emo] - probs.get("neutral", 0)) * 1.5 + 0.2 * max(energy, 0), 0, 1)) if emo != "neutral" else 0.0
    r = dict(emotion=emo, emotion_th=EMO_TH[emo], probs={k: round(p, 3) for k, p in probs.items()}, valence=round(v, 3), arousal=round(a, 3),
             intensity=round(inten, 3), confidence=round(probs[emo], 3), evidence=face.get("evidence", []) if emo == face.get("emotion") else [],
             sources=dict(face=face.get("emotion"), words=[w for w, _ in lex], energy=round(energy, 2)))
    r["explanation_th"] = describe(r, notes)
    return r
