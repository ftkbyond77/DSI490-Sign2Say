"""Face branch (v11): facial EXPRESSION from the face picture — an independent late-fusion branch next to the sign model.

    crops (48x48 gray face pictures, cut by the client / by `crop_from_points`)  →  ONNX classifier (models/face-model-*-v1)
    → raw frame result {expression, confidence, probs, quality, valid} → `ExpressionSmoother` (EMA + hysteresis) → segment summary
    → `speech_affect` (what the voice may use).

Rules the code enforces (tests/test_face_branch.py):
  • numpy + onnxruntime only; it never imports or touches the sign pipeline, and `FaceBranch.process` never raises (a failure = an
    all-`unknown` answer + a log line) — the sign result is computed without it and is identical with it on, off or crashed;
  • no Sign Model input (landmarks, bs, face points) is changed: the branch only READS points to place a crop;
  • "unknown" is a first-class answer: no face, small / dark / blurred / occluded face, low calibrated confidence or a small top-2 margin;
  • an expression is evidence about the face, not a sign: a yes/no question's raised brows are grammar — `speech_affect` keeps them out of
    "surprise" with the grammar flag the sign pipeline already reports (it reads that flag, it does not write it).

Model folder (models/face-model-<arch>-v1/): face_fer.onnx, config.json (classes, temperature, crop geometry, thresholds, smoother), model_card.json.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("face-fer")
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]      # FER2013 order (= the model's output order)
EMO_TH = {"angry": "โกรธ", "disgust": "รังเกียจ", "fear": "กลัว", "happy": "มีความสุข", "sad": "เศร้า", "surprise": "ประหลาดใจ", "neutral": "เป็นกลาง",
          "unknown": "ไม่แน่ใจ"}
# the voice (Speech v4) takes these 7 names; the FER2013 names are the same words
TO_SPEECH = {c: c for c in CLASSES}
_VA = {"neutral": (0.0, 0.0), "happy": (0.8, 0.5), "sad": (-0.7, -0.4), "angry": (-0.6, 0.8), "fear": (-0.6, 0.7), "disgust": (-0.7, 0.3), "surprise": (0.2, 0.8)}


def softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / T
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ------------------------------------------------------------------ crop geometry (shared by browser, server and the notebook)
def crop_box(face68, w, h, scale=1.22, shift_up=0.16):
    """Square face box (x0, y0, side) in pixels from the 68 MediaPipe face points (x, y in 0..1). The 68 points stop at the brows, so the
    centre moves up by `shift_up` of the face height; side = the larger face extent × scale. Same rule as webapp/web/lib/faceBranch.ts."""
    p = np.asarray(face68, np.float32)
    x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    cx, cy = (x0 + x1) / 2 * w, ((y0 + y1) / 2 - (y1 - y0) * shift_up) * h
    side = max((x1 - x0) * w, (y1 - y0) * h) * scale
    return cx - side / 2, cy - side / 2, side


def crop48(img, box):
    """img (gray or BGR/RGB uint8 [H,W(,3)]) + box → uint8 [48,48] gray (area-averaged; outside the picture = edge pixels)."""
    import cv2
    x0, y0, s = box
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sc = 48.0 / max(s, 1.0)
    M = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc]], np.float32)
    # area interpolation when shrinking: pre-blur by the shrink factor, then an affine warp (border = replicate)
    if sc < 0.5:
        k = int(1 / sc) | 1
        img = cv2.GaussianBlur(img, (k, k), 0.5 * (1 / sc))
    return cv2.warpAffine(img, M, (48, 48), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crops_from_video(path, raw, every_s=0.2, scale=1.22, shift_up=0.16, **_):
    """Server-side video: the face pictures at the frames where MediaPipe (modules/mediapipe_pose.extract → `raw`) found a face, one per
    `every_s`. Read only: `raw` and the video are not changed. → list of dict(t [s], g [2304 bytes], side_px, occluded)."""
    import cv2
    t, face = np.asarray(raw["t"], np.float64), np.asarray(raw["face"])
    cap = cv2.VideoCapture(str(path))
    fps = float(raw.get("src_fps", 25.0)) if raw.get("src_fps", 0) else 25.0
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    want, last = {}, -1e9
    for i in range(len(t)):
        if t[i] - last >= every_s - 1e-6 and np.abs(face[i]).sum() > 0:
            want[int(round(t[i] * fps))] = i; last = t[i]
    out, k = [], 0
    try:
        while want and k <= max(want):
            if not cap.grab():
                break
            if k in want:
                ok, fr = cap.retrieve()
                if ok:
                    i = want[k]
                    box = crop_box(face[i], W, H, scale, shift_up)
                    g = crop48(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), box)
                    hands = [raw[h][i] for h in ("lh", "rh") if h in raw]
                    out.append(dict(t=float(t[i]), g=g.tobytes(), side_px=float(box[2]), occluded=hand_overlap(face[i], hands)))
            k += 1
    finally:
        cap.release()
    return out


def hand_overlap(face68, hands, margin=0.05):
    """Fraction of a hand's points inside the face box → 0..1 (the biggest of the two hands). hands = list of [[x, y(, z)]…] (0..1)."""
    p = np.asarray(face68, np.float32)
    x0, y0, x1, y1 = p[:, 0].min() - margin, p[:, 1].min() - margin * 2, p[:, 0].max() + margin, p[:, 1].max() + margin
    best = 0.0
    for h in hands:
        a = np.asarray(h, np.float32)
        if a.ndim == 2 and len(a) and np.abs(a[:, :2]).sum() > 0:
            best = max(best, float(((a[:, 0] > x0) & (a[:, 0] < x1) & (a[:, 1] > y0) & (a[:, 1] < y1)).mean()))
    return best


# ------------------------------------------------------------------ face quality (kept apart from the classifier's confidence)
_LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)


def face_quality(g48, side_px=None, occluded=0.0):
    """0..1 from the crop and what the client knows: face size in the camera picture, light (too dark / burnt), contrast, sharpness,
    hand over the face. Not a probability — a gate: low quality → `unknown`."""
    g = np.asarray(g48, np.float32) / 255.0
    mean, std = float(g.mean()), float(g.std())
    q_light = min(1.0, mean / 0.15) * min(1.0, (1.0 - mean) / 0.15)
    q_contrast = min(1.0, std / 0.10)
    lap = (_LAP[1, 1] * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    q_sharp = min(1.0, float(lap.var()) / 0.0012)
    q_size = 1.0 if side_px is None else float(np.clip((side_px - 28) / (72 - 28), 0.0, 1.0))
    q_occ = 1.0 - float(np.clip(occluded * 1.6, 0.0, 1.0))
    return float(np.clip(q_size * q_light * (0.5 + 0.5 * q_contrast) * (0.5 + 0.5 * q_sharp) * q_occ, 0.0, 1.0))


# ------------------------------------------------------------------ the classifier
class FaceFER:
    def __init__(self, model_dir, providers=None, threads=2):
        import onnxruntime as ort
        d = Path(model_dir)
        self.dir, self.cfg = d, json.loads((d / "config.json").read_text(encoding="utf-8"))
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(d / self.cfg.get("onnx", "face_fer.onnx")), so, providers=providers or ["CPUExecutionProvider"])
        self.classes = self.cfg.get("classes", CLASSES)
        self.T = float(self.cfg.get("temperature", 1.0))
        ps = self.cfg.get("prior_shift") or {}                                  # deployment prior: logit bonus per class (neutral: signers rest on a neutral face)
        self.bias = np.array([float(ps.get(c, 0.0)) for c in self.classes], np.float64)
        self.unk = {**dict(min_conf=0.45, min_margin=0.10, min_quality=0.25), **self.cfg.get("unknown", {})}
        self.version = self.cfg.get("name", d.name)

    def logits(self, crops):
        x = np.asarray(crops, np.float32).reshape(-1, 1, 48, 48)
        return self.sess.run(None, {"face": x})[0]

    def probs(self, crops):
        return softmax(self.logits(crops) + self.T * self.bias, self.T)

    def frame_results(self, crops, quality=None, t=None):
        """crops [N,48,48] uint8 (+ per-crop quality / time) → list of raw frame results (the structure in the module doc)."""
        p = self.probs(crops)
        out = []
        for i in range(len(p)):
            q = 1.0 if quality is None else float(quality[i])
            top = np.argsort(-p[i])[:2]
            conf, margin = float(p[i, top[0]]), float(p[i, top[0]] - p[i, top[1]])
            valid = q >= self.unk["min_quality"] and conf >= self.unk["min_conf"] and margin >= self.unk["min_margin"]
            out.append(dict(t=None if t is None else float(t[i]), expression=self.classes[top[0]] if valid else "unknown", top=self.classes[top[0]],
                            confidence=conf if valid else 0.0, top_confidence=conf, margin=margin, probs={c: float(p[i, k]) for k, c in enumerate(self.classes)},
                            quality=q, valid=bool(valid)))
        return out


# ------------------------------------------------------------------ temporal smoothing (face branch only)
class ExpressionSmoother:
    """EMA of the class probabilities (time constant `tau_s`) + hysteresis for the shown label: a new class must lead the current one by
    `switch_margin` for `min_hold_s` before it is shown (or lead by `strong` at once); an invalid stretch longer than `lost_s` → unknown.
    Works on timestamps (seconds), so the camera frame rate / the sampling cadence does not matter. `state()` / `from_state()` make it
    stateless for the API (the client carries the small state between calls)."""

    def __init__(self, classes=CLASSES, tau_s=0.35, switch_margin=0.12, min_hold_s=0.3, strong=0.80, lost_s=1.5, min_conf=0.45):
        self.classes, self.tau, self.margin, self.hold, self.strong, self.lost, self.min_conf = list(classes), tau_s, switch_margin, min_hold_s, strong, lost_s, min_conf
        self.reset()

    def reset(self):
        self.ema = None; self.cur = "unknown"; self.cand = None; self.cand_t = 0.0; self.t = None; self.last_valid = None

    def state(self):
        return dict(ema=None if self.ema is None else [round(float(v), 4) for v in self.ema], cur=self.cur, cand=self.cand, cand_t=self.cand_t, t=self.t,
                    last_valid=self.last_valid)

    @classmethod
    def from_state(cls, st, **kw):
        s = cls(**kw)
        if st:
            s.ema = None if st.get("ema") is None else np.asarray(st["ema"], np.float64)
            s.cur, s.cand, s.cand_t, s.t, s.last_valid = st.get("cur", "unknown"), st.get("cand"), st.get("cand_t", 0.0), st.get("t"), st.get("last_valid")
        return s

    def update(self, r):
        """r = a raw frame result (with `t` in seconds) → smoothed dict(expression, confidence, probs, valid, switched)."""
        t = float(r["t"]) if r.get("t") is not None else (self.t or 0.0) + 0.1
        dt = 0.1 if self.t is None else max(t - self.t, 1e-3)
        self.t = t
        if r["valid"]:
            p = np.array([r["probs"][c] for c in self.classes], np.float64)
            a = 1 - math.exp(-dt / self.tau)
            self.ema = p if self.ema is None else self.ema + a * (p - self.ema)
            self.last_valid = t
        elif self.last_valid is None or t - self.last_valid > self.lost:
            self.cur, self.cand = "unknown", None                           # lost for too long → unknown (the EMA is kept for a short return)
        switched = False
        if self.ema is not None and self.last_valid is not None and t - self.last_valid <= self.lost:
            top = int(np.argmax(self.ema)); top_c = self.classes[top]
            if self.cur == "unknown":
                if self.ema[top] >= self.min_conf:
                    self.cur, self.cand, switched = top_c, None, True
            elif top_c != self.cur:
                lead = float(self.ema[top] - self.ema[self.classes.index(self.cur)])
                if lead >= self.strong - 0.5 and self.ema[top] >= self.strong:               # strong evidence: no wait
                    self.cur, self.cand, switched = top_c, None, True
                elif lead >= self.margin:
                    if self.cand != top_c:
                        self.cand, self.cand_t = top_c, t
                    elif t - self.cand_t >= self.hold:
                        self.cur, self.cand, switched = top_c, None, True
                else:
                    self.cand = None
            else:
                self.cand = None
        if self.cur == "unknown" or self.ema is None:
            return dict(expression="unknown", confidence=0.0, probs=None, valid=False, switched=switched)
        k = self.classes.index(self.cur)
        return dict(expression=self.cur, confidence=float(self.ema[k]), probs={c: float(self.ema[i]) for i, c in enumerate(self.classes)}, valid=True,
                    switched=switched)


# ------------------------------------------------------------------ the branch the API calls
UNKNOWN_FRAME = dict(expression="unknown", confidence=0.0, probs=None, quality=0.0, valid=False)


class FaceBranch:
    """Stateless façade for the API: raw → smoothed → segment summaries. Every public method returns plain dicts and never raises."""

    def __init__(self, model_dir, **kw):
        self.model = FaceFER(model_dir, **kw)
        self.sm_cfg = {**dict(tau_s=0.35, switch_margin=0.12, min_hold_s=0.3, strong=0.80, lost_s=1.5, min_conf=self.model.unk["min_conf"]),
                       **self.model.cfg.get("smoother", {})}
        self.errors = 0

    def _frames(self, crops):
        """crops: list of dict(t [s], g (48x48 uint8 or b64 of it), side_px?, occluded?) → (t, G, quality) arrays."""
        G, Q, T = [], [], []
        for c in crops:
            g = c["g"]
            if isinstance(g, (bytes, bytearray)):
                g = np.frombuffer(g, np.uint8)
            g = np.asarray(g, np.uint8).reshape(48, 48)
            G.append(g); T.append(float(c["t"])); Q.append(face_quality(g, c.get("side_px"), c.get("occluded", 0.0)))
        return np.array(T), np.stack(G), np.array(Q)

    def live(self, crops, state=None):
        """Live display: crops since the last call (+ the state returned by it) → raw frames + the smoothed label + the new state."""
        try:
            if not crops:
                st = ExpressionSmoother.from_state(state, classes=self.model.classes, **self.sm_cfg)
                return dict(ok=True, frames=[], smoothed=dict(UNKNOWN_FRAME) if not state else self._shown(st), state=st.state(), model=self.model.version)
            t, G, Q = self._frames(crops)
            fr = self.model.frame_results(G, Q, t)
            sm = ExpressionSmoother.from_state(state, classes=self.model.classes, **self.sm_cfg)
            shown = [sm.update(f) for f in fr]
            return dict(ok=True, frames=[{k: f[k] for k in ("t", "expression", "top", "confidence", "quality", "valid")} for f in fr],
                        smoothed=shown[-1], state=sm.state(), model=self.model.version)
        except Exception as e:  # noqa: BLE001
            self.errors += 1
            log.exception("face branch (live) failed: %s", e)
            return dict(ok=False, error=f"{type(e).__name__}: {e}", frames=[], smoothed=dict(UNKNOWN_FRAME), state=None, model=self.model.version)

    def _shown(self, sm):
        return dict(expression=sm.cur, confidence=0.0, probs=None, valid=sm.cur != "unknown")

    def process(self, crops, segments=None, t0=None):
        """A whole utterance: crops (t in seconds, same clock as the sign frames; t0 = the first sign frame's time) → frame-level
        smoothed track, the utterance summary and one summary per sign segment (`segments`: [{t0, t1}] in seconds from the utterance start).
        The summary is a vote of the VALID frames weighted by confidence × quality — never one frame."""
        t_start = time.perf_counter()
        try:
            if not crops:
                return self._empty("no face crops")
            t, G, Q = self._frames(crops)
            t0 = float(t[0]) if t0 is None else float(t0)
            rel = t - t0
            fr = self.model.frame_results(G, Q, rel)
            sm = ExpressionSmoother(self.model.classes, **self.sm_cfg)
            track = [dict(sm.update(f), t=float(f["t"]), raw=f["expression"], raw_confidence=f["confidence"], quality=f["quality"]) for f in fr]
            for tr in track:
                tr.pop("probs", None)
            out = dict(ok=True, model=self.model.version, n_frames=len(fr), summary=self._summary(fr, rel, None), track=track,
                       per_segment=[self._summary(fr, rel, (float(s["t0"]), float(s["t1"]))) for s in (segments or [])],
                       latency_ms=round((time.perf_counter() - t_start) * 1000, 1), sign_modified=False)
            return out
        except Exception as e:  # noqa: BLE001
            self.errors += 1
            log.exception("face branch failed: %s", e)
            return self._empty(f"{type(e).__name__}: {e}")

    def _empty(self, why):
        return dict(ok=False, error=why, model=self.model.version, n_frames=0, summary=self._summary([], np.zeros(0), None), track=[], per_segment=[], sign_modified=False)

    def _summary(self, fr, rel, span):
        m = np.ones(len(fr), bool) if span is None else (rel >= span[0]) & (rel < max(span[1], span[0] + 0.2))
        idx = np.where(m)[0]
        ok = [i for i in idx if fr[i]["valid"]]
        base = dict(expression="unknown", confidence=0.0, probs=None, valid_frac=0.0, n=int(len(idx)), quality=0.0, stable=0.0)
        if not len(idx):
            return base
        base["quality"] = float(np.mean([fr[i]["quality"] for i in idx])); base["valid_frac"] = len(ok) / len(idx)
        if len(ok) < max(2, int(0.25 * len(idx))):
            return base
        w = np.array([fr[i]["top_confidence"] * fr[i]["quality"] for i in ok])
        P = np.array([[fr[i]["probs"][c] for c in self.model.classes] for i in ok])
        mean = (P * w[:, None]).sum(0) / max(w.sum(), 1e-9)
        k = int(np.argmax(mean))
        votes = np.mean([fr[i]["top"] == self.model.classes[k] for i in ok])
        if mean[k] < self.model.unk["min_conf"]:
            return dict(base, probs={c: float(mean[i]) for i, c in enumerate(self.model.classes)})
        return dict(base, expression=self.model.classes[k], confidence=float(mean[k]), stable=float(votes),
                    probs={c: float(mean[i]) for i, c in enumerate(self.model.classes)})


# ------------------------------------------------------------------ speech layer: what the voice may use
def speech_affect(fer_summary, body_affect=None, grammar=None, min_conf=0.5, w_fer=0.5):
    """(emotion, intensity, source) for the voice from the face branch's utterance summary and, optionally, the v9 blendshape affect
    (`face.affect`) — or (None, None, reason). Only a confident non-neutral emotion is voiced, subtly. A yes/no question's raised brows
    (grammar flag from the sign pipeline, read only) are not surprise. The sign text is never touched."""
    probs = {c: 0.0 for c in CLASSES}
    n = 0.0
    if fer_summary and fer_summary.get("probs") and fer_summary.get("valid_frac", 0) >= 0.25 and fer_summary.get("expression") != "unknown":
        for c, v in fer_summary["probs"].items():
            probs[c] += w_fer * v
        n += w_fer
    if body_affect and body_affect.get("probs"):
        for c, v in body_affect["probs"].items():
            c = TO_SPEECH.get(c, c)
            if c in probs:
                probs[c] += (1 - w_fer) * v
        n += 1 - w_fer
    if n <= 0:
        return None, None, "no face evidence"
    probs = {c: v / n for c, v in probs.items()}
    if grammar and grammar.get("question_yesno") and probs["surprise"] > 0:
        probs["neutral"] += 0.5 * probs["surprise"]; probs["surprise"] *= 0.5
    emo = max(probs, key=probs.get)
    if emo == "neutral" or probs[emo] < min_conf:
        return None, None, "neutral / not confident"
    inten = float(np.clip((probs[emo] - probs["neutral"]) * 1.5, 0.0, 1.0)) if emo != "neutral" else 0.0
    return TO_SPEECH[emo], round(0.25 + 0.5 * inten, 2), "face"
