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
An ENSEMBLE folder lists `members` in config.json: each one an ONNX file with its own input (`fer48`: our 48x48 gray graph; `hse`: an
HSEmotion / AffectNet net that takes a 96x96 gray crop, resized to its own size, 3 channels, ImageNet normalisation, its own class order).
The frame probability = mean of the members' probabilities (each: logits / T + prior). A crop sent without its 96x96 version is scored by
the 48x48 members only, so an old client still works.
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
def brow_features(pts):
    """68 face points in PIXELS (iBUG order, image-left first) → [inner-brow gap, inner-brow height], both divided by the distance between
    the eye centres and measured in the eyes' own frame (head roll removed). A frown = both smaller than at the person's rest."""
    p = np.asarray(pts, np.float64).reshape(68, 2)
    el, er = p[36:42].mean(0), p[42:48].mean(0)
    d = er - el; iod = float(np.hypot(*d))
    if iod < 1e-6:
        return np.array([np.nan, np.nan])
    ux = d / iod; uy = np.array([-ux[1], ux[0]])                      # eye line, and its normal (pointing down the face)
    gap = float(np.dot(p[22] - p[21], ux)) / iod
    hgt = float(np.mean([np.dot(e - b, uy) for b, e in ((p[20], el), (p[21], el), (p[22], er), (p[23], er))])) / iod
    return np.array([gap, hgt])


def crop_box(face68, w, h, scale=1.22, shift_up=0.16):
    """Square face box (x0, y0, side) in pixels from the 68 MediaPipe face points (x, y in 0..1). The 68 points stop at the brows, so the
    centre moves up by `shift_up` of the face height; side = the larger face extent × scale. Same rule as webapp/web/lib/faceBranch.ts."""
    p = np.asarray(face68, np.float32)
    x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    cx, cy = (x0 + x1) / 2 * w, ((y0 + y1) / 2 - (y1 - y0) * shift_up) * h
    side = max((x1 - x0) * w, (y1 - y0) * h) * scale
    return cx - side / 2, cy - side / 2, side


def crop48(img, box, n=48):
    """img (gray or BGR/RGB uint8 [H,W(,3)]) + box → uint8 [n,n] gray (area-averaged; outside the picture = edge pixels). n = 48 or 96."""
    import cv2
    x0, y0, s = box
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sc = float(n) / max(s, 1.0)
    M = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc]], np.float32)
    # area interpolation when shrinking: pre-blur by the shrink factor, then an affine warp (border = replicate)
    if sc < 0.5:
        k = int(1 / sc) | 1
        img = cv2.GaussianBlur(img, (k, k), 0.5 * (1 / sc))
    return cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crops_from_video(path, raw, every_s=0.2, scale=1.22, shift_up=0.16, with96=False, **_):
    """Server-side video: the face pictures at the frames where MediaPipe (modules/mediapipe_pose.extract → `raw`) found a face, one per
    `every_s`. Read only: `raw` and the video are not changed. → list of dict(t [s], g [2304 bytes], side_px, occluded) (+ g96 [9216 bytes] if with96)."""
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
                    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    g = crop48(gray, box)
                    hands = [raw[h][i] for h in ("lh", "rh") if h in raw]
                    out.append(dict(t=float(t[i]), g=g.tobytes(), side_px=float(box[2]), occluded=hand_overlap(face[i], hands)))
                    if with96:
                        out[-1]["g96"] = crop48(gray, box, 96).tobytes()
                    out[-1]["pts"] = (np.asarray(face[i], np.float64)[:, :2] * [W, H]).ravel()
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
_IMNET_MEAN, _IMNET_STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


class _Member:
    """One net of the model folder. kind 'fer48': our graph, [N,1,48,48] 0..255 in, our class order out. kind 'hse': HSEmotion (AffectNet)
    graph, 96x96 gray crop -> `size` px bicubic, 3 channels, /255, ImageNet mean/std; outputs in `order` (extra classes such as contempt dropped)."""

    def __init__(self, d, m, classes, so, providers):
        import onnxruntime as ort
        self.kind, self.T = m.get("kind", "fer48"), float(m.get("temperature", 1.0))
        self.sess = ort.InferenceSession(str(d / m["onnx"]), so, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.size = int(m.get("size", 260))
        order = m.get("order", classes)
        self.idx = np.array([order.index(c) for c in classes])
        self.has_feat = len(self.sess.get_outputs()) > 1                       # a graph that also returns its penultimate features
        self.last_feat = None

    def logits(self, crops48, crops96):
        if self.kind == "fer48":
            return self.sess.run(None, {self.inp: np.asarray(crops48, np.float32).reshape(-1, 1, 48, 48)})[0][:, self.idx]
        import cv2
        out, feats = [], []
        for i in range(0, len(crops96), 16):
            x = np.stack([cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_CUBIC) for g in crops96[i:i + 16]]).astype(np.float32) / 255.0
            x = ((np.repeat(x[..., None], 3, -1) - _IMNET_MEAN) / _IMNET_STD).transpose(0, 3, 1, 2)
            r = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})
            out.append(r[0])
            if self.has_feat:
                feats.append(r[1])
        self.last_feat = np.concatenate(feats).astype(np.float32) if feats else None
        return np.concatenate(out)[:, self.idx]


class PictureHead:
    """RAVDESS-trained head over a short WINDOW of crops (Face-Lab/rav_head.py; the same features it was trained on): mean of the B2
    penultimate features + mean / max of the members' plain probabilities (B2 without contempt, v2 at `v2_temperature`; no deployment
    prior). It is linear, so each crop is reduced once to 21 numbers (7 = its feature part of the logits, 7 + 7 = its B2 / v2
    probabilities) and a window = the mean / max of those → the live state carries ~10 small rows, not 1408-d features.
    Config (`head` in config.json): npz, window_s, min_quality, v2_temperature, emotions (the head's class order), prior_shift."""

    def __init__(self, d, cfg, classes):
        z = np.load(d / cfg["npz"])
        self.mu, self.sd, self.W, self.b = (z[k].astype(np.float64) for k in ("mu", "sd", "W", "b"))
        self.win, self.min_q, self.Tv = float(cfg.get("window_s", 2.0)), float(cfg.get("min_quality", 0.25)), float(cfg.get("v2_temperature", 0.7))
        emo = list(cfg["emotions"]); classes = list(classes)
        self.from_cls = [classes.index(e) for e in emo]                      # model order → head order
        self.to_cls = [emo.index(c) for c in classes]                        # head order → model order
        self.nf = len(self.mu) - 4 * len(emo)
        ps = cfg.get("prior_shift") or {}
        self.bias = np.array([float(ps.get(c, 0.0)) for c in classes], np.float64)

    def crop_vecs(self, feat, zb, zv):
        """B2 features [N,1408] + B2 / v2 logits [N,K] (model order, contempt already dropped) → [N, 3K] per-crop rows (head order)."""
        c = ((np.asarray(feat, np.float64) - self.mu[:self.nf]) / self.sd[:self.nf]) @ self.W[:self.nf]
        return np.c_[c, softmax(zb)[:, self.from_cls], softmax(np.asarray(zv) / self.Tv)[:, self.from_cls]]

    def window_probs(self, V):
        """rows [n, 3K] of one window → probabilities [K] in the model's class order."""
        V = np.asarray(V, np.float64); k = V.shape[1] // 3
        pb, pv = V[:, k:2 * k], V[:, 2 * k:]
        r = np.r_[pb.mean(0), pb.max(0), pv.mean(0), pv.max(0)]
        L = V[:, :k].mean(0) + ((r - self.mu[self.nf:]) / self.sd[self.nf:]) @ self.W[self.nf:] + self.b
        return softmax(L[self.to_cls] + self.bias)

    def apply(self, P, V, t, q, hist):
        """frame probabilities [N,K] + their rows (NaN = no 96 px crop → the ensemble answer stays) + times / qualities + the recent rows
        `hist` ([t, q, *row], updated in place) → probabilities from the head over each frame's trailing `window_s`."""
        P = np.array(P, np.float64)
        for i in range(len(P)):
            ti, ok = float(t[i]), bool(np.isfinite(V[i]).all())
            if ok:
                hist.append([round(ti, 3), round(float(q[i]), 3), *np.round(V[i], 5).tolist()])
            hist[:] = [h for h in hist if ti - self.win < h[0] <= ti + 1e-3]
            if not ok:
                continue
            good = [h[2:] for h in hist if h[1] >= self.min_q] or [hist[-1][2:]]
            P[i] = self.window_probs(good)
        return P


class FaceFER:
    def __init__(self, model_dir, providers=None, threads=2):
        d = Path(model_dir)
        self.dir, self.cfg = d, json.loads((d / "config.json").read_text(encoding="utf-8"))
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.log_severity_level = 3
        self.classes = self.cfg.get("classes", CLASSES)
        members = self.cfg.get("members") or [dict(onnx=self.cfg.get("onnx", "face_fer.onnx"), kind="fer48", temperature=self.cfg.get("temperature", 1.0))]
        self.members = [_Member(d, m, self.classes, so, providers or ["CPUExecutionProvider"]) for m in members]
        self.sess = self.members[0].sess                                         # (the API reports its providers)
        self.wants_96 = any(m.kind != "fer48" for m in self.members)
        self.T = float(self.cfg.get("temperature", 1.0))
        ps = self.cfg.get("prior_shift") or {}                                  # deployment prior: logit bonus per class (neutral: signers rest on a neutral face)
        self.bias = np.array([float(ps.get(c, 0.0)) for c in self.classes], np.float64)
        self.unk = {**dict(min_conf=0.45, min_margin=0.10, min_quality=0.25), **self.cfg.get("unknown", {})}
        self.version = self.cfg.get("name", d.name)
        self.head = PictureHead(d, self.cfg["head"], self.classes) if self.cfg.get("head") else None
        self._vecs = None

    def logits(self, crops):
        """the FIRST member's logits on 48x48 crops (a single-net folder: the model's logits; used by the lab scripts)."""
        return self.members[0].logits(crops, None)

    def probs(self, crops, crops96=None):
        """crops [N,48,48] (+ crops96: [N,96,96] or a list with None where a crop has no 96x96 version) → [N,K] probabilities with the
        deployment prior. Single net: softmax((z + T*bias) / T) as before. Ensemble: mean over the members that have their input."""
        if len(self.members) == 1 and self.members[0].kind == "fer48":
            return softmax(self.logits(crops) + self.T * self.bias, self.T)
        n = len(crops)
        has96 = np.array([crops96 is not None and crops96[i] is not None for i in range(n)], bool)
        tot, cnt = np.zeros((n, len(self.classes))), np.zeros((n, 1))
        zv = zb = fb = None
        for m in self.members:
            if m.kind == "fer48":
                z = m.logits(crops, None); zv = z if zv is None else zv
                tot += softmax(z / m.T + self.bias); cnt += 1
            elif has96.any():
                z = m.logits(None, np.stack([np.asarray(crops96[i], np.uint8).reshape(96, 96) for i in np.where(has96)[0]]))
                tot[has96] += softmax(z / m.T + self.bias); cnt[has96] += 1
                if m.has_feat and fb is None:
                    zb, fb = z, m.last_feat
        self._vecs = None                                                   # the head's per-crop rows (frame_results uses them)
        if self.head is not None and fb is not None and zv is not None:
            self._vecs = np.full((n, 3 * len(self.classes)), np.nan)
            self._vecs[has96] = self.head.crop_vecs(fb, zb, zv[has96])
        return tot / np.maximum(cnt, 1)

    def frame_results(self, crops, quality=None, t=None, crops96=None, head_hist=None):
        """crops [N,48,48] uint8 (+ per-crop quality / time, + 96x96 crops for an ensemble) → list of raw frame results (module doc).
        A folder with a `head`: a frame with its 96 px crop gets the head's answer over its trailing window (`head_hist` = the recent
        rows carried between live calls, updated in place; None = a fresh window, e.g. a whole utterance)."""
        p = self.probs(crops, crops96)
        q = np.ones(len(p)) if quality is None else np.asarray(quality, np.float64)
        if self._vecs is not None:
            tt = np.arange(len(p)) * 0.2 if t is None else np.asarray(t, np.float64)
            p = self.head.apply(p, self._vecs, tt, q, [] if head_hist is None else head_hist)
        return [dict(t=None if t is None else float(t[i]), **self.judge(p[i], float(q[i]))) for i in range(len(p))]

    def judge(self, p, q):
        """one frame's probabilities [K] + quality → the frame fields (the unknown rule: quality, confidence, margin)."""
        top = np.argsort(-p)[:2]
        conf, margin = float(p[top[0]]), float(p[top[0]] - p[top[1]])
        valid = q >= self.unk["min_quality"] and conf >= self.unk["min_conf"] and margin >= self.unk["min_margin"]
        return dict(expression=self.classes[top[0]] if valid else "unknown", top=self.classes[top[0]], confidence=conf if valid else 0.0,
                    top_confidence=conf, margin=margin, probs={c: float(p[k]) for k, c in enumerate(self.classes)}, quality=q, valid=bool(valid))


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


class BrowCue:
    """Brow-furrow cue from the 68 face points: the picture models read a plain frown (brows drawn together / lowered, the rest of the face
    calm) as neutral. Measure: the inner-brow HEIGHT above the eyes (on RAVDESS it separates angry from neutral, AUC 0.92; the gap between
    the brows does not). Relative to the PERSON's own resting brows: the baseline = the `base_pct` percentile of their recent heights
    (carried in the live state, or the utterance's own frames), a frown = how far below it, in units of `unit` (× eye distance). Only a
    frame the models call neutral is touched (a confident happy / sad stays as it is): past `thr` its probabilities move toward `target`
    (angry) by up to `w_max`. LIVE DISPLAY ONLY (the web shows angry as "ขมวดคิ้ว"): a lowered brow is also Thai Sign Language grammar
    (wh-questions), so the utterance summary that sets the voice's emotion (`process` → `speech_affect`) never uses this cue. Config (`brow` in config.json): thr, span, w_max, unit, base_pct, min_base, keep, target."""

    def __init__(self, cfg, classes):
        self.thr, self.span, self.w_max = float(cfg.get("thr", 2.0)), float(cfg.get("span", 1.5)), float(cfg.get("w_max", 0.6))
        self.unit = float(cfg.get("unit", 0.02))
        self.pct, self.min_base, self.keep = float(cfg.get("base_pct", 85)), int(cfg.get("min_base", 6)), int(cfg.get("keep", 60))
        self.k, self.neutral = list(classes).index(cfg.get("target", "angry")), list(classes).index("neutral")

    def score(self, feat, hist):
        """feature [gap, height] + recent heights → frown score (0 = at rest, > 0 = brows lower), or None without a baseline yet."""
        h = np.asarray([x for x in hist if np.isfinite(x)], np.float64)
        if len(h) < self.min_base or not np.isfinite(feat[1]):
            return None
        return float((np.percentile(h, self.pct) - feat[1]) / self.unit)

    def apply(self, fr, feats, hist, judge):
        """raw frame results + their features (rows of NaN = no points) + the person's history (list, updated in place) → frames with
        the cue applied (`brow` = the score). `judge(p, q)` re-derives expression / validity with the model's unknown rule."""
        for f, x in zip(fr, feats):
            if not np.isfinite(x).all():
                continue
            s = self.score(x, hist)
            hist.append(round(float(x[1]), 4)); del hist[:-self.keep]
            f["brow"] = None if s is None else round(s, 2)
            p = np.array(list(f["probs"].values()))
            if s is None or s <= self.thr or int(np.argmax(p)) != self.neutral:
                continue
            w = self.w_max * min(1.0, (s - self.thr) / self.span)
            p = p * (1 - w); p[self.k] += w
            f.update(judge(p, f["quality"]))
        return fr


class FaceBranch:
    """Stateless façade for the API: raw → smoothed → segment summaries. Every public method returns plain dicts and never raises."""

    def __init__(self, model_dir, **kw):
        self.model = FaceFER(model_dir, **kw)
        self.brow = BrowCue(self.model.cfg["brow"], self.model.classes) if self.model.cfg.get("brow") else None
        self.sm_cfg = {**dict(tau_s=0.35, switch_margin=0.12, min_hold_s=0.3, strong=0.80, lost_s=1.5, min_conf=self.model.unk["min_conf"]),
                       **self.model.cfg.get("smoother", {})}
        self.errors = 0

    def _frames(self, crops):
        """crops: list of dict(t [s], g (48x48 uint8 or its bytes), g96? (96x96 uint8 or its bytes), side_px?, occluded?)
        → (t, G, quality, G96 = list with None where a crop has no 96x96 version)."""
        G, Q, T, G96, self._feats = [], [], [], [], []
        for c in crops:
            pts = c.get("pts")
            self._feats.append(brow_features(np.asarray(pts, np.float64)) if pts is not None and np.size(pts) == 136 else np.array([np.nan, np.nan]))
            g = c["g"]
            if isinstance(g, (bytes, bytearray)):
                g = np.frombuffer(g, np.uint8)
            g = np.asarray(g, np.uint8).reshape(48, 48)
            h = c.get("g96")
            if h is not None:
                h = np.frombuffer(h, np.uint8) if isinstance(h, (bytes, bytearray)) else np.asarray(h, np.uint8)
                h = h.reshape(96, 96) if h.size == 9216 else None
            G.append(g); G96.append(h); T.append(float(c["t"])); Q.append(face_quality(g, c.get("side_px"), c.get("occluded", 0.0)))
        return np.array(T), np.stack(G), np.array(Q), G96

    def live(self, crops, state=None):
        """Live display: crops since the last call (+ the state returned by it) → raw frames + the smoothed label + the new state."""
        try:
            if not crops:
                st = ExpressionSmoother.from_state(state, classes=self.model.classes, **self.sm_cfg)
                return dict(ok=True, frames=[], smoothed=dict(UNKNOWN_FRAME) if not state else self._shown(st), state=st.state(), model=self.model.version)
            t, G, Q, G96 = self._frames(crops)
            head_hist = list((state or {}).get("head") or [])
            fr = self.model.frame_results(G, Q, t, G96, head_hist)
            hist = list((state or {}).get("brow") or [])
            if self.brow:
                self.brow.apply(fr, self._feats, hist, self.model.judge)
            sm = ExpressionSmoother.from_state(state, classes=self.model.classes, **self.sm_cfg)
            shown = [sm.update(f) for f in fr]
            st = sm.state()
            if self.brow:
                st["brow"] = hist
            if self.model.head is not None:
                st["head"] = head_hist
            return dict(ok=True, frames=[{k: f.get(k) for k in ("t", "expression", "top", "confidence", "quality", "valid", "brow")} for f in fr],
                        smoothed=shown[-1], state=st, model=self.model.version)
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
            t, G, Q, G96 = self._frames(crops)
            t0 = float(t[0]) if t0 is None else float(t0)
            rel = t - t0
            fr = self.model.frame_results(G, Q, rel, G96)
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
    if grammar and grammar.get("question_wh") and probs["angry"] > 0:       # wh-question brows (lowered / drawn together) are grammar, not anger
        probs["neutral"] += 0.5 * probs["angry"]; probs["angry"] *= 0.5
    emo = max(probs, key=probs.get)
    if emo == "neutral" or probs[emo] < min_conf:
        return None, None, "neutral / not confident"
    inten = float(np.clip((probs[emo] - probs["neutral"]) * 1.5, 0.0, 1.0)) if emo != "neutral" else 0.0
    return TO_SPEECH[emo], round(0.25 + 0.5 * inten, 2), "face"
