"""Weight of the picture model vs the blendshape model in the voice's emotion (modules/face_fer.speech_affect, w_fer = 0.5 today).
 RAVDESS (24 actors, out-of-fold, cache/fuse24_probs.npz from fuse24.py): F1 of the fused probabilities and of the voice decision
   (neutral unless a non-neutral class is on top with ≥ 0.5).
 Team clips (18 clips, real_check frames every 0.5 s): 2 s windows (4 frames, hop 1 s) run like the app — the deployed head
   (models/face-model-ravhead-v1, one crop per call, its window state) → FaceBranch._summary; the shipped blendshape model
   (models/face-model) on the window's blendshapes + face points → speech_affect(summary, blendshape, w_fer).
   Reported: on windows whose frames were all judged neutral, how often the voice gets an emotion; windows with a judged smile: voiced happy.
  python voice_weight.py → results/voice_weight.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import FaceFER, speech_affect   # noqa: E402
from modules.face_model import EMOTIONS, FaceEmotion, features   # noqa: E402
from compare24 import metrics                         # noqa: E402
import real_check                                     # noqa: E402

W = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
NEU = EMOTIONS.index("neutral")


def summary(m, fr):
    """FaceBranch._summary for a whole window"""
    ok = [f for f in fr if f["valid"]]
    base = dict(expression="unknown", probs=None, valid_frac=len(ok) / max(len(fr), 1))
    if len(ok) < max(2, int(0.25 * len(fr))):
        return base
    w = np.array([f["top_confidence"] * f["quality"] for f in ok]); P = np.array([[f["probs"][c] for c in m.classes] for f in ok])
    mean = (P * w[:, None]).sum(0) / max(w.sum(), 1e-9); k = int(np.argmax(mean)); probs = {c: float(mean[i]) for i, c in enumerate(m.classes)}
    return dict(base, probs=probs, expression=m.classes[k] if mean[k] >= m.unk["min_conf"] else "unknown")


def main():
    rep = dict(note=__doc__, ravdess={}, team={})
    z = np.load(ROOT / "Face-Lab" / "cache" / "fuse24_probs.npz")
    PH, PB, y, whole, inten = z["head"], z["blend"], z["y"], z["whole"], z["inten"]
    print("w_fer | RAVDESS fused F1 | voice F1 | voice recall")
    for w in W:
        P = w * PH + (1 - w) * PB
        k = P.argmax(1); k[(k == NEU) | (P.max(1) < 0.5)] = NEU
        a, v = metrics(P, y, whole, inten), metrics(np.eye(7)[k], y, whole, inten)
        rep["ravdess"][str(w)] = dict(fused=a, voice=v)
        print(f"{w:4.1f} | {a['clips_macro_f1']:.3f} | {v['clips_macro_f1']:.3f} |", v["per_class_recall"], flush=True)

    # team clips, app-like
    S = np.load(real_check.CACHE, allow_pickle=True); pts = np.load(ROOT / "Face-Lab" / "cache" / "real_pts.npz")
    assert bool(pts["aligned"])
    jl = real_check.judged(S["clip"], S["t"], S["q"]); s96 = [cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in S["col"]]
    m = FaceFER(ROOT / "models" / "face-model-ravhead-v1"); fe = FaceEmotion.load(ROOT / "models" / "face-model")
    wins = []                                                          # (summary, blendshape probs dict, kind)
    for c in sorted(set(S["clip"])):
        idx = np.where(S["clip"] == c)[0]; idx = idx[np.argsort(S["t"][idx])]; hist = []
        fr = [m.frame_results(S["g"][i:i + 1], [S["q"][i]], [float(S["t"][i])], [s96[i]], hist)[0] for i in idx]
        for s in range(0, max(1, len(idx) - 3), 2):
            j = idx[s:s + 4]
            if len(j) < 3:
                continue
            lab = jl[j]
            kind = "neutral" if (lab == 6).all() else ("smile" if (lab == 3).any() else None)
            if kind is None:
                continue
            p = fe.proba(features(S["bs"][j], pts["pts"][j]))[0]
            wins.append((summary(m, fr[s:s + 4]), {e: float(p[i]) for i, e in enumerate(EMOTIONS)}, kind))
    nn = sum(k == "neutral" for *_, k in wins); ns = sum(k == "smile" for *_, k in wins)
    print(f"team windows: {nn} neutral, {ns} with a smile")
    print("w_fer | neutral windows voiced as an emotion | smile windows voiced happy")
    for w in W:
        out = [(speech_affect(sm, {"probs": bp}, w_fer=w)[0], k) for sm, bp, k in wins] if 0 < w else \
              [(speech_affect(None, {"probs": bp}, w_fer=0.0)[0], k) for sm, bp, k in wins]
        fa = [e for e, k in out if k == "neutral" and e is not None]
        hp = sum(e == "happy" for e, k in out if k == "smile")
        rep["team"][str(w)] = dict(neutral_windows=nn, voiced_emotion=len(fa), rate=round(len(fa) / max(nn, 1), 3),
                                   as_={e: fa.count(e) for e in sorted(set(fa))}, smile_windows=ns, voiced_happy=hp)
        print(f"{w:4.1f} | {len(fa)}/{nn} = {len(fa) / max(nn, 1):.1%} {rep['team'][str(w)]['as_']} | {hp}/{ns}", flush=True)
    (ROOT / "Face-Lab" / "results" / "voice_weight.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
