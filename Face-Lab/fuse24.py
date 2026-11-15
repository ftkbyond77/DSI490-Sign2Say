"""The whole face system as the app combines it, on ALL 24 RAVDESS actors (same samples / metrics as rav_head.py):
   voice emotion = speech_affect(picture summary, blendshape affect)  — 0.5 · picture + 0.5 · blendshape probabilities,
   then neutral when the top class is neutral or < 0.5 (only a confident emotion is voiced).
 main  : picture = EfficientNet-B0 v1 (FER2013, prior 0.6)            + blendshape model (friend's recipe)
 ours  : picture = RAVDESS head (+ neutral signing, prior 0.5)       + the same blendshape model
Picture summary = the frames' probabilities weighted by top confidence × quality (FaceBranch._summary); the head's answer per sample is
its out-of-fold logits (cache/rav_oof.npz from rav_prior.py). Blendshape: out-of-fold LR (L2 0.01, temperature 1.6) on RAVDESS only — the
friend's 720 neutral signing samples are not on this machine (that recipe scores 0.592 here vs 0.567 shipped, so blendshape is a bit flattered).
  python fuse24.py → results/fuse24.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import FaceFER, softmax   # noqa: E402
from modules.face_model import EMOTIONS, features  # noqa: E402
from face_model import fit_lr                   # noqa: E402
from compare24 import EMO, metrics, spans       # noqa: E402
import rav_head as R                            # noqa: E402

B0 = ROOT / "models" / "face-model-efficientnetb0-v1"
NEU = EMOTIONS.index("neutral")


def voice(P, min_conf=0.5):
    """speech_affect's rule as a 7-way decision: neutral unless a non-neutral class is on top with ≥ min_conf"""
    k = P.argmax(1); k[(k == NEU) | (P.max(1) < min_conf)] = NEU
    return np.eye(len(EMOTIONS))[k]


def main():
    m = FaceFER(B0); cls = m.classes; to_e = [cls.index(e) for e in EMOTIONS]
    bias = m.T * np.array([float((m.cfg.get("prior_shift") or {}).get(c, 0.0)) for c in cls])
    XB, PB0, y, actor, whole, inten = [], [], [], [], [], []
    for f in sorted(R.DATA.glob("actor_*.npz")):
        z = np.load(f); fps = float(z["fps"]); a = int(z["actor"])
        Z = np.concatenate([m.logits(z["crop_g"][i:i + 256]) for i in range(0, len(z["crop_g"]), 256)])
        pf = softmax(Z + bias, m.T)[:, to_e]
        bs_all, face_all, t_all = z["bs"].astype(np.float32), z["face"].astype(np.float32), z["t"]
        cq, ct, cc = z["crop_q"], z["crop_t"], z["crop_clip"]
        for ci in range(len(z["name"])):
            a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1])
            bs, face = bs_all[a0:a1], face_all[a0:a1].copy(); face[..., 0] *= 1280 / 720
            tt = t_all[a0:a1]
            for k, (s, en) in enumerate(spans(a1 - a0, fps)):
                sel = (cc == ci) & (cq >= 0.25) & (ct >= tt[s] - 1e-6) & (ct <= tt[min(en, len(tt)) - 1] + 1e-6)
                if not sel.any():
                    sel = (cc == ci) & (cq >= 0.25)
                if not sel.any():
                    sel = cc == ci
                w = pf[sel].max(1) * np.maximum(cq[sel], 1e-3)
                PB0.append((pf[sel] * w[:, None]).sum(0) / w.sum())
                XB.append(features(bs[s:en], face[s:en]))
                y.append(EMOTIONS.index(EMO[int(z["emotion"][ci])])); actor.append(a); whole.append(k == 0); inten.append(int(z["intensity"][ci]))
        print(f.name, flush=True)
    XB, PB0, y, actor, whole, inten = map(np.array, (XB, PB0, y, actor, whole, inten))
    o = np.load(ROOT / "Face-Lab" / "cache" / "rav_oof.npz", allow_pickle=True)
    assert (o["y"] == y).all() and (o["actor"] == actor).all(), "sample order differs from rav_oof.npz"
    Lh = o["L"].copy(); Lh[:, NEU] += 0.5; PH = softmax(Lh)                       # ours, as deployed (prior 0.5)
    acts = np.unique(actor); LB = np.zeros((len(XB), len(EMOTIONS)))
    for te in (acts[i::6] for i in range(6)):
        mk = np.isin(actor, te); LB[mk] = fit_lr(XB[~mk], y[~mk], 0.01).logits(XB[mk])
    PBS = softmax(LB / 1.6)
    np.savez_compressed(ROOT / "Face-Lab" / "cache" / "fuse24_probs.npz", head=PH, blend=PBS, b0=PB0, y=y, whole=whole, inten=inten, actor=actor)
    rows = {"blendshape alone": PBS, "main: B0 v1 alone": PB0, "ours: head alone": PH,
            "main: B0 v1 + blendshape (0.5/0.5)": (PB0 + PBS) / 2, "ours: head + blendshape (0.5/0.5)": (PH + PBS) / 2}
    rep = dict(n_clips=int(whole.sum()), note=__doc__)
    for name, P in rows.items():
        rep[name] = r = metrics(P, y, whole, inten)
        print(f"{name:38s} F1 {r['clips_macro_f1']:.3f} acc {r['clips_acc']:.3f} n/e {r['neutral_vs_emotion_acc']:.3f}", r["per_class_recall"], flush=True)
        if "+" in name:
            rep[name + " → voice"] = v = metrics(voice(P), y, whole, inten)
            print(f"{'   → voice (neutral unless ≥ 0.5)':38s} F1 {v['clips_macro_f1']:.3f} acc {v['clips_acc']:.3f} n/e {v['neutral_vs_emotion_acc']:.3f}", v["per_class_recall"], flush=True)
    (ROOT / "Face-Lab" / "results" / "fuse24.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
