"""EfficientNet-B0 v1 (models/face-model-efficientnetb0-v1, FER2013, the bundle on main before Face v2) zero-shot on ALL 24 RAVDESS actors,
the same samples / metrics as compare24.py, ensemble24.py and rav_head.py (a sample = mean of its 48 px crops' probabilities).
Shown without a prior and with the shipped neutral prior (config prior_shift, logit bonus × temperature as FaceFER.probs).
  python b0v1_24.py → results/b0v1_24.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import FaceFER, softmax   # noqa: E402
from modules.face_model import EMOTIONS         # noqa: E402
from compare24 import EMO, metrics, spans       # noqa: E402
import rav_head as R                            # noqa: E402

B0 = ROOT / "models" / "face-model-efficientnetb0-v1"


def main():
    m = FaceFER(B0); cls = m.classes; to_e = [cls.index(e) for e in EMOTIONS]
    shipped = float((m.cfg.get("prior_shift") or {}).get("neutral", 0.0))
    priors = sorted({0.0, shipped})
    P = {p: [] for p in priors}; y, whole, inten = [], [], []
    for f in sorted(R.DATA.glob("actor_*.npz")):
        z = np.load(f); fps = float(z["fps"])
        Z = np.concatenate([m.logits(z["crop_g"][i:i + 256]) for i in range(0, len(z["crop_g"]), 256)])
        t_all, cq, ct, cc = z["t"], z["crop_q"], z["crop_t"], z["crop_clip"]
        prob = {p: softmax(Z + m.T * np.array([p if c == "neutral" else 0.0 for c in cls]), m.T) for p in priors}
        for ci in range(len(z["name"])):
            a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1]); tt = t_all[a0:a1]
            for k, (s, en) in enumerate(spans(a1 - a0, fps)):
                sel = (cc == ci) & (cq >= 0.25) & (ct >= tt[s] - 1e-6) & (ct <= tt[min(en, len(tt)) - 1] + 1e-6)
                if not sel.any():
                    sel = (cc == ci) & (cq >= 0.25)
                if not sel.any():
                    sel = cc == ci
                for p in priors:
                    P[p].append(prob[p][sel].mean(0)[to_e])
                y.append(EMOTIONS.index(EMO[int(z["emotion"][ci])])); whole.append(k == 0); inten.append(int(z["intensity"][ci]))
        print(f.name, flush=True)
    y, whole, inten = map(np.array, (y, whole, inten))
    rep = dict(bundle=B0.name, temperature=m.T, n_clips=int(whole.sum()))
    for p in priors:
        rep[f"prior {p}"] = r = metrics(np.array(P[p]), y, whole, inten)
        print(f"prior {p}: F1 {r['clips_macro_f1']:.3f} acc {r['clips_acc']:.3f}", r["per_class_recall"], flush=True)
    (ROOT / "Face-Lab" / "results" / "b0v1_24.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
