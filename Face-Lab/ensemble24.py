"""The deployed ensemble (face-model-ensemble-v2ferplus-b2, zero-shot) and ensemble + brow cue on ALL 24 RAVDESS actors, the same
samples / metrics as compare24.py and rav_head.py, from the cached member logits (cache/ravdess24_feats; identical to running the bundle:
v2 at T 0.7, B2 without contempt, + neutral prior 0.75, mean of the two).   python ensemble24.py → results/ensemble24.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import BrowCue, brow_features, softmax   # noqa: E402
from modules.face_model import EMOTIONS                        # noqa: E402
from compare24 import EMO, metrics, spans                      # noqa: E402
import rav_head as R                                           # noqa: E402

ENS = ROOT / "Face-Lab" / "candidates" / "face-model-ensemble-v2ferplus-b2"


def main():
    cfg = json.loads((ENS / "config.json").read_text(encoding="utf-8")); cls = cfg["classes"]
    bias = np.array([cfg["prior_shift"].get(c, 0.0) for c in cls]); to7 = [R.HSE.index(c) for c in cls]; to_e = [cls.index(e) for e in EMOTIONS]
    cue = BrowCue(cfg["brow"], cls); rng = np.random.RandomState(0)
    Pp, Pb, y, whole, inten = [], [], [], [], []
    for f in sorted(R.DATA.glob("actor_*.npz")):
        z = np.load(f); e = np.load(R.FEAT / f.name); fps = float(z["fps"])
        P = (softmax(e["v2"] / 0.7 + bias) + softmax(e["b2"][:, to7] + bias)) / 2
        face_all, t_all, cq, ct, cc = z["face"].astype(np.float32), z["t"], z["crop_q"], z["crop_t"], z["crop_clip"]
        Q = P.copy(); hist = []
        for ci in rng.permutation(len(z["name"])):
            a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1])
            for j in np.where(cc == ci)[0]:
                k = a0 + int(np.argmin(np.abs(t_all[a0:a1] - ct[j])))
                ft = brow_features(face_all[k] * [1280, 720]) if np.abs(face_all[k]).sum() > 0 else np.array([np.nan, np.nan])
                fr = [dict(probs={c: float(Q[j, i]) for i, c in enumerate(cls)}, quality=float(cq[j]))]
                cue.apply(fr, [ft], hist, lambda p, q: dict(probs={c: float(p[i]) for i, c in enumerate(cls)}, quality=q))
                Q[j] = [fr[0]["probs"][c] for c in cls]
        for ci in range(len(z["name"])):
            a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1]); tt = t_all[a0:a1]; lab = EMOTIONS.index(EMO[int(z["emotion"][ci])])
            for k, (s, en) in enumerate(spans(a1 - a0, fps)):
                sel = (cc == ci) & (cq >= 0.25) & (ct >= tt[s] - 1e-6) & (ct <= tt[min(en, len(tt)) - 1] + 1e-6)
                if not sel.any():
                    sel = (cc == ci) & (cq >= 0.25)
                if not sel.any():
                    sel = cc == ci
                Pp.append(P[sel].mean(0)[to_e]); Pb.append(Q[sel].mean(0)[to_e]); y.append(lab); whole.append(k == 0); inten.append(int(z["intensity"][ci]))
    Pp, Pb, y, whole, inten = map(np.array, (Pp, Pb, y, whole, inten))
    rep = dict(n_clips=int(whole.sum()), ensemble=metrics(Pp, y, whole, inten), ensemble_brow=metrics(Pb, y, whole, inten))
    for k in ("ensemble", "ensemble_brow"):
        print(k, rep[k])
    (ROOT / "Face-Lab" / "results" / "ensemble24.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
