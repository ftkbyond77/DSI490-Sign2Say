"""What a neutral prior shift on the RAVDESS head costs on RAVDESS: the same CV as rav_head.py ('picture_head + neutral signing',
l2 0.1, folds actors[i::6]), out-of-fold logits + shift on neutral → the same metrics.   python rav_prior.py → results/rav_prior.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "Face-Lab"))
import rav_head as R                                       # noqa: E402
from modules.face_model import EMOTIONS                    # noqa: E402
from face_model import fit_lr                              # noqa: E402
from compare24 import EMO, metrics, spans                  # noqa: E402


def main():
    to7 = [R.HSE.index(c) for c in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]]
    cls = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]; e_idx = [cls.index(e) for e in EMOTIONS]
    XP, y, actor, whole, inten = [], [], [], [], []
    for f in sorted(R.DATA.glob("actor_*.npz")):
        z = np.load(f); e = np.load(R.FEAT / f.name); fps = float(z["fps"]); a = int(z["actor"])
        pb, pv = R.member_probs(e["b2"], e["v2"], to7, e_idx); feat = e["feat"].astype(np.float32)
        t_all, cq, ct, cc = z["t"], z["crop_q"], z["crop_t"], z["crop_clip"]
        for ci in range(len(z["name"])):
            a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1]); tt = t_all[a0:a1]; lab = EMOTIONS.index(EMO[int(z["emotion"][ci])])
            for k, (s, en) in enumerate(spans(a1 - a0, fps)):
                sel = (cc == ci) & (cq >= 0.25) & (ct >= tt[s] - 1e-6) & (ct <= tt[min(en, len(tt)) - 1] + 1e-6)
                if not sel.any():
                    sel = (cc == ci) & (cq >= 0.25)
                if not sel.any():
                    sel = cc == ci
                XP.append(R.picfeat(feat[sel], pb[sel], pv[sel])); y.append(lab); actor.append(a); whole.append(k == 0); inten.append(int(z["intensity"][ci]))
    XP, y, actor, whole, inten = map(np.array, (XP, y, actor, whole, inten))
    Xd = R.sign_neutral(to7, e_idx); acts = np.unique(actor); folds = [acts[i::6] for i in range(6)]
    L = np.zeros((len(XP), len(EMOTIONS)))
    for te in folds:
        m = np.isin(actor, te)
        L[m] = fit_lr(np.r_[XP[~m], Xd], np.r_[y[~m], np.full(len(Xd), 0)], 0.1).logits(XP[m]); print("fold", flush=True)
    rep = {}
    for ps in (0.0, 0.5, 1.0):
        Lb = L.copy(); Lb[:, 0] += ps
        rep[f"neutral_prior_{ps}"] = r = metrics(Lb, y, whole, inten); print(ps, {k: v for k, v in r.items() if k != "per_class_recall"}, r["per_class_recall"], flush=True)
    (ROOT / "Face-Lab" / "results" / "rav_prior.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
