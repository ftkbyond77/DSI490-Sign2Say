"""Picture face model (Face-Lab ensemble + brow cue) vs the blendshape face model (models/face-model, scripts/face_model.py) on
RAVDESS, all 24 actors, SAME frames (cache/ravdess24 from ravdess24.py) and the SAME protocol / metrics as scripts/face_model.py:
  * labels from the file name, calm folded into neutral; samples = every clip + its 2 s windows (hop 1 s)
  * blendshape model: its recipe (features → class-balanced multinomial LR, L2 0.01, temperature on out-of-fold logits) in
    leave-actors-out CV, folds = actors[i::6] (6 folds × 4 actors). Without the 720 in-domain neutral signing samples of the shipped
    model (not on this machine) = the report's "RAVDESS only" column (acc 0.592), used as the reproduction check.
  * picture model: not trained on RAVDESS, so every actor is unseen; a clip / window = the mean of its crops' probabilities (quality ≥ 0.25);
    the brow cue runs as live: per actor, clips in a random order, the baseline from that actor's earlier frames.
Metrics: clips acc / macro-F1 (mean over the 7 classes), windows acc / macro-F1, strong-intensity acc, neutral-vs-emotion acc, valence agreement.
  python compare24.py   → results/compare24.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from modules.face_fer import BrowCue, FaceFER, brow_features   # noqa: E402
from modules.face_model import EMOTIONS, VA, features          # noqa: E402
from face_model import fit_lr, macro_f1                         # noqa: E402

DATA = ROOT / "Face-Lab" / "cache" / "ravdess24"
ENS = ROOT / "Face-Lab" / "candidates" / "face-model-ensemble-v2ferplus-b2"
EMO = {1: "neutral", 2: "neutral", 3: "happy", 4: "sad", 5: "angry", 6: "fear", 7: "disgust", 8: "surprise"}   # calm → neutral


def spans(n, fps, win_s=2.0, hop_s=1.0):
    w, h = int(win_s * fps), int(hop_s * fps)
    return [(0, n)] + [(s, s + w) for s in range(0, max(1, n - w) + 1, h)]


def metrics(L, y, whole, inten):
    pw, pa = L[whole].argmax(1), L.argmax(1); strong = whole & (inten == 2)
    val = lambda k: np.sign([VA[EMOTIONS[i]][0] for i in k])
    return dict(clips_acc=round(float(np.mean(pw == y[whole])), 3), clips_macro_f1=round(macro_f1(y[whole], pw), 3),
                windows_acc=round(float(np.mean(pa == y)), 3), windows_macro_f1=round(macro_f1(y, pa), 3),
                strong_intensity_acc=round(float(np.mean(L[strong].argmax(1) == y[strong])), 3),
                neutral_vs_emotion_acc=round(float(np.mean((pw == 0) == (y[whole] == 0))), 3),
                valence_sign_agreement=round(float(np.mean(val(pw) == val(y[whole]))), 3),
                per_class_recall={EMOTIONS[k]: round(float(np.mean(pw[y[whole] == k] == k)), 3) for k in range(len(EMOTIONS))})


def main():
    files = sorted(DATA.glob("actor_*.npz"))
    m = FaceFER(ENS); cls = m.classes; to_e = np.array([cls.index(e) for e in EMOTIONS])      # picture-model columns in EMOTIONS order
    cue = BrowCue(m.cfg["brow"], cls)
    rng = np.random.RandomState(0)
    X, y, actor, whole, inten = [], [], [], [], []
    Pp, Pb = [], []                                                    # picture model per sample: without / with the brow cue
    for f in files:
        z = np.load(f); fps = float(z["fps"]); a = int(z["actor"])
        bs_all, face_all, t_all = z["bs"].astype(np.float32), z["face"].astype(np.float32), z["t"]
        cg, cg96, cq, ct, cc = z["crop_g"], z["crop_g96"], z["crop_q"], z["crop_t"], z["crop_clip"]
        P = m.probs(cg, list(cg96))
        # brow cue, live-style per actor
        Q = P.copy(); order = rng.permutation(len(z["name"])); hist = []
        for ci in order:
            a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1])
            for j in np.where(cc == ci)[0]:
                k = a0 + int(np.argmin(np.abs(t_all[a0:a1] - ct[j])))
                feat = brow_features(face_all[k] * [1280, 720]) if np.abs(face_all[k]).sum() > 0 else np.array([np.nan, np.nan])
                fr = [dict(probs={c: float(Q[j, i]) for i, c in enumerate(cls)}, quality=float(cq[j]))]
                cue.apply(fr, [feat], hist, lambda p, q: dict(probs={c: float(p[i]) for i, c in enumerate(cls)}, quality=q))
                Q[j] = [fr[0]["probs"][c] for c in cls]
        for ci in range(len(z["name"])):
            a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1])
            bs, face = bs_all[a0:a1], face_all[a0:a1].copy(); face[..., 0] *= 1280 / 720
            lab = EMOTIONS.index(EMO[int(z["emotion"][ci])])
            tt = t_all[a0:a1]
            for k, (s, e) in enumerate(spans(a1 - a0, fps)):
                X.append(features(bs[s:e], face[s:e])); y.append(lab); actor.append(a); whole.append(k == 0); inten.append(int(z["intensity"][ci]))
                sel = (cc == ci) & (cq >= 0.25) & (ct >= tt[s] - 1e-6) & (ct <= tt[min(e, len(tt)) - 1] + 1e-6)
                Pp.append(P[sel].mean(0)[to_e] if sel.any() else np.full(len(EMOTIONS), np.nan))
                Pb.append(Q[sel].mean(0)[to_e] if sel.any() else np.full(len(EMOTIONS), np.nan))
        print(f"actor {a:02d} done", flush=True)
    X, y, actor, whole, inten = np.array(X), np.array(y), np.array(actor), np.array(whole), np.array(inten)
    Pp, Pb = np.array(Pp), np.array(Pb)
    acts = np.unique(actor); folds = [acts[i::6] for i in range(6)]
    L = np.zeros((len(X), len(EMOTIONS)))
    for te in folds:
        mm = np.isin(actor, te)
        L[mm] = fit_lr(X[~mm], y[~mm], 0.01).logits(X[mm])
    ok = ~np.isnan(Pp).any(1)                                       # samples where the picture model saw a face (the same set for all rows)
    np.savez_compressed(ROOT / "Face-Lab" / "cache" / f"compare24_{len(acts)}actors.npz", X=X, y=y, actor=actor, whole=whole, inten=inten, Pp=Pp, Pb=Pb, L=L)
    # fusion (stacking): the blendshape features + the picture model's log-probabilities, same recipe and the same folds
    fus = {}
    for name, P in (("fusion_blendshape_picture", Pp), ("fusion_blendshape_picture_brow", Pb)):
        Xf = np.c_[X[ok], np.log(np.clip(P[ok], 1e-4, 1))]; Lf = np.zeros((ok.sum(), len(EMOTIONS)))
        for te in folds:
            mm = np.isin(actor[ok], te)
            Lf[mm] = fit_lr(Xf[~mm], y[ok][~mm], 0.01).logits(Xf[mm])
        fus[name] = metrics(Lf, y[ok], whole[ok], inten[ok])
    rep = dict(**fus,n_actors=len(acts), n_clips=int(whole.sum()), n_samples=int(len(X)), picture_coverage=round(float(ok.mean()), 4),
               blendshape_lr_ravdess_only=metrics(L, y, whole, inten),
               blendshape_lr_ravdess_only_same_samples=metrics(L[ok], y[ok], whole[ok], inten[ok]),
               picture_ensemble=metrics(Pp[ok], y[ok], whole[ok], inten[ok]),
               picture_ensemble_brow=metrics(Pb[ok], y[ok], whole[ok], inten[ok]),
               reference_shipped_blendshape_model=json.loads((ROOT / "models/face-model/model_card.json").read_text(encoding="utf-8"))["cv"])
    rep["reference_shipped_blendshape_model"].pop("confusion", None)
    (ROOT / "Face-Lab" / "results" / f"compare24_{len(acts)}actors.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    for k in ("blendshape_lr_ravdess_only", "blendshape_lr_ravdess_only_same_samples", "picture_ensemble", "picture_ensemble_brow", *fus):
        print(k, {kk: v for kk, v in rep[k].items() if kk != "per_class_recall"})
        print("   recall", rep[k]["per_class_recall"])


if __name__ == "__main__":
    main()
