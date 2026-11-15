"""Can the picture head also win on FER2013? Retrain the head (same features, same fit_lr, L2 0.1) with FER2013 TRAIN added:
  A  FER2013 train only
  B  FER2013 train + RAVDESS + neutral signing (FER rows weighted 1)
  C  same as B, FER rows down-weighted so FER2013 and RAVDESS weigh the same in total
Scored on (1) the FER2013 test folder (never used to train or choose), (2) RAVDESS 24 actors, leave-actors-out (actors[i::6], FER train
always in the training side), (3) the team clips through the app path (check_ravhead.app_path), neutral prior 0.5 as deployed.
FER2013 images are 48 px: enlarged to 96 px for B2, a "window" of one image. Features cached in cache/fer_feats.npz.
  python fer_head.py → results/fer_head.json (heads not kept: none was used)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import FaceFER                     # noqa: E402
from modules.face_model import EMOTIONS                  # noqa: E402
from face_model import fit_lr                            # noqa: E402
from compare24 import metrics                            # noqa: E402
from data import CLASSES, load_raw                       # noqa: E402
from evaluate import classification                      # noqa: E402
import rav_head as R                                     # noqa: E402
from rav_prior import load_samples                       # noqa: E402
import check_ravhead as C                                # noqa: E402
import real_check                                        # noqa: E402

CACHE = ROOT / "Face-Lab" / "cache" / "fer_feats.npz"
PRIOR = 0.5
NEU = EMOTIONS.index("neutral")


def fer_feats():
    if CACHE.exists():
        z = np.load(CACHE); return {s: (z[f"{s}_f"].astype(np.float32), z[f"{s}_b2"], z[f"{s}_v2"], z[f"{s}_y"]) for s in ("train", "test")}
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.log_severity_level = 3
    b2 = ort.InferenceSession(str(R.B2F), so, providers=["CPUExecutionProvider"]); v2 = FaceFER(R.V2)
    raw, out = load_raw(), {}
    for s in ("train", "test"):
        X, y, _ = raw[s]; F, ZB = [], []
        for i in range(0, len(X), 512):
            f, zb = R._b2_run(b2, np.stack([cv2.resize(g, (96, 96), interpolation=cv2.INTER_CUBIC) for g in X[i:i + 512]]))
            F.append(f.astype(np.float16)); ZB.append(zb); print(s, i, flush=True)
        ZV = np.concatenate([v2.logits(X[i:i + 512]) for i in range(0, len(X), 512)]).astype(np.float32)
        out[s] = (np.concatenate(F), np.concatenate(ZB), ZV, y)
    np.savez_compressed(CACHE, **{f"{s}_{k}": v for s, t in out.items() for k, v in zip(("f", "b2", "v2", "y"), t)})
    return {s: (t[0].astype(np.float32), *t[1:]) for s, t in out.items()}


def main():
    XP, y, actor, whole, inten, meta, to7, e_idx = load_samples()
    Xd = R.sign_neutral(to7, e_idx); yd = np.full(len(Xd), NEU)
    lab = np.array([EMOTIONS.index(c) for c in CLASSES])
    ff = {}
    for s, (f, zb, zv, yy) in fer_feats().items():
        pb, pv = R.member_probs(zb, zv, to7, e_idx)
        ff[s] = (np.c_[f, pb, pb, pv, pv], lab[yy])
    (XF, yF), (XT, yT) = ff["train"], ff["test"]
    n_rav = len(XP) + len(Xd)
    variants = {"A": None, "B": 1.0, "C": n_rav / len(XF)}
    acts = np.unique(actor); folds = [acts[i::6] for i in range(6)]
    S = np.load(real_check.CACHE, allow_pickle=True); jl = real_check.judged(S["clip"], S["t"], S["q"])
    s96 = list(np.stack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in S["col"]]))
    m = FaceFER(ROOT / "models" / "face-model-ravhead-v1"); m.head.bias = np.array([PRIOR if c == "neutral" else 0.0 for c in m.classes])
    rep = dict(note=__doc__, n_fer_train=len(XF), n_fer_test=len(XT), n_ravdess=len(XP), n_signing=len(Xd), prior=PRIOR, variants={})

    yT_fer = np.array([CLASSES.index(EMOTIONS[k]) for k in yT]); to_c = [EMOTIONS.index(c) for c in CLASSES]

    def fer_test(head):
        L = head.logits(XT); L[:, NEU] += PRIOR
        r = classification(yT_fer, L[:, to_c])
        return dict(accuracy=r["accuracy"], macro_f1=r["macro_f1"], per_class_recall={c: r["per_class"][c]["recall"] for c in CLASSES})

    for name, wf in variants.items():
        print("== variant", name, flush=True)
        if wf is None:
            head = fit_lr(XF, yF, 0.1)
            L = head.logits(XP); L[:, NEU] += PRIOR
        else:
            L = np.zeros((len(XP), len(EMOTIONS)))
            for te in folds:
                k = np.isin(actor, te)
                Xtr = np.r_[XP[~k], Xd, XF]; ytr = np.r_[y[~k], yd, yF]; w = np.r_[np.ones((~k).sum() + len(Xd)), np.full(len(XF), wf)]
                L[k] = fit_lr(Xtr, ytr, 0.1, w=w).logits(XP[k]); print(" fold", flush=True)
            L[:, NEU] += PRIOR
            head = fit_lr(np.r_[XP, Xd, XF], np.r_[y, yd, yF], 0.1, w=np.r_[np.ones(n_rav), np.full(len(XF), wf)])
        rav = metrics(L, y, whole, inten)
        out = ROOT / "Face-Lab" / "candidates" / f"fer_head_{name}.npz"; head.save(out)
        z = np.load(out); m.head.mu, m.head.sd, m.head.W, m.head.b = (z[k].astype(np.float64) for k in ("mu", "sd", "W", "b"))
        team = C.app_path(m, S, s96, jl)
        rep["variants"][name] = r = dict(fer_weight=wf, fer2013_test=fer_test(head), ravdess=rav, team=team)
        print(name, "FER test F1", round(r["fer2013_test"]["macro_f1"], 3), "acc", round(r["fer2013_test"]["accuracy"], 3),
              "| RAVDESS F1", round(rav["clips_macro_f1"], 3), "| team wrong", team["wrong"], team["wrong_as"], "smiles", team["smiles"], flush=True)
        (ROOT / "Face-Lab" / "results" / "fer_head.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
