"""Round 8 evaluation on the laptop: the out-of-fold outputs of ft_rav.py (Colab) → the SAME samples / metrics as rav_head.py, against the
current head (results/rav_prior.json, 0.723 at prior 0.5) and the blendshape face model (0.591); then the "all" model (best.onnx) as a
bundle on the team clips through the app path.
  python rav_ft_eval.py --dir runs/rav_ft      → results/rav_ft_eval.json  (+ candidates/face-model-ravft-v1/ when best.onnx is there)
Variants (each with a neutral prior grid, applied as a logit bonus like the deployed models):
  ft            fine-tuned B2 alone, a sample = mean of its crops' probabilities
  ft+v2         fine-tuned B2 + our v2 48 px member (mean of the two), as the deployed ensemble
  ft_head       the LR head on the fine-tuned features (out-of-fold features; note: each fold's feature net saw the other folds' actors)"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent; ROOT = H.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(H))
from modules.face_fer import FaceFER, softmax       # noqa: E402
from modules.face_model import EMOTIONS             # noqa: E402
from face_model import fit_lr                       # noqa: E402
from compare24 import metrics, spans                # noqa: E402
import rav_head as R                                # noqa: E402

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
TO7 = [R.HSE.index(c) for c in CLASSES]; TO_E = [CLASSES.index(e) for e in EMOTIONS]
PRIORS = (0.0, 0.5, 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(H / "runs" / "rav_ft")); a = ap.parse_args()
    d = Path(a.dir)
    pk = np.load(H / "cache" / "rav_ft_pack.npz")
    N = len(pk["label"]); Z = np.full((N, 8), np.nan, np.float32); Fe = np.zeros((N, 1408), np.float32)
    for f in sorted(d.glob("fold_*.npz")):
        z = np.load(f); Z[z["idx"]] = z["logits"]; Fe[z["idx"]] = z["feat"]
    rav = pk["actor"] > 0
    assert not np.isnan(Z[rav]).any(), "a fold is missing"
    # the pack's RAVDESS crops are in the same order as cache/ravdess24 (actor files sorted, crops in order) → the v2 logits from the feature cache
    zv = np.concatenate([np.load(R.FEAT / f.name)["v2"] for f in sorted(R.DATA.glob("actor_*.npz"))])
    assert len(zv) == rav.sum()
    off = 0; S = []                              # samples: (crop indices into the pack, label, actor, whole, intensity)
    for f in sorted(R.DATA.glob("actor_*.npz")):
        z = np.load(f); fps = float(z["fps"]); t_all, cq, ct, cc = z["t"], z["crop_q"], z["crop_t"], z["crop_clip"]
        from compare24 import EMO
        for ci in range(len(z["name"])):
            a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1]); tt = t_all[a0:a1]
            for k, (s, en) in enumerate(spans(a1 - a0, fps)):
                sel = (cc == ci) & (cq >= 0.25) & (ct >= tt[s] - 1e-6) & (ct <= tt[min(en, len(tt)) - 1] + 1e-6)
                if not sel.any():
                    sel = (cc == ci) & (cq >= 0.25)
                if not sel.any():
                    sel = cc == ci
                S.append((off + np.where(sel)[0], EMOTIONS.index(EMO[int(z["emotion"][ci])]), int(z["actor"]), k == 0, int(z["intensity"][ci])))
        off += len(cq)
    y, actor, whole, inten = (np.array([s[i] for s in S]) for i in (1, 2, 3, 4))
    rep = {}
    for p in PRIORS:
        b = np.array([p if c == "neutral" else 0.0 for c in CLASSES])
        pf = softmax(Z[:, TO7] + b); pv = softmax(zv / 0.7 + b)
        P1 = np.array([pf[i].mean(0)[TO_E] for i, *_ in S])
        P2 = np.array([((pf[i] + pv[i]) / 2).mean(0)[TO_E] for i, *_ in S])
        rep[f"ft prior {p}"] = metrics(P1, y, whole, inten); rep[f"ft+v2 prior {p}"] = metrics(P2, y, whole, inten)
    # head on the out-of-fold features (same picfeat recipe, our v2 probabilities at T 0.7, B2 → the fine-tuned one)
    pb = softmax(Z[:, TO7])[:, TO_E]; pv0 = softmax(zv / 0.7)[:, TO_E]
    X = np.array([R.picfeat(Fe[i], pb[i], pv0[i]) for i, *_ in S])
    acts = np.unique(actor); folds = [acts[i::6] for i in range(6)]; L = np.zeros((len(X), 7))
    for te in folds:
        m = np.isin(actor, te); L[m] = fit_lr(X[~m], y[~m], 0.1).logits(X[m])
    for p in PRIORS:
        Lb = L.copy(); Lb[:, 0] += p; rep[f"ft_head prior {p}"] = metrics(Lb, y, whole, inten)
    for k, v in rep.items():
        print(f"{k:22s} F1 {v['clips_macro_f1']:.3f} acc {v['clips_acc']:.3f} | {v['per_class_recall']}", flush=True)
    # the all-actor model as a bundle: v2 + fine-tuned B2 (no head), then the team clips (app path)
    best = d / "best.onnx"
    if best.exists():
        import check_ravhead as C, real_check, cv2
        out = H / "candidates" / "face-model-ravft-v1"; out.mkdir(parents=True, exist_ok=True)
        src = ROOT / "models" / "face-model-ravhead-v1"
        shutil.copy(src / "face_fer.onnx", out); shutil.copy(best, out / "enet_b2_8_ravft.onnx")
        cfg = json.loads((src / "config.json").read_text(encoding="utf-8")); cfg.pop("head", None); cfg.pop("prior_note", None)
        cfg["name"] = "face-model-ravft-v1"; cfg["members"][1]["onnx"] = "enet_b2_8_ravft.onnx"
        cfg["members"][1]["note"] = "HSEmotion B2 fine-tuned on RAVDESS 24 actors + neutral dictionary signing (ft_rav.py, round 8)"
        S2 = np.load(real_check.CACHE, allow_pickle=True); jl = real_check.judged(S2["clip"], S2["t"], S2["q"])
        s96 = list(np.stack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in S2["col"]]))
        for p in PRIORS:
            cfg["prior_shift"] = {"neutral": p}; (out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
            rep[f"team ft+v2 prior {p}"] = r = C.app_path(FaceFER(out), S2, s96, jl); print("team prior", p, r, flush=True)
    (H / "results" / "rav_ft_eval.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
