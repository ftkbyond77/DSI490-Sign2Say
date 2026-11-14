"""Picture-model features + a RAVDESS-trained head, under the SAME protocol as the blendshape face model (scripts/face_model.py):
leave-actors-out CV, folds = actors[i::6], samples = every clip + its 2 s windows (hop 1 s), calm → neutral, class-balanced multinomial
LR (`fit_lr`, the same code), L2 chosen from the same kind of grid by the same rule (clips macro-F1 + windows macro-F1).

Per crop (cache/ravdess24 from ravdess24.py, quality ≥ 0.25): HSEmotion B2 penultimate features [1408] + its 8 logits, and our v2
48 px logits [7]. A sample = mean of its crops' B2 features + mean / max of the members' probabilities.
Variants:  blendshape (the friend's model, reproduced)  ·  picture head  ·  picture head + blendshape features (fusion)
  python rav_head.py            → results/rav_head.json   (features cached per actor in cache/ravdess24_feats/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import FaceFER, softmax, _IMNET_MEAN, _IMNET_STD   # noqa: E402
from modules.face_model import EMOTIONS, features                        # noqa: E402
from face_model import fit_lr                                             # noqa: E402
from compare24 import EMO, metrics, spans                                 # noqa: E402

DATA = ROOT / "Face-Lab" / "cache" / "ravdess24"
FEAT = ROOT / "Face-Lab" / "cache" / "ravdess24_feats"
V2 = ROOT / "Face-Lab" / "candidates" / "face-model-efficientnetb0-v2-ferplus"
B2F = ROOT / "data_test" / "hsemotion" / "enet_b2_8_feat.onnx"
HSE = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def extract(f, v2, b2):
    import cv2
    out = FEAT / f.name
    if out.exists():
        return np.load(out)
    z = np.load(f)
    G, G96 = z["crop_g"], z["crop_g96"]
    zv = v2.logits(G)
    feat, zb = _b2_run(b2, G96)
    np.savez_compressed(out, feat=feat.astype(np.float16), b2=zb, v2=zv.astype(np.float32))
    return np.load(out)


def _b2_run(b2, G96):
    """HSEmotion B2 on 96x96 gray crops → (penultimate features [N,1408], logits [N,8])."""
    import cv2
    F, Z = [], []
    for i in range(0, len(G96), 16):
        x = np.stack([cv2.resize(g, (260, 260), interpolation=cv2.INTER_CUBIC) for g in G96[i:i + 16]]).astype(np.float32) / 255.0
        x = ((np.repeat(x[..., None], 3, -1) - _IMNET_MEAN) / _IMNET_STD).transpose(0, 3, 1, 2)
        lo, fe = b2.run(None, {"input": np.ascontiguousarray(x)}); Z.append(lo); F.append(fe)
    return np.concatenate(F), np.concatenate(Z).astype(np.float32)


def member_probs(zb, zv, to7, e_idx):
    """B2 logits [N,8] + v2 logits [N,7] → their probabilities in EMOTIONS order (contempt dropped, v2 at T 0.7)."""
    pb = softmax(zb)[:, to7]; pb /= pb.sum(1, keepdims=True)
    return pb[:, e_idx], softmax(zv / 0.7)[:, e_idx]


def sign_neutral(to7, e_idx, fps=5.0, win_s=2.0, hop_s=1.0):
    """cache/sign_neutral.npz (sign_neutral.py) → neutral samples: every clip + its 2 s windows, as the RAVDESS samples."""
    p = ROOT / "Face-Lab" / "cache" / "sign_neutral.npz"
    if not p.exists():
        return np.zeros((0, 1408 + 28))
    z = np.load(p); pb, pv = member_probs(z["b2"], z["v2"], to7, e_idx); feat = z["feat"].astype(np.float32)
    X = []
    for c in np.unique(z["clip"]):
        idx = np.where((z["clip"] == c) & (z["q"] >= 0.25))[0]
        if not len(idx):
            continue
        for s, e in spans(len(idx), fps, win_s, hop_s):
            k = idx[s:e]
            X.append(picfeat(feat[k], pb[k], pv[k]))
    return np.array(X)


def picfeat(feat, pb, pv):
    return np.r_[feat.mean(0), pb.mean(0), pb.max(0), pv.mean(0), pv.max(0)]


def signers(head, v2, b2, to7, e_idx, win=1.0):
    """The 18 signer clips (real_check): per judged frame, a ±1 s window of that clip → the head. Neutral frames should stay neutral."""
    import cv2
    import real_check
    S = np.load(real_check.CACHE, allow_pickle=True)
    jl = real_check.judged(S["clip"], S["t"], S["q"])
    G96 = np.stack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in S["col"]])
    zv = v2.logits(S["g"])
    feat, zb = _b2_run(b2, G96)
    pb, pv = member_probs(zb, zv, to7, e_idx)
    pred = np.full(len(jl), -1)
    for i in np.where(jl >= 0)[0]:
        sel = (S["clip"] == S["clip"][i]) & (np.abs(S["t"] - S["t"][i]) <= win) & (S["q"] >= 0.25)
        if not sel.any():
            sel = np.arange(len(jl)) == i
        pred[i] = int(np.argmax(head.logits(picfeat(feat[sel], pb[sel], pv[sel])[None])[0]))
    n, h = jl == 6, jl == 3
    N, H = EMOTIONS.index("neutral"), EMOTIONS.index("happy")
    return dict(neutral_frames=int(n.sum()), shown_neutral=round(float((pred[n] == N).mean()), 3),
                false_emotion=round(float((pred[n] != N).mean()), 3), smiles=f"{int((pred[h] == H).sum())}/{int(h.sum())}",
                wrong_as={EMOTIONS[k]: int((pred[n] == k).sum()) for k in range(len(EMOTIONS)) if k != N and (pred[n] == k).any()})


def main():
    import onnxruntime as ort
    FEAT.mkdir(parents=True, exist_ok=True)
    v2 = FaceFER(V2); so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.log_severity_level = 3
    b2 = ort.InferenceSession(str(B2F), so, providers=["CPUExecutionProvider"])
    to7 = [HSE.index(c) for c in v2.classes]; e_idx = [v2.classes.index(e) for e in EMOTIONS]      # → EMOTIONS order
    XB, XP, y, actor, whole, inten = [], [], [], [], [], []
    for f in sorted(DATA.glob("actor_*.npz")):
        z = np.load(f); e = extract(f, v2, b2); fps = float(z["fps"]); a = int(z["actor"])
        pb, pv = member_probs(e["b2"], e["v2"], to7, e_idx)
        feat = e["feat"].astype(np.float32)
        bs_all, face_all, t_all = z["bs"].astype(np.float32), z["face"].astype(np.float32), z["t"]
        cq, ct, cc = z["crop_q"], z["crop_t"], z["crop_clip"]
        for ci in range(len(z["name"])):
            a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1])
            bs, face = bs_all[a0:a1], face_all[a0:a1].copy(); face[..., 0] *= 1280 / 720
            tt = t_all[a0:a1]; lab = EMOTIONS.index(EMO[int(z["emotion"][ci])])
            for k, (s, en) in enumerate(spans(a1 - a0, fps)):
                sel = (cc == ci) & (cq >= 0.25) & (ct >= tt[s] - 1e-6) & (ct <= tt[min(en, len(tt)) - 1] + 1e-6)
                if not sel.any():
                    sel = (cc == ci) & (cq >= 0.25)
                if not sel.any():
                    sel = cc == ci
                XB.append(features(bs[s:en], face[s:en]))
                XP.append(np.r_[feat[sel].mean(0), pb[sel].mean(0), pb[sel].max(0), pv[sel].mean(0), pv[sel].max(0)])
                y.append(lab); actor.append(a); whole.append(k == 0); inten.append(int(z["intensity"][ci]))
        print(f"actor {a:02d}", flush=True)
    XB, XP, y, actor, whole, inten = map(np.array, (XB, XP, y, actor, whole, inten))
    acts = np.unique(actor); folds = [acts[i::6] for i in range(6)]
    from face_model import macro_f1

    def cv(X, grid):
        best = None
        for l2 in grid:
            L = np.zeros((len(X), len(EMOTIONS)))
            for te in folds:
                m = np.isin(actor, te)
                L[m] = fit_lr(X[~m], y[~m], l2).logits(X[m])
            sc = macro_f1(y[whole], L[whole].argmax(1)) + macro_f1(y, L.argmax(1))
            print(f"   l2 {l2:g}: clips F1 {macro_f1(y[whole], L[whole].argmax(1)):.3f}", flush=True)
            if best is None or sc > best[0]:
                best = (sc, l2, L)
        return best[1], best[2]

    rep = dict(n_actors=len(acts), n_clips=int(whole.sum()))
    Xd = sign_neutral(to7, e_idx)
    print(f"neutral signing samples: {len(Xd)}", flush=True)

    def cv_dom(X, grid):                          # + the neutral signing samples in every training fold (never tested on)
        best = None
        for l2 in grid:
            L = np.zeros((len(X), len(EMOTIONS)))
            for te in folds:
                m = np.isin(actor, te)
                L[m] = fit_lr(np.r_[X[~m], Xd], np.r_[y[~m], np.full(len(Xd), EMOTIONS.index("neutral"))], l2).logits(X[m])
            sc = macro_f1(y[whole], L[whole].argmax(1)) + macro_f1(y, L.argmax(1))
            print(f"   l2 {l2:g}: clips F1 {macro_f1(y[whole], L[whole].argmax(1)):.3f}", flush=True)
            if best is None or sc > best[0]:
                best = (sc, l2, L)
        return best[1], best[2]

    if len(Xd):
        name = "picture_head + neutral signing"
        print(name, flush=True)
        l2, L = cv_dom(XP, (0.03, 0.1, 0.3))
        rep[name] = dict(l2=l2, n_neutral_signing=len(Xd), **metrics(L, y, whole, inten))
        print(" ", {k: v for k, v in rep[name].items() if k != "per_class_recall"}, "| recall", rep[name]["per_class_recall"], flush=True)
        head = fit_lr(np.r_[XP, Xd], np.r_[y, np.full(len(Xd), EMOTIONS.index("neutral"))], l2)
        head.save(ROOT / "Face-Lab" / "candidates" / "rav_head_signing.npz")
        rep["signers_with_neutral_signing"] = signers(head, v2, b2, to7, e_idx)
        print("  signers", rep["signers_with_neutral_signing"], flush=True)
    for name, X, grid in (("blendshape_lr (friend recipe)", XB, (0.01,)),
                          ("picture_head (B2 features + member probs)", XP, (0.03, 0.1, 0.3, 1.0)),
                          ("fusion (picture head + blendshape)", np.c_[XP, XB], (0.03, 0.1, 0.3, 1.0))):
        print(name, flush=True)
        l2, L = cv(X, grid)
        rep[name] = dict(l2=l2, **metrics(L, y, whole, inten))
        print(" ", {k: v for k, v in rep[name].items() if k != "per_class_recall"}, "\n   recall", rep[name]["per_class_recall"], flush=True)
        if name.startswith("picture_head"):
            head = fit_lr(X, y, l2)                                          # the deployable head: every RAVDESS actor
            head.save(ROOT / "Face-Lab" / "candidates" / "rav_head.npz")
            rep["signers"] = signers(head, v2, b2, to7, e_idx)
            print("  signers", rep["signers"], flush=True)
    (ROOT / "Face-Lab" / "results" / f"rav_head_{len(acts)}actors.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
