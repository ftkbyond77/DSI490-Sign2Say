"""Head-only fine-tune of HSEmotion EfficientNet-B2 (AffectNet) on FER+ soft labels, on the laptop CPU (no GPU needed).

The B2 backbone stays frozen: its pooled 1408-d features are extracted once with onnxruntime (an extra graph output in front of the last
Gemm, cached in cache/b2_feats.npz), then only the final linear layer (8 outputs, started from the published weights) is retrained:
  loss = soft-label CE on our 7 classes (FER+ votes)  +  kd * KL(original B2 8-class output || new)  +  l2 * ||W - W0||^2
Inputs as at deployment: gray crop -> bicubic to 260 -> 3 channels -> ImageNet normalisation (FER2013 48 px; RAVDESS / signers 96 px).
Selection: FER+ val Macro-F1 + 0.5 x RAVDESS actors 1-2 clip Macro-F1; actors 3-4 + signer clips = test (results/b2_head.json).

  python b2_head.py         (base anaconda env: onnx, onnxruntime, cv2, torch for data loading; the head is trained in numpy)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Face-Lab"))
import build_ferplus as fp   # noqa: E402
import data                  # noqa: E402
import real_check            # noqa: E402
from video_eval import scores  # noqa: E402

SRC = ROOT / "data_test" / "hsemotion" / "enet_b2_8.onnx"
FEAT_ONNX = ROOT / "data_test" / "hsemotion" / "enet_b2_8_feat.onnx"
CACHE = ROOT / "Face-Lab" / "cache" / "b2_feats.npz"
HSE_ORDER = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TO7 = np.array([HSE_ORDER.index(c) for c in data.CLASSES])
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
NU = data.CLASSES.index("neutral")


def feat_model():
    m = onnx.load(str(SRC))
    gemm = [n for n in m.graph.node if n.op_type == "Gemm"][-1]
    m.graph.output.append(onnx.helper.make_tensor_value_info(gemm.input[0], onnx.TensorProto.FLOAT, None))
    onnx.save(m, str(FEAT_ONNX))
    W = next(onnx.numpy_helper.to_array(i) for i in m.graph.initializer if i.name == gemm.input[1])
    b = next(onnx.numpy_helper.to_array(i) for i in m.graph.initializer if i.name == gemm.input[2])
    return W.astype(np.float32), b.astype(np.float32)                   # [8,1408], [8]


def extract(sess, X, size=260, bs=16):
    """uint8 gray [N,H,W] -> (features [N,1408], original logits [N,8])."""
    F, Z = [], []
    for i in range(0, len(X), bs):
        x = np.stack([cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC) for g in X[i:i + bs]]).astype(np.float32) / 255.0
        x = ((np.repeat(x[..., None], 3, -1) - MEAN) / STD).transpose(0, 3, 1, 2)
        z, f = sess.run(None, {"input": np.ascontiguousarray(x)})
        F.append(f.reshape(len(f), -1)); Z.append(z)
        if i % 3200 == 0:
            print(f"  {i}/{len(X)}", flush=True)
    return np.concatenate(F), np.concatenate(Z)


def gray96(col):
    return np.stack([cv2.resize(cv2.cvtColor(c, cv2.COLOR_RGB2GRAY), (96, 96), interpolation=cv2.INTER_AREA) for c in col])


def build():
    W, b = feat_model()
    o = ort.SessionOptions(); o.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(FEAT_ONNX), o, providers=["CPUExecutionProvider"])
    D = data.prepare(42)
    R = np.load(ROOT / "Face-Lab" / "cache" / "ravdess_faces.npz"); okr = R["q"] >= 0.25
    S = np.load(real_check.CACHE, allow_pickle=True)
    out = dict(W0=W, b0=b)
    for k, X in (("val", D["val"][0]), ("rav", gray96(R["col"][okr])), ("sign", gray96(S["col"])), ("train", D["train"][0])):
        print(k, len(X), flush=True)
        out[f"{k}_f"], out[f"{k}_z"] = extract(sess, X)
    np.savez(CACHE, **out, rav_clip=R["clip"][okr], rav_label=R["label"][okr], rav_actor=R["actor"][okr])


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def train_head(C, t, kd, l2, epochs=30, lr=0.05, seed=0):
    """full-batch-ish Adam on the linear head (numpy). t = soft targets [N,7] over our classes."""
    rng = np.random.default_rng(seed)
    W, b = C["W0"].copy(), C["b0"].copy(); X = C["train_f"]; pt = softmax(C["train_z"] / 2.0)
    mW, vW, mb, vb = 0, 0, 0, 0; step = 0
    for ep in range(epochs):
        for idx in np.array_split(rng.permutation(len(X)), 50):
            x = X[idx]; z = x @ W.T + b
            g = np.zeros_like(z)
            p7 = softmax(z[:, TO7]); g[:, TO7] += (p7 - t[idx])                       # d CE / d z (7 classes)
            if kd > 0:
                g += kd * 2.0 * (softmax(z / 2.0) - pt[idx])                          # d (T^2 KL) / d z, T = 2
            g /= len(idx)
            gW = g.T @ x + 2 * l2 * (W - C["W0"]); gb = g.sum(0)
            step += 1
            mW = 0.9 * mW + 0.1 * gW; vW = 0.999 * vW + 0.001 * gW ** 2
            mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb ** 2
            c1, c2 = 1 - 0.9 ** step, 1 - 0.999 ** step
            W -= lr * 0.01 * (mW / c1) / (np.sqrt(vW / c2) + 1e-8)
            b -= lr * 0.01 * (mb / c1) / (np.sqrt(vb / c2) + 1e-8)
    return W, b


def main():
    if not CACHE.exists():
        build()
    C = dict(np.load(CACHE))
    D = data.prepare(42); raw = data.load_raw()
    vt, ok_all = fp.soft_targets(fp.load({"train": raw["train"][2]})["train"])
    ytr = D["train"][1]
    t = np.eye(7, dtype=np.float32)[ytr]; ok = ok_all[D["train_idx"]]; t[ok] = vt[D["train_idx"]][ok]
    ok_va = ok_all[D["val_idx"]]; yva = vt[D["val_idx"]].argmax(1)
    clip, lab, act = C["rav_clip"], C["rav_label"], C["rav_actor"]
    S = np.load(real_check.CACHE, allow_pickle=True); jl = real_check.judged(S["clip"], S["t"], S["q"]); n, h = jl == NU, jl == data.CLASSES.index("happy")

    def rav(z7, actors, pr=0.0):
        m = np.isin(act, actors); P = softmax(z7[m] + np.where(np.arange(7) == NU, pr, 0)); cl = clip[m]
        cs = sorted(set(cl)); cp = np.stack([P[cl == c].mean(0) for c in cs]).argmax(1); cy = np.array([lab[m][cl == c][0] for c in cs])
        return scores(cy, cp)

    def ev(W, b):
        zv = (C["val_f"] @ W.T + b)[:, TO7]; f1 = scores(yva[ok_va], zv[ok_va].argmax(1))["macro_f1"]
        zr = (C["rav_f"] @ W.T + b)[:, TO7]; zs = (C["sign_f"] @ W.T + b)[:, TO7]
        return f1, rav(zr, [1, 2])["macro_f1"], zr, zs

    res = dict(note=__doc__.split("\n\n")[0], runs={})
    f1, rf, zr0, zs0 = ev(C["W0"], C["b0"])
    print(f"original B2: FER+ val F1 {f1:.4f}  RAV-dev F1 {rf:.3f}", flush=True)
    cands = {"original": (C["W0"], C["b0"], f1, rf)}
    for kd in (0.0, 0.5, 2.0):
        for l2 in (0.0, 1e-3):
            for ep in (5, 15):
                W, b = train_head(C, t, kd, l2, epochs=ep)
                f1, rf, _, _ = ev(W, b)
                print(f"kd {kd} l2 {l2} ep {ep}: FER+ val F1 {f1:.4f}  RAV-dev F1 {rf:.3f}  score {f1 + 0.5 * rf:.4f}", flush=True)
                cands[f"kd{kd}_l2{l2}_ep{ep}"] = (W, b, f1, rf)
    best = max((k for k in cands if k != "original"), key=lambda k: cands[k][2] + 0.5 * cands[k][3])
    print("selected:", best)
    np.savez(ROOT / "Face-Lab" / "candidates" / "b2_head_best.npz", W=cands[best][0], b=cands[best][1])
    for k in ("original", best):
        W, b = cands[k][:2]; _, _, zr, zs = ev(W, b)
        res["runs"][k] = dict(fer_plus_val_f1=cands[k][2], rav_dev_f1=cands[k][3], test={})
        for pr in (0.0, 0.5, 1.0, 1.5):
            s = rav(zr, [3, 4], pr); ps = softmax(zs + np.where(np.arange(7) == NU, pr, 0)).argmax(1)
            row = dict(rav_test_f1=s["macro_f1"], expr_recall=s["expression_recall"], neutral_kept=s["neutral_kept"],
                       signer_neutral=round(float((ps[n] == NU).mean()), 3), smiles=f"{int((ps[h] == 3).sum())}/{int(h.sum())}")
            res["runs"][k]["test"][f"prior_{pr}"] = row
            print(f"  {k:22s} p{pr}: RAV a3-4 F1 {row['rav_test_f1']:.3f} exprRec {row['expr_recall']:.3f} neuKept {row['neutral_kept']} | signers neutral {row['signer_neutral']:.3f} smiles {row['smiles']}")
        np.save(Path(r"C:/Users/ppich/AppData/Local/Temp/claude/5217bdba97ed1d7b/6d4dfe47-d5d4-4a83-adc3-4d62bb20a2b4/scratchpad") / f"b2head_{k}_zr.npy", zr)
        np.save(Path(r"C:/Users/ppich/AppData/Local/Temp/claude/5217bdba97ed1d7b/6d4dfe47-d5d4-4a83-adc3-4d62bb20a2b4/scratchpad") / f"b2head_{k}_zs.npy", zs)
    (ROOT / "Face-Lab" / "results" / "b2_head.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
