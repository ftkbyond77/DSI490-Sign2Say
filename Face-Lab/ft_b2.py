"""Fine-tune HSEmotion EfficientNet-B2 (AffectNet, Savchenko 2022) on FER+ soft labels, on Colab.

Why: the off-the-shelf B2 already beats our FER2013 models on real video (RAVDESS), but it over-reacts on signers. Starting from its
AffectNet weights and showing it FER+ (whose crowd labels are calmer) may keep the real-face skill and add the FER+ judgement of neutral.
Forgetting is held back with a small learning rate, few epochs and (optionally) distillation to the original B2's 8-class output.

The trainable model comes from the published ONNX file (onnx2torch), so no old timm pickle is needed. Its input = RGB 260x260, ImageNet
normalisation; our gray crops are copied to 3 channels (gray 96 px crops scored as well as colour in hse_eval.py).
Checkpoint = best FER+ val Macro-F1 + --sel-w * RAVDESS-dev (actors 1-2) clip Macro-F1; actors 3-4 stay unseen (tested on the laptop).

  python ft_b2.py --tag b2_ft_kd1 --epochs 4 --lr 2e-5 --kd 1.0
  -> runs/<tag>/best.onnx (+ results/<tag>.json)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import build_ferplus as fp
import data

ROOT = Path(__file__).resolve().parent
HSE_ORDER = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TO7 = [HSE_ORDER.index(c) for c in data.CLASSES]                      # our 7 classes inside the 8 outputs (contempt left out)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NU = data.CLASSES.index("neutral")


def to_input(y, dev):
    """[N,1,R,R] gray in 0..1 -> normalised [N,3,R,R]."""
    return (y.expand(-1, 3, -1, -1) - MEAN.to(dev)) / STD.to(dev)


def gray_batches(X, res, dev, bs=128):
    """uint8 [N,H,W] -> normalised [b,3,res,res] batches (bicubic resize, like the laptop/server path)."""
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(dev).float().unsqueeze(1) / 255.0
        x = F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False).clamp(0, 1)
        yield to_input(x, dev)


@torch.no_grad()
def logits(net, X, res, dev):
    net.eval()
    out = []
    for x in gray_batches(X, res, dev):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(net(x).float().cpu())
    return torch.cat(out).numpy()


def macro_f1(y, p, classes=range(7)):
    f = []
    for c in classes:
        if (y == c).sum() == 0:
            continue
        tp, fp_, fn = ((p == c) & (y == c)).sum(), ((p == c) & (y != c)).sum(), ((p != c) & (y == c)).sum()
        f.append(2 * tp / (2 * tp + fp_ + fn) if tp else 0.0)
    return float(np.mean(f))


def rav_scores(z7, rav):
    P = np.exp(z7 - z7.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    clips = sorted(set(rav["clip"]))
    cp = np.stack([P[rav["clip"] == c].mean(0) for c in clips]).argmax(1)
    cy = np.array([rav["label"][rav["clip"] == c][0] for c in clips])
    return dict(rav_dev_macro_f1=round(macro_f1(cy, cp), 4), rav_dev_expr_recall=round(float((cp[cy != NU] != NU).mean()), 4),
                rav_dev_neutral_kept=round(float((cp[cy == NU] == NU).mean()), 4), rav_dev_acc=round(float((cp == cy).mean()), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "hse" / "enet_b2_8.onnx"))
    ap.add_argument("--res", type=int, default=260)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5, help="backbone LR (the last linear layer gets 10x)")
    ap.add_argument("--head-only", action="store_true", help="train only the last linear layer")
    ap.add_argument("--kd", type=float, default=1.0, help="weight of KL(original B2 || student) on all 8 outputs, temperature 2; 0 = off")
    ap.add_argument("--neutral-down", type=float, default=1.0)
    ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--sel-w", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    import onnx
    import onnx2torch
    import onnxruntime as ort

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    net = onnx2torch.convert(onnx.load(a.onnx)).to(dev)
    # conversion check against onnxruntime on 2 random inputs
    xc = np.random.randn(2, 3, a.res, a.res).astype(np.float32)
    ref = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"]).run(None, {"input": xc})[0]
    net.eval()
    with torch.no_grad():
        diff = float(np.abs(net(torch.from_numpy(xc).to(dev)).cpu().numpy() - ref).max())
    print(f"onnx2torch max |diff| = {diff:.2e}", flush=True)
    assert diff < 1e-2, "conversion does not match the ONNX model"
    teacher = copy.deepcopy(net).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    lin = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    head = lin[-1] if lin else None
    print("last linear:", head, flush=True)

    D = data.prepare(42)
    (Xtr, ytr), (Xva, yva) = D["train"], D["val"]
    raw = data.load_raw()
    vt, ok_all = fp.soft_targets(fp.load({"train": raw["train"][2]})["train"])
    t = np.eye(7, dtype=np.float32)[ytr]
    ok = ok_all[D["train_idx"]]; t[ok] = vt[D["train_idx"]][ok]
    if a.neutral_down != 1.0:
        t[:, NU] *= a.neutral_down; t /= t.sum(1, keepdims=True)
    ok_va = ok_all[D["val_idx"]]; yva_fp = vt[D["val_idx"]].argmax(1)
    R = np.load(ROOT / "cache" / "rav_dev96.npz")
    import cv2                                                         # crops stored as JPEG q95 (upload size limit)
    rav = dict(g96=np.stack([cv2.imdecode(R["jpg"][R["off"][i]:R["off"][i + 1]], cv2.IMREAD_GRAYSCALE) for i in range(len(R["off"]) - 1)]), clip=R["clip"], label=R["label"])
    xtr = torch.from_numpy(Xtr).to(dev).float().unsqueeze(1); ttr = torch.from_numpy(t).to(dev)

    def evaluate(tag):
        zv = logits(net, Xva[ok_va], a.res, dev)[:, TO7]
        zr = logits(net, rav["g96"], a.res, dev)[:, TO7]
        row = dict(epoch=tag, val_fp_macro_f1=round(macro_f1(yva_fp[ok_va], zv.argmax(1)), 4), val_fp_acc=round(float((zv.argmax(1) == yva_fp[ok_va]).mean()), 4), **rav_scores(zr, rav))
        row["score"] = round(row["val_fp_macro_f1"] + a.sel_w * row["rav_dev_macro_f1"], 4)
        return row

    log = [evaluate(0)]; print(json.dumps(log[-1]), flush=True)                   # epoch 0 = the original B2
    best = log[-1]["score"]; out = ROOT / "runs" / a.tag; out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "best_state.pt"); best_ep = 0

    if a.head_only:
        for p in net.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        groups = [dict(params=list(head.parameters()), lr=a.lr * 10)]
    else:
        hp = set(id(p) for p in head.parameters())
        groups = [dict(params=[p for p in net.parameters() if id(p) not in hp], lr=a.lr), dict(params=list(head.parameters()), lr=a.lr * 10)]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4); base = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler()
    spe = math.ceil(len(xtr) / a.bs); total = spe * a.epochs; step = 0; t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        for m in net.modules():                                       # BatchNorm statistics stay as learned on AffectNet
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        perm = torch.randperm(len(xtr), device=dev); tl = 0.0
        for i in range(0, len(xtr), a.bs):
            ix = perm[i:i + a.bs]
            f = min(1.0, (step + 1) / 200) * (0.5 * (1 + math.cos(math.pi * step / total)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            x = to_input(data.augment(xtr[ix], a.res, a.aug).clamp(0, 1), dev)
            with torch.autocast("cuda", dtype=torch.float16):
                z = net(x).float()
                if a.kd > 0:
                    with torch.no_grad():
                        zt = teacher(x).float()
            loss = -(ttr[ix] * F.log_softmax(z[:, TO7], 1)).sum(1).mean()
            if a.kd > 0:
                loss = loss + a.kd * 4.0 * F.kl_div(F.log_softmax(z / 2, 1), F.softmax(zt / 2, 1), reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 2.0)
            scaler.step(opt); scaler.update(); step += 1; tl += float(loss)
        row = evaluate(ep); row["loss"] = round(tl / spe, 4); row["min"] = round((time.time() - t0) / 60, 1); log.append(row)
        print(json.dumps(row), flush=True)
        if row["score"] > best:
            best, best_ep = row["score"], ep
            torch.save(net.state_dict(), out / "best_state.pt")

    net.load_state_dict(torch.load(out / "best_state.pt")); net.eval().float()
    torch.onnx.export(net, torch.randn(1, 3, a.res, a.res, device=dev), str(out / "best.onnx"), input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}, opset_version=17)
    chk = ort.InferenceSession(str(out / "best.onnx"), providers=["CPUExecutionProvider"]).run(None, {"input": xc})[0]
    with torch.no_grad():
        ediff = float(np.abs(net(torch.from_numpy(xc).to(dev)).cpu().numpy() - chk).max())
    res = dict(tag=a.tag, args=vars(a), conversion_diff=diff, export_diff=ediff, best_epoch=best_ep, best_score=best, log=log)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    b, l0 = log[best_ep], log[0]
    print(f"[{a.tag}] best ep{best_ep}: FER+ val F1 {l0['val_fp_macro_f1']:.4f} -> {b['val_fp_macro_f1']:.4f} | RAV-dev F1 {l0['rav_dev_macro_f1']:.3f} -> {b['rav_dev_macro_f1']:.3f}"
          f" exprRec {l0['rav_dev_expr_recall']:.3f} -> {b['rav_dev_expr_recall']:.3f} | export diff {ediff:.1e}")


if __name__ == "__main__":
    main()
