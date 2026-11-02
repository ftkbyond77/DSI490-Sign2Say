"""Fine-tune a pretrained backbone on FER2013 (local GPU). Two stages in one run:
  A  head only (backbone frozen) for --head-epochs,
  B  the upper part of the backbone (from --unfreeze, a fraction of its blocks) + head, lower backbone LR, warm-up + cosine.
Checkpoint = best validation Macro-F1 (EMA weights). Test data is never touched here.

  python train.py --arch mobilenet_v3_large --res 96 --epochs 14 --tag pilot_mbv3
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data
from evaluate import calibration, classification, fit_temperature, predict_logits, softmax
from models import FerNet

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--unfreeze", type=float, default=0.5, help="stage B trains blocks >= this fraction of the backbone (0 = all)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-back", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--cw-power", type=float, default=0.5, help="class weight = (1/freq)^power; 0 = unweighted")
    ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--mix", type=float, default=0.0, help="CutMix probability")
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    D = data.prepare(a.seed)
    (Xtr, ytr), (Xva, yva) = D["train"], D["val"]
    xtr = torch.from_numpy(Xtr).to(dev).float().unsqueeze(1)           # the whole training set lives on the GPU (60 MB)
    ytr_t = torch.from_numpy(ytr).to(dev)
    w = torch.from_numpy(data.class_weights(ytr, a.cw_power)).to(dev)
    net = FerNet(a.arch, a.res).to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad = False
    out = ROOT / "runs" / a.tag
    out.mkdir(parents=True, exist_ok=True)

    scaler = torch.amp.GradScaler()
    spe = math.ceil(len(xtr) / a.bs)
    log, best, bad, t0 = [], (-1.0, 0), 0, time.time()
    opt = sched = None
    for ep in range(1, a.epochs + 1):
        stage = "A" if ep <= a.head_epochs else "B"
        if ep == 1 or ep == a.head_epochs + 1:                       # (re)build the optimiser when the stage changes
            net.set_trainable(1.0 if stage == "A" else a.unfreeze)
            groups = [dict(params=[p for p in net.head.parameters()], lr=a.lr_head)]
            if stage == "B":
                groups.append(dict(params=[p for p in net.features.parameters() if p.requires_grad], lr=a.lr_back))
            opt = torch.optim.AdamW(groups, weight_decay=a.wd)
            base = [g["lr"] for g in opt.param_groups]
            total = (a.head_epochs if stage == "A" else a.epochs - a.head_epochs) * spe
            step = 0
        net.train()
        perm = torch.randperm(len(xtr), device=dev)
        tl, nb = 0.0, 0
        for i in range(0, len(xtr), a.bs):
            ix = perm[i:i + a.bs]
            xb, yb = data.augment(xtr[ix], a.res, a.aug), ytr_t[ix]
            warm = min(1.0, (step + 1) / (spe if stage == "B" else 1))
            f = warm * (0.5 * (1 + math.cos(math.pi * step / total)) * 0.98 + 0.02)
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            lam = 1.0
            if a.mix > 0 and np.random.rand() < a.mix:                    # CutMix
                lam = float(np.random.beta(1, 1)); r = torch.randperm(len(xb), device=dev)
                s = int(a.res * math.sqrt(1 - lam)); x0, y0 = np.random.randint(0, a.res - s + 1, 2)
                xb = xb.clone(); xb[:, :, y0:y0 + s, x0:x0 + s] = xb[r][:, :, y0:y0 + s, x0:x0 + s]
                lam = 1 - s * s / a.res / a.res; yb2 = yb[r]
            with torch.autocast("cuda", dtype=torch.float16):
                z = net.forward_res(xb.contiguous(memory_format=torch.channels_last))
            loss = F.cross_entropy(z.float(), yb, weight=w, label_smoothing=a.smooth)
            if lam < 1.0:
                loss = lam * loss + (1 - lam) * F.cross_entropy(z.float(), yb2, weight=w, label_smoothing=a.smooth)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 2.0)
            scaler.step(opt); scaler.update()
            step += 1
            with torch.no_grad():                                         # EMA of the weights (+ BN statistics)
                d = min(a.ema, (1 + ep * spe) / (10 + ep * spe))
                pe, pn = list(ema.parameters()), [p.detach() for p in net.parameters()]
                torch._foreach_mul_(pe, d); torch._foreach_add_(pe, pn, alpha=1 - d)
                for be, bn in zip(ema.buffers(), net.buffers()):
                    be.copy_(bn)
            tl += float(loss); nb += 1
        zv = predict_logits(ema, Xva, dev)
        m = classification(yva, softmax(zv))
        row = dict(epoch=ep, stage=stage, loss=tl / nb, val_acc=m["accuracy"], val_macro_f1=m["macro_f1"], val_uar=m["uar"], min_recall=min(v["recall"] for v in m["per_class"].values()),
                   sec=round(time.time() - t0, 1), vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        log.append(row)
        print(json.dumps(row), flush=True)
        if m["macro_f1"] > best[0]:
            best, bad = (m["macro_f1"], ep), 0
            torch.save(dict(state=ema.state_dict(), arch=a.arch, res=a.res, args=vars(a), epoch=ep), out / "best.pt")
        else:
            bad += 1
            if ep > a.head_epochs and bad >= a.patience:
                print("early stop", flush=True)
                break

    ck = torch.load(out / "best.pt", map_location=dev)
    ema.load_state_dict(ck["state"])
    zv = predict_logits(ema, Xva, dev)
    T = fit_temperature(zv, yva)
    res = dict(tag=a.tag, args=vars(a), best_epoch=best[1], best_val_macro_f1=best[0], temperature=T, train_minutes=round((time.time() - t0) / 60, 2), log=log,
               val=classification(yva, softmax(zv, T)), val_calibration_raw=calibration(yva, softmax(zv)), val_calibration_T=calibration(yva, softmax(zv, T)),
               n_train=int(len(Xtr)), n_val=int(len(Xva)), excluded=D["excluded"], peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               env=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"[{a.tag}] best val macro-F1 {best[0]:.4f} @ep{best[1]}  acc {res['val']['accuracy']:.4f}  UAR {res['val']['uar']:.4f}  T={T:.2f}  {res['train_minutes']} min")


if __name__ == "__main__":
    main()
