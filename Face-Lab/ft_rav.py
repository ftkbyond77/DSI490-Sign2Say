"""Round 8 (Colab GPU): fine-tune HSEmotion EfficientNet-B2 (AffectNet) on RAVDESS 24 actors + neutral dictionary signing, under the SAME
leave-actors-out folds as rav_head.py / the blendshape face model (folds = actors[i::6]), so the result compares with 0.723 / 0.591.

Per fold: train on the other 20 actors (+ every neutral signing crop), predict the 4 held-out actors' crops → out-of-fold logits [8] and
penultimate features [1408] for every RAVDESS crop (cache/ → evaluated on the laptop by rav_ft_eval.py with the same samples / metrics).
Fold "all": train on all 24 actors → best.onnx with two outputs (output = 8 logits, feat = 1408 features), a drop-in for enet_b2_8_feat.onnx.
Loss = class-balanced CE on our 7 classes (contempt left out) + kd · KL(original B2 ‖ student) at T 2 (keeps the AffectNet skill);
BatchNorm statistics frozen; GPU augmentation as ft_b2.py; quality ≥ 0.25 crops only for training.
A fold whose output already exists is skipped (a Colab disconnect resumes where it stopped).

  python ft_rav.py --out /content/drive/MyDrive/face-v2/out/rav_ft [--folds 0,1,2,3,4,5,all] [--epochs 2] [--lr 3e-5] [--kd 0.5]
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

import data

ROOT = Path(__file__).resolve().parent
HSE_ORDER = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TO7 = [HSE_ORDER.index(c) for c in data.CLASSES]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_input(y, dev):
    return (y.expand(-1, 3, -1, -1) - MEAN.to(dev)) / STD.to(dev)


def load_pack(path):
    import cv2
    z = dict(np.load(path)); jpg, off = z["jpg"], z["off"]                 # read each array once (an NpzFile re-reads on every access)
    G = np.stack([cv2.imdecode(jpg[off[i]:off[i + 1]], cv2.IMREAD_GRAYSCALE) for i in range(len(off) - 1)])
    return G, {k: z[k] for k in ("label", "actor", "clip", "intensity", "q", "t")}


class GlobalMaxPool(torch.nn.Module):
    def forward(self, x):
        return torch.amax(x, dim=tuple(range(2, x.dim())), keepdim=True)


def register_globalmaxpool():
    """onnx2torch has no GlobalMaxPool converter (B2 has one, before Flatten → Gemm)."""
    from onnx2torch.node_converters.registry import add_converter
    from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node
    for v in (1, 22):
        try:
            add_converter(operation_type="GlobalMaxPool", version=v)(
                lambda node, graph: OperationConverterResult(torch_module=GlobalMaxPool(), onnx_mapping=onnx_mapping_from_node(node)))
        except ValueError:                                                 # already registered (a newer onnx2torch)
            pass


class Tap(torch.nn.Module):
    """net → (logits, the last linear layer's input = the penultimate features)."""

    def __init__(self, net, lin):
        super().__init__(); self.net, self.lin, self._f = net, lin, None
        lin.register_forward_hook(lambda m, i, o: setattr(self, "_f", i[0]))

    def forward(self, x):
        z = self.net(x)
        return z, self._f


@torch.no_grad()
def predict(tap, G, res, dev, bs=128):
    tap.eval(); Z, Fe = [], []
    for i in range(0, len(G), bs):
        x = torch.from_numpy(G[i:i + bs]).to(dev).float().unsqueeze(1) / 255.0
        x = to_input(F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False).clamp(0, 1), dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            z, f = tap(x)
        Z.append(z.float().cpu()); Fe.append(f.float().cpu())
    return torch.cat(Z).numpy(), torch.cat(Fe).numpy()


def train_one(base_net, lin_idx, G, y, a, dev, log):
    net = copy.deepcopy(base_net).to(dev)
    teacher = copy.deepcopy(base_net).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    lin = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][lin_idx]
    hp = set(id(p) for p in lin.parameters())
    opt = torch.optim.AdamW([dict(params=[p for p in net.parameters() if id(p) not in hp], lr=a.lr), dict(params=list(lin.parameters()), lr=a.lr * 10)],
                            weight_decay=1e-4)
    base = [g["lr"] for g in opt.param_groups]; scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    cnt = np.bincount(y, minlength=7).astype(np.float32); cw = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / 7, device=dev)
    X = torch.from_numpy(G).to(dev).float().unsqueeze(1); Y = torch.from_numpy(y.astype(np.int64)).to(dev)
    spe = math.ceil(len(X) / a.bs); total = spe * a.epochs; step = 0; t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        for m in net.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        perm = torch.randperm(len(X), device=dev); tl = 0.0
        for i in range(0, len(X), a.bs):
            ix = perm[i:i + a.bs]
            f = min(1.0, (step + 1) / 100) * (0.5 * (1 + math.cos(math.pi * step / total)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            x = to_input(data.augment(X[ix], a.res, a.aug).clamp(0, 1), dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                z = net(x).float()
                if a.kd > 0:
                    with torch.no_grad():
                        zt = teacher(x).float()
            loss = F.cross_entropy(z[:, TO7], Y[ix], weight=cw)
            if a.kd > 0:
                loss = loss + a.kd * 4.0 * F.kl_div(F.log_softmax(z / 2, 1), F.softmax(zt / 2, 1), reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            scaler.step(opt); scaler.update(); step += 1; tl += float(loss)
            if a.max_steps and step >= a.max_steps:
                break
        log.append(dict(epoch=ep, loss=round(tl / spe, 4), min=round((time.time() - t0) / 60, 1))); print("   ", log[-1], flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    return net.eval(), lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "hse" / "enet_b2_8.onnx"))
    ap.add_argument("--pack", default=str(ROOT / "cache" / "rav_ft_pack.npz"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "rav_ft"))
    ap.add_argument("--folds", default="0,1,2,3,4,5,all")
    ap.add_argument("--res", type=int, default=260)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-5, help="backbone LR (the last linear layer gets 10x)")
    ap.add_argument("--kd", type=float, default=0.5)
    ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop each fold after this many steps")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many crops")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dummy", action="store_true", help="smoke test without onnx2torch: a tiny CNN stands in for B2")
    a = ap.parse_args()
    import onnxruntime as ort

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    xc = np.random.randn(2, 3, a.res, a.res).astype(np.float32)
    if a.dummy:
        base_net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 5, 4), torch.nn.BatchNorm2d(8), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
                                       torch.nn.Flatten(), torch.nn.Linear(8, 8)).eval(); diff = 0.0
    else:
        import onnx
        import onnx2torch
        register_globalmaxpool()
        base_net = onnx2torch.convert(onnx.load(a.onnx)).eval()
        ref = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"]).run(None, {"input": xc})[0]
        with torch.no_grad():
            diff = float(np.abs(base_net(torch.from_numpy(xc)).numpy() - ref).max())
        assert diff < 1e-2, diff
    print(f"onnx2torch max |diff| = {diff:.2e} on {dev}", flush=True)
    lins = [m for m in base_net.modules() if isinstance(m, torch.nn.Linear)]; lin_idx = len(lins) - 1

    G, M = load_pack(a.pack)
    if a.limit:
        keep = np.r_[np.where(M["actor"] > 0)[0][:a.limit], np.where(M["actor"] == 0)[0][:a.limit // 10]]
        G, M = G[keep], {k: v[keep] for k, v in M.items()}
    acts = np.unique(M["actor"][M["actor"] > 0]); folds = {str(i): acts[i::6] for i in range(6)}
    print(f"{len(G)} crops, actors {list(acts)}", flush=True)
    good = M["q"] >= 0.25
    for fk in a.folds.split(","):
        done = out / (f"fold_{fk}.npz" if fk != "all" else "best.onnx")
        if done.exists():
            print(f"fold {fk}: done already", flush=True); continue
        test = folds.get(fk, np.array([], int))
        tr = good & ~np.isin(M["actor"], test)
        print(f"fold {fk}: test actors {list(test)}, train crops {int(tr.sum())}", flush=True)
        log = []; t0 = time.time()
        net, lin = train_one(base_net, lin_idx, G[tr], M["label"][tr], a, dev, log)
        tap = Tap(net, lin)
        if fk != "all":
            te = np.where(np.isin(M["actor"], test))[0]
            Z, Fe = predict(tap, G[te], a.res, dev)
            np.savez_compressed(done, idx=te, logits=Z.astype(np.float32), feat=Fe.astype(np.float16), log=json.dumps(log))
        else:
            tap = tap.float().to(dev).eval()
            kw = dict(input_names=["input"], output_names=["output", "feat"], opset_version=17,
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}, "feat": {0: "batch_size"}})
            try:
                torch.onnx.export(tap, torch.randn(1, 3, a.res, a.res, device=dev), str(done), dynamo=False, **kw)    # TorchScript exporter (the hook is traced)
            except TypeError:
                torch.onnx.export(tap, torch.randn(1, 3, a.res, a.res, device=dev), str(done), **kw)
            chk = ort.InferenceSession(str(done), providers=["CPUExecutionProvider"]).run(None, {"input": xc})
            tap.eval()                                                      # export restores the module's previous (training) mode
            with torch.no_grad():
                zz, ff = tap(torch.from_numpy(xc).to(dev))
            (out / "all_log.json").write_text(json.dumps(dict(log=log, export_diff=float(np.abs(zz.cpu().numpy() - chk[0]).max()),
                                                                 feat_diff=float(np.abs(ff.cpu().numpy() - chk[1]).max()), args=vars(a))))
        print(f"fold {fk}: {(time.time() - t0) / 60:.1f} min", flush=True)
        del net, tap
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
