"""Evaluation helpers for the face model: classification metrics, calibration, selective prediction (unknown), latency / memory."""
from __future__ import annotations

import time

import numpy as np

from data import CLASSES


def softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / T
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def confusion(y, p, k=len(CLASSES)):
    m = np.zeros((k, k), np.int64)
    np.add.at(m, (y, p), 1)
    return m


def classification(y, prob):
    """Accuracy, Macro-F1, Weighted-F1, UAR (= balanced accuracy), per-class P / R / F1, confusion matrix."""
    p = prob.argmax(1)
    cm = confusion(y, p)
    tp = np.diag(cm).astype(float)
    sup, pred = cm.sum(1).astype(float), cm.sum(0).astype(float)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    pre = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    f1 = np.divide(2 * pre * rec, pre + rec, out=np.zeros_like(tp), where=(pre + rec) > 0)
    return dict(n=int(len(y)), accuracy=float(tp.sum() / cm.sum()), macro_f1=float(f1.mean()), weighted_f1=float((f1 * sup).sum() / sup.sum()),
                uar=float(rec.mean()), balanced_accuracy=float(rec.mean()),
                per_class={c: dict(precision=float(pre[i]), recall=float(rec[i]), f1=float(f1[i]), support=int(sup[i])) for i, c in enumerate(CLASSES)},
                confusion=cm.tolist())


def calibration(y, prob, bins=15):
    """Expected Calibration Error (top-label, equal-width bins), Brier score (multi-class), NLL; reliability-diagram data."""
    conf, pred = prob.max(1), prob.argmax(1)
    ok = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece, rel = 0.0, []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
            rel.append(dict(lo=float(a), hi=float(b), n=int(m.sum()), conf=float(conf[m].mean()), acc=float(ok[m].mean())))
    onehot = np.eye(prob.shape[1])[y]
    return dict(ece=float(ece), brier=float(((prob - onehot) ** 2).sum(1).mean()), nll=float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-9, 1)).mean()),
                mean_confidence=float(conf.mean()), reliability=rel)


def fit_temperature(logits, y):
    """Temperature scaling on validation logits: T minimising NLL (grid + refinement)."""
    best = (1e9, 1.0)
    for T in np.concatenate([np.linspace(0.3, 3.0, 55), np.linspace(3.0, 6.0, 7)]):
        p = softmax(logits, T)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()
        best = min(best, (nll, float(T)))
    return best[1]


def selective(y, prob, thresholds=None):
    """Accuracy / macro-F1 of the answered frames and coverage when the model may say `unknown` below a confidence threshold."""
    conf, pred = prob.max(1), prob.argmax(1)
    out = []
    for t in (thresholds if thresholds is not None else np.round(np.arange(0.0, 0.96, 0.05), 2)):
        m = conf >= t
        if m.sum() < 10:
            continue
        c = classification(y[m], prob[m])
        out.append(dict(threshold=float(t), coverage=float(m.mean()), accuracy=c["accuracy"], macro_f1=c["macro_f1"], uar=c["uar"]))
    return out


def predict_logits(net, X, device, batch=512, res=None, flip=False):
    """X uint8 [N,48,48] -> logits [N,7] (the model's own deployment path; flip=True averages the mirrored crop's logits)."""
    import torch
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch]).to(device).float().unsqueeze(1)
            z = net(x).float()
            if flip:
                z = (z + net(torch.flip(x, dims=[3])).float()) / 2
            out.append(z.cpu().numpy())
    return np.concatenate(out)


def latency_torch(net, device, batch=1, n=200, warm=30):
    """ms per call of the PyTorch model (synchronised), median / p95, FP16 autocast on GPU."""
    import torch
    net.eval().to(device)
    x = torch.randint(0, 256, (batch, 1, 48, 48), device=device).float()
    ts = []
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
        for i in range(warm + n):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter(); net(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warm:
                ts.append((time.perf_counter() - t) * 1000)
    return dict(median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)), batch=batch)


def latency_ort(path, providers=("CPUExecutionProvider",), batch=1, n=300, warm=30, threads=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=list(providers))
    x = np.random.randint(0, 256, (batch, 1, 48, 48)).astype(np.float32)
    ts = []
    for i in range(warm + n):
        t = time.perf_counter(); s.run(None, {"face": x})
        if i >= warm:
            ts.append((time.perf_counter() - t) * 1000)
    return dict(provider=providers[0], median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)), batch=batch, threads=threads)
