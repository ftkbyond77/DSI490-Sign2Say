"""v4 L4b: continuous sign segmentation — "which frames are a word?".

Self-supervised from isolated TTRS clips (no manual boundaries): each TTRS clip is rest → sign → rest, so its active
span is a sign. Synthetic sentences = 2–7 signs of the SAME signer, time-compressed 1.0–2.5× (conversational pace),
joined by interpolated transitions (keypoint space), with rest at the ends; affine + jitter + hand-dropout augmentation.

Frame tagger: [Uni-Sign contextual features (768) ‖ kinematics (18)] → BiGRU → {0 outside, 1 sign-begin, 2 inside}.
Decoding: inside-probability runs, split at begin peaks, length limits."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kinematics import N_FEAT, frame_features, heuristic_segments
from .unisign import collate_parts, prepare_parts, upsample
from .utils import get_logger

log = get_logger("segment")


# ----------------------------------------------------------------------------- synthetic sentences
def sign_span(kp, sc, hw, pad=0):
    """Active span of an isolated clip using per-hand raise/motion (robust to one-handed signs)."""
    _, sig = frame_features(kp, sc, hw)
    rest = sig["rest_ref"]
    raise_ = np.maximum(sig["L"]["height"] - rest[0], sig["R"]["height"] - rest[1])
    act = (raise_ > 0.25) | (sig["motion"] > 0.12)
    if act.sum() < 3:
        return None
    i0, i1 = np.flatnonzero(act)[[0, -1]]
    return max(0, i0 - pad), min(len(kp), i1 + 1 + pad)


def _resample(kp, sc, factor):
    n = len(kp)
    m = max(2, int(round(n / factor)))
    t = np.linspace(0, n - 1, m)
    i0 = np.floor(t).astype(int); i1 = np.minimum(i0 + 1, n - 1); w = (t - i0)[:, None, None]
    return (kp[i0] * (1 - w) + kp[i1] * w).astype(np.float32), np.minimum(sc[i0], sc[i1])


def _affine(kp, rng):
    ang = np.deg2rad(rng.uniform(-8, 8)); s = rng.uniform(0.85, 1.15)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s
    c = kp.reshape(-1, 2).mean(0)
    out = (kp - c) @ R.T + c + rng.uniform(-0.05, 0.05, 2)
    return out.astype(np.float32)


def compress_signing_space(kp, hw, alpha):
    """Move elbows, wrists and whole hands toward a chest anchor: new = anchor + alpha·(old − anchor)."""
    H, W = hw
    p = kp * np.array([W, H], np.float32)
    mid = (p[:, 5] + p[:, 6]) / 2
    sw = np.linalg.norm(p[:, 5] - p[:, 6], axis=1, keepdims=True)
    anchor = mid + np.concatenate([np.zeros_like(sw), 0.9 * sw], 1)
    out = p.copy()
    for wr, el, hand in ((9, 7, slice(91, 112)), (10, 8, slice(112, 133))):
        shift = (1 - alpha) * (anchor - p[:, wr])
        out[:, wr] += shift; out[:, el] += 0.5 * shift
        out[:, hand] += shift[:, None, :]
    return (out / np.array([W, H], np.float32)).astype(np.float32)


class SentenceSynth:
    def __init__(self, clips: list[dict], seed=0):
        """clips: dict(kp, sc, hw, span=(a,b), signer, word)."""
        self.by_signer = {}
        for c in clips:
            self.by_signer.setdefault(c["signer"], []).append(c)
        self.signers = [s for s, v in self.by_signer.items() if len(v) >= 8]
        self.rng = np.random.RandomState(seed)

    def sample(self):
        rng = self.rng
        pool = self.by_signer[self.signers[rng.randint(len(self.signers))]]
        k = rng.randint(2, 8)
        picks = [pool[i] for i in rng.choice(len(pool), k, replace=False)]
        hw = picks[0]["hw"]
        KP, SC, Y, words = [], [], [], []
        first = picks[0]
        a0 = first["span"][0]
        rest_n = rng.randint(3, 10)
        if a0 > 0:  # leading rest
            idx = np.linspace(0, max(a0 - 1, 0), rest_n).astype(int)
            KP.append(first["kp"][idx]); SC.append(first["sc"][idx]); Y.append(np.zeros(rest_n, np.int64))
        for j, c in enumerate(picks):
            a, b = c["span"]
            kp, sc = _resample(c["kp"][a:b], c["sc"][a:b], rng.uniform(1.0, 2.5))
            if KP:  # transition from previous last frame to this first frame
                n_tr = rng.randint(1, 5)
                prev_k, prev_s = KP[-1][-1], SC[-1][-1]
                w = np.linspace(0, 1, n_tr + 2)[1:-1, None, None]
                KP.append((prev_k[None] * (1 - w) + kp[0][None] * w).astype(np.float32))
                SC.append(np.minimum(prev_s, sc[0])[None].repeat(n_tr, 0)); Y.append(np.zeros(n_tr, np.int64))
                if rng.rand() < 0.1:  # occasional pause mid-sentence
                    n_p = rng.randint(3, 8)
                    KP.append(np.repeat(KP[-1][-1:], n_p, 0)); SC.append(np.repeat(SC[-1][-1:], n_p, 0)); Y.append(np.zeros(n_p, np.int64))
            y = np.full(len(kp), 2, np.int64); y[0] = 1
            KP.append(kp); SC.append(sc); Y.append(y); words.append(c["word"])
        last = picks[-1]
        b = last["span"][1]
        if b < len(last["kp"]):
            idx = np.linspace(b, len(last["kp"]) - 1, rng.randint(3, 10)).astype(int)
            n_tr = 2
            w = np.linspace(0, 1, n_tr + 2)[1:-1, None, None]
            KP.append((KP[-1][-1][None] * (1 - w) + last["kp"][idx[0]][None] * w).astype(np.float32))
            SC.append(np.repeat(SC[-1][-1:], n_tr, 0)); Y.append(np.zeros(n_tr, np.int64))
            KP.append(last["kp"][idx]); SC.append(last["sc"][idx]); Y.append(np.zeros(len(idx), np.int64))
        kp = np.concatenate(KP); sc = np.concatenate(SC).astype(np.float32); y = np.concatenate(Y)
        if rng.rand() < 0.7:  # compact conversational signing: pull hands toward the chest
            kp = compress_signing_space(kp, hw, rng.uniform(0.55, 1.0))
        kp = _affine(kp, rng) + rng.normal(0, 0.002, kp.shape).astype(np.float32)
        if rng.rand() < 0.3:  # one hand unreliable for a stretch
            h = slice(91, 112) if rng.rand() < 0.5 else slice(112, 133)
            s0 = rng.randint(0, len(kp)); s1 = min(len(kp), s0 + rng.randint(5, 30))
            sc[s0:s1, h] = 0.0
        return dict(kp=kp, sc=sc, hw=hw, y=y, words=words)


# ----------------------------------------------------------------------------- features
@torch.no_grad()
def sequence_features(enc, kp, sc, hw):
    """→ [T, 768+18] at the pose frame rate (Uni-Sign run at 2× and sub-sampled back)."""
    kin, _ = frame_features(kp, sc, hw)
    k2, s2 = upsample(kp, sc, 2)
    parts, mask = collate_parts([prepare_parts(k2, s2)], max_len=None)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enc.device == "cuda"):
        f = enc.frames(parts, mask, "ctx")["ctx"][0].float()
    ctx = f[0::2][:len(kp)].cpu().numpy()
    if len(ctx) < len(kp):
        ctx = np.concatenate([ctx, np.repeat(ctx[-1:], len(kp) - len(ctx), 0)])
    return np.concatenate([ctx, kin], 1).astype(np.float32)


@torch.no_grad()
def sequence_features_batch(enc, seqs, bs=12):
    """Batched version of sequence_features for a list of dict(kp, sc, hw) → list of float16 arrays."""
    kins = [frame_features(s["kp"], s["sc"], s["hw"])[0] for s in seqs]
    parts = [prepare_parts(*upsample(s["kp"], s["sc"], 2)) for s in seqs]
    order = np.argsort([len(p["body"]) for p in parts])
    out = [None] * len(seqs)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        P, mask = collate_parts([parts[j] for j in idx], max_len=None)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enc.device == "cuda"):
            f = enc.frames(P, mask, "ctx")["ctx"].float().cpu().numpy()
        for k, j in enumerate(idx):
            T = len(seqs[j]["kp"])
            ctx = f[k, 0::2][:T]
            if len(ctx) < T:
                ctx = np.concatenate([ctx, np.repeat(ctx[-1:], T - len(ctx), 0)])
            out[j] = np.concatenate([ctx, kins[j]], 1).astype(np.float16)
    return out


def train_segmenter_cached(enc, synth, val_synth, n_train=2400, epochs=15, bs=24, lr=1e-3, device="cuda"):
    """Precompute a pool of synthetic sentences once (the expensive encoder pass), then train the tagger for many epochs."""
    t0 = time.time()
    train = [synth.sample() for _ in range(n_train)]
    feats = []
    for i in range(0, n_train, 240):
        feats += sequence_features_batch(enc, train[i:i + 240])
        log.info("seg pool %d/%d (%.0fs)", len(feats), n_train, time.time() - t0)
    for s, x in zip(train, feats):
        s["x"] = x; s.pop("kp"); s.pop("sc")
    val = [val_synth.sample() for _ in range(150)]
    for s, x in zip(val, sequence_features_batch(enc, val)):
        s["x"] = x.astype(np.float32)
    model = Segmenter().to(device)
    steps = epochs * (n_train // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    wts = torch.tensor([1.0, 4.0, 1.0], device=device)
    rng = np.random.RandomState(0)
    hist, best, best_state, it = [], -1, None, 0
    for ep in range(epochs):
        perm = rng.permutation(n_train)
        for b in range(0, n_train - bs + 1, bs):
            batch = [train[j] for j in perm[b:b + bs]]
            L = max(len(s["x"]) for s in batch)
            X = torch.zeros(bs, L, batch[0]["x"].shape[1]); Y = torch.full((bs, L), -100, dtype=torch.long)
            for i, s in enumerate(batch):
                X[i, :len(s["x"])] = torch.from_numpy(s["x"].astype(np.float32)); Y[i, :len(s["x"])] = torch.from_numpy(s["y"])
            X, Y = X.to(device), Y.to(device)
            loss = F.cross_entropy(model(X).reshape(-1, 3), Y.reshape(-1), weight=wts, ignore_index=-100)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); it += 1
        r = evaluate_segmenter(model, val, device)
        hist.append(dict(step=it, epoch=ep, loss=float(loss), **r))
        log.info("seg epoch %d loss %.3f | val(held-out signer) boundary F1 %.3f frame acc %.3f count err %.2f (%.0fs)", ep, float(loss),
                 r["boundary_f1"], r["frame_acc"], r["count_abs_err"], time.time() - t0)
        if r["boundary_f1"] > best:
            best, best_state = r["boundary_f1"], {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist, val


class Segmenter(nn.Module):
    def __init__(self, d_in=768 + N_FEAT, d=192, layers=2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(d, d, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * d, 3)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.out(h)


def train_segmenter(enc, synth: SentenceSynth, val_synth: SentenceSynth, steps=2500, bs=16, lr=1e-3, device="cuda"):
    model = Segmenter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    wts = torch.tensor([1.0, 4.0, 1.0], device=device)
    val = [val_synth.sample() for _ in range(120)]
    for v in val:
        v["x"] = sequence_features(enc, v["kp"], v["sc"], v["hw"])
    hist, t0 = [], time.time()
    for it in range(steps):
        batch = [synth.sample() for _ in range(bs)]
        xs = [torch.from_numpy(sequence_features(enc, b["kp"], b["sc"], b["hw"])) for b in batch]
        L = max(len(x) for x in xs)
        X = torch.zeros(bs, L, xs[0].shape[1]); Y = torch.full((bs, L), -100, dtype=torch.long)
        for i, (x, b) in enumerate(zip(xs, batch)):
            X[i, :len(x)] = x; Y[i, :len(x)] = torch.from_numpy(b["y"])
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits.reshape(-1, 3), Y.reshape(-1), weight=wts, ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if it % 250 == 0 or it == steps - 1:
            r = evaluate_segmenter(model, val, device)
            hist.append(dict(step=it, loss=float(loss), **r))
            log.info("seg %d loss %.3f | val boundary F1 %.3f frame acc %.3f count err %.2f (%.0fs)", it, float(loss),
                     r["boundary_f1"], r["frame_acc"], r["count_abs_err"], time.time() - t0)
    return model, hist, val


def decode_segments(prob, fps=12.5, thr_in=0.5, thr_begin=0.35, min_len=3, max_len_s=3.0):
    """prob [T,3] softmax → [(f0,f1)]; inside = p1+p2, split runs at begin peaks."""
    p_in = prob[:, 1] + prob[:, 2]
    p_in = np.convolve(p_in, np.ones(3) / 3, "same")
    inside = p_in > thr_in
    segs, t, T = [], 0, len(prob)
    while t < T:
        if not inside[t]:
            t += 1; continue
        s = t
        while t < T and inside[t]:
            t += 1
        cuts = [s]
        pb = prob[s:t, 1]
        for i in range(1, len(pb) - 1):
            if pb[i] > thr_begin and pb[i] >= pb[i - 1] and pb[i] >= pb[i + 1] and s + i - cuts[-1] >= min_len:
                cuts.append(s + i)
        cuts.append(t)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a >= min_len:
                segs.append((a, b))
    max_len = int(max_len_s * fps)
    out = []
    for a, b in segs:  # very long runs → split evenly (rare)
        n = int(np.ceil((b - a) / max_len))
        edges = np.linspace(a, b, n + 1).astype(int)
        out += [(int(x), int(y)) for x, y in zip(edges[:-1], edges[1:])]
    return out


def gt_segments(y):
    segs, t = [], 0
    while t < len(y):
        if y[t] == 0:
            t += 1; continue
        s = t; t += 1
        while t < len(y) and y[t] == 2:
            t += 1
        segs.append((s, t))
    return segs


def boundary_f1(pred, gt, tol=2):
    """Match segments by start AND end within ±tol frames."""
    used, tp = set(), 0
    for a, b in pred:
        for j, (c, d) in enumerate(gt):
            if j not in used and abs(a - c) <= tol and abs(b - d) <= tol:
                used.add(j); tp += 1; break
    p = tp / max(len(pred), 1); r = tp / max(len(gt), 1)
    return 2 * p * r / max(p + r, 1e-9)


@torch.no_grad()
def evaluate_segmenter(model, val, device="cuda", heuristic=False):
    model.eval()
    f1s, accs, cerr = [], [], []
    for v in val:
        gt = gt_segments(v["y"])
        if heuristic:
            _, sig = frame_features(v["kp"], v["sc"], v["hw"])
            pred = heuristic_segments(sig)
            acc = np.nan
        else:
            prob = model(torch.from_numpy(v["x"])[None].to(device)).softmax(-1)[0].cpu().numpy()
            pred = decode_segments(prob)
            acc = float(((prob.argmax(1) > 0) == (v["y"] > 0)).mean())
        f1s.append(boundary_f1(pred, gt)); accs.append(acc); cerr.append(abs(len(pred) - len(gt)))
    model.train()
    return dict(boundary_f1=float(np.mean(f1s)), frame_acc=float(np.nanmean(accs)) if not heuristic else float("nan"),
                count_abs_err=float(np.mean(cerr)))
