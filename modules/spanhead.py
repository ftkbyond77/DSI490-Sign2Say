"""Span head — one-pass spotting for real time (v7; knowledge distillation, teacher → student).

Problem: the sign encoder (Uni-Sign ST-GCN + mT5 encoder, ~170 MFLOP per frame) embeds one window at a time. Spotting a 5 s
utterance needs ~500–1,000 candidate windows → 3–6 windows/s per CPU core → tens of seconds. Not real time.
Solution: the encoder already runs ONCE over the whole utterance to give the tagger its contextual frame states H [T,768].
A small student head maps the frame states of any span [a, b) to the embedding the full encoder would give that window alone:

    student(H[a:b]) = norm( W2 · gelu(W1 · LN([attn-pool(H[a:b]) ‖ mean(H[a:b]) ‖ H[a] ‖ H[b−1] ‖ log len])) + mean(H[a:b]) )

trained to match the teacher's isolated-window embeddings (cosine + in-batch contrastive, so rankings are preserved). The decoder
segments the utterance with student scores (≈ free), then only the few chosen segments are re-embedded exactly by the encoder
for the final word ranking and the calibration features.
Inference is plain NumPy (weights in models/span_head.npz).
"""
from __future__ import annotations

import numpy as np

D = 768


def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def span_features(H, wins, q):
    """[N, 4·768+1] features of N spans of one sequence (attention pool with query q, mean, first, last, log length) — vectorised
    with cumulative sums (softmax over a span = exp(a − max over the sequence), exact up to float rounding)."""
    H = np.asarray(H, np.float32)
    W = np.asarray(wins, np.int64).reshape(-1, 2)
    a = H @ q
    e = np.exp((a - a.max()).astype(np.float64))[:, None]
    H64 = H.astype(np.float64); z = np.zeros((1, H.shape[1]))
    cs = np.concatenate([z, np.cumsum(H64, 0)]); ce = np.concatenate([z, np.cumsum(e * H64, 0)]); cw = np.concatenate([[0.0], np.cumsum(e[:, 0])])
    s_, t_ = W[:, 0], W[:, 1]
    L = (t_ - s_).astype(np.float32)[:, None]
    attn = (ce[t_] - ce[s_]) / np.maximum((cw[t_] - cw[s_])[:, None], 1e-30)
    return np.concatenate([attn, (cs[t_] - cs[s_]) / L, H[s_], H[t_ - 1], np.log(L) / 4.0], 1).astype(np.float32)


class SpanHeadNP:
    """NumPy inference of the trained head (models/span_head.npz)."""

    def __init__(self, path):
        z = np.load(path)
        self.p = {k: z[k].astype(np.float32) for k in z.files}

    def __call__(self, H, wins):
        if not len(wins):
            return np.zeros((0, D), np.float32)
        p = self.p
        X = span_features(H.astype(np.float32), wins, p["q"])
        h = _gelu(_ln(X, p["ln_g"], p["ln_b"]) @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        z = h + X[:, D:2 * D]
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)


def train_span_head(samples, epochs=8, bs=256, lr=1e-3, device="cuda", log=None, val=None):
    """samples: list of (H [T,768] float16, wins [(a,b)], teacher Z [N,768]). Returns numpy weights dict."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Parameter(torch.zeros(D))
            self.ln = nn.LayerNorm(4 * D + 1)
            self.l1 = nn.Linear(4 * D + 1, 1024)
            self.l2 = nn.Linear(1024, D)
            nn.init.zeros_(self.l2.weight); nn.init.zeros_(self.l2.bias)

    head = Head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(0)

    def batch_forward(items):
        """vectorised span pooling: every window of every sequence gathered into [N, Lmax, 768] with a mask."""
        Xs, Ts = [], []
        for H, wins, Zt in items:
            Ht = torch.from_numpy(H.astype(np.float32)).to(device)
            W = torch.as_tensor(np.asarray(wins), device=device)
            L = (W[:, 1] - W[:, 0])
            Lm = int(L.max())
            idx = W[:, :1] + torch.arange(Lm, device=device)[None]
            m = torch.arange(Lm, device=device)[None] < L[:, None]
            G = Ht[idx.clamp(max=len(Ht) - 1)]                       # [N, Lm, D]
            a = (G @ head.q).masked_fill(~m, -1e9)
            w = torch.softmax(a, 1)
            attn = (w[..., None] * G).sum(1)
            mean = (G * m[..., None]).sum(1) / L[:, None]
            first, last = Ht[W[:, 0]], Ht[W[:, 1] - 1]
            X = torch.cat([attn, mean, first, last, (torch.log(L.float()) / 4.0)[:, None]], 1)
            Xs.append(X); Ts.append(torch.from_numpy(Zt.astype(np.float32)).to(device))
        X, T = torch.cat(Xs), torch.cat(Ts)
        Z = F.normalize(head.l2(F.gelu(head.l1(head.ln(X)), approximate="tanh")) + X[:, D:2 * D], dim=-1)
        return Z, F.normalize(T, dim=-1)

    def evaluate(items):
        head.eval()
        with torch.no_grad():
            cs = []
            for i in range(0, len(items), 16):
                Z, T = batch_forward(items[i:i + 16])
                cs.append((Z * T).sum(1).cpu().numpy())
        head.train()
        return float(np.concatenate(cs).mean()) if cs else 0.0
    for ep in range(epochs):
        perm = rng.permutation(len(samples))
        for i in range(0, len(perm), 16):
            Z, T = batch_forward([samples[j] for j in perm[i:i + 16]])
            l_cos = (1 - (Z * T).sum(1)).mean()
            k = min(len(Z), 512)
            sel = torch.randperm(len(Z), device=device)[:k]
            logits = Z[sel] @ T[sel].t() / 0.05
            l_nce = F.cross_entropy(logits, torch.arange(k, device=device))
            loss = l_cos + 0.2 * l_nce
            opt.zero_grad(); loss.backward(); opt.step()
        if log:
            log.info("span head epoch %d: loss %.4f  val cos(student, teacher) %.4f", ep, float(loss), evaluate(val) if val else -1)
    sd = {k: v.detach().cpu().numpy() for k, v in head.state_dict().items()}
    return dict(q=sd["q"], ln_g=sd["ln.weight"], ln_b=sd["ln.bias"], W1=sd["l1.weight"].T.copy(), b1=sd["l1.bias"], W2=sd["l2.weight"].T.copy(), b2=sd["l2.bias"])
