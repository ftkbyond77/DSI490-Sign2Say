"""v11 prototype-CTC — joint segmentation + recognition (prompt-opt-v11 §3A.3).

Per-frame logits = γ · cos(W h_t, proto_f) for every FORM prototype of the label space + a learned not-a-sign prototype (the CTC blank);
form logits are max-pooled to their word. Loss: CTC over the stage-3 sentences + 0.5 × retrieval loss (CosFace of the pooled span
embedding against the same prototypes, keeps the bank space). New words enter by appending a prototype (no retraining).
Encoder: frozen except its last 2 blocks (lr 3e-5); γ, blank and the projection W at lr 3e-4. Drift check: isolated R@1 on TUNE may
not drop > 0.5 points, otherwise the encoder is frozen and only W / γ / blank train.
"""
from __future__ import annotations

import numpy as np


def build(encoder, protos, form_word, n_words):
    """encoder = lmenc LandmarkEncoder; protos [F, emb] form prototypes (L2-normed, from the bank); form_word [F] word index."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ProtoCTC(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = encoder
            d = encoder.dim
            self.proj = nn.Linear(d, protos.shape[1])
            self.register_buffer("P", F.normalize(torch.as_tensor(protos, dtype=torch.float32), dim=-1))
            self.register_buffer("fw", torch.as_tensor(form_word, dtype=torch.long))
            self.blank = nn.Parameter(F.normalize(torch.randn(protos.shape[1]), dim=0))
            self.log_g = nn.Parameter(torch.tensor(np.log(16.0), dtype=torch.float32))
            self.n_words = n_words

        def frame_logits(self, x, m):
            """[B,T,1+n_words] (index 0 = blank)"""
            h = self.enc.frames(x, m)
            e = F.normalize(self.proj(h).float(), dim=-1)
            g = self.log_g.exp()
            sf = g * (e @ self.P.T)                                      # [B,T,F]
            sw = torch.full((*sf.shape[:2], self.n_words), -1e4, device=sf.device, dtype=sf.dtype)
            sw = sw.scatter_reduce(2, self.fw.expand(*sf.shape[:2], -1), sf, "amax")
            sb = g * (e @ F.normalize(self.blank, dim=0))[..., None]
            return torch.cat([sb, sw], -1), h

        def forward(self, x, m):
            return self.frame_logits(x, m)

    return ProtoCTC()


def ctc_loss(logits, m, targets):
    """logits [B,T,1+W], m [B,T], targets list of word-index lists (0-based words → CTC labels +1)"""
    import torch
    import torch.nn.functional as F
    lp = F.log_softmax(logits.float(), -1).transpose(0, 1)               # [T,B,C]
    il = m.sum(1).long()
    tl = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat = torch.tensor([w + 1 for t in targets for w in t], dtype=torch.long)
    return F.ctc_loss(lp, flat, il, tl, blank=0, reduction="mean", zero_infinity=True)


def greedy(logits, min_len=3):
    """[T,1+W] → [(f0, f1, word)] collapsed runs of the same non-blank argmax"""
    a = np.asarray(logits).argmax(-1)
    out, t, T = [], 0, len(a)
    while t < T:
        if a[t] == 0:
            t += 1; continue
        s, w = t, a[t]; t += 1
        while t < T and a[t] == w:
            t += 1
        if t - s >= min_len:
            out.append((s, t, int(w) - 1))
    return out
