"""v11 Path B segmentation (prompt-opt-v11 §3A.2): MS-TCN boundary head on the FROZEN v11 encoder's frame states.

Input per frame: encoder frame states H [T, d] (modules/lmenc.frames_long) ‖ the 13 kinematic features (modules/segment.kin_features).
Model: 4 stages × 10 dilated residual 1-D conv layers (dilation 1 … 512, 64 channels), stage k>1 refines the softmax of stage k−1
(Farha & Gall 2019). Classes: 0 non-sign · 1 begin · 2 inside · 3 end.
Loss: class-balanced focal loss (γ 2) on every stage + truncated-MSE smoothing loss 0.15 on log-probabilities; frames of signs ≤ 12
frames weighted × 3. To the existing decoder it looks like the v10 tagger: `as_tagger_prob` → [T, 3] (non-sign, begin, inside).
"""
from __future__ import annotations

import numpy as np

N_CLS = 4


def labels_from_bounds(T, bounds, short=12):
    """gold sign spans [(f0, f1)] → frame classes [T] and frame weights [T] (×3 inside short signs)"""
    y = np.zeros(T, np.int64); w = np.ones(T, np.float32)
    for a, b in bounds:
        a, b = max(0, int(a)), min(T, int(b))
        if b - a < 2:
            continue
        y[a:b] = 2; y[a] = 1; y[b - 1] = 3
        if b - a <= short:
            w[a:b] = 3.0
    return y, w


def as_tagger_prob(p4):
    """[T,4] → [T,3] in the v10 tagger's classes (non-sign, begin, inside); 'end' frames count as inside"""
    p4 = np.asarray(p4, np.float32)
    return np.stack([p4[:, 0], p4[:, 1], p4[:, 2] + p4[:, 3]], 1)


def segments_from_prob(p4, thr=0.5, min_len=4):
    """greedy decode (for boundary F1 / segment F1 of the head alone): runs of P(sign) ≥ thr, split at strong 'begin' peaks"""
    p = np.asarray(p4)
    sign = (1 - p[:, 0]) >= thr
    segs, t, T = [], 0, len(p)
    while t < T:
        if not sign[t]:
            t += 1; continue
        s = t; t += 1
        while t < T and sign[t] and not (p[t, 1] > 0.5 and t - s >= min_len):
            t += 1
        if t - s >= min_len:
            segs.append((s, t))
    return segs


def build(d_in, ch=64, stages=4, layers=10):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Res(nn.Module):
        def __init__(self, dil):
            super().__init__()
            self.c = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil); self.o = nn.Conv1d(ch, ch, 1); self.d = nn.Dropout(0.3)

        def forward(self, x, m):
            return (x + self.d(self.o(F.relu(self.c(x))))) * m

    class Stage(nn.Module):
        def __init__(self, din):
            super().__init__()
            self.i = nn.Conv1d(din, ch, 1)
            self.L = nn.ModuleList([Res(2 ** k) for k in range(layers)])
            self.out = nn.Conv1d(ch, N_CLS, 1)

        def forward(self, x, m):
            h = self.i(x) * m
            for l in self.L:
                h = l(h, m)
            return self.out(h) * m

    class MSTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = dict(d_in=d_in, ch=ch, stages=stages, layers=layers)
            self.norm = nn.LayerNorm(d_in)
            self.S = nn.ModuleList([Stage(d_in)] + [Stage(N_CLS) for _ in range(stages - 1)])

        def forward(self, x, m):
            """x [B,T,d_in], m [B,T] → list of logits [B,T,4] per stage"""
            x = self.norm(x).transpose(1, 2); mm = m[:, None]
            outs, h = [], x
            for k, s in enumerate(self.S):
                o = s(h if k == 0 else F.softmax(outs[-1], 1) * mm, mm)
                outs.append(o)
            return [o.transpose(1, 2) for o in outs]

    return MSTCN()


def loss(outs, y, w, m, cls_w, smooth=0.15, gamma=2.0):
    import torch
    import torch.nn.functional as F
    tot = 0.0
    for o in outs:
        lp = F.log_softmax(o, -1)
        nll = -lp.gather(-1, y.clamp(min=0)[..., None]).squeeze(-1)
        pt = torch.exp(-nll)
        foc = cls_w[y.clamp(min=0)] * (1 - pt) ** gamma * nll * w * m
        tot = tot + foc.sum() / m.sum().clamp(min=1)
        d = (lp[:, 1:] - lp[:, :-1].detach()) ** 2
        d = d.clamp(max=16).mean(-1) * m[:, 1:] * m[:, :-1]
        tot = tot + smooth * d.sum() / m.sum().clamp(min=1)
    return tot / len(outs)


def boundary_f1(pred, gold, tol=3):
    """both ends within ±tol frames (v10 definition, scripts/decision.score)"""
    used, tp = set(), 0
    for a, b in pred:
        for j, (c, d) in enumerate(gold):
            if j not in used and abs(a - c) <= tol and abs(b - d) <= tol:
                used.add(j); tp += 1; break
    return 2 * tp / max(len(pred) + len(gold), 1)


def segment_f1(pred, gold, iou=0.3):
    used, tp = set(), 0
    for a, b in pred:
        for j, (c, d) in enumerate(gold):
            inter = max(0, min(b, d) - max(a, c)); uni = max(b, d) - min(a, c)
            if j not in used and uni > 0 and inter / uni >= iou:
                used.add(j); tp += 1; break
    return 2 * tp / max(len(pred) + len(gold), 1)
