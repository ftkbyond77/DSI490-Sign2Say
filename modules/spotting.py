"""v4 L3b: recognition-driven sign spotting.

The boundary tagger (segment.py) is trained on synthetic concatenations of dictionary clips; on fluent real signing it
often sees one long "inside" run with no begin peaks, and the old decoder then cut that run into equal chunks
(chunk boundaries unrelated to the signs). Spotting instead asks the vocabulary directly:

  1. active region   = frames the tagger (or the kinematic hand-raise signal) marks as signing
  2. candidate windows = every [a, b) on a 2-frame grid, 0.5–2.4 s long, mostly inside the active region
  3. each window is embedded exactly like a dictionary clip and scored against the bank
     gain = standardised top-1 score (how much the best word stands out from the whole vocabulary) − λ
  4. semi-Markov DP picks non-overlapping windows maximising Σ gain − γ·(uncovered active frames)
  5. neighbouring picks with the same best word are merged

Every pick is still a *segment with a nearest known word*; whether the word is trusted is decided later by the
calibrated tiers, so an unknown sign is reported as "[?] ≈ nearest" instead of a guess.
"""
from __future__ import annotations

import numpy as np

from .recognize import embed
from .unisign import prepare_parts, upsample


def active_frames(prob=None, sig=None, T=None, thr=0.5):
    if prob is not None:
        p = np.convolve(prob[:, 1] + prob[:, 2], np.ones(3) / 3, "same")
        return p > thr
    raise_ = np.maximum(sig["raise_l"], sig["raise_r"]) if sig is not None else np.ones(T)
    return raise_ > 0.2


def candidate_windows(act, lens, stride=2, min_active=0.6):
    T = len(act)
    out = []
    for a in range(0, T, stride):
        for L in lens:
            b = a + L
            if b > T:
                b = T
                if b - a < min(lens):
                    continue
            if act[a:b].mean() >= min_active:
                out.append((a, b))
    return sorted(set(out))


def window_scores(enc, bank, kp, sc, wins, pad=1, bs=128):
    parts = []
    for a, b in wins:
        a0, b0 = max(0, a - pad), min(len(kp), b + pad)
        k, s = upsample(kp[a0:b0], sc[a0:b0], 2)
        parts.append(prepare_parts(k, s, scale_ref=(kp, sc)))
    Z = embed(enc, parts, "ctx", bs=bs)
    return Z, bank.word_scores(Z)


def dp_select(wins, gain, act, gap_cost, stride=2):
    """Max Σ gain(picked) − gap_cost·(uncovered active frames), picks non-overlapping, on the stride grid."""
    T = len(act)
    nodes = sorted(set(range(0, T, stride)) | {T} | {b for _, b in wins} | {a for a, _ in wins})
    pos = {n: i for i, n in enumerate(nodes)}
    ending = {}
    for w, (a, b) in enumerate(wins):
        ending.setdefault(b, []).append((a, w))
    cum = np.concatenate([[0], np.cumsum(act)])
    f = np.full(len(nodes), -1e18); f[0] = 0.0
    back = [None] * len(nodes)
    for i in range(1, len(nodes)):
        n, pn = nodes[i], nodes[i - 1]
        f[i] = f[i - 1] - gap_cost * (cum[n] - cum[pn]); back[i] = ("gap", i - 1, None)
        for a, w in ending.get(n, []):
            v = f[pos[a]] + gain[w]
            if v > f[i]:
                f[i] = v; back[i] = ("win", pos[a], w)
    picks, i = [], len(nodes) - 1
    while i > 0:
        kind, j, w = back[i]
        if kind == "win":
            picks.append(w)
        i = j
    return picks[::-1]


def window_table(enc, bank, kp, sc, act, lens=tuple(range(6, 31, 2)), stride=2, topk=5):
    """Embed + score all candidate windows once; keep only what decoding needs (top-k, vocabulary mean/std)."""
    wins = candidate_windows(act, lens, stride)
    if not wins:
        return dict(wins=[], top_idx=np.zeros((0, topk), int), top_sc=np.zeros((0, topk)), mu=np.zeros(0), sd=np.zeros(0))
    _, S = window_scores(enc, bank, kp, sc, wins)
    o = np.argsort(-S, axis=1)[:, :topk]
    return dict(wins=wins, top_idx=o, top_sc=np.take_along_axis(S, o, 1), mu=S.mean(1), sd=S.std(1) + 1e-6)


def decode(table, act, vocab, lam=4.0, gap_cost=0.1, stride=2, merge=True, min_len=6, prob=None, beta=0.0, agg=None, slack=2, agg_min_len=10):
    """gain(window) = z − λ + β·(tagger begin evidence at the window start, 0 when no tagger)."""
    wins_all = table["wins"]
    if not wins_all:
        return []
    keep = np.array([b - a >= min_len for a, b in wins_all])
    if not keep.any():
        return []
    idx = np.flatnonzero(keep)
    wins = [wins_all[i] for i in idx]
    top1_all, top2_all = table["top_sc"][:, 0], table["top_sc"][:, 1]
    z_all = (top1_all - table["mu"]) / table["sd"]
    gain = z_all[idx] - lam
    if prob is not None and beta:
        pb = np.maximum.reduce([np.roll(prob[:, 1], k) for k in (-1, 0, 1)])
        gain = gain + beta * np.array([pb[a] for a, _ in wins])
    picks = [int(idx[w]) for w in dp_select(wins, gain, act, gap_cost, stride)]
    wins, z, top1, top2 = wins_all, z_all, top1_all, top2_all
    segs = []
    for w in picks:
        a, b = wins[w]
        wi = int(table["top_idx"][w, 0])
        if merge and segs and segs[-1]["_wi"] == wi and a - segs[-1]["f1"] <= stride:
            s = segs[-1]
            if z[w] > s["z"]:
                s.update(z=float(z[w]), score=float(top1[w]), margin=float(top1[w] - top2[w]), _w=w)
            s["f1"] = int(b)
            continue
        segs.append(dict(f0=int(a), f1=int(b), _wi=wi, _w=w, z=float(z[w]), score=float(top1[w]), margin=float(top1[w] - top2[w])))
    for s in segs:
        w = s.pop("_w"); wi = s.pop("_wi")
        if agg == "sum":  # temporal ensemble: Σ top-5 scores over every window inside the segment (±slack frames)
            acc, best = {}, {}
            for v, (x, y) in enumerate(wins):
                if x >= s["f0"] - slack and y <= s["f1"] + slack and y - x >= agg_min_len:
                    for j, sc_ in zip(table["top_idx"][v], table["top_sc"][v]):
                        acc[j] = acc.get(j, 0.0) + float(sc_); best[j] = max(best.get(j, -9.0), float(sc_))
            order = sorted(acc, key=lambda j: -acc[j])[:5]
            s["candidates"] = [dict(word=vocab[j], score=best[j], support=round(acc[j], 3)) for j in order]
            s["window_top1"] = vocab[wi]
        else:
            s["candidates"] = [dict(word=vocab[j], score=float(x)) for j, x in zip(table["top_idx"][w], table["top_sc"][w])]
        s["nearest"] = s["candidates"][0]["word"]
    return segs
