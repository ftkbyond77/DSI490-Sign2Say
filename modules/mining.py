"""v4 weakly-supervised vocabulary mining from YouTube lessons (no manual labels).

For every spoken lexicon word (caption/ASR time t) the teacher usually signs it within a few seconds.
Candidate windows inside [t−1.5 s, t+3.0 s] are embedded with the pretrained sign encoder and scored against the
TTRS vocabulary (≈4.6k words). A window becomes a new instance of word w only if w ranks in the top-K of the whole
vocabulary (chance ≈ K/4,600) and beats the other candidates of that hit. Instances come from new signers/cameras,
which is exactly the diversity TTRS lacks."""
from __future__ import annotations

import numpy as np

from .kinematics import frame_features
from .recognize import embed
from .unisign import prepare_parts, upsample
from .utils import get_logger

log = get_logger("mining")


def candidate_windows(n_frames, t_center_f, lo_f, hi_f, lengths=(8, 12, 16, 22), stride=3):
    out = []
    a0, a1 = max(0, t_center_f + lo_f), min(n_frames, t_center_f + hi_f)
    for L in lengths:
        for s in range(a0, max(a0 + 1, a1 - L + 1), stride):
            if s + L <= n_frames:
                out.append((s, s + L))
    return out


def mine_clip(enc, bank, kp, sc, hw, hits, t_start, fps=12.5, window=(-1.5, 3.0), rank_k=5, min_raise=0.2,
              stop_words=frozenset()):
    """hits: [{word,t0,t1}] in video seconds; t_start = pose t0 (seconds). Returns accepted instances."""
    T = len(kp)
    _, sig = frame_features(kp, sc, hw)
    rest = sig["rest_ref"]
    raise_ = np.maximum(sig["L"]["height"] - rest[0], sig["R"]["height"] - rest[1])
    # merge repeated mentions of the same word within 5 s
    hits = sorted([h for h in hits if h["word"] in bank.widx and h["word"] not in stop_words], key=lambda h: h["t0"])
    merged = []
    for h in hits:
        if merged and merged[-1]["word"] == h["word"] and h["t0"] - merged[-1]["t0"] < 5:
            continue
        merged.append(h)
    wins, owner = [], []
    for hi, h in enumerate(merged):
        tc = int(round((h["t0"] - t_start) * fps))
        if tc + window[1] * fps < 0 or tc + window[0] * fps >= T:
            continue
        for a, b in candidate_windows(T, tc, int(window[0] * fps), int(window[1] * fps)):
            if raise_[a:b].mean() < min_raise:  # hands not in signing space
                continue
            wins.append((a, b)); owner.append(hi)
    if not wins:
        return []
    uniq = sorted(set(wins))
    parts = [prepare_parts(*upsample(kp[a:b], sc[a:b], 2), scale_ref=(kp, sc)) for a, b in uniq]
    Z = embed(enc, parts, "ctx", bs=256)
    zi = {w: i for i, w in enumerate(uniq)}
    S = bank.word_scores(Z)                         # [W_uniq, vocab]
    ranks_all = np.argsort(-S, axis=1)[:, :rank_k]
    best = {}
    for (a, b), hi in zip(wins, owner):
        w = merged[hi]["word"]; wid = bank.widx[w]; r = zi[(a, b)]
        if wid not in ranks_all[r]:
            continue
        rank = int(np.flatnonzero(ranks_all[r] == wid)[0])
        cand = (rank, -float(S[r, wid]), a, b)
        if hi not in best or cand < best[hi]:
            best[hi] = cand
    out = []
    for hi, (rank, negs, a, b) in best.items():
        h = merged[hi]
        top = bank.vocab[int(np.argmax(S[zi[(a, b)]]))]
        out.append(dict(word=h["word"], f0=int(a), f1=int(b), t0=round(t_start + a / fps, 2), t1=round(t_start + b / fps, 2),
                        speech_t=h["t0"], rank=rank, score=-negs, top1=top, emb=Z[zi[(a, b)]]))
    # one window may be claimed by several words → keep the better one
    out.sort(key=lambda x: (x["rank"], -x["score"]))
    kept = []
    for o in out:
        if all(min(o["f1"], k["f1"]) - max(o["f0"], k["f0"]) <= 0.5 * (o["f1"] - o["f0"]) for k in kept):
            kept.append(o)
    return kept
