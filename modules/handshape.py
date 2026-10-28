"""Hand-shape + hand-location cue (v8.1): a second, independent opinion on which word a sign segment is.

The sign encoder ranks words from learned embeddings; for an unseen signer it often puts the right word 2nd–5th when two signs
share a place (ฉัน / ชอบ at the chest, ชื่อ / เพื่อน). This module describes a span directly from the landmarks — finger bend
angles, finger-tip distances, palm direction, where each hand is relative to the shoulders and the nose, how it moves over the three
thirds of the span, and the two hands' relation — and compares it with the same description of every bank clip.

    D = descriptor(kp_iso, sc, f0, f1)          # [DIM] raw description of one span (canonical right-dominant landmarks)
    hb = HandBank.fit(D_bank, c, signer)         # metric (standardise + LDA fitted on training signers only) + projected bank
    S = hb.scores(D_query)                       # [N, C] per-word similarity (top-2 mean, like runtime.Bank), -2 = no example
Used by SignPipeline.back (fusion with the encoder row, weight chosen on tune data by scripts/decision.py).
"""
from __future__ import annotations

import numpy as np

from .schema import LH, RH, SH_L, SH_R

FING = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
PER_FRAME = 28                                     # position (2) + hand shape (26)
DIM = 2 * (PER_FRAME * 5 + 6) + 4


def hand_shape(h):
    """h [T,21,2] → [T,26]: 15 joint-bend cosines, 5 tip–wrist and 4 thumb-tip distances (hand-size units), palm direction."""
    w = h[:, :1]
    size = np.linalg.norm(h[:, 9] - h[:, 0], axis=1)[:, None] + 1e-6
    out = []
    for f in FING:
        v = np.diff(np.concatenate([w, h[:, f]], 1), axis=1)
        v = v / (np.linalg.norm(v, axis=2, keepdims=True) + 1e-6)
        out.append((v[:, :-1] * v[:, 1:]).sum(2))
    out.append(np.linalg.norm(h[:, [4, 8, 12, 16, 20]] - w, axis=2) / size)
    out.append(np.linalg.norm(h[:, [8, 12, 16, 20]] - h[:, 4:5], axis=2) / size)
    out.append((h[:, 9] - h[:, 0]) / size)
    return np.concatenate(out, 1)


def descriptor(kp, sc, f0=None, f1=None, thr=0.3):
    """kp iso [T,133,2] (canonical: dominant = right hand), sc [T,133] → [DIM]. Per hand (dominant first), over the frames where
    it is seen: mean / std of [position, shape], the means of the three temporal thirds, net displacement, path length, position
    relative to the nose, seen fraction; then the hands' relative position (mean) and distance (mean, std)."""
    T = len(kp)
    f0 = 0 if f0 is None else max(0, int(f0)); f1 = T if f1 is None else min(T, int(f1))
    if f1 - f0 < 3:
        f0, f1 = 0, T
    k, s = kp[f0:f1], sc[f0:f1]
    ok = (s[:, SH_L] > thr) & (s[:, SH_R] > thr)
    sw = float(np.median(np.linalg.norm(k[:, SH_L] - k[:, SH_R], axis=1)[ok])) if ok.any() else 0.2
    mid = np.median(((k[:, SH_L] + k[:, SH_R]) / 2)[ok], 0) if ok.any() else np.full(2, 0.5)
    nose = np.median(k[:, 0][s[:, 0] > thr], 0) if (s[:, 0] > thr).any() else mid
    F, M = [], []
    for hand in (RH, LH):
        h = k[:, hand]
        F.append(np.concatenate([(h[:, 0] - mid) / max(sw, 1e-3), hand_shape(h)], 1)); M.append(s[:, hand].mean(1) > thr)
    out = []
    for X, m in zip(F, M):
        if m.sum() >= 2:
            X = X[m]; n = len(X)
            thirds = [X[: max(1, n // 3)].mean(0), X[n // 3: max(n // 3 + 1, 2 * n // 3)].mean(0), X[2 * n // 3:].mean(0)]
            p = X[:, :2]
            out += [X.mean(0), X.std(0), *thirds, p[-1] - p[0], [np.linalg.norm(np.diff(p, axis=0), axis=1).sum()],
                    p.mean(0) + (mid - nose) / max(sw, 1e-3), [m.mean()]]
        else:
            out += [np.zeros(PER_FRAME)] * 5 + [np.zeros(2), [0.0], np.zeros(2), [0.0]]
    both = M[0] & M[1]
    rel = F[0][both, :2] - F[1][both, :2] if both.sum() >= 2 else np.zeros((1, 2))
    d = np.linalg.norm(rel, axis=1)
    out += [rel.mean(0), [d.mean(), d.std()]]
    return np.nan_to_num(np.concatenate([np.ravel(o) for o in out])).astype(np.float32)


def _norm(X):
    return (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)).astype(np.float32)


class HandBank:
    """Projected bank descriptors grouped by concept (same top-2-mean rule as runtime.Bank)."""

    def __init__(self, P, c, n_concepts, tf, allowed=None):
        P, c = np.asarray(P, np.float32), np.asarray(c, np.int64)
        keep = np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)
        if allowed is not None:
            keep &= np.isin(c, np.asarray(sorted(allowed)))
        o = np.argsort(c[keep], kind="stable")
        self.P, self.c = P[keep][o], c[keep][o]
        self.tf, self.n_concepts = tf, n_concepts
        self.present, self.starts = np.unique(self.c, return_index=True)
        self.seg = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, len(self.c)]))

    @staticmethod
    def fit_transform(D, c, signer, shrink=0.1, dim=64):
        """Standardise + shrinkage LDA on training-signer clips: within-word scatter over words with ≥ 2 signers (how the same
        sign varies between people) vs between-word scatter → the directions that separate words across signers."""
        from scipy.linalg import eigh
        D, c = np.asarray(D, np.float64), np.asarray(c)
        mu, sd = D.mean(0), D.std(0) + 1e-3
        X = (D - mu) / sd
        import pandas as pd
        ns = pd.DataFrame(dict(c=c, s=np.asarray(signer))).groupby("c").s.nunique()
        multi = ns[ns >= 2].index.values
        W = np.zeros((X.shape[1],) * 2); M = []; n = 0
        for k in multi:
            A = X[c == k]; m = A.mean(0); M.append(m); A = A - m; W += A.T @ A; n += len(A)
        W /= max(n, 1); M = np.array(M); B = M.T @ M / max(len(M), 1)
        Wr = (1 - shrink) * W + shrink * np.trace(W) / len(W) * np.eye(len(W))
        _, U = eigh(B, Wr)
        return dict(mu=mu.astype(np.float32), sd=sd.astype(np.float32), A=U[:, ::-1][:, :dim].astype(np.float32))

    @staticmethod
    def project(D, tf):
        D = np.atleast_2d(np.asarray(D, np.float32))
        return _norm(((D - tf["mu"]) / tf["sd"]) @ tf["A"])

    @classmethod
    def load(cls, npz, n_concepts, allowed=None, fold=None):
        """fold: runtime.variant_lut (sign variants read as their word)."""
        z = np.load(npz)
        tf = dict(mu=z["mu"], sd=z["sd"], A=z["A"])
        return cls(z["P"], z["c"] if fold is None else fold[z["c"]], n_concepts, tf, allowed)

    def scores(self, D):
        """raw descriptors [N, DIM] → [N, C]: mean of the top-2 projected similarities per concept, -2 where none."""
        S = self.project(D, self.tf) @ self.P.T
        t1 = np.maximum.reduceat(S, self.starts, axis=1)
        S2 = np.where(S >= t1[:, self.seg] - 1e-7, -2.0, S)
        t2 = np.maximum.reduceat(S2, self.starts, axis=1)
        out = np.full((len(S), self.n_concepts), -2.0, np.float32)
        out[:, self.present] = np.where(t2 > -1.5, (t1 + t2) / 2, t1)
        return out


def fuse(enc, hand, w):
    """Encoder row(s) [N, C] + hand-cue row(s) [N, C] → fused ranking score in encoder units: each row is z-scored over the words
    both know, the hand cue added with weight w, then mapped back to the encoder's scale (mean / std of its row) so temperatures
    and score features keep their meaning. Words without a hand example keep the hand cue at its mean (z = 0)."""
    enc, hand = np.atleast_2d(enc), np.atleast_2d(hand)
    ve = enc > -1.5
    vh = (hand > -1.5) & ve
    out = np.full(enc.shape, -2.0, np.float32)
    for i in range(len(enc)):
        e = enc[i, ve[i]]
        me, se = e.mean(), e.std() + 1e-6
        ze = (enc[i] - me) / se
        h = hand[i, vh[i]]
        zh = np.where(vh[i], (hand[i] - h.mean()) / (h.std() + 1e-6), 0.0) if vh[i].sum() > 2 else np.zeros_like(ze)
        out[i] = np.where(ve[i], me + se * (ze + w * zh), -2.0)
    return out
