"""Exemplar selection — which recorded clip becomes THE avatar sign for each concept.

Sources (authoritative national dictionaries only): TTRS and th-sl.com (NADT), plus the TSL51 researcher's isolated signs.
YouTube lessons are NOT motion sources (audit: overlays, picture-in-picture, child learners, unknown authority) — they stay
evaluation references.

No source prior: when a concept has clips from several signers, the clip closest to all the others (the DTW medoid of a
signer-independent trajectory: wrists relative to the shoulders, handshape relative to the palm, left-dominant signers mirrored)
wins, so an idiosyncratic or regional variant is not chosen just because one source has more clips. With one or two clips there
is no consensus to measure → the better-quality clip wins and the IR says `consensus: null`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths
from .quality import L_SH, R_SH, L_WR, R_WR, iso3, load_raw, present

SOURCES = ["ttrs", "th_sl", "tsl51_user_sign"]
OUT = paths.WORK / "select"


def features(raw, f0, f1, dominant, n=32):
    """signer-independent trajectory [n, 14] of the span: dominant / other wrist (sw units, mid-shoulder origin, mirrored so
    the dominant hand is always 'right') + dominant fingertips relative to the wrist (palm lengths)."""
    hw = tuple(int(v) for v in raw["hw"])
    P = iso3(raw["pose"], hw)[f0:f1]
    if len(P) < 3:
        return None
    sw = np.median(np.linalg.norm(P[:, L_SH, :2] - P[:, R_SH, :2], axis=1)) or 1.0
    mid = (P[:, L_SH, :2] + P[:, R_SH, :2]) / 2
    dw, ow = (R_WR, L_WR) if dominant == "right" else (L_WR, R_WR)
    hd = iso3(raw["rh" if dominant == "right" else "lh"], hw)[f0:f1]
    sgn = 1.0 if dominant == "right" else -1.0                       # mirror x for left-dominant signers
    a = (P[:, dw, :2] - mid) / sw; b = (P[:, ow, :2] - mid) / sw
    a[:, 0] *= sgn; b[:, 0] *= sgn
    m = present(hd)
    tips = np.zeros((len(P), 10))
    if m.any():
        rel = hd[:, [4, 8, 12, 16, 20], :2] - hd[:, :1, :2]
        size = np.median(np.linalg.norm(hd[m, 9, :2] - hd[m, 0, :2], axis=1)) or 1.0
        rel = rel / size
        rel[..., 0] *= sgn
        idx = np.flatnonzero(m)
        for j in range(10):                                         # fill untracked frames by interpolation
            tips[:, j] = np.interp(np.arange(len(P)), idx, rel[m].reshape(len(idx), 10)[:, j])
    x = np.concatenate([a, b, 0.5 * tips], 1)
    t = np.linspace(0, len(x) - 1, n)
    return np.stack([np.interp(t, np.arange(len(x)), x[:, j]) for j in range(x.shape[1])], 1)


def dtw(a, b, band=8):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    C = np.linalg.norm(a[:, None] - b[None], axis=-1)
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)


def quality_score(r):
    """0..1, higher = cleaner (used to break ties and to rank 1–2 clip concepts)."""
    s = 1.0
    s -= 0.5 * max(0.0, 1.0 - (r.get("hand_rate_dom") or 0))
    s -= 0.6 * min(1.0, (r.get("finger_jitter") or 0) / 0.25) * 0.5
    s -= 0.3 * min(1.0, (r.get("swap_frac") or 0) / 0.08)
    s -= 0.02 * min(10, r.get("hand_gap_dom") or 0)
    s -= 0.25 if r.get("tier") == "B" else 0.0
    return float(np.clip(s, 0, 1))


def fill_rows(t, have):
    """eye-checked fill-ins for concepts no dictionary covers (t2s/data/youtube_fill.csv, cloud_s3/agent_data/reverse_harvest + harvest_decisions.csv)."""
    rows = []
    f = paths.PKG / "data" / "youtube_fill.csv"
    if f.exists():
        dec = pd.read_csv(f)
        ok = set(dec[dec.decision == "accept"].clip_id)
        y = t[t.clip_id.isin(ok) & ~t.concept.isin(have) & t.tier.notna()]
        for r in y.to_dict("records"):
            a = max(0, int(r["yt_f0"]) - 10)
            rows.append(dict(concept=r["concept"], clip_id=r["clip_id"], dataset="youtube_agent", signer=r["signer"], tier=r["tier"],
                             quality=round(quality_score(r), 3), dominant=r["dominant"], mirrored=r["dominant"] == "left",
                             span_f0=int(r["span_f0"]), span_f1=int(r["span_f1"]), n_candidates=1, n_signers=1, consensus=None,
                             gloss=r["gloss"], sign_variant=None, sense=None, raw_file=r["raw_file"], source_file=r["source_file"],
                             trim_a=a, trim_b=int(r["yt_f1"]) + 10))
            have.add(r["concept"])
    h = paths.HARVEST / "qc.csv"
    d = paths.PKG / "data" / "harvest_decisions.csv"
    if h.exists() and d.exists():
        q = pd.read_csv(h).merge(pd.read_csv(d)[["id", "decision", "word"]], on=["id", "word"])
        q = q[(q.decision == "accept") & q.tier.notna() & ~q.word.isin(have)]
        ph = pd.read_csv(h).merge(pd.read_csv(d)[["id", "decision", "word"]], on=["id", "word"])
        for r in ph[(ph.decision == "phrase") & ph.tier.notna()].drop_duplicates("word").to_dict("records"):
            rows.append(dict(concept=f"phrase:{r['word']}", clip_id=f"yth:{r['id']}", dataset="youtube_phrase", signer=r.get("channel"),
                             tier=r["tier"], quality=round(quality_score(r), 3), dominant=r["dominant"], mirrored=r["dominant"] == "left",
                             span_f0=int(r["span_f0"]), span_f1=int(r["span_f1"]), n_candidates=1, n_signers=1, consensus=None,
                             gloss=r["word"], sign_variant=None, sense=None, raw_file=r["raw_file"], source_file=r["video"]))
        for r in q.sort_values("finger_jitter").drop_duplicates("word").to_dict("records"):
            rows.append(dict(concept=r["word"], clip_id=f"yth:{r['id']}", dataset="youtube_harvest", signer=r.get("channel"), tier=r["tier"],
                             quality=round(quality_score(r), 3), dominant=r["dominant"], mirrored=r["dominant"] == "left",
                             span_f0=int(r["span_f0"]), span_f1=int(r["span_f1"]), n_candidates=1, n_signers=1, consensus=None,
                             gloss=r["word"], sign_variant=None, sense=None, raw_file=r["raw_file"], source_file=r["video"]))
    return pd.DataFrame(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(paths.WORK / "audit" / "clips_quality.parquet")
    c = t[t.dataset.isin(SOURCES) & t.tier.notna() & t.concept.notna()].copy()
    c["q"] = [quality_score(r) for r in c.to_dict("records")]
    rows, alts = [], []
    feats = {}
    for concept, g in c.groupby("concept"):
        recs = g.to_dict("records")
        best_tier = "A" if (g.tier == "A").any() else "B"
        recs = [r for r in recs if r["tier"] == best_tier]
        one = {}                                                    # one vote per signer (TSL51 has ~10 takes of one person)
        for r in sorted(recs, key=lambda r: -r["q"]):
            one.setdefault(r["signer"], r)
        recs = list(one.values())
        cons = None
        if len(recs) >= 3:
            F = []
            for r in recs:
                if r["clip_id"] not in feats:
                    raw = load_raw(r["raw_file"])
                    feats[r["clip_id"]] = features(raw, int(r["span_f0"]), int(r["span_f1"]), r["dominant"])
                F.append(feats[r["clip_id"]])
            ok = [i for i, f in enumerate(F) if f is not None]
            recs = [recs[i] for i in ok]; F = [F[i] for i in ok]
            n = len(F)
            D = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    D[i, j] = D[j, i] = dtw(F[i], F[j])
            mean_d = D.sum(1) / max(n - 1, 1)
            score = mean_d + 0.3 * np.array([1 - r["q"] for r in recs])
            k = int(np.argmin(score))
            cons = float(np.exp(-mean_d[k] / 0.6))                   # 1 = everyone signs it the same way
            order = np.argsort(score)
        else:
            order = np.argsort([-r["q"] for r in recs])
            k = int(order[0])
        ch = recs[k]
        rows.append(dict(concept=concept, clip_id=ch["clip_id"], dataset=ch["dataset"], signer=ch["signer"], tier=ch["tier"],
                         quality=round(ch["q"], 3), dominant=ch["dominant"], mirrored=ch["dominant"] == "left",
                         span_f0=int(ch["span_f0"]), span_f1=int(ch["span_f1"]), n_candidates=len(g),
                         n_signers=int(g.signer.nunique()), consensus=cons, gloss=ch["gloss"], sign_variant=ch["sign_variant"],
                         sense=ch["sense"], raw_file=ch["raw_file"], source_file=ch["source_file"]))
        for j in order[1:3]:
            alts.append(dict(concept=concept, clip_id=recs[j]["clip_id"], rank=int(list(order).index(j))))
    sel = pd.DataFrame(rows)
    sel = pd.concat([sel, fill_rows(t, set(sel.concept))], ignore_index=True)
    sel.to_parquet(OUT / "exemplars.parquet", index=False)
    pd.DataFrame(alts).to_parquet(OUT / "alternatives.parquet", index=False)
    print(len(sel), "concepts;", sel.tier.value_counts().to_dict(), "by source", sel.dataset.value_counts().to_dict(),
          "mirrored", int(sel.mirrored.sum()), "with consensus", int(sel.consensus.notna().sum()),
          "median consensus", round(float(sel.consensus.median()), 3))
    return sel
