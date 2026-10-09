"""Phase 3 — ground the composer's timing in recorded signing (→ models/timing.json).

1. Reach model (how long a hand takes to travel d shoulder widths to a sign): measured on the LIFT phase of dictionary clips —
   from movement onset (wrist speed > 15 % of the lift's peak) to the first hold (speed minimum after the peak) — in the
   signers' own landmarks; robust linear fit D = c0 + c1·d.
2. Sentence tempo (citation form vs connected signing, same person): TSL51's researcher signs every word in isolation AND in
   sentences. tempo = (sentence signing span) / (Σ her isolated stroke durations + Σ predicted transitions); < 1 means words are
   signed faster inside sentences — the composer plays dictionary strokes at 1/tempo of their citation speed.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .. import paths
from ..data.quality import L_SH, R_SH, L_WR, R_WR, iso3, load_raw


def _wrist_traj(raw):
    hw = tuple(int(v) for v in raw["hw"])
    P = iso3(raw["pose"], hw)
    sw = float(np.median(np.linalg.norm(P[:, L_SH, :2] - P[:, R_SH, :2], axis=1)))
    W = P[:, [L_WR, R_WR], :2] / sw
    if len(W) >= 7:
        W = savgol_filter(W, 7, 2, axis=0)
    return W


def lift_phase(raw, f0, f1):
    """(duration s, distance sw) of the rest → first-hold movement of the hand that moves more, or None."""
    W = _wrist_traj(raw)
    sp = np.linalg.norm(np.diff(W, axis=0), axis=-1)                    # [T-1, 2] per frame
    k = int(np.argmax(sp[f0:f1].sum(0)))
    s = sp[:, k]
    a0 = max(0, f0 - 15)
    seg = s[a0:min(len(s), f0 + 40)]
    if len(seg) < 8:
        return None
    pk = int(np.argmax(seg))
    if seg[pk] < 0.02:
        return None
    on = pk
    while on > 0 and seg[on] > 0.15 * seg[pk]:
        on -= 1
    end = pk
    while end + 1 < len(seg) and not (seg[end + 1] > seg[end] and seg[end] < 0.35 * seg[pk]):
        end += 1
        if seg[end] < 0.12 * seg[pk]:
            break
    if end - on < 3:
        return None
    d = float(np.linalg.norm(W[a0 + end, k] - W[a0 + on, k]))
    return (end - on) / 25.0, d


def reach_model(n=2000, seed=0):
    sel = pd.read_parquet(paths.WORK / "select" / "exemplars.parquet")
    sel = sel[sel.dataset.isin(["ttrs", "th_sl"])].sample(min(n, len(sel)), random_state=seed)
    rows = []
    for r in sel.itertuples():
        raw = load_raw(r.raw_file)
        x = lift_phase(raw, int(r.span_f0), int(r.span_f1))
        if x and 0.3 < x[1] < 4.0:
            rows.append(dict(dataset=r.dataset, dur=x[0], dist=x[1]))
    d = pd.DataFrame(rows)
    # least squares (durations are quantised to 40 ms, so a median-of-slopes fit collapses to 0); per-sextile medians kept as evidence
    c1, c0 = np.polyfit(d.dist, d.dur, 1)
    bins = d.groupby(pd.qcut(d.dist, 6), observed=True).dur.median()
    return dict(c0=float(c0), c1=float(c1), n=len(d), median_dur=float(d.dur.median()), median_dist=float(d.dist.median()),
                by_distance={f"{iv.left:.2f}-{iv.right:.2f}": float(v) for iv, v in bins.items()}), d


def sentence_tempo(c0, c1):
    """TSL51: same researcher, isolated words vs sentences."""
    from ..motion.compose import FPS  # noqa: F401
    q = pd.read_parquet(paths.WORK / "audit" / "clips_quality.parquet")
    words = q[(q.dataset == "tsl51_user_sign") & q.span_s.notna() & q.fatal.isna()].copy()
    words["g"] = words.gloss.str.replace(r"_var_\d+$", "", regex=True)
    lifts = []
    for r in words.itertuples():                       # citation strokes = span − lift − drop (drop ≈ lift)
        x = lift_phase(load_raw(r.raw_file), int(r.span_f0), int(r.span_f1))
        lifts.append(x[0] if x else np.nan)
    words["lift"] = lifts
    L = float(np.nanmedian(words.lift))
    words["stroke"] = (words.span_s - 2 * words.lift.fillna(L)).clip(lower=0.2)
    iso = words.groupby("g").stroke.median()
    sents = q[(q.dataset == "tsl51_user_sentence") & q.span_s.notna() & q.fatal.isna()]
    rows = []
    for r in sents.itertuples():
        gl = [re.sub(r"_var_\d+$", "", x) for x in r.gloss.split("__")]
        if not all(g in iso.index for g in gl):
            continue
        strokes = float(sum(iso[g] for g in gl))
        trans = (len(gl) - 1) * (c0 + c1 * 1.0)
        core = float(r.span_s) - 2 * L
        rows.append(dict(clip=r.clip_id, n=len(gl), sentence_s=float(r.span_s), core_s=core, isolated_strokes_s=strokes,
                         lift_s=L, ratio=core / (strokes + trans)))
    d = pd.DataFrame(rows)
    return float(d.ratio.median()), d


def phrase_tempo():
    """conversational pace: 28+ everyday phrases recorded by an adult signer (harvest 2026-10-10) vs our citation-speed generation of
    the same Thai sentence (all words signable): real = a·strokes + b·transitions (non-negative least squares)."""
    from scipy.optimize import nnls
    from ..motion import compose
    from ..pipeline import Engine
    saved = dict(compose.timing())
    compose.timing.cache_clear()
    e = Engine(english=False)
    dec = pd.read_csv(paths.PKG / "data" / "harvest_decisions.csv")
    qc = pd.read_csv(paths.HARVEST / "qc.csv")
    ph = dec[dec.decision == "phrase"].merge(qc[["id", "word", "span_s"]], on=["id", "word"]).drop_duplicates("word")
    rows = []
    for r in ph.itertuples():
        ir, mo = e.run(r.word, phrasebook=False)
        segs = [x for x in mo["segments"] if x["kind"] != "transition"]
        if not segs or any(u.kind != "sign" for u in ir.units):
            continue
        a, b = segs[0]["start"], segs[-1]["end"]
        inner = [x for x in mo["segments"] if x["kind"] == "transition" and a <= x["start"] < b]
        rows.append(dict(phrase=r.word, n=len(segs), real_s=r.span_s, strokes_s=sum(x["end"] - x["start"] for x in segs) / 25,
                         trans_s=sum(x["end"] - x["start"] for x in inner) / 25, generated_s=(b - a) / 25))
    d = pd.DataFrame(rows)
    (coef, _) = nnls(d[["strokes_s", "trans_s"]].values, d.real_s.values)
    return dict(n=len(d), median_ratio=round(float((d.real_s / d.generated_s).median()), 3), stroke_factor=round(float(coef[0]), 3),
                transition_factor=round(float(coef[1]), 3), timing_used=saved), d


def run():
    rm, d = reach_model()
    tempo, ds = sentence_tempo(rm["c0"], rm["c1"])
    # TSL51 sentences are NOT faster than citation strokes + transitions (ratio > 1) → strokes are played at citation speed
    out = dict(c0=round(rm["c0"], 3), c1=round(rm["c1"], 3), tempo=1.0, stroke_speed=round(float(np.clip(1.0 / tempo, 1.0, 1.6)), 3),
               min_s=0.3, max_s=0.6, rest_c0=round(rm["c0"] + 0.1, 3), rest_c1=round(rm["c1"], 3), clause_hold_s=0.25, hold_cap_s=0.24,
               text_pause_s=0.9, letter_hold_s=0.30, letter_trans_s=0.12,
               evidence=dict(reach=rm, sentence_tempo=dict(median_ratio=round(tempo, 3), n=len(ds),
                                                            note="TSL51 researcher: sentence span / (isolated spans + transitions)")))
    # conversational pace (real everyday phrases) is ~1.7× faster than citation strokes; transitions nearly vanish (co-articulation).
    # Applied a little gentler than measured (an avatar is read, not overheard): strokes ×1.5, transitions ≈ 0.25 s, holds ≤ 0.12 s.
    try:
        pt, dp = phrase_tempo()
        dp.to_csv(paths.WORK / "eval" / "phrase_tempo.csv", index=False, encoding="utf-8-sig")
        out["evidence"]["conversational_tempo"] = {k: v for k, v in pt.items() if k != "timing_used"}
        out.update(stroke_speed=1.5, c0=0.25, c1=0.003, min_s=0.18, max_s=0.4, hold_cap_s=0.12)
    except FileNotFoundError:
        pass
    (paths.MODELS / "timing.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (paths.WORK / "eval").mkdir(parents=True, exist_ok=True)
    d.to_csv(paths.WORK / "eval" / "reach_samples.csv", index=False)
    ds.to_csv(paths.WORK / "eval" / "sentence_tempo.csv", index=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out
