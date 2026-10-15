"""Sentence-level non-manual features measured on recorded sentences (v2, result_v2.md §5.5) — the evidence behind
t2s/motion/nmm.py. Nothing here is assumed from other sign languages: every overlay the avatar uses cites a number from this run.

Data: TSL51 user sentences (252 takes of 76 gloss sequences, ONE signer, gold TSL gloss order) and the 40 conversational phrases
recorded by adult signers on YouTube (several signers; the only yes/no questions in the lake). Per clip, on the cleaned MediaPipe
landmarks inside the signing span (manifest / harvest span):
  head yaw    (nose − mid-ears) x / ear distance;  shake = RMS of yaw after removing its 0.6 s moving average (oscillation, not a
              turn), + the number of yaw reversals larger than 0.02
  head pitch  (nose − mid-ears) y / ear distance (nod / forward tilt)
  brows       mean(browInnerUp, browOuterUp L/R) − the clip's own neutral (first frames before the span)
in 5 equal time bins of the span (where in the sentence a mark sits) and in the last 40 % (where TSL puts negation / WH).
Groups: negative (a ไม่… sign), WH (อะไร ที่ไหน ใคร เมื่อไร ทำไม อย่างไร เท่าไร), topic-first (the first sign is the object /
place, not the subject or a time word), plain statements. Tests: Mann–Whitney U, one-sided, vs plain statements.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import paths
from ..data.quality import load_raw
from ..motion.clean import clean

WH = {"อะไร", "ที่ไหน", "ใคร", "เมื่อไร", "ทำไม", "อย่างไร", "เท่าไร", "ไหน"}
PERSON = {"ฉัน", "คุณ", "เขา", "เธอ", "เรา", "น้อง", "พี่", "ยาย", "ตา", "แม่", "พ่อ", "แฟน", "เพื่อน", "ครู", "หมอ", "ลูก", "ป้า", "ลุง"}
TIME = {"พรุ่งนี้", "วันนี้", "เมื่อวาน", "ตอนนี้"}
BS = ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"]


def _bs_index():
    from ..motion.retarget import BS as IDX
    return [IDX[b] for b in BS]


def features(raw, f0, f1):
    cr, _ = clean(raw)
    pose = cr["pose"]
    T = len(pose)
    f0, f1 = max(0, int(f0)), min(T, int(f1))
    if f1 - f0 < 10:
        return None
    ears = (pose[:, 7, :2] + pose[:, 8, :2]) / 2
    ed = np.linalg.norm(pose[:, 7, :2] - pose[:, 8, :2], axis=1)
    ok = (pose[:, 0, 3] > 0.5) & (pose[:, 7, 3] > 0.3) & (pose[:, 8, 3] > 0.3) & (ed > 1e-4)
    yaw = np.where(ok, (pose[:, 0, 0] - ears[:, 0]) / np.maximum(ed, 1e-4), np.nan)
    pitch = np.where(ok, (pose[:, 0, 1] - ears[:, 1]) / np.maximum(ed, 1e-4), np.nan)
    yaw = pd.Series(yaw).interpolate(limit_direction="both").to_numpy()
    pitch = pd.Series(pitch).interpolate(limit_direction="both").to_numpy()
    k = 15                                                          # 0.6 s at 25 fps
    trend = pd.Series(yaw).rolling(k, center=True, min_periods=1).mean().to_numpy()
    osc = yaw - trend
    bs = np.nan_to_num(cr["bs"][:, _bs_index()].astype(float)).mean(1)
    base = np.median(bs[:max(3, f0)]) if f0 >= 3 else np.percentile(bs, 20)
    brow = bs - base
    pbase = np.median(pitch[:max(3, f0)]) if f0 >= 3 else np.median(pitch)
    s = slice(f0, f1)
    n = f1 - f0
    bins = np.array_split(np.arange(f0, f1), 5)
    last = slice(f0 + int(0.6 * n), f1)
    d = np.diff(osc[s])
    rev = int(((d[1:] * d[:-1]) < 0).sum()) if len(d) > 2 else 0
    amp_ok = np.abs(osc[s]).max() > 0.02
    return dict(shake=float(np.sqrt(np.mean(osc[s] ** 2))), shake_last=float(np.sqrt(np.mean(osc[last] ** 2))),
                reversals=rev if amp_ok else 0, brow_last=float(brow[last].mean()), brow_first=float(brow[bins[0]].mean()),
                brow_bins=[float(brow[b].mean()) for b in bins], pitch_last=float(np.nanmean(pitch[last] - pbase)),
                pitch_bins=[float(np.nanmean(pitch[b] - pbase)) for b in bins], frames=n)


def _group(glosses):
    g = glosses.split()
    if any(x.startswith("ไม่") for x in g):
        return "negative"
    if any(x in WH for x in g):
        return "wh"
    first = g[0]
    if first in TIME:
        first = g[1] if len(g) > 1 else first
    return "statement_subject_first" if first in PERSON else "topic_first"


def tsl51():
    m = pd.read_parquet(paths.MANIFEST)
    s = m[m.dataset == "tsl51_user_sentence"]
    rows = []
    for r in s.itertuples():
        f = paths.mp_raw_path(r.clip_id, r.dataset)
        if not f.exists():
            continue
        raw = load_raw(f)
        x = features(raw, r.span_f0 if r.span_f0 == r.span_f0 else 0, r.span_f1 if r.span_f1 == r.span_f1 else len(raw["t"]))
        if x:
            rows.append(dict(clip=r.clip_id, glosses=r.lemma, group=_group(r.lemma), source="tsl51", signer=r.signer, **x))
    return pd.DataFrame(rows)


def phrases():
    q = pd.read_csv(paths.HARVEST / "qc.csv")
    d = pd.read_csv(paths.PKG / "data" / "harvest_decisions.csv")
    ph = q.merge(d[d.decision == "phrase"][["id", "word"]], on=["id", "word"]).drop_duplicates("word")
    rows = []
    for r in ph.itertuples():
        f = paths.HARVEST / "pose_mp_raw" / f"{r.id}.npz"
        if not f.exists():
            continue
        x = features(load_raw(f), r.span_f0, r.span_f1)
        if not x:
            continue
        w = r.word
        typ = ("yn" if any(p in w for p in ("หรือยัง", "ไหม", "มั้ย", "หรือเปล่า", "ใช่ไหม")) else
               "wh" if any(p in w for p in WH | {"กี่"}) else "negative" if "ไม่" in w else "statement")
        rows.append(dict(clip=r.id, glosses=w, group=typ, source="youtube_phrase", signer=r.channel, **x))
    return pd.DataFrame(rows)


def _mw(a, b, alt="greater"):
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return None
    return float(mannwhitneyu(a, b, alternative=alt).pvalue)


def run():
    t, p = tsl51(), phrases()
    out = dict(n_tsl51=len(t), n_phrases=len(p), groups_tsl51=t.group.value_counts().to_dict(), groups_phrases=p.group.value_counts().to_dict())
    st = t[t.group == "statement_subject_first"]
    neg = t[t.group == "negative"]
    wh = t[t.group == "wh"]
    top = t[t.group == "topic_first"]
    out["negation"] = dict(shake_last_neg=float(neg.shake_last.median()), shake_last_statement=float(st.shake_last.median()),
                           p_shake_last=_mw(neg.shake_last, st.shake_last), reversals_neg=float(neg.reversals.median()),
                           reversals_statement=float(st.reversals.median()), p_reversals=_mw(neg.reversals, st.reversals),
                           brow_last_neg=float(neg.brow_last.median()), brow_last_statement=float(st.brow_last.median()),
                           p_brow_down=_mw(neg.brow_last, st.brow_last, "less"))
    out["wh"] = dict(brow_bins_wh=np.median(np.stack(wh.brow_bins), 0).round(4).tolist() if len(wh) else None,
                     brow_bins_statement=np.median(np.stack(st.brow_bins), 0).round(4).tolist(),
                     p_brow_last=_mw(wh.brow_last, st.brow_last))
    out["topic"] = dict(brow_first_topic=float(top.brow_first.median()), brow_first_subject=float(st.brow_first.median()),
                        p_brow_first=_mw(top.brow_first, st.brow_first), n_topic=len(top), n_subject=len(st))
    if len(p):
        yn, ps = p[p.group == "yn"], p[p.group == "statement"]
        out["phrases"] = dict(n_yn=len(yn), n_statement=len(ps), n_wh=int((p.group == "wh").sum()),
                              brow_last_yn=float(yn.brow_last.median()) if len(yn) else None,
                              brow_last_statement=float(ps.brow_last.median()) if len(ps) else None,
                              p_brow_last_yn=_mw(yn.brow_last, ps.brow_last),
                              pitch_last_yn=float(yn.pitch_last.median()) if len(yn) else None,
                              pitch_last_statement=float(ps.pitch_last.median()) if len(ps) else None,
                              p_pitch_yn=_mw(yn.pitch_last, ps.pitch_last))
    (paths.WORK / "eval").mkdir(parents=True, exist_ok=True)
    pd.concat([t, p]).to_csv(paths.WORK / "eval" / "nmm_sentences.csv", index=False, encoding="utf-8-sig")
    (paths.WORK / "eval" / "nmm_evidence.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
