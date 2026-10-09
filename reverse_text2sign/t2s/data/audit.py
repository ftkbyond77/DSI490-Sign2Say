"""Phase 1 — dataset audit for text→sign. One table (data/audit/clips_quality.parquet) with every clip that could become avatar motion:
its source, labels and the quality metrics of t2s.data.quality, plus the rule failures. Contact sheets per failure reason are drawn by
`sheets()` so every kind of noise is looked at by eye before a threshold is trusted."""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from .quality import clip_quality, failures, load_raw, tier

OUT = paths.WORK / "audit"
MOTION_SOURCES = ["ttrs", "th_sl", "tsl51_user_sign", "tsl51_user_sentence"]   # sources with video-derived MediaPipe (x, y, z + face)


def _agent_rows():
    """v7/v8 YouTube single-word harvest (agent_data/youtube_words) with the v8 audit verdicts."""
    q = pd.read_csv(paths.AGENT / "youtube_words" / "qc" / "quality_v8.csv")
    rows = []
    for r in q.itertuples():
        f = paths.MP_RAW / "agent" / (r.clip_id.replace(":", "_") + ".npz")
        rows.append(dict(clip_id=r.clip_id, dataset="youtube_agent", kind="isolated", signer=r.signer, gloss=r.word, lemma=r.word,
                         concept=r.concept, sign_variant=None, sense=None, category=r.channel if isinstance(r.channel, str) else None,
                         source_file=str(Path("agent_data/youtube_words") / r.video), raw_file=str(f), width=r.width, height=r.height,
                         v8_keep=bool(r.keep), v8_reason=r.reason if isinstance(r.reason, str) else None,
                         yt_f0=r.f0, yt_f1=r.f1, cuts_in_span=r.cuts_in_span, faces_max=r.faces_max))
    return pd.DataFrame(rows)


def clip_table():
    c = pd.read_parquet(paths.MANIFEST)
    c = c[c.dataset.isin(MOTION_SOURCES)].copy()
    c["raw_file"] = [str(paths.mp_raw_path(i, d)) for i, d in zip(c.clip_id, c.dataset)]
    c["source_file"] = [str(paths.video_path(i, d, s).relative_to(paths.DATA_ROOT)) for i, d, s in zip(c.clip_id, c.dataset, c.source_file)]
    keep = ["clip_id", "dataset", "kind", "signer", "gloss", "lemma", "concept", "sign_variant", "sense", "category", "source_file",
            "raw_file", "width", "height", "role", "exclude_reason", "n_persons_max", "multi_person_frac", "span_f0", "span_f1"]
    c = c[keep].rename(columns={"span_f0": "m_span_f0", "span_f1": "m_span_f1", "exclude_reason": "prep_exclude"})
    a = _agent_rows()
    return pd.concat([c, a], ignore_index=True)


def _one(args):
    cid, f, f0, f1 = args
    p = Path(f)
    if not p.exists():
        return cid, {"fatal": "no_mediapipe"}
    raw = load_raw(p)
    if f0 is not None and not np.isnan(f0):                 # YouTube lessons: only the annotated sign span (+0.4 s each side)
        a, b = max(0, int(f0) - 10), min(len(raw["t"]), int(f1) + 10)
        raw = {k: (v[a:b] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(raw["t"]) else v) for k, v in raw.items()}
    try:
        from ..motion.clean import clean
        cr, rep = clean(raw)                                # quality is measured on what the avatar would actually play
        q = clip_quality(cr)
        q.update({f"clean_{k}": v for k, v in rep.items() if k != "sw"})
        return cid, q
    except Exception as e:                                  # noqa: BLE001 — a broken file is a finding, not a crash
        return cid, {"fatal": f"error:{type(e).__name__}"}


def run(procs=6):
    OUT.mkdir(parents=True, exist_ok=True)
    t = clip_table()
    f0 = t.get("yt_f0", pd.Series([None] * len(t)))
    jobs = [(r.clip_id, r.raw_file, (f0.iloc[i] if r.dataset == "youtube_agent" else None),
             (t.yt_f1.iloc[i] if r.dataset == "youtube_agent" else None)) for i, r in enumerate(t.itertuples())]
    with Pool(procs) as pool:
        res = dict(pool.map(_one, jobs, chunksize=64))
    q = pd.DataFrame([dict(clip_id=k, **v) for k, v in res.items()])
    t = t.merge(q, on="clip_id", how="left")
    t["fail"] = [("|".join(failures(r)) or None) for r in t.to_dict("records")]
    ok_src = t["prep_exclude"].isna() & (t.get("v8_keep", True) != False)  # noqa: E712
    t["tier"] = [tier(r) if ok else None for r, ok in zip(t.to_dict("records"), ok_src)]
    t["usable"] = t["fail"].isna() & ok_src
    t.to_parquet(OUT / "clips_quality.parquet", index=False)
    summary(t)
    return t


def summary(t):
    rows = []
    for ds, d in t.groupby("dataset"):
        rows.append(dict(dataset=ds, clips=len(d), usable=int(d.usable.sum()), concepts=d.concept.nunique(),
                         usable_concepts=d[d.usable].concept.nunique(), tier_b=int((d.tier == "B").sum()),
                         concepts_a_or_b=d[d.tier.notna()].concept.nunique(), signers=d.signer.nunique(),
                         left_dominant=float((d.dominant == "left").mean()),
                         median_span_s=float(d.span_s.median()), median_hand_rate=float(d.hand_rate_dom.median())))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    reasons = (t.assign(fail=t["fail"].fillna("")).fail.str.split("|").explode())
    reasons = reasons[reasons != ""]
    rc = t.loc[reasons.index, "dataset"].to_frame().assign(reason=reasons.values).value_counts().rename("clips").reset_index()
    rc.to_csv(OUT / "fail_reasons.csv", index=False)
    print(s.to_string(index=False))
    print(rc.to_string(index=False))


# ----------------------------------------------------------------------------- contact sheets (looking at the noise)
def _frame(video, t_s, size=180):
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return np.zeros((size, size, 3), np.uint8)
    h, w = fr.shape[:2]
    s = size / max(h, w)
    fr = cv2.resize(fr, (int(w * s), int(h * s)))
    out = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size - fr.shape[0]) // 2, (size - fr.shape[1]) // 2
    out[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    return out


def contact_sheet(rows, path, title="", cols=8, size=180, frames_per_clip=1):
    """rows: records with source_file, span_f0/span_f1, label → one jpg grid (middle of the sign span; frames_per_clip spread)."""
    import cv2
    tiles = []
    for r in rows:
        vid = paths.DATA_ROOT / r["source_file"]
        f0 = r.get("span_f0") if r.get("span_f0") == r.get("span_f0") else 0
        f1 = r.get("span_f1") if r.get("span_f1") == r.get("span_f1") else 50
        off = (r.get("yt_f0") or 0) if r.get("dataset") == "youtube_agent" and r.get("yt_f0") == r.get("yt_f0") else 0
        off = max(0, off - 10) if off else 0
        for k in range(frames_per_clip):
            fr = f0 + (f1 - f0) * (k + 1) / (frames_per_clip + 1)
            im = _frame(vid, (off + fr) / 25.0, size)
            lab = f"{r.get('concept','')}"[:14]
            cv2.rectangle(im, (0, size - 34), (size, size), (0, 0, 0), -1)
            cv2.putText(im, (r.get("fail") or "ok")[:26], (3, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
            cv2.putText(im, f"{r['dataset'][:8]} {r['clip_id'][-10:]}", (3, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
            tiles.append(im)
    if not tiles:
        return None
    n = len(tiles); rows_n = (n + cols - 1) // cols
    sheet = np.zeros((rows_n * size, cols * size, 3), np.uint8)
    for i, im in enumerate(tiles):
        sheet[(i // cols) * size:(i // cols + 1) * size, (i % cols) * size:(i % cols + 1) * size] = im
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tofile(str(path))
    return path


def sheets(n=32, seed=0):
    """One sheet per (dataset, failure reason) and one random 'usable' sheet per dataset."""
    t = pd.read_parquet(OUT / "clips_quality.parquet")
    rng = np.random.default_rng(seed)
    made = []
    for ds, d in t.groupby("dataset"):
        ok = d[d.usable]
        if len(ok):
            pick = ok.iloc[rng.choice(len(ok), min(n, len(ok)), replace=False)]
            made.append(contact_sheet(pick.to_dict("records"), OUT / "sheets" / f"{ds}__usable.jpg"))
        reasons = d["fail"].dropna().str.split("|").explode()
        for reason in reasons.unique():
            dd = d.loc[reasons[reasons == reason].index.unique()]
            pick = dd.iloc[rng.choice(len(dd), min(n, len(dd)), replace=False)]
            made.append(contact_sheet(pick.to_dict("records"), OUT / "sheets" / f"{ds}__{reason}.jpg"))
    return [m for m in made if m]
