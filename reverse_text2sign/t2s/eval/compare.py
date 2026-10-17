"""Real signer vs avatar, aligned sign by sign (v2, result_v2.md §5.7 — the prompt's "5 images, top real person, bottom avatar").

v1 showed evenly spaced frames of both, so a column could show the avatar at one sign and the person at another. Here the avatar
signs the sentence FROM WORDS (phrasebook off — the recorded phrase itself is never shown as the avatar) and the real recording
of the same sentence is aligned to it by dynamic time warping on what both share: the two wrists relative to the shoulders and
the dominant hand's finger tips relative to its wrist (shoulder-width / palm units, the same normalisation for both). For every
sign of the avatar the column shows the middle of its stroke (bottom) and the real frame DTW matched to it (top), labelled with
the gloss, plus the per-column distance (how close the hands are at that moment, shoulder widths) — a number the eye can check.
Needs the web app (docker stack on :8081) and Playwright.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from .. import paths
from ..data.quality import L_SH, R_SH, L_WR, R_WR, iso3, load_raw, present
from ..motion.clean import clean, mirror
from ..motion.codec import decode
from ..motion.library import _fk_2d
from ..motion.rig import get_rig
from .render import Renderer, _label, _video_frame

TIPS = [4, 8, 12, 16, 20]
OUT = paths.REPORT / "image"


def real_features(raw):
    cr, _ = clean(raw)
    hw = tuple(int(v) for v in cr["hw"])
    P = iso3(cr["pose"], hw)[:, :, :2]
    mid = (P[:, L_SH] + P[:, R_SH]) / 2
    sw = np.median(np.linalg.norm(P[:, L_SH] - P[:, R_SH], axis=1))
    lw, rw = (P[:, L_WR] - mid) / sw, (P[:, R_WR] - mid) / sw
    H = iso3(cr["rh"], hw)[:, :, :2]
    ok = present(cr["rh"])
    palm = np.median(np.linalg.norm(H[ok, 9] - H[ok, 0], axis=1)) if ok.any() else 1.0
    tips = (H[:, TIPS] - H[:, :1]) / max(palm, 1e-6)
    tips[~ok] = np.nan
    F = np.concatenate([lw, rw, 0.5 * tips.reshape(len(P), -1)], 1)
    for c in range(F.shape[1]):                                 # short tracking gaps: interpolate; never-tracked: 0
        x = F[:, c]
        bad = ~np.isfinite(x)
        if bad.all():
            F[:, c] = 0
        elif bad.any():
            F[bad, c] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return F


def avatar_features(q, rig):
    A, _ = _fk_2d(q, rig)
    lw, rw = A["leftHand"], A["rightHand"]
    palm = np.median(np.linalg.norm(A["rightMiddleProximal"] - A["rightHand"], axis=1))
    tips = np.stack([(A[f"right{f}Tip"] - A["rightHand"]) / palm for f in ("Thumb", "Index", "Middle", "Ring", "Little")], 1)
    return np.concatenate([lw, rw, 0.5 * tips.reshape(len(q), -1)], 1)


def dtw(A, B):
    """path [(i, j)] aligning A [n, d] and B [m, d] (Euclidean, standard steps)."""
    n, m = len(A), len(B)
    D = np.linalg.norm(A[:, None] - B[None], axis=-1)
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return path[::-1], D


def sheet(R, text, raw_file, video, span, mirrored=False, size=230, name="cmp"):
    rig = get_rig()
    res = R.api("/api/translate", {"text": text, "phrasebook": False})
    R.load(res)
    q, _ = decode(res["motion"])
    segs = [s for s in res["motion"]["segments"] if s["kind"] in ("sign", "fingerspell", "phrase")]
    a0, a1 = segs[0]["start"], segs[-1]["end"]
    raw = load_raw(raw_file)
    if mirrored:
        raw = mirror(raw)
    f0, f1 = max(0, int(span[0]) - 4), min(len(raw["t"]), int(span[1]) + 4)
    FA = avatar_features(q, rig)[a0:a1]
    FR = real_features(raw)[f0:f1]
    path, D = dtw(FA, FR)
    match = {}
    for i, j in path:
        match.setdefault(i, []).append(j)
    top, bot, scores = [], [], []
    for s in segs:
        k = (s["start"] + s["end"]) // 2
        j = int(np.median(match[k - a0])) + f0
        scores.append(float(np.linalg.norm(FA[k - a0][:4] - FR[j - f0][:4]) / 2))      # wrists, shoulder widths (per hand)
        t_real = float(raw["t"][min(j, len(raw["t"]) - 1)])
        v = _video_frame(video, t_real, size)
        if mirrored:
            v = cv2.flip(v, 1)
        av = cv2.resize(R.frame(k / 25.0), (size, size))
        top.append(_label(v, s["label"]))
        bot.append(_label(av, f"{s['label']}  Δ {scores[-1]:.2f}"))
    img = np.vstack([np.hstack(top), np.hstack(bot)])
    gl = " · ".join(s["label"] for s in segs)
    head = np.full((34, img.shape[1], 3), 255, np.uint8)
    img = np.vstack([_label(head, f"{text}  →  {gl}   ·  Δ = wrist distance (shoulder widths)"), img])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    cv2.imencode(".png", img)[1].tofile(str(p))
    return dict(text=text, glosses=gl, file=str(p.relative_to(paths.HERE)), wrist_delta_mean=round(float(np.mean(scores)), 3),
                wrist_delta=[round(x, 3) for x in scores], dtw_cost=round(float(np.mean([D[i, j] for i, j in path])), 3),
                dur_avatar=round((a1 - a0) / 25, 2), dur_real=round((f1 - f0) / 25, 2))


def run(base_url="http://localhost:8081"):
    import pandas as pd
    q = pd.read_csv(paths.HARVEST / "qc.csv")
    d = pd.read_csv(paths.PKG / "data" / "harvest_decisions.csv")
    ph = q.merge(d[d.decision == "phrase"][["id", "word"]], on=["id", "word"]).drop_duplicates("word").set_index("word")
    items = []
    m = pd.read_parquet(paths.MANIFEST)
    pairs = json.loads((paths.PREP / "params" / "tsl51_thai_sentences.json").read_text(encoding="utf-8"))
    # TSL51: the same signer recorded the words (in the library) and the sentence — differences show OUR system, not a variant
    for gloss in ("คุณ ชื่อ อะไร", "พรุ่งนี้ คุณ ข้าว กิน ไป ที่ไหน", "ขอบคุณ ฉัน สบายดี", "กรุงเทพ ฉัน เที่ยว ชอบ"):
        s = m[(m.dataset == "tsl51_user_sentence") & (m.lemma == gloss)].iloc[0]
        items.append((pairs[gloss], paths.mp_raw_path(s.clip_id, s.dataset), paths.video_path(s.clip_id, s.dataset, s.source_file),
                      (s.span_f0, s.span_f1), False))
    # another signer, conversational speed (YouTube recorded phrase): the hard case — other variants, other tempo
    r = ph.loc["คุณกินข้าวหรือยัง"]
    items.append(("คุณกินข้าวหรือยัง", paths.HARVEST / "pose_mp_raw" / f"{r.id}.npz", paths.HARVEST / "videos" / f"{r.id}.mp4",
                  (r.span_f0, r.span_f1), r.dominant == "left"))
    R = Renderer(base_url, size=(560, 560))
    out = []
    try:
        for k, (text, raw, video, span, mir) in enumerate(items, 1):
            out.append(sheet(R, text, raw, video, span, mir, name=f"cmp_{k}"))
    finally:
        R.close()
    (paths.WORK / "eval").mkdir(parents=True, exist_ok=True)
    (paths.WORK / "eval" / "compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
