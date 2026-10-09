"""Targeted harvest for words the dictionaries do not cover (user-approved, 2026-10-10): YouTube single-word Thai sign-language
lessons whose title IS the word. Every step filters noise before anything can become an avatar sign:

  1. candidates   forward project's existing search/crawl results + new searches ("ภาษามือ คำว่า X", "ภาษามือ X"); the title,
                  cleaned by the forward parser (scripts/harvest_words.title_core), must equal the target word exactly;
                  1.5–45 s long; known child-learner / overlay channels from the audit are skipped.
  2. download     ≤ 480p mp4, ≤ 3 per word, polite (sequential, pauses) — no bot-check bypassing.
  3. extract      the same MediaPipe Holistic extractor as all other sources (forward modules.mediapipe_pose).
  4. QC + eye     t2s.data.quality tiers, then a contact sheet that is looked at clip by clip (adult signer, no picture overlay,
                  one sign, label plausible); the decisions are written to t2s/data/youtube_fill.csv.
th-sl.com is NOT harvested (its robots.txt reserves rights against AI agents). TTRS was checked: 5,045 entries online vs 5,043
already held — nothing new there.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths

OUT = paths.HARVEST
FWD_SEARCH = [paths.AGENT / "youtube_words" / "search_raw.jsonl", paths.AGENT / "youtube_words" / "channels_raw.jsonl"]
# channels whose lessons are signed by children / scouts or carry picture overlays (Phase-1 eye check, youtube_fill_*.jpg)
SKIP_SIGNERS = {"yt_person_039", "yt_person_030"}


def _fwd():
    sys.path.insert(0, str(paths.REPO))
    from scripts import harvest_words as hw  # forward project's title parser (shared code, unchanged)
    return hw


def _run(cmd, timeout=180):
    import os
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env).stdout


def candidates(words, per_query=10, new_search=True):
    hw = _fwd()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in FWD_SEARCH:
        if f.exists():
            rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    cache = OUT / "search_raw.jsonl"
    if cache.exists():
        rows += [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(rows).drop_duplicates("id")
    df["core"] = df.title.map(hw.title_core)
    have = set(df[df.core.isin(words)].core)
    if new_search:
        with open(cache, "a", encoding="utf-8") as f:
            for w in words:
                if w in have:
                    continue
                for q in (f"ภาษามือ คำว่า {w}", f"ภาษามือ {w}"):
                    for r in hw.search(q, per_query):
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    time.sleep(1.5)
        rows = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines() if x.strip()]
        df = pd.concat([df, pd.DataFrame(rows)]).drop_duplicates("id")
        df["core"] = df.title.map(hw.title_core)
    c = df[df.core.isin(words) & df.duration.between(1.5, 45) & ~df.title.str.contains("สะกด")].copy()
    c["word"] = c.core
    c.to_csv(OUT / "candidates.csv", index=False, encoding="utf-8-sig")
    return c


def download(c, per_word=3):
    vids = OUT / "videos"
    vids.mkdir(parents=True, exist_ok=True)
    got = []
    for w, g in c.groupby("word"):
        for r in g.sort_values("duration").head(per_word).itertuples():
            out = vids / f"{r.id}.mp4"
            if not out.exists():
                _run(["yt-dlp", "-q", "--no-warnings", "-f", "bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]/bv*[height<=480]/b",
                      "--no-playlist", "-o", str(out), f"https://www.youtube.com/watch?v={r.id}"])
                time.sleep(2.0)
            if out.exists() and out.stat().st_size > 10_000:
                got.append(dict(id=r.id, word=w, channel=r.channel, title=r.title, video=str(out)))
    d = pd.DataFrame(got)
    d.to_csv(OUT / "downloaded.csv", index=False, encoding="utf-8-sig")
    return d


def extract(d):
    sys.path.insert(0, str(paths.REPO))
    from modules.mediapipe_pose import extract as mp_extract, save_raw
    out = OUT / "pose_mp_raw"
    out.mkdir(parents=True, exist_ok=True)
    for r in d.itertuples():
        f = out / f"{r.id}.npz"
        if not f.exists():
            raw = mp_extract(r.video, fps=25.0, max_side=960, hand_fallback=True)
            if raw is not None:
                save_raw(f, raw)
    return out


def qc(d):
    """quality metrics of every downloaded lesson (whole clip = one sign)."""
    from ..motion.clean import clean
    from .quality import clip_quality, load_raw, tier
    rows = []
    for r in d.itertuples():
        f = OUT / "pose_mp_raw" / f"{r.id}.npz"
        if not f.exists():
            rows.append(dict(id=r.id, word=r.word, fatal="no_mediapipe")); continue
        raw = load_raw(f)
        q = clip_quality(clean(raw)[0])
        q.update(id=r.id, word=r.word, channel=r.channel, title=r.title, video=r.video, raw_file=str(f), tier=tier(q))
        rows.append(q)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "qc.csv", index=False, encoding="utf-8-sig")
    return t
