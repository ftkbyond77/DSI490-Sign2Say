"""Reference clips to learn how a sentence is signed (for a person preparing a live test) — not test data and never used by the model.

  python scripts/study_clips.py --out data_test/jack_test --sentences "สวัสดี ฉัน ชื่อ J A C K" "วันนี้ ฉัน จะ มา นำเสนอ งาน"

For every word: up to `--per_word` clear clips from different signers — dictionary videos (th_sl, TTRS), the TSL51 researcher, the
harvested YouTube word lessons (cut to the labelled sign span) and, for single letters, the held letter from an alphabet lesson
(agent_data/en_alpha, th_alpha). Clips are re-encoded to H.264 so any player / browser opens them, and README.md lists them.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from modules.utils import DATA, get_logger

log = get_logger("study")
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def cut(src: Path, dst: Path, t0=None, t1=None):
    cmd = [FFMPEG, "-y", "-loglevel", "error"]
    if t0 is not None:
        cmd += ["-ss", f"{max(0.0, t0):.2f}"]
    cmd += ["-i", str(src)]
    if t0 is not None and t1 is not None:
        cmd += ["-t", f"{max(0.3, t1 - t0):.2f}"]
    cmd += ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True, timeout=120)


def word_sources(word):
    """[(label, video path, t0, t1)] best first: th_sl / TTRS dictionaries, TSL51 researcher, YouTube lessons (span)."""
    out = []
    m = pd.read_parquet(DATA / "data_prep" / "manifest" / "clips.parquet")
    w = m[(m.concept == word) & (m.kind == "isolated") & m.dataset.isin(["th_sl", "ttrs", "tsl51_user_sign"]) & m.source_file.notna()]
    w = w[w.source_file.str.endswith(".mp4")]
    for ds in ("th_sl", "ttrs", "tsl51_user_sign"):
        for r in w[w.dataset == ds].drop_duplicates("signer").itertuples():
            out.append((f"{ds}_{r.signer}", DATA / r.source_file, None, None))
    # TSL51 researcher: the poses are stored as landmarks, the original video is TSL51/videos/user_sign/<clip>.mp4
    t51 = m[(m.concept == word) & (m.dataset == "tsl51_user_sign")]
    for r in t51.head(1).itertuples():
        v = DATA / "raw_data" / "TSL51" / "videos" / "user_sign" / f"{r.clip_id.split(':', 1)[1]}.mp4"
        if v.exists():
            out.insert(min(1, len(out)), (f"tsl51_researcher_{v.stem}", v, None, None))
    yt = DATA / "agent_data" / "youtube_words"
    if (yt / "items.csv").exists():
        it = pd.read_csv(yt / "items.csv")
        it = it[(it.concept == word) & (it.status == "accepted")]
        for r in it.drop_duplicates("signer").itertuples():
            v = yt / r.video
            if v.exists():
                out.append((f"youtube_{v.stem}", v, float(r.f0) / 25.0 - 0.3, float(r.f1) / 25.0 + 0.3))
    return out


def letter_sources(letter):
    out = []
    for s in ("en_alpha", "th_alpha"):
        d = DATA / "agent_data" / s
        if not (d / "items.csv").exists():
            continue
        it = pd.read_csv(d / "items.csv")
        it = it[it.letter == letter].sort_values("agree", ascending=False).drop_duplicates("channel_id")
        for r in it.itertuples():
            v = d / "videos" / f"{r.video}.mp4"
            if v.exists():
                out.append((f"{s}_{r.video}", v, r.f0 / 25.0 - 0.4, r.f1 / 25.0 + 0.4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_test/jack_test")
    ap.add_argument("--sentences", nargs="+", required=True)
    ap.add_argument("--per_word", type=int, default=3)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    words = list(dict.fromkeys(w for s in a.sentences for w in s.split()))
    readme = ["# Study clips (how each word is signed)", "", "Reference only: never used to train or tune the model.", "",
              "Sentences:", ""] + [f"* {s}" for s in a.sentences] + ["", "| word | clips | note |", "|---|---|---|"]
    for i, w in enumerate(words, 1):
        is_letter = len(w) == 1 or w.startswith("สระ")
        src = letter_sources(w) if is_letter else word_sources(w)
        d = out / f"{i:02d}_{w}"
        d.mkdir(exist_ok=True)
        made = []
        for label, v, t0, t1 in src:
            if len(made) >= a.per_word:
                break
            dst = d / f"{label}.mp4"
            try:
                if not dst.exists():
                    cut(v, dst, t0, t1)
                made.append(dst.name)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: %s", dst.name, e)
        note = "" if made else "no clip in any source (see the chat summary)"
        readme.append(f"| {w} | {', '.join(made) if made else '-'} | {note} |")
        log.info("%s: %d clips", w, len(made))
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
