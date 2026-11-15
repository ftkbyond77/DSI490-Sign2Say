"""Stream a SUBSET of airesearch/thai-ser (no full 41 h download) -> thai-ser/audio/*.{flac,wav} + thai-ser/meta.csv

  python fetch_thaiser.py --n 3000
Prints the dataset's columns first; audio bytes are written as-is (no decoding, so no torchcodec/ffmpeg needed).
"""
import argparse, csv, json, os
from pathlib import Path
from datasets import load_dataset, Audio

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--split", default="train")
ap.add_argument("--out", default="thai-ser")
a = ap.parse_args()

ds = load_dataset("airesearch/thai-ser", split=a.split, streaming=True)
acol = next((c for c, f in ds.features.items() if isinstance(f, Audio)), None)
print("columns:", {c: str(f)[:80] for c, f in ds.features.items()})
if acol:
    ds = ds.cast_column(acol, Audio(decode=False))
out = Path(a.out); (out / "audio").mkdir(parents=True, exist_ok=True)
rows = []
for i, ex in enumerate(ds):
    if i >= a.n:
        break
    meta = {k: v for k, v in ex.items() if k != acol}
    if acol:
        au = ex[acol]
        name = Path(au.get("path") or f"{i}.flac").name
        name = f"{i:05d}_{name}" if not name.startswith(f"{i:05d}") else name
        (out / "audio" / name).write_bytes(au["bytes"])
        meta["file"] = f"audio/{name}"
    rows.append({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in meta.items()})
    if i % 250 == 0:
        print(i, {k: str(v)[:40] for k, v in meta.items()}, flush=True)
cols = sorted({k for r in rows for k in r})
with open(out / "meta.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
print(f"saved {len(rows)} clips -> {out}/meta.csv")
