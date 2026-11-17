"""v12 TPH 1B: MediaPipe Holistic (production extractor, normalize=False) on the two new sources → data_prep/pose_mp_raw + pose_mp.

  python scripts/v12_extract.py items                 # build data_prep/params/v12_video_items.csv (priority order of prompt 1B)
  python scripts/v12_extract.py run [--workers 9] [--stage 1..6] [--limit N]

Sources: royal_institute (cloud_s3/sign-ราชบัณฑิต) and etsl (cloud_s3/sign-etsl-chula). Clip ids: rid:<stem> / etsl:<stem>.
Priority: 1 in-vocabulary words → 2 RID single-person sentences → 3 e-TSL phrases / sentences → 4 remaining words → 5 classifier → 6 letters.
Two-person question-answer clips (stage 7) need the signer lock (scripts/v12_signer_lock.py writes a crop column) before extraction.
The 202 clips already extracted by EA-v12 (cache/v12_ea/ood_raw) are reused (same extractor, same settings).
Resumable: finished clips are skipped.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from modules.utils import PREP, safe_id

RID_DIR, ETSL_DIR = ROOT / "cloud_s3" / "sign-ราชบัณฑิต", ROOT / "cloud_s3" / "sign-etsl-chula"
ITEMS = PREP / "params" / "v12_video_items.csv"
OUT_RAW, OUT = PREP / "pose_mp_raw", PREP / "pose_mp"


def clip_id(src, file):
    return f"{src}:{Path(file).stem}"


def build_items():
    T = json.load(open(PREP / "manifest" / "tiers.json", encoding="utf-8")); W = set(T["words"])
    R = pd.read_csv(RID_DIR / "manifest.csv"); E = pd.read_csv(ETSL_DIR / "manifest.csv")
    rows = []
    for r in R[R.file.notna()].itertuples():
        dialog = r.kind == "sentence" and isinstance(r.caption, str) and ("ถาม" in r.caption)
        if r.kind == "word":
            st = 1 if r.word in W else 4
        elif r.kind == "sentence":
            st = 7 if dialog else 2
        else:
            st = 5
        rows.append(dict(clip_id=clip_id("rid", r.file), dataset="royal_institute", video=str(RID_DIR / r.file), stage=st, kind=r.kind,
                         word=r.word, dialog=bool(dialog), dup_tsl51=str(r.status).startswith("already")))
    for r in E[E.file.notna()].itertuples():
        if r.type == "word":
            st = 1 if r.word_th in W else 4
        elif r.type in ("phrase", "sentence"):
            st = 1 if (r.word_th in W and r.n_signs == 1) else 3
        else:
            st = 6
        rows.append(dict(clip_id=clip_id("etsl", r.file), dataset="etsl", video=str(ETSL_DIR / r.file), stage=st, kind=r.type,
                         word=r.word_th, dialog=False, dup_tsl51=False))
    D = pd.DataFrame(rows)
    D["out_name"] = [safe_id(c) for c in D.clip_id]
    D["exists"] = [Path(v).exists() for v in D.video]
    D = D.sort_values(["stage", "dataset", "clip_id"], kind="stable").reset_index(drop=True)
    ITEMS.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(ITEMS, index=False, encoding="utf-8")
    print(D.groupby(["stage", "dataset"]).size().to_string()); print("missing videos", int((~D.exists).sum()))
    return D


def reuse_ea(D):
    """Copy the EA-v12 extractions (same extractor) into the store and write their Standard Schema files."""
    from modules.mediapipe_pose import load_raw, to_standard
    from modules.schema import save
    sel = json.load(open(ROOT / "artifacts/eval/ea_v12/ood_sel.json", encoding="utf-8"))
    by_video = {str(Path(v)).lower(): (c, ds, o) for v, c, ds, o in zip(D.video, D.clip_id, D.dataset, D.out_name)}
    n = 0
    for s in sel:
        f = ROOT / "cache/v12_ea/ood_raw" / f"{s['key']}.npz"
        hit = by_video.get(str(Path(s["video"])).lower())
        if not f.exists() or hit is None:
            continue
        cid, ds, o = hit
        dst_raw, dst = OUT_RAW / ds / f"{o}.npz", OUT / ds / f"{o}.npz"
        if dst_raw.exists() and dst.exists():
            continue
        dst_raw.parent.mkdir(parents=True, exist_ok=True); dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst_raw)
        save(dst, to_standard(load_raw(dst_raw), src=ds)); n += 1
    print("reused EA-v12 extractions:", n, flush=True)


def _one(item):
    from extract_mp import _one as one
    return one(item)


def run(a):
    D = pd.read_csv(ITEMS) if ITEMS.exists() else build_items()
    reuse_ea(D)
    D = D[D.exists & D.stage.isin([int(s) for s in a.stage.split(",")])]
    items = []
    for r in D.itertuples():
        if (OUT_RAW / r.dataset / f"{r.out_name}.npz").exists() and (OUT / r.dataset / f"{r.out_name}.npz").exists():
            continue
        items.append((r.clip_id, r.dataset, r.video, r.out_name, None))
    if a.limit:
        items = items[:a.limit]
    print(f"to extract: {len(items)} (stages {a.stage}), workers {a.workers}", flush=True)
    log = PREP / "params" / "v12_extract_log.jsonl"
    t0, done = time.time(), 0
    with ProcessPoolExecutor(a.workers) as ex, open(log, "a", encoding="utf-8") as fl:
        futs = [ex.submit(_one, it) for it in items]
        for f in as_completed(futs):
            cid, status, n, info = f.result(); done += 1
            fl.write(json.dumps(dict(clip_id=cid, status=status, frames=n, info=info), ensure_ascii=False, default=float) + "\n"); fl.flush()
            if done % 25 == 0 or done == len(items):
                el = time.time() - t0
                print(f"{done}/{len(items)} {el/60:.1f} min, eta {el/done*(len(items)-done)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("items")
    p = sub.add_parser("run"); p.add_argument("--workers", type=int, default=9); p.add_argument("--stage", default="1,2,3,4,5,6")
    p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    build_items() if a.cmd == "items" else run(a)
