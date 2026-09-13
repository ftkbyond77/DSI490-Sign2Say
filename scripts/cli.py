"""ThaiSLM command line.

  python scripts/cli.py manifests                 # adapters → manifests + splits
  python scripts/cli.py bgpool                    # background pool for chroma-key augmentation
  python scripts/cli.py cache --set ttrs          # tracks + embedding cache (resumable)
  python scripts/cli.py cache --set youtube
  python scripts/cli.py infer --video data_test/x.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cmd_manifests(a):
    from modules.data import build_manifests, make_splits
    clips, _ = build_manifests(force=True)
    make_splits(clips)


def cmd_bgpool(a):
    import subprocess
    import numpy as np
    import cv2
    import pandas as pd
    from modules.utils import ART
    clips = pd.read_parquet(ART / "manifests" / "clips.parquet")
    yt = clips[(clips.source == "youtube")]
    rng = np.random.RandomState(0)
    imgs = []
    for r in yt.sample(min(len(yt), a.count), random_state=0, replace=len(yt) < a.count).itertuples():
        t = rng.uniform(0.05, 0.95) * r.duration_s
        out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(ROOT / r.media_path), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
        if len(out) != r.width * r.height * 3:
            continue
        img = np.frombuffer(out, np.uint8).reshape(r.height, r.width, 3)
        if r.height > r.width:  # vertical: take a landscape band
            y0 = rng.randint(0, r.height - r.width * 9 // 16)
            img = img[y0:y0 + r.width * 9 // 16]
        imgs.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
    p = ART / "backgrounds" / "bg_pool.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.stack(imgs))
    print("bg pool", len(imgs), p)


def cmd_cache(a):
    import pandas as pd
    from modules.articulators import parse_regions
    from modules.encoder import run_cache
    from modules.utils import ART, load_cfg
    cfg = load_cfg("data")
    clips = pd.read_parquet(ART / "manifests" / "clips.parquet")
    sp = json.loads((ART / "manifests" / "splits.json").read_text(encoding="utf-8"))
    blocked = set(sp["leaked_realworld_in_corpus"])
    if a.set == "ttrs":
        sel = clips[clips.source == "ttrs"]
    else:
        sel = clips[clips.source == "youtube"]
    sel = sel[~sel.clip_id.isin(blocked)]
    if a.limit:
        sel = sel.head(a.limit)
    yc = cfg["youtube"]
    items = []
    for r in sel.itertuples():
        if r.source == "ttrs":
            items.append((r.clip_id, str(ROOT / r.media_path), parse_regions(r.mask_regions), None, True, 0.0, None))
        else:
            start = yc["skip_intro_s"] if r.duration_s > 3 * yc["skip_intro_s"] else 0.0
            zone = parse_regions(r.signer_zone)
            items.append((r.clip_id, str(ROOT / r.media_path), None, zone, False, start, yc["max_seconds_per_video"]))
    vt, vo = cfg["cache"]["versions_ttrs"], cfg["cache"]["versions_other"]
    run_cache(items, lambda cid: vt if cid.startswith("ttrs:") else vo, workers=a.workers, threads=a.threads)


def cmd_infer(a):
    """Deployment defaults: session-relative DP decoder (no absolute threshold), session centering on, full vocabulary."""
    from modules.pipeline import VideoPipeline
    from modules.utils import ART
    vp = VideoPipeline(ART / "checkpoints" / "islr" / f"{a.tag}.pt", ART / "prototypes" / f"{a.tag}.npz",
                       decoder=a.decoder, decoder_kw=dict(delta=a.delta) if a.decoder == "dp_rel" else dict(penalty=a.delta),
                       window_lengths=(12, 16, 20, 24, 32), tts=None if a.no_audio else "local", tta_flip=a.tta,
                       affect="auto", center=not a.no_center)
    out = Path(a.out) if a.out else ROOT / "result_reporting" / "inference" / "final" / Path(a.video).stem
    r = vp.run(a.video, out)
    summary = dict(text=r["text"], sentences=[{k: s[k] for k in ("sentence_th", "glosses", "nmm", "affect", "uncertain", "provider")}
                                              for s in r["sentences"]],
                   glosses=[(g["lemma"], round(g["conf"], 2), g["t0"], g["t1"], g["alt"][:4]) for g in r["gloss_timeline"]],
                   timings_ms=r["timings_ms"], out_dir=str(out))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifests")
    b = sub.add_parser("bgpool"); b.add_argument("--count", type=int, default=400)
    c = sub.add_parser("cache")
    c.add_argument("--set", choices=["ttrs", "youtube"], default="ttrs")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--threads", type=int, default=1)
    i = sub.add_parser("infer")
    i.add_argument("--video", required=True); i.add_argument("--tag", default="islr_ssl_fast"); i.add_argument("--out", default="")
    i.add_argument("--tta", action="store_true"); i.add_argument("--no_audio", action="store_true")
    i.add_argument("--decoder", default="dp_rel"); i.add_argument("--delta", type=float, default=0.1); i.add_argument("--no_center", action="store_true")
    a = ap.parse_args()
    {"manifests": cmd_manifests, "bgpool": cmd_bgpool, "cache": cmd_cache, "infer": cmd_infer}[a.cmd](a)


if __name__ == "__main__":
    main()
