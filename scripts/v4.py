"""ThaiSLM v4 command line (pose-first, weakly-supervised vocabulary, Uni-Sign encoder).

  python scripts/v4.py pose --set pilot|ttrs|youtube     # RTMW whole-body keypoints (resumable)
  python scripts/v4.py pose_test                          # data_test videos (inference-style signer lock)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from modules.utils import ART, get_logger, read_json

log = get_logger("v4")
MAN = ART / "manifests"


def pilot_ids():
    c = pd.read_parquet(MAN / "clips_signers.parquet")
    sp = read_json(MAN / "splits.json")
    t = c[c.source == "ttrs"]
    ids = set(sp["seen_val_queries"] + sp["seen_test_queries"] + sp["val_clips"] + sp["test_clips"])
    units = set(sp["seen_val_units"] + sp["seen_test_units"])
    ids |= set(t[t.sign_variant_id.isin(units)].clip_id)
    words = ["ฉัน", "ปลอบใจ", "เพื่อน", "ร้องไห้", "กิน", "รับประทาน", "ข้าว", "ด้วยกัน", "ร่วมกัน"]
    ids |= set(t[t.lemma.isin(words)].clip_id)
    rest = sorted(set(sp["train_islr"]) - ids)
    rng = np.random.RandomState(0)
    ids |= set(rng.choice(rest, 700, replace=False))
    return sorted(ids)


def cmd_pose(a):
    from modules.encoder import track_path
    from modules.wholebody import run_corpus
    c = pd.read_parquet(MAN / "clips_signers.parquet").set_index("clip_id")
    if a.set == "pilot":
        ids = pilot_ids()
    elif a.set == "ttrs":
        ids = c[c.source == "ttrs"].index.tolist()
    elif a.set == "youtube_labelled":  # lessons with caption/ASR word hits or a title label
        cv4 = pd.read_parquet(ART / "manifests_v4" / "clips_v4.parquet")
        sw = pd.read_parquet(ART / "manifests_v4" / "speech_words.parquet")
        ids = sorted(set(sw.clip_id) | set(cv4[cv4.label_source == "title"].clip_id))
    else:  # lessons kept by v4 cleaning (no broadcasts / songs / alphabet drills)
        from modules.lexicon import FINGERSPELL_KW, SONG_KW
        y = c[(c.source == "youtube") & ~c.subset.isin(["bigsign", "youtube_continuous"])]
        ids = [cid for cid, t in zip(y.index, y.title) if not any(k in (t or "").lower() for k in FINGERSPELL_KW + SONG_KW)]
    items = []
    for cid in ids:
        tp = track_path(cid)
        if not tp.exists():
            continue
        tr = np.load(tp)
        items.append((cid, str(ROOT / c.loc[cid, "media_path"]), str(tp), float(tr["start_s"]), len(tr["kpts"])))
    run_corpus(items, workers=a.workers)


def cmd_pose_test(a):
    from modules.wholebody import POSE_DIR, extract_video
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted((ROOT / "data_test").glob("*.mp4")):
        for fps in (12.5, 25.0):
            d = extract_video(p, fps=fps)
            out = POSE_DIR / f"test_{p.stem}_{int(fps * 10)}.npz"
            np.savez_compressed(out, **{k: v for k, v in d.items() if k != "frames"})
            log.info("%s fps=%s frames=%d hand-conf L %.2f R %.2f", p.name, fps, len(d["kp"]),
                     float(d["sc"][:, 91:112].mean()), float(d["sc"][:, 112:133].mean()))


def e1_restricted(Zq, q_units, Zg, g_units, topm=2):
    """E1 on a restricted gallery: instance bank of gallery clips → rank of the query's unit."""
    from modules.recognize import VocabBank
    bank = VocabBank(Zg, g_units)
    S = bank.word_scores(Zq, topm)
    tgt = np.array([bank.widx[u] for u in q_units])
    rank = (S > S[np.arange(len(tgt)), tgt][:, None]).sum(1)
    return dict(R1=float((rank < 1).mean()), R5=float((rank < 5).mean()), R10=float((rank < 10).mean()),
                MRR=float((1 / (rank + 1)).mean()), median_rank=float(np.median(rank) + 1), n=len(tgt), gallery=len(bank.vocab))


def cmd_eval_enc(a):
    """Pilot: compare pose encoders vs the v1 RGB model on identical E1 queries/galleries."""
    import torch
    from modules.recognize import active_span, embed, load_pose, segment_parts
    from modules.unisign import UniSignEncoder
    from modules.utils import write_json
    from modules.wholebody import pose_path
    c = pd.read_parquet(MAN / "clips_signers.parquet").set_index("clip_id")
    sp = read_json(MAN / "splits.json")
    have = {cid for cid in c.index if pose_path(cid).exists()}
    res = {}
    for part in ("val", "test"):
        qs = [q for q in sp[f"seen_{part}_queries"] if q in have]
        gal = [g for g in sp["train_islr"] if g in have]
        q_units = [c.loc[q, "sign_variant_id"] for q in qs]
        g_units = [c.loc[g, "sign_variant_id"] for g in gal]
        keep = [i for i, u in enumerate(q_units) if u in set(g_units)]
        qs = [qs[i] for i in keep]; q_units = [q_units[i] for i in keep]
        cache = {}
        for cid in qs + gal:
            kp, sc, hw, _ = load_pose(pose_path(cid))
            cache[cid] = (kp, sc, hw)
        for ck in a.ckpts.split(","):
            enc = UniSignEncoder(ck, use_mt5=True).eval()
            for up in (2, 1):
                parts = {}
                for cid, (kp, sc, hw) in cache.items():
                    t0, t1 = active_span(kp, sc, hw) if a.active else (0, len(kp))
                    parts[cid] = segment_parts(kp, sc, t0, t1, up=up)
                for level in ("gcn", "ctx"):
                    for pool in ("mean", "meanmax"):
                        if up == 1 and pool == "meanmax":
                            continue
                        Zq = embed(enc, [parts[q] for q in qs], level, pool=pool)
                        Zg = embed(enc, [parts[g] for g in gal], level, pool=pool)
                        r = e1_restricted(Zq, q_units, Zg, g_units)
                        key = f"{part}|{ck}|up{up}|{level}|{pool}"
                        res[key] = r
                        log.info("%-40s R1 %.3f R5 %.3f R10 %.3f MRR %.3f med %.0f (n=%d, gallery=%d)", key, r["R1"], r["R5"],
                                 r["R10"], r["MRR"], r["median_rank"], r["n"], r["gallery"])
            del enc; torch.cuda.empty_cache()
        # v1 RGB model on the same queries / gallery
        from modules.encoder import FeatureStore
        from modules.evalkit import embed_clips
        from modules.pipeline import load_islr
        m = load_islr(ART / "checkpoints" / "islr" / "islr_ssl_fast.pt")
        st = FeatureStore("ttrs")
        Zq = embed_clips(m.embed, st, qs); Zg = embed_clips(m.embed, st, gal)
        r = e1_restricted(Zq, q_units, Zg, g_units)
        res[f"{part}|v1_islr_ssl_fast"] = r
        log.info("%-40s R1 %.3f R5 %.3f R10 %.3f MRR %.3f med %.0f", f"{part}|v1_islr_ssl_fast", r["R1"], r["R5"], r["R10"], r["MRR"], r["median_rank"])
    write_json(ART / "reports" / f"v4_eval_enc_{a.tag}.json", res)


def cmd_asr(a):
    from modules.lexicon import FINGERSPELL_KW, SONG_KW, best_vtt, run_asr
    c = pd.read_parquet(MAN / "clips_signers.parquet")
    y = c[(c.source == "youtube") & ~c.subset.isin(["bigsign", "youtube_continuous"])]
    items = []
    for r in y.itertuples():
        tl = (r.title or "").lower()
        if any(k in tl for k in FINGERSPELL_KW + SONG_KW):
            continue
        if best_vtt(r.media_path)[0] is None:
            items.append((r.clip_id, r.media_path))
    log.info("ASR for %d lessons without Thai captions", len(items))
    run_asr(items)


def cmd_schema(a):
    from modules.lexicon import MAN4, build_schema_v4
    asr_p = MAN4 / "asr_whisper.json"
    asr = json.loads(asr_p.read_text(encoding="utf-8")) if asr_p.exists() else None
    build_schema_v4(asr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pose"); p.add_argument("--set", default="pilot"); p.add_argument("--workers", type=int, default=8)
    sub.add_parser("pose_test"); sub.add_parser("asr"); sub.add_parser("schema")
    e = sub.add_parser("eval_enc"); e.add_argument("--ckpts", default="wlasl,csl_stage1"); e.add_argument("--active", type=int, default=1); e.add_argument("--tag", default="pilot")
    a = ap.parse_args()
    dict(pose=cmd_pose, pose_test=cmd_pose_test, eval_enc=cmd_eval_enc, asr=cmd_asr, schema=cmd_schema)[a.cmd](a)


if __name__ == "__main__":
    main()
