"""pitty: metrics + per-video confidence timeline for a folder of real videos (data_test, our own recorded clips, ...).

Usage (from Main_demo/, env thaislm):
    python pitty/scripts/my_eval.py --set data_test                       # the 5 data_test videos (refs from modules.lexicon)
    python pitty/scripts/my_eval.py --set mine --videos pitty/my_clips --refs pitty/my_clips/refs.csv
refs.csv: two columns `file,ref` — ref = the signed words in signing order, space separated (e.g. `A01_take1.mp4,สวัสดี แม่ สบายดี`).
Outputs (git-ignored): cache/pitty_eval/<set>/{metrics.json, per_video.csv, <video>_timeline.png}; landmarks cached in cache/pitty_eval/mp/.
Metrics follow the team table: WER, words right / wrong / missing / extra and top-5 hit — all from the WER alignment (pitty/scripts/
wer_ops.py; the old order-free Recall@1 / @5 counted a word found anywhere in the output), accepted words correct, count error, latency (Segment F1 needs hand-marked
sign boundaries — only TSL51 has them, so it is not computed for real videos)."""
import argparse, csv, json, sys, time, logging
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
OUT = ROOT / "cache" / "pitty_eval"
STATUS_COL = {"accepted": "#16a34a", "uncertain": "#f59e0b", "unknown": "#dc2626"}


def load_refs(a):
    from modules.lexicon import DATA_TEST_REFS
    if a.set.startswith("data_test"):
        return {ROOT / "data_test" / f"{k}.mp4": v for k, v in DATA_TEST_REFS.items()}
    vids = Path(a.videos)
    have = {p.stem: p for p in vids.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")}
    out = {}
    with open(a.refs, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            stem = Path(r["file"]).stem                     # any video extension; takes like "<name>_take2" / "<name> (2)" also match
            for k, p in have.items():
                if k == stem or k.startswith(stem + "_") or k.startswith(stem + " ("):
                    out[p] = r["ref"].split()
            if not any(k == stem or k.startswith(stem + "_") or k.startswith(stem + " (") for k in have):
                if (OUT / "mp" / f"{a.mp_set or a.set}__{stem}.npz").exists():     # video replaced, landmarks still cached
                    out[vids / r["file"]] = r["ref"].split()
                else:
                    print("ยังไม่มีไฟล์:", r["file"])
    return out


def canon_fn():
    from modules.lexicon import TEST_SURFACE
    cm = json.loads((ROOT / "cloud_s3/data_prep/manifest/concepts.json").read_text(encoding="utf-8"))["concept"]
    return lambda w: cm.get(TEST_SURFACE.get(w, w), TEST_SURFACE.get(w, w))


def run(a):
    from modules.mediapipe_pose import extract, load_raw, save_raw
    from modules.pipeline import SignPipeline
    from modules.decode import wer
    sys.path.insert(0, str(Path(__file__).parent)); from wer_ops import align
    canon = canon_fn()
    pipe = SignPipeline(backend="onnx", device="cpu", vocab=a.vocab, use_llm=False)
    import os
    if os.environ.get("SPOT_OVERRIDE"):      # pitty: decoder settings override (JSON, "post" merged) — e.g. the pre-tuning baseline
        ov = json.loads(os.environ["SPOT_OVERRIDE"]); post = dict(pipe.spot.get("post", {}), **ov.pop("post", {})); pipe.spot = dict(pipe.spot, **ov, post=post)
    if os.environ.get("ACCEPT_MIN"):
        pipe.spot_all = dict(pipe.spot_all, accept_min=float(os.environ["ACCEPT_MIN"])); pipe.cal = dict(pipe.cal, tau_accept=float(os.environ["ACCEPT_MIN"]))
    od = OUT / a.set; od.mkdir(parents=True, exist_ok=True)
    rows, full = [], {}
    for video, ref in load_refs(a).items():
        stem = video.stem
        mp = OUT / "mp" / f"{a.mp_set or a.set}__{stem}.npz"
        if not mp.exists():
            src = ROOT / "cache/data_test_mp" / f"{stem}.npz"           # data_test landmarks already extracted by `main.py test`
            if a.set.startswith("data_test") and src.exists():
                mp = src
            else:
                print("MediaPipe:", video.name); save_raw(mp, extract(video))
        raw = load_raw(mp)
        if a.trim_start or a.trim_end:                       # pitty: drop the rest before / after signing (seconds)
            tt = np.asarray(raw["t"], float); keep = (tt >= tt[0] + a.trim_start) & (tt <= tt[-1] - a.trim_end)
            raw = {k: (v[keep] if hasattr(v, "shape") and v.shape[:1] == tt.shape else v) for k, v in raw.items()}
        t = time.perf_counter(); r = pipe.run_landmarks(raw, title=stem, alphabet=a.alphabet); ms = (time.perf_counter() - t) * 1000
        refc = [canon(w) for w in ref]
        segs = [s for s in r["segments"] if s["status"] != "transition"]
        top1 = [s["nearest"] for s in segs]
        top5 = [[c["word"] for c in s["candidates"][:5]] for s in segs]
        acc = [s["nearest"] for s in segs if s["status"] == "accepted"]
        ops = align(refc, top1)
        rows.append(dict(video=stem, ref=" ".join(refc), top1=" ".join(top1), accepted=" ".join(acc),
                         WER=wer(refc, top1), right=sum(o[0] == "C" for o in ops), wrong=sum(o[0] == "S" for o in ops),
                         missing=sum(o[0] == "D" for o in ops), extra=sum(o[0] == "I" for o in ops),
                         top5_hit=sum(o[0] in "CS" and o[1] in top5[o[3]] for o in ops),
                         n_ref=len(refc), n_seg=len(segs), count_err=abs(len(segs) - len(refc)),
                         accepted_n=len(acc), accepted_correct=sum(w in refc for w in acc),
                         latency_s=ms / 1000, duration_s=r.get("duration_s"), sentence=r.get("sentence") or "",
                         mirrored=bool(r.get("mirrored_to_right_hand")), face=", ".join(k for k in ("question_yesno", "question_wh", "question_furrow", "negation") if (r.get("face") or {}).get(k)), emotion=(r.get("face") or {}).get("emotion_th", "")))
        full[stem] = dict(ref=refc, segments=[dict(t0=s["t0"], t1=s["t1"], f0=s["f0"], f1=s["f1"], status=s["status"], p=round(float(s.get("p_correct", 0)), 3),
                                                   cands=[[c["word"], round(float(c["score"]), 3)] for c in s["candidates"][:3]]) for s in segs],
                          video=str(video), t=np.asarray(raw["t"], float).tolist())
        timeline_png(full[stem], od / f"{stem}_timeline.png", stem)
        x = rows[-1]; print(f"{stem}: WER {x['WER']:.2f}  right {x['right']}/{x['n_ref']}  wrong {x['wrong']}  missing {x['missing']}  extra {x['extra']}  accepted {x['accepted_correct']}/{x['accepted_n']}")
    n = sum(r["n_ref"] for r in rows)
    summary = dict(set=a.set, vocab=a.vocab, videos=len(rows), ref_words=n,
                   WER=float(np.mean([r["WER"] for r in rows])),
                   right=f"{sum(r['right'] for r in rows)}/{n}", wrong=sum(r["wrong"] for r in rows), missing=sum(r["missing"] for r in rows),
                   extra=sum(r["extra"] for r in rows), top5_hit=f"{sum(r['top5_hit'] for r in rows)}/{n}",
                   accepted=f"{sum(r['accepted_correct'] for r in rows)}/{sum(r['accepted_n'] for r in rows)}",
                   count_err=float(np.mean([r["count_err"] for r in rows])), latency_s=float(np.mean([r["latency_s"] for r in rows])),
                   accept_min=max(pipe.cal["tau_accept"], pipe.spot_all.get("accept_min", 0)),
                   p_sign_min=pipe.spot["post"].get("p_sign_min"), min_hand_frac=pipe.spot["post"].get("min_hand_frac", 0.0))
    (od / "metrics.json").write_text(json.dumps(dict(summary=summary, videos=rows, detail={k: {kk: vv for kk, vv in v.items() if kk != "t"} for k, v in full.items()}),
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    with open(od / "per_video.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("summary:", json.dumps(summary, ensure_ascii=False))
    return summary


def timeline_png(d, path, title):
    """One column per detected sign: a frame from the middle of it, time + P(correct) coloured by status, top-3 words as bars."""
    import cv2, matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Tahoma", "Leelawadee UI", "sans-serif"]
    segs = d["segments"]
    if not segs:
        return
    t = np.asarray(d["t"]); scale = 1000.0 if t.max() > 1000 else 1.0
    cap = cv2.VideoCapture(d["video"])
    n = len(segs)
    fig = plt.figure(figsize=(2.3 * n, 4.6))
    for i, s in enumerate(segs):
        mid = (s["t0"] + s["t1"]) / 2 + float(t[0]) / scale        # t0/t1 are seconds on the pipeline's 25 fps grid (≠ raw frame index)
        cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
        ok, img = cap.read()
        ax = fig.add_axes([i / n + 0.006, 0.42, 1 / n - 0.012, 0.44])
        if ok:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        col = STATUS_COL.get(s["status"], "#6b7280")
        ax.set_title(f"{i + 1}. {s['t0']:.2f}–{s['t1']:.2f}s  p={s['p']:.2f}", fontsize=10, color=col, fontweight="bold")
        bx = fig.add_axes([i / n + 0.006, 0.04, 1 / n - 0.012, 0.34]); bx.axis("off"); bx.set_xlim(0, 1); bx.set_ylim(0, 3)
        for j, (w, sc) in enumerate(s["cands"]):
            y = 2.55 - j
            bx.barh(y, max(sc, 0.02), height=0.75, color=col if j == 0 else "#cbd5e1")
            bx.text(0.03, y, f"{w}  {sc:.3f}", va="center", fontsize=10, color="white" if j == 0 else "#0f172a", fontweight="bold" if j == 0 else "normal")
    cap.release()
    fig.suptitle(f"{title}   ·   เฉลย: {' '.join(d['ref'])}", fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="data_test")
    ap.add_argument("--videos"); ap.add_argument("--refs")
    ap.add_argument("--vocab", default="conversation")
    ap.add_argument("--mp_set", default="", help="reuse landmarks extracted for another set")
    ap.add_argument("--trim_start", type=float, default=0.0); ap.add_argument("--trim_end", type=float, default=0.0)
    ap.add_argument("--alphabet", default="en", help="fingerspelling: en (web default) / th / both / off")
    run(ap.parse_args())
