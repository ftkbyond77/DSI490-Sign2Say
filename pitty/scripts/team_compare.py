"""pitty: compare models on the team's Drive test videos (Pitty 7 · Boom 6 · Jack 5), same metrics as my_eval.py / team metrics.

Steps (the notebook pitty/notebooks/pitty_team_compare.ipynb runs them all; models run as separate processes because pitty's
and Boom's code both live in a package called `modules`):
    python pitty/scripts/team_compare.py download                      # the 18 videos from Drive (needs `pip install gdown`)
    python pitty/scripts/team_compare.py setup_boom                    # origin/boom code → cache/team_eval/boom_code (+ this checkout's model files)
    python pitty/scripts/team_compare.py extract                       # MediaPipe once per video (pitty code) + canonical refs
    python pitty/scripts/team_compare.py eval --label pitty --code_root .
    python pitty/scripts/team_compare.py eval --label boom_combined --code_root cache/team_eval/boom_code \
           --override '{"post.p_sign_min": 0.55, "act_thr": 0.6, "accept_min": 0.7}'
Every model reads the same landmark files, so differences come from the model code / settings only.
Videos: cache/team_eval/videos/<person>/<file> (Drive folder 1S375gAp0rc5pAnDKgEyehnO0C9guzU9g). Refs: pitty/team_compare/refs.csv.
Output: cache/team_eval/results/<label>/{metrics.json, per_video.csv, <person>__<video>_timeline.png}."""
import argparse, csv, json, os, sys, time, logging
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "cache" / "team_eval"
REFS = ROOT / "pitty" / "team_compare" / "refs.csv"
logging.disable(logging.WARNING)


def key(person, file):
    return f"{person}__{Path(file).stem}"


def download(a):
    """Each video by its Drive file id (refs.csv drive_id); file names come from refs.csv, so a '?' in a Drive name is no problem."""
    try:
        import gdown
    except ImportError:
        sys.exit("ต้องติดตั้ง gdown ก่อน: pip install gdown  (หรือโหลดโฟลเดอร์จาก Drive เองมาไว้ที่ cache/team_eval/videos/<คน>/)")
    for r in csv.DictReader(open(REFS, encoding="utf-8-sig")):
        v = E / "videos" / r["person"] / r["file"]
        if v.exists():
            continue
        v.parent.mkdir(parents=True, exist_ok=True)
        print("download:", v.relative_to(E), flush=True)
        gdown.download(id=r["drive_id"], output=str(v), quiet=True)
        if not v.exists():
            sys.exit(f"โหลดไม่ได้: {r['file']} — ต้องมีสิทธิ์เปิดโฟลเดอร์ Drive ของทีม")


def setup_boom(a):
    """Boom's code from origin/boom (read only: no branch switch, nothing committed). Model weights are Git LFS files that the
    boom branch shares with this checkout byte for byte, so they are copied from here instead of pulled again."""
    import io, shutil, subprocess, tarfile
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, check=True, capture_output=True).stdout
    git("fetch", "origin", "boom")
    ref = a.boom_ref; commit = git("rev-parse", ref).decode().strip()
    dst = E / "boom_code"
    if (dst / "BOOM_COMMIT.txt").exists() and (dst / "BOOM_COMMIT.txt").read_text().strip() == commit:
        print("boom_code already at", commit); return
    shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", ref, "modules", "models", "main.py"))) as t:
        t.extractall(dst)
    lfs = [l for l in git("lfs", "ls-files", "-n").decode().splitlines() if l.startswith("models/")]
    changed = git("diff", "--name-only", "HEAD", ref, "--", *lfs).decode().split()
    if changed:
        sys.exit(f"น้ำหนักโมเดลใน {ref} ต่างจาก checkout นี้: {changed} — ต้อง git lfs pull ของ branch boom แยก")
    for f in lfs:
        src = ROOT / f
        if src.stat().st_size < 1024 and src.read_bytes().startswith(b"version https://git-lfs"):
            sys.exit(f"{f} ยังเป็น LFS pointer — รัน git lfs pull ก่อน")
        shutil.copy2(src, dst / f)
    (dst / "BOOM_COMMIT.txt").write_text(commit)
    print("boom_code:", commit)


def extract(a):
    sys.path.insert(0, str(ROOT))
    from modules.mediapipe_pose import extract as mp_extract, save_raw
    from modules.lexicon import TEST_SURFACE
    cm = json.loads((ROOT / "cloud_s3/data_prep/manifest/concepts.json").read_text(encoding="utf-8"))["concept"]
    canon = lambda w: cm.get(TEST_SURFACE.get(w, w), TEST_SURFACE.get(w, w))
    vocab = set(json.loads((ROOT / "models/vocab_conversation.json").read_text(encoding="utf-8"))["concepts"])
    (E / "mp").mkdir(parents=True, exist_ok=True)
    items = []
    for r in csv.DictReader(open(REFS, encoding="utf-8-sig")):
        k = key(r["person"], r["file"]); v = E / "videos" / r["person"] / r["file"]
        if not v.exists():
            print("ยังไม่มีไฟล์:", v); continue
        mp = E / "mp" / f"{k}.npz"
        if not mp.exists():
            t = time.perf_counter(); save_raw(mp, mp_extract(v)); print(f"MediaPipe {k}: {time.perf_counter() - t:.1f}s", flush=True)
        ref = [canon(w) for w in r["ref"].split()]
        # a word the model can say: in the conversation vocabulary, or one fingerspelled letter
        known = [w in vocab or (len(w) == 1 and not w.isspace()) for w in ref]
        items.append(dict(key=k, person=r["person"], file=r["file"], video=str(v), mp=str(mp), ref=ref, known=known, note=r.get("note", "")))
    (E / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} videos → {E / 'items.json'}")


def apply_override(pipe, ov, cal0):
    """Same convention as Boom's test_engine.py: 'post.x' → spot['post'][x]; accept_min → tau_accept = max(calibration tau, value)."""
    for k, v in ov.items():
        if k == "accept_min":
            pipe.spot_all = dict(pipe.spot_all, accept_min=v); pipe.cal = dict(pipe.cal, tau_accept=max(float(cal0.get("tau_accept", 0)), v))
        elif k.startswith("post."):
            pipe.spot = dict(pipe.spot, post=dict(pipe.spot.get("post", {}), **{k.split(".", 1)[1]: v}))
        else:
            pipe.spot = dict(pipe.spot, **{k: v})


def evaluate(a):
    code = Path(a.code_root).resolve()
    sys.path.insert(0, str(Path(__file__).parent)); from wer_ops import align
    sys.path.insert(0, str(code))
    os.environ.setdefault("THAISLM_SYMBOLIC_WEIGHT", "0")        # Boom's hand-shape fusion is off by default on her branch too
    from modules.mediapipe_pose import load_raw
    from modules.pipeline import SignPipeline
    import modules.pipeline as mpipe
    assert Path(mpipe.__file__).resolve().is_relative_to(code), f"loaded the wrong code: {mpipe.__file__}"
    items = json.loads((E / "items.json").read_text(encoding="utf-8"))
    pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False)
    cal_json = json.loads((code / "models/segment_calibration.json").read_text(encoding="utf-8"))
    cal0 = cal_json.get("conversation", cal_json)
    ov = json.loads(a.override or "{}")
    apply_override(pipe, ov, cal0)
    od = E / "results" / a.label; od.mkdir(parents=True, exist_ok=True)
    pipe.run_landmarks(load_raw(items[0]["mp"]), alphabet=a.alphabet)      # warm-up so the first video's latency is not a cold start
    rows, detail = [], {}
    for it in items:
        raw = load_raw(it["mp"]); ref = it["ref"]
        t = time.perf_counter(); r = pipe.run_landmarks(raw, title=it["key"], alphabet=a.alphabet); sec = time.perf_counter() - t
        segs = [s for s in r["segments"] if s["status"] != "transition"]
        top1 = [s["nearest"] for s in segs]; top5 = [[c["word"] for c in s["candidates"][:5]] for s in segs]
        acc = [s["nearest"] for s in segs if s["status"] == "accepted"]
        ops = align(ref, top1); n = len(ref)
        S, D, I, C = (sum(o[0] == x for o in ops) for x in "SDIC")
        kref = [w for w, k in zip(ref, it["known"]) if k]; kops = align(kref, top1) if kref else []
        rows.append(dict(person=it["person"], video=Path(it["file"]).stem, ref=" ".join(ref), top1=" ".join(top1), accepted=" ".join(acc),
                         n_ref=n, n_oov=n - len(kref), WER=(S + D + I) / n, right=C, wrong=S, missing=D, extra=I,
                         top5_hit=sum(o[0] in "CS" and o[1] in top5[o[3]] for o in ops),
                         recall1=C / n, recall5=sum(o[0] in "CS" and o[1] in top5[o[3]] for o in ops) / n,
                         WER_known=(sum(o[0] in "SDI" for o in kops) / len(kref)) if kref else None,
                         n_seg=len(segs), count_err=abs(len(segs) - n), accepted_n=len(acc), accepted_correct=sum(w in ref for w in acc),
                         latency_s=sec, duration_s=r.get("duration_s"), mirrored=bool(r.get("mirrored_to_right_hand"))))
        detail[it["key"]] = dict(ref=ref, video=it["video"], t=np.asarray(raw["t"], float).tolist(),
                                 segments=[dict(t0=s["t0"], t1=s["t1"], f0=s["f0"], f1=s["f1"], status=s["status"], p=round(float(s.get("p_correct", 0)), 3),
                                                cands=[[c["word"], round(float(c["score"]), 3)] for c in s["candidates"][:3]]) for s in segs])
        if a.timelines:
            sys.path.insert(0, str(ROOT / "pitty/scripts")); import my_eval     # plotting only; no `modules` import inside
            my_eval.timeline_png(detail[it["key"]], od / f"{it['key']}_timeline.png", f"{a.label} · {it['key']}")
        x = rows[-1]; print(f"{it['key']}: WER {x['WER']:.2f} right {C}/{n} wrong {S} missing {D} extra {I} accepted {x['accepted_correct']}/{x['accepted_n']}", flush=True)
    settings = dict(act_thr=pipe.spot.get("act_thr"), p_sign_min=pipe.spot["post"].get("p_sign_min"),
                    min_hand_frac=pipe.spot["post"].get("min_hand_frac", 0.0), tau_accept=pipe.cal["tau_accept"],
                    trim_rest=bool(pipe.spot_all.get("trim_rest")), look_alike=bool(pipe.spot_all.get("look_alike")))
    (od / "metrics.json").write_text(json.dumps(dict(label=a.label, code_root=str(code), override=ov, alphabet=a.alphabet, settings=settings,
                                                     videos=rows, detail={k: {kk: vv for kk, vv in v.items() if kk != "t"} for k, v in detail.items()}),
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    with open(od / "per_video.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("settings:", json.dumps(settings, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "setup_boom", "extract", "eval"])
    ap.add_argument("--boom_ref", default="origin/boom")
    ap.add_argument("--label"); ap.add_argument("--code_root", default=str(ROOT))
    ap.add_argument("--override", default="", help="JSON settings patch, Boom's test_engine convention")
    ap.add_argument("--alphabet", default="en", help="fingerspelling: en (web default, as in my_eval) / th / both / off")
    ap.add_argument("--timelines", type=int, default=1)
    a = ap.parse_args()
    dict(download=download, setup_boom=setup_boom, extract=extract, eval=evaluate)[a.step](a)
