"""Boom-model vs Pitty-model vs Jack-model on the same data (notebook: modeling_compare.ipynb at the repo root).

    python scripts/modeling_compare.py setup                               # models/<name>/ (if missing) + code of pitty, boom
    python scripts/modeling_compare.py extract                             # MediaPipe once per data_test/team-drive video
    python scripts/modeling_compare.py eval --model Jack-model --set tsl51 # one model × one test set → cache/modeling_compare/results/

Each model = models/<name>/ (a complete model_dir: SignPipeline(model_dir=...)) + the code of its branch. Each runs in its own
process (all three codes are a package called `modules`) and reads the same landmark files → differences come from the model only.
  Jack-model  = jack_master (v8.1)            code: this checkout
  Pitty-model = origin/pitty                  code: cache/modeling_compare/code/pitty (git archive of the branch)
  Boom-model  = origin/boom + Boom's frozen "Combined" setting (Boom/phase2.1combined/FROZEN_TEST.json, baked into its spotting.json)
Sets: tsl51 = TSL51 sentence_test (126 sentences, 1×, bank of training signers only, fingerspelling off — no letters in TSL51)
      team  = data_test/team-drive/Eval-Test/<person>/* (18 real videos, production bank, English fingerspelling on = web default)
"""
import argparse, csv, io, json, os, shutil, subprocess, sys, tarfile, time, logging
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "cache" / "modeling_compare"
TEAM = ROOT / "data_test" / "team-drive" / "Eval-Test"
REFS = ROOT / "pitty" / "team_compare" / "refs.csv"
MODELS = {          # commit = the version that was compared (models/<name>/ is built from that commit's models/)
    "Boom-model": dict(branch="boom", commit="04cc177899dda09e0d04260bd2eda7c67113c543", code=E / "code" / "boom",
                       setting={"conversation.post.p_sign_min": 0.55, "conversation.act_thr": 0.6, "accept_min": 0.7}),
    "Pitty-model": dict(branch="pitty", commit="4aeb7b9ab0ae9a2f1c0b9aab9e0c8664322852cf", code=E / "code" / "pitty", setting={}),
    "Jack-model": dict(branch="jack_master", commit="c6a1cfe71f07bd5f845f199a89cb6010029cb283", code=ROOT, setting={}),
}
# Jack's clips = data_test/iphone_test/Sentence (same files); sign order given by the owner (modules/lexicon.py IPHONE_TEST_REFS)
JACK_REFS = {"สวัสดี ฉันชื่อ J ฉันชอบแมว": "IMG_3302", "ฉันสบายดี คุณสบายดี_": "IMG_3303", "วันนี้ฉันนำเสนองานเรื่องภาษามือกับ AI": "IMG_3304",
             "วันนี้ผมอยากเรียนภาษามือ": "IMG_3305", "พรุ่งนี้ผมจะไปเที่ยวประเทศจีน": "IMG_3307"}
logging.disable(logging.WARNING)


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True).stdout


def lfs_file(oid):
    """A local file with this LFS object id (a pulled LFS path of this checkout, or any file already under models/)."""
    import hashlib
    for line in git("lfs", "ls-files", "-l").decode().splitlines():
        o, _, path = line.split(" ", 2)
        if o == oid and (ROOT / path).exists():
            return ROOT / path
    for f in (ROOT / "models").rglob("*"):
        if f.is_file() and f.stat().st_size > 1024 and hashlib.sha256(f.read_bytes()).hexdigest() == oid:
            return f
    sys.exit(f"LFS object {oid[:12]} is not in this checkout — git lfs pull")


def setup(a):
    """models/<name>/ (in git) is built only when missing: the models/ of its commit (read only, no checkout; Boom's frozen setting
    written into its spotting.json). Pitty / Boom code (modules/, main.py of their commit) → cache/modeling_compare/code/<name>."""
    for name, m in MODELS.items():
        md, ref = ROOT / "models" / name, m["commit"]
        if not (md / "model_card.json").exists():
            tmp = E / "export" / name
            shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
            with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", ref, "models"))) as t:
                t.extractall(tmp)
            md.mkdir(parents=True, exist_ok=True)
            for f in sorted((tmp / "models").rglob("*")):
                rel = f.relative_to(tmp / "models")
                if not f.is_file():
                    continue
                (md / rel).parent.mkdir(parents=True, exist_ok=True)
                if f.stat().st_size < 1024 and f.read_bytes().startswith(b"version https://git-lfs"):
                    shutil.copy2(lfs_file(f.read_text().split("sha256:")[1].split()[0]), md / rel)
                else:
                    shutil.copy2(f, md / rel)
            if m["setting"]:
                sp = json.loads((md / "spotting.json").read_text(encoding="utf-8"))
                for k, v in m["setting"].items():
                    d = sp
                    for p in k.split(".")[:-1]:
                        d = d[p]
                    d[k.split(".")[-1]] = v
                (md / "spotting.json").write_text(json.dumps(sp, ensure_ascii=False, indent=1), encoding="utf-8")
            shutil.rmtree(tmp)
            (md / "model_card.json").write_text(json.dumps(dict(name=name, branch=m["branch"], commit=ref,
                code="this checkout" if m["code"] == ROOT else f"modules/ of {m['branch']} @ {ref[:8]}",
                setting=m["setting"] or "branch models/spotting.json as is"), ensure_ascii=False, indent=1), encoding="utf-8")
            print(name, "->", md.relative_to(ROOT), flush=True)
        if m["code"] != ROOT and not (m["code"] / "modules").exists():
            m["code"].mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", ref, "modules", "main.py"))) as t:
                t.extractall(m["code"])
            # the branch code finds its files at <code>/models (modules.utils.MODEL) → a junction to models/<name>
            subprocess.run(["cmd", "/c", "mklink", "/J", str(m["code"] / "models"), str(md)], check=True, capture_output=True)
            print(name, "code ->", m["code"].relative_to(ROOT), flush=True)


def team_items():
    sys.path.insert(0, str(ROOT))
    from modules.lexicon import IPHONE_TEST_REFS, TEST_SURFACE
    cm = json.loads((ROOT / "cloud_s3/data_prep/manifest/concepts.json").read_text(encoding="utf-8"))["concept"]
    canon = lambda w: cm.get(TEST_SURFACE.get(w, w), TEST_SURFACE.get(w, w))
    items = []
    for r in csv.DictReader(open(REFS, encoding="utf-8-sig")):
        stem = Path(r["file"]).stem
        ref = IPHONE_TEST_REFS["sentence"][JACK_REFS[stem]] if r["person"] == "Jack" else [canon(w) for w in r["ref"].split()]
        items.append(dict(id=f"{r['person']}/{stem}", person=r["person"], video=str(TEAM / r["person"] / r["file"]),
                          mp=str(E / "mp" / f"{r['person']}__{stem}.npz"), ref=ref))
    return items


def extract(a):
    sys.path.insert(0, str(ROOT))
    from modules.mediapipe_pose import extract as mp_extract, save_raw
    (E / "mp").mkdir(parents=True, exist_ok=True)
    for it in team_items():
        if not Path(it["mp"]).exists():
            t = time.perf_counter()
            save_raw(it["mp"], mp_extract(it["video"], fps=25.0, normalize=False))       # = SignPipeline.landmarks_from_video
            print(f"MediaPipe {it['id']}: {time.perf_counter() - t:.0f}s", flush=True)
    (E / "team_items.json").write_text(json.dumps(team_items(), ensure_ascii=False, indent=1), encoding="utf-8")


def tsl51_items():
    A = json.loads((ROOT / "artifacts/runs/seq-v7/tsl51_alignment.json").read_text(encoding="utf-8"))
    return [dict(id=cid, ref=list(a["glosses"]), gold=[(g["f0"], g["f1"]) for g in a["segments"]],
                 pose=str(ROOT / "cloud_s3/data_prep/pose_mp/tsl51_user_sentence" / f"{cid.split(':', 1)[1]}.npz"))
            for cid, a in A.items() if cid.startswith("t51s:") and a["role"] == "sentence_test"]


def evaluate(a):
    m = MODELS[a.model]; code = Path(m["code"]).resolve()
    sys.path.insert(0, str(code))
    os.environ.setdefault("THAISLM_SYMBOLIC_WEIGHT", "0")        # Boom's hand-shape fusion: off, as in Boom's frozen test
    from modules.pipeline import SignPipeline
    from modules.schema import SH_L, SH_R, canonical_hands
    import modules.pipeline as mp_
    assert Path(mp_.__file__).resolve().is_relative_to(code), f"wrong code loaded: {mp_.__file__}"
    kw = dict(bank_file=ROOT / "artifacts/eval/bank_eval.npz") if a.set == "tsl51" else {}
    pipe = SignPipeline(model_dir=ROOT / "models" / a.model, backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False, **kw)
    if a.set == "tsl51":            # poses are already in the standard schema → bypass the MediaPipe → schema step (pitty sentence_eval.py)
        items, alphabet, cur = tsl51_items(), "off", {}
        def standardise(_):
            kp, sc, hw = cur["kp"], cur["sc"], cur["hw"]
            present = sc[:, [0, SH_L, SH_R]].mean(1) > 0.5
            sc = sc.copy(); sc[~present] = 0.0; kp = kp.copy(); kp[~present] = 0.0
            kp, sc, mirrored = canonical_hands(kp, sc, hw)
            return kp, sc, hw, present, mirrored
        pipe.standardise = standardise
        def load(it):
            z = np.load(it["pose"]); cur.update(kp=z["kp"].astype(np.float32), sc=z["sc"].astype(np.float32), hw=tuple(int(v) for v in z["hw"]))
            return None
    else:
        from modules.mediapipe_pose import load_raw
        items, alphabet = json.loads((E / "team_items.json").read_text(encoding="utf-8")), "en"
        load = lambda it: load_raw(it["mp"])
    pipe.run_landmarks(load(items[0]), alphabet=alphabet)          # warm-up (first-call latency)
    out = []
    for it in items:
        raw = load(it)
        t = time.perf_counter(); r = pipe.run_landmarks(raw, alphabet=alphabet); sec = time.perf_counter() - t
        segs = [dict(f0=int(s["f0"]), f1=int(s["f1"]), t0=s["t0"], t1=s["t1"], word=s["nearest"], status=s["status"],
                     p=round(float(s.get("p_correct", 0.0)), 3), top5=[c["word"] for c in s["candidates"][:5]])
                for s in r["segments"] if s["status"] != "transition"]
        out.append(dict({k: v for k, v in it.items() if k not in ("pose", "mp", "video")}, segs=segs, latency_s=round(sec, 3),
                        duration_s=r.get("duration_s")))
        print(f"{it['id']}: {' '.join(s['word'] for s in segs)}", flush=True)
    vocab = sorted(pipe.concepts[c] for c in pipe.bank.concepts_present)
    od = E / "results"; od.mkdir(parents=True, exist_ok=True)
    (od / f"{a.model}__{a.set}.json").write_text(json.dumps(dict(model=a.model, set=a.set, alphabet=alphabet,
        card=json.loads((ROOT / "models" / a.model / "model_card.json").read_text(encoding="utf-8")),
        vocab=vocab, items=out), ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["setup", "extract", "eval"])
    ap.add_argument("--model", choices=list(MODELS)); ap.add_argument("--set", choices=["tsl51", "team"])
    a = ap.parse_args()
    dict(setup=setup, extract=extract, eval=evaluate)[a.step](a)
