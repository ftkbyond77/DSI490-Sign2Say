"""Write lab.ipynb — the post-training lab (v8): data quality, hands, latent space (zero-shot vs v7 vs v8), continuous windows,
segmentation, results on unseen people, speech, production. Run: python scripts/make_lab.py && python scripts/make_lab.py --execute"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = []


def md(s):
    CELLS.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))


def code(s):
    CELLS.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))


md("""
# ThaiSLM v8 — lab: clean data → tracked hands → Thai-only encoder → continuous signing → real people → speech

Kernel: conda env `hugging`. This notebook only **reads** what the pipeline produced (`artifacts/reports/*.json`, `models/`,
`cloud_s3/data_prep/`, `artifacts/runs/`) plus a few small live computations (UMAP, one inference). Nothing here trains or tunes.
Report: `result_reporting/result_v8.md`.

| section | what you see |
|---|---|
| 1 | data quality: the YouTube audit (rules + eye check), label merges, vocabulary sized by evidence |
| 2 | hands: why the browser lost them, the fallback, the per-utterance tracking report |
| 3 | latent space: zero-shot (ASL/CSL prior) vs v7 vs v8 (Thai only), prior bias, signers per word |
| 4 | continuous windows: robustness to window boundaries (the v8 failure found and fixed) |
| 5 | tagger + segmentation on real sentences |
| 6 | results: TSL51 sentences, data_test, external people, natural phrases |
| 7 | speech: Thai voices measured by a speech recogniser |
| 8 | production: load, infer, sizes, latency; diagnosing your own recording |
""")
code("""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
plt.rcParams["font.family"] = ["Tahoma", "Leelawadee UI", "DejaVu Sans"]
from modules.utils import ART, MODEL, PREP, read_json
REP = ART / "reports"
def rep(name):
    f = REP / f"{name}.json"
    return read_json(f) if f.exists() else None
cfg = read_json(MODEL / "config.json"); cfg
""")
md("## 1. Data quality first")
code("""
q = pd.read_csv(PREP.parent / "agent_data" / "youtube_words" / "qc" / "quality_v8.csv")
print(f"YouTube word clips: {len(q)} measured · kept {int(q.keep.sum())} · dropped {int((~q.keep).sum())}")
display(q[~q.keep].reason.str.replace("eye:", "eye check: ").value_counts().rename("dropped because").to_frame())
display(q[["cuts_in_span", "faces_multi_frac", "zoom_ratio", "shoulder_frac", "hand_frac_span", "wrist_out_frac", "label_rank"]].describe().T.round(3))
""")
code("""
sheet = PREP.parent / "agent_data" / "youtube_words" / "qc" / "sheets_v8" / "review1" / "sheet_009.jpg"
if sheet.exists():
    display(Markdown("Eye check: each YouTube clip (top row) next to a dictionary reference of the same word by another signer (bottom row). "
                     "Busy ceremony graphics with a tiny signer (#49–51) were dropped; #52 (สนุก) matches its reference and was kept."))
    display(Image(str(sheet), width=1000))
""")
code("""
import re
clips = pd.read_parquet(PREP / "shards" / "clips.parquet")
prim = clips[clips.view == "primary"]
old = pd.read_parquet(PREP / "manifest" / "clips.parquet")
src = old.concept.dropna()
notes = src[src.str.contains(r"\\((?:ภาษามือโรงเรียน|พจนานุกรมสารสนเทศ|ภาษามือวิทยาลัยราชสุดา|อายุเท่ากันหรือน้อยกว่า)", regex=True)]
print("concepts that only differed by the name of their source dictionary:", notes.nunique(), "e.g.", list(notes.unique()[:3]))
print("concepts before → after merging:", old[old.use].concept.nunique(), "→", prim.concept.nunique())
iso = prim[(prim.kind == "isolated") & prim.concept.notna()]
sig = iso.groupby("concept").signer.nunique()
fig, ax = plt.subplots(figsize=(7, 3))
vc = sig.clip(upper=6).value_counts().sort_index()
ax.bar([str(i) if i < 6 else "6+" for i in vc.index], vc.values, color="#4f46e5"); ax.set_yscale("log")
ax.set_xlabel("different people signing the concept"); ax.set_ylabel("concepts"); ax.set_title("most of the 9.6k concepts have ONE signer")
plt.tight_layout(); plt.show()
for name in ("conversation", "daily"):
    f = MODEL / f"vocab_{name}.json"
    if f.exists():
        v = read_json(f); s = pd.Series(v["signers"])
        print(f"{name}: {len(v['concepts'])} words · median signers {s.median():.0f} · ≥ 2 signers {int((s >= 2).sum())}")
""")
md("## 2. Hands — lost in the browser, recovered by the fallback")
code("""
import gzip, glob
from modules.mediapipe_pose import from_browser, hand_report
def live_stats(files):
    n = {"left": [0, 0], "right": [0, 0]}; fps = []
    for f in files:
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
        r = hand_report(from_browser(d["frames"], (d["height"], d["width"])))
        for s in n:
            n[s][0] += r[s]["n"]; n[s][1] += (r[s]["hit"] or 0) * r[s]["n"]
        fps.append(r["fps"])
    return {s: round(v[1] / max(v[0], 1), 3) for s, v in n.items()}, float(np.median(fps))
old = sorted(glob.glob(str(ROOT / "cache" / "replay_src" / "*.json.gz"))) + sorted(glob.glob(str(ROOT / "cache" / "live_requests" / "20261004_*.json.gz")))
print("v7 browser, your webcam (raised-hand frames with 21 points, median fps):", live_stats(old))
ext = rep("external_v8") or rep("external_v7")
if ext:
    h = ext["hand_tracking"]
    for mode in ("holistic", "fallback"):
        print(mode, {s: round(sum((v[mode][s]["hit"] or 0) * v[mode][s]["n"] for v in h.values()) / max(sum(v[mode][s]["n"] for v in h.values()), 1), 3)
                     for s in ("left", "right")}, "longest gap", max(max(v[mode]["left"]["max_gap"], v[mode]["right"]["max_gap"]) for v in h.values()), "frames")
""")
md("## 3. Latent space — Thai-only encoder vs the ASL/CSL-pretrained one")
code("""
iso_ = rep("isolated_v8")
prot = pd.DataFrame({m: {k: v.get("R1", v.get("auc")) for k, v in p.items() if isinstance(v, dict)} for m, p in iso_["protocols"].items()})
display(prot.round(3))
display(Markdown("**Prior bias** — share of a model's test errors that are the zero-shot (ASL/CSL) model's *same* wrong answer"))
display(pd.DataFrame(iso_["prior_bias"]).T.round(3))
display(Markdown("**R@1 by number of people signing the word in the bank**"))
display(pd.DataFrame({m: {r["depth"]: r["R1"] for r in v} for m, v in iso_["recall_by_depth"].items()}).round(3))
display(pd.DataFrame(iso_["latent"]).T.round(3))
""")
code("""
import umap
idx = pd.read_parquet(ART / "runs" / "enc-v8" / "emb_index.parquet")
conv = read_json(MODEL / "vocab_conversation.json")["concepts"]
top = idx[idx.concept.isin(conv)].concept.value_counts()
top = [c for c in top.index if idx[idx.concept == c].signer.nunique() >= 4][:10]
sel = idx[idx.concept.isin(top)].groupby("concept", group_keys=False).apply(lambda g: g.sample(min(len(g), 50), random_state=0))
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for k, (run, name) in enumerate((("zeroshot-on-v8", "zero-shot Uni-Sign (CSL → ASL)"), ("v7-on-v8", "v7 (fine-tuned Uni-Sign)"), ("enc-v8", "v8 (Thai only)"))):
    Z = np.load(ART / "runs" / run / "emb.npy").astype(np.float32)
    U = umap.UMAP(n_neighbors=15, min_dist=0.2, metric="cosine", random_state=0).fit_transform(Z[sel.index.values])
    for c in top:
        m = (sel.concept == c).values; ax[k].scatter(U[m, 0], U[m, 1], s=8, label=c)
    ax[k].set_title(name); ax[k].set_xticks([]); ax[k].set_yticks([])
ax[2].legend(fontsize=8, markerscale=2, bbox_to_anchor=(1.0, 1.0)); plt.tight_layout(); plt.show()
""")
md("## 4. Continuous windows — the v8 failure, found in tuning and fixed")
code("""
for role in ("tagger_train", "sentence_test"):
    r = rep(f"continuous_windows_{role}")
    if r:
        display(Markdown(f"**{'tune sentences (decisions)' if role == 'tagger_train' else 'test sentences (report only)'}** — {r['n_windows']} word windows of an unseen signer"))
        display(pd.DataFrame({m: {k: v["R1"] for k, v in x.items()} for m, x in r["models"].items()}).T.round(3))
""")
md("## 5. Tagger and segmentation on real sentences")
code("""
import pickle
seq = ART / "runs" / cfg["sequence_run"]
D = pickle.load(open(seq / "seq_tables.pkl", "rb"))
items = [it for it in D["items"] if it["kind"] == "real" and it["split"] == "test" and it["speed"] == 1.0]
from modules.segment import gt_segments
fig, axs = plt.subplots(3, 1, figsize=(12, 7))
for ax, it in zip(axs, items[:3]):
    p = it["prob"].astype(np.float32); t = np.arange(len(p)) / 25
    ax.plot(t, p[:, 1] + p[:, 2], label="P(sign)"); ax.plot(t, p[:, 1], label="P(begin)", lw=0.8)
    for g0, g1 in gt_segments(it["y"]):
        ax.axvspan(g0 / 25, g1 / 25, color="#16a34a", alpha=0.15)
    ax.set_title(" ".join(it["glosses"]), fontsize=9); ax.set_ylim(0, 1.05)
axs[0].legend(fontsize=8); axs[-1].set_xlabel("seconds (green = aligned reference words)"); plt.tight_layout(); plt.show()
""")
md("## 6. Results — the same protocols as v7")
code("""
sp = rep("spotting")
for vocab in ("conversation", "daily"):
    if sp and vocab in sp:
        r = sp[vocab]
        t = pd.DataFrame({k: {m: r[k][m] for m in ("seg_f1", "exact_count_rate", "word_recall_top1", "word_precision_top1", "WER")} for k in r if k.startswith(("real_test", "synth_test"))}).T
        display(Markdown(f"**Continuous signing — {vocab} vocabulary (TSL51 test templates, never used for any decision)**")); display(t.round(3))
for name, title in (("infer_v8", "data_test"), ("external_v8", "external people (public videos, no source)"), ("phrases_v8", "natural phrases of an unseen signer"), ("selftest_v8", "self-made TSL sentences")):
    r = rep(name)
    if not r:
        continue
    for vocab, res in (r.get("results", r)).items():
        if isinstance(res, dict) and "summary" in res:
            display(Markdown(f"**{title} — {vocab}**")); display(pd.Series(res["summary"]).to_frame("v8").T)
""")
md("## 7. Speech — Thai voices measured by a speech recogniser")
code("""
tb = rep("tts_bench")
if tb:
    display(pd.DataFrame([{k: v[k] for k in ("voice", "cer", "rtf", "pitch_sd_semitones")} for v in tb["voices"]]).set_index("voice"))
    print("asr judge:", tb["asr"])
""")
md("## 8. Production — load, infer, inspect")
code("""
from modules.pipeline import SignPipeline
pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False)   # ONNX Runtime only — what the web API runs
r = pipe.run(ROOT / "data_test" / "พ่อดื่มน้ำ.mp4")
print("evidence:", r["fusion"]["evidence_gloss"]); print("sentence:", r["sentence"] or "(withheld)"); print("hands:", r["hand_tracking"]); print("ms:", r["timings_ms"])
""")
code("""
from modules import lmenc
m = lmenc.load_model(MODEL / "encoder.pt")
print(f"sign encoder: {sum(p.numel() for p in m.parameters())/1e6:.1f} M parameters ({len(m.members)} Thai-only members) · frame states {m.dim} · embedding {m.emb_dim}")
display(pd.Series({p.name: f"{p.stat().st_size/1e6:.1f} MB" for p in sorted(list(MODEL.glob('*.pt')) + list(MODEL.glob('*.npz')) + list((MODEL / 'onnx').glob('*.onnx')))}, name="size"))
from modules.mediapipe_pose import load_raw
raw = load_raw(ROOT / "cache" / "data_test_mp" / "พ่อดื่มน้ำ.npz")
for n in (60, 125, 250):
    sub = {k: (v[:n] if np.ndim(v) > 0 and len(v) == len(raw["t"]) else v) for k, v in raw.items()}
    t = time.perf_counter(); pipe.run_landmarks(sub); print(f"{n/25:.1f} s of signing → {(time.perf_counter()-t)*1000:.0f} ms (CPU)")
""")
md("""
### Diagnosing your own recording

```bash
python main.py diagnose path/to/my_video.mp4 --intended "ฉัน รัก เพื่อน"      # or a saved live request: cache/live_requests/<file>.json.gz
```
→ `result_reporting/diagnose/<name>/report.md` + `tracking.png`: hand tracking over time, each segment's top-5 with P(correct) and
the nearest reference clips, and for each intended word one of: landmark · segmentation · not_in_vocab · confusable · sign_differs · ok.
Your intended word is never treated as a verified label.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    nb = dict(cells=CELLS, metadata=dict(kernelspec=dict(display_name="Python 3 (hugging)", language="python", name="python3"),
                                         language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
    (ROOT / "lab.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"lab.ipynb: {len(CELLS)} cells")
    if a.execute:
        import nbformat
        from nbclient import NotebookClient
        nb = nbformat.read(ROOT / "lab.ipynb", as_version=4)
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}, allow_errors=True).execute()
        nbformat.write(nb, ROOT / "lab.ipynb")
        errs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
        print(f"executed: {len(errs)} cells with errors", [e.get("ename") for e in errs])


if __name__ == "__main__":
    main()
