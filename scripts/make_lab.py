"""Write lab.ipynb — the post-training lab (v7): data, latent space, frame tagger, segmentation, results, self-made sentence tests,
face cues, and how to use the production model. Run: python scripts/make_lab.py && python scripts/make_lab.py --execute"""
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
# ThaiSLM v7 — lab: data → latent space → tagger → segmentation → results → real use

Kernel: conda env `hugging`. This notebook only **reads** what the pipeline produced (`artifacts/reports/*.json`, `models/`,
`cloud_s3/data_prep/`) plus a few small live computations (UMAP, one inference). Nothing here trains or tunes anything.

| section | what you see |
|---|---|
| 1 | data: one extractor (MediaPipe) for everything, daily-conversation word depth, harvested YouTube signers |
| 2 | poses: MediaPipe vs RTMW of the same video, handedness canonicalisation |
| 3 | latent space: UMAP of sign embeddings (zero-shot ASL/CSL prior vs v7), neighbourhood purity, prior bias |
| 4 | frame tagger: P(sign) / P(begin) over real sentences, span head (one-pass) vs exact window scores |
| 5 | segmentation: decoder output vs the reference, transition filter |
| 6 | results: held-out signers, continuous sentences (1× / 1.5× / 2×), data_test |
| 7 | self-made sentence videos (held-out signers, TSL grammar) → full system from the pixels |
| 8 | face channel: blendshapes → question / negation / emotion |
| 9 | production: load the model, inference, sizes, architecture, latency, web API |
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
md("## 1. Data — one extractor, one schema")
code("""
clips = pd.read_parquet(PREP / "shards" / "clips.parquet")
prim = clips[clips.view == "primary"]
summary = prim.groupby(["dataset", "extractor"]).agg(clips=("clip_id", "size"), signers=("signer", "nunique"), concepts=("concept", "nunique"),
                                                      hours=("n_frames", lambda f: round(f.sum() / 25 / 3600, 2)))
display(summary)
print("RTMW views of the same videos (extra positives for training, never used at inference):", int((clips.view == "rtmw").sum()))
""")
code("""
dd = pd.read_csv(ROOT / "vocab" / "daily_conversation_depth.csv")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
b = [0, 1, 2, 3, 5, 10, 100]
ax[0].hist(dd.signers.clip(upper=99), bins=b, rwidth=0.9, color="#4f46e5"); ax[0].set_xscale("symlog"); ax[0].set_title("daily-conversation words: signers per word (v7 bank)")
ax[0].set_xlabel("signers"); ax[0].set_ylabel("words")
cat = dd.groupby("category").signers.median().sort_values()
ax[1].barh(cat.index, cat.values, color="#16a34a"); ax[1].set_title("median signers per word, by category")
plt.tight_layout(); plt.show()
print("daily words with >= 3 signers:", int((dd.signers >= 3).sum()), "/", len(dd), "· not in any source:", list(dd[dd.signers == 0].word))
""")
code("""
it = Path(PREP).parent / "agent_data" / "youtube_words" / "items.csv"
if it.exists():
    ag = pd.read_csv(it)
    display(ag.groupby(["status"]).size().rename("clips"))
    display(ag[ag.status == "rejected"].reason.str.split("(").str[0].value_counts().rename("rejected because"))
    acc = ag[ag.status == "accepted"]
    print(f"accepted {len(acc)} clips · {acc.concept.nunique()} words · {acc.signer.nunique()} people (channels merged by face)")
    display(acc.sample(min(8, len(acc)), random_state=0)[["title", "word", "concept", "channel", "signer", "span_s", "consistency", "role"]])
    q = it.parent / "qc" / "agent_signers.jpg"
    if q.exists():
        display(Markdown("Signer identities of the harvested channels (one row per person; last rows = data_test people, checked for leakage)"))
        display(Image(str(q), width=900))
""")
md("## 2. Poses — the extractor gap, and why v7 removes it")
code("""
from modules.schema import load, iso
def skel(ax, kp, sc, title):
    p = kp.copy()
    for a, b in [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]:
        ok = (sc[:, a] > 0.3) & (sc[:, b] > 0.3)
        for t in np.flatnonzero(ok)[::6]:
            ax.plot(p[t, [a, b], 0], -p[t, [a, b], 1], color="#94a3b8", lw=0.6)
    for sl, col in ((slice(91, 112), "#0891b2"), (slice(112, 133), "#db2777")):
        m = sc[:, sl].mean(1) > 0.3
        ax.scatter(p[m, sl][:, :, 0].mean(1), -p[m, sl][:, :, 1].mean(1), s=4, color=col)
    ax.set_title(title, fontsize=9); ax.set_aspect("equal"); ax.axis("off")
cid = prim[(prim.dataset == "ttrs") & prim.aux_clip.notna()].iloc[0]
store = np.load(PREP / "shards" / "pose_store.npz", allow_pickle=True)
from modules.encoder import PoseStore
S = PoseStore(PREP / "shards" / "pose_store.npz")
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for k, (c, name) in enumerate(((cid.clip_id, "MediaPipe (production extractor)"), (cid.aux_clip, "RTMW-X (v6 extractor)"))):
    kp, sc, hw = S.get(S.row[c])
    skel(ax[k], iso(kp, hw), sc, f"{cid.concept} — {name}")
plt.show()
iso_ = rep("isolated")
if iso_:
    pc = [r for r in iso_["table"] if r["protocol"] == "extractor_pair_cos"]
    print("same video, MediaPipe vs RTMW → cosine of the two embeddings:", {r["metric"]: (round(r["zero_shot"], 3), round(r["v7"], 3)) for r in pc}, "(zero-shot, v7)")
print("clips mirrored to right-hand-dominant:", int(prim.mirrored.sum()), "of", len(prim))
""")
md("## 3. Latent space — does the model learn Thai signs, or lean on the ASL/CSL prior?")
code("""
run = ART / "runs" / cfg["encoder_run"]
idx = pd.read_parquet(run / "emb_index.parquet")
Z1, Z0 = np.load(run / "emb.npy").astype(np.float32), np.load(run / "emb_zero_shot.npy").astype(np.float32)
conv = read_json(MODEL / "vocab_conversation.json")["concepts"]
top = idx[idx.concept.isin(conv) & (idx.kind == "isolated")].concept.value_counts()
top = [c for c in top.index if idx[idx.concept == c].signer.nunique() >= 4][:12]
sel = idx[idx.concept.isin(top)].groupby("concept", group_keys=False).apply(lambda g: g.sample(min(len(g), 60), random_state=0))
import umap
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for k, (Z, name) in enumerate(((Z0, "zero-shot (Uni-Sign CSL→ASL prior)"), (Z1, "v7 fine-tuned"))):
    U = umap.UMAP(n_neighbors=15, min_dist=0.2, metric="cosine", random_state=0).fit_transform(Z[sel.index.values])
    for c in top:
        m = (sel.concept == c).values
        ax[k].scatter(U[m, 0], U[m, 1], s=9, label=c)
    ax[k].set_title(name); ax[k].set_xticks([]); ax[k].set_yticks([])
ax[1].legend(fontsize=8, markerscale=2, bbox_to_anchor=(1.0, 1.0))
plt.tight_layout(); plt.show()
if iso_:
    display(pd.DataFrame(iso_["latent"]).T.round(3))
    display(Markdown("**Prior bias** (of v7's errors on held-out signers, how many are the zero-shot ASL/CSL model's *same* wrong answer):"))
    display(pd.Series(iso_["prior_bias"]).round(3))
""")
md("## 4. Frame tagger and the one-pass span head")
code("""
import pickle
seq = ART / "runs" / cfg["sequence_run"]
D = pickle.load(open(seq / "seq_tables.pkl", "rb"))
items = [it for it in D["items"] if it["kind"] == "real" and it["split"] == "test" and it["speed"] == 1.0]
from modules.segment import gt_segments
fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=False)
for ax, it in zip(axs, items[:3]):
    p = it["prob"].astype(np.float32); t = np.arange(len(p)) / 25
    ax.plot(t, p[:, 1] + p[:, 2], label="P(sign)"); ax.plot(t, p[:, 1], label="P(begin)", lw=0.8)
    for g0, g1 in gt_segments(it["y"]):
        ax.axvspan(g0 / 25, g1 / 25, color="#16a34a", alpha=0.15)
    ax.set_title(" ".join(it["glosses"]), fontsize=9); ax.set_ylim(0, 1.05)
axs[0].legend(fontsize=8); axs[-1].set_xlabel("seconds (green = aligned reference words)")
plt.tight_layout(); plt.show()
sp = rep("spotting")
if sp:
    c = sp["conversation"]
    print("one-pass (span head) vs per-window teacher on the TSL51 test templates:")
    display(pd.DataFrame({"one-pass span head (deployed)": {k: c[k]["word_recall_top1"] for k in c if k.startswith("real_test")},
                          "teacher windows (slow)": {k: v["word_recall_top1"] for k, v in c["teacher_windows_on_test"].items() if k.startswith("real_test")}}).round(3))
""")
md("## 5. Segmentation — decoder vs reference, and the transition filter")
code("""
from modules.pipeline import timeline
inf = rep("infer_final")
for stem in ["พ่อดื่มน้ำ", "ฉันรักเพื่อน"]:
    f = ROOT / "result_reporting" / "inference" / "conversation" / stem / "timeline.png"
    if f.exists():
        display(Markdown(f"**{stem}** (data_test) — grey = transition (not used in the sentence)")); display(Image(str(f)))
""")
md("## 6. Results")
code("""
if iso_:
    rows = {k: {n: v.get("R1") for n, v in d.items()} for k, d in iso_["comparison"].items()}
    display(Markdown("**Isolated signs, held-out signers (R@1, same queries and same bank for every encoder)**"))
    display(pd.DataFrame(rows).T.round(3))
    display(Markdown("**Recall by bank depth (signers per word)**")); display(pd.DataFrame(iso_["recall_by_depth"]).round(3))
if sp:
    for vocab in ("conversation", "full"):
        r = sp[vocab]
        t = pd.DataFrame({k: {m: r[k][m] for m in ("seg_f1", "exact_count_rate", "word_recall_top1", "word_precision_top1", "WER")} for k in r if k.startswith(("real_test", "synth_test"))}).T
        display(Markdown(f"**Continuous signing — {vocab} vocabulary (TSL51 test templates never used for any decision)**")); display(t.round(3))
""")
code("""
if inf:
    for vocab, res in inf["results"].items():
        display(Markdown(f"**data_test — {vocab} vocabulary**: {json.dumps(res['summary'], ensure_ascii=False)}"))
        display(pd.DataFrame({k: dict(reference=" ".join(v["ref"]), top1=" ".join(v["top1"]), sentence=v["sentence"] or f"(withheld) {v['tentative'] or ''}",
                                      R1=v["recall_top1"], R5=v["recall_top5"]) for k, v in res["videos"].items()}).T)
""")
md("""
## 7. Self-made sentence videos (TSL grammar) → the full system from the pixels

Sentences are written in **TSL word order** (topic/object first, verb late, question last — the pattern of TSL51 and the
interpreters' videos), e.g. `พ่อ น้ำ ดื่ม` → *พ่อดื่มน้ำ*, `วันนี้ ฉัน โรงเรียน ไป` → *วันนี้ฉันไปโรงเรียน*. Each word is a real video clip of a
**held-out signer** (test roles: never in training, and this evaluation uses the training-signer bank only), cut to its signing span
and joined with a short cross-fade (the hand transition between signs), at natural (1.3×) and fast phone pace (1.8×). The system
then runs from the pixels: MediaPipe → schema → tagger → spotting → sentence.
""")
code("""
st = rep("selftest")
if st:
    for vocab in ("conversation",):
        display(Markdown(f"**{vocab}**: {json.dumps(st[vocab]['summary'], ensure_ascii=False)}"))
        display(pd.DataFrame({k: s for k, s in st[vocab]["by_speed"].items()}).T[["recall_top1", "recall_top5", "ref_words_in_sentence", "words_in_sentence_correct", "words_in_sentence", "count_abs_err"]])
        v = st[vocab]["videos"]
        display(pd.DataFrame({k: dict(TSL_order=" ".join(x["ref"]), meaning=x["thai_reference"], speed=x["speed"], predicted=" ".join(x["top1"]),
                                      sentence=x["sentence"] or f"(withheld) {x['tentative'] or ''}") for k, x in list(v.items())[:30]}).T)
""")
code("""
import cv2
d = ROOT / "cache" / "selftest"
vids = sorted(d.glob("*x1.3.mp4"))[:3]
for v in vids:
    cap = cv2.VideoCapture(str(v)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fr = []
    for i in np.linspace(0, n - 1, 8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if ok: fr.append(cv2.resize(f, (160, int(160 * f.shape[0] / f.shape[1]))))
    cap.release()
    if fr:
        plt.figure(figsize=(14, 2.2)); plt.imshow(cv2.cvtColor(np.concatenate(fr, 1), cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.title(v.stem, fontsize=9); plt.show()
""")
md("## 8. Face channel — blendshapes → question / negation / emotion")
code("""
from modules.mediapipe_pose import load_raw, BLENDSHAPES
from modules.face import analyse
f = ROOT / "cache" / "data_test_mp" / "ไปทานข้าวด้วยกันมั้ย.npz"
if f.exists():
    raw = load_raw(f); B = {n: i for i, n in enumerate(BLENDSHAPES)}
    t = raw["t"]
    plt.figure(figsize=(12, 3))
    for n in ("browInnerUp", "browOuterUpLeft", "browDownLeft", "jawOpen", "mouthSmileLeft"):
        plt.plot(t, raw["bs"][:, B[n]], label=n)
    plt.legend(fontsize=8); plt.xlabel("seconds"); plt.title("ไปทานข้าวด้วยกันมั้ย — face blendshapes (MediaPipe)"); plt.show()
    print(json.dumps({k: v for k, v in analyse(raw).items() if k != "per_segment"}, ensure_ascii=False, indent=1))
""")
md("## 9. Production — load, infer, inspect")
code("""
from modules.pipeline import SignPipeline
pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False)   # ONNX Runtime only — what the web API runs
r = pipe.run(ROOT / "data_test" / "พ่อดื่มน้ำ.mp4")                                         # MediaPipe → … → sentence
print("evidence:", r["fusion"]["evidence_gloss"]); print("sentence:", r["sentence"] or "(withheld)"); print("timings (ms):", r["timings_ms"])
""")
code("""
# the same call from landmarks (what the browser sends): frames = [{t, pose[33×4], lh[21×3], rh[21×3], face[68×2], bs[52]}, …]
from modules.mediapipe_pose import extract
raw = extract(ROOT / "data_test" / "ฉันรักเพื่อน.mp4")
t = time.perf_counter(); r = pipe.run_landmarks(raw); print(f"{(time.perf_counter()-t)*1000:.0f} ms for {r['duration_s']} s of signing →", r["fusion"]["evidence_gloss"])
""")
code("""
import onnx, torch
sizes = {p.name: f"{p.stat().st_size/1e6:.1f} MB" for p in sorted(list(MODEL.glob('*.pt')) + list(MODEL.glob('*.npz')) + list((MODEL / 'onnx').glob('*.onnx')) + list((MODEL / 'mediapipe').glob('*')))}
display(pd.Series(sizes, name="size"))
from modules.encoder import load_encoder
enc = load_encoder(MODEL, MODEL / "encoder.pt", device="cpu")
n_all = sum(p.numel() for p in enc.parameters()); n_gcn = sum(p.numel() for p in enc.pose.parameters()); n_head = sum(p.numel() for p in enc.head.parameters())
print(f"sign encoder: {n_all/1e6:.1f} M parameters (ST-GCN {n_gcn/1e6:.1f} M · mT5 encoder {(n_all-n_gcn-n_head)/1e6:.1f} M · head {n_head/1e6:.2f} M)")
from modules.sequence import Tagger
tg = Tagger(); print(f"tagger (BiGRU): {sum(p.numel() for p in tg.parameters())/1e6:.2f} M parameters")
sh = np.load(MODEL / "span_head.npz"); print("span head:", {k: v.shape for k, v in sh.items()})
m = onnx.load(str(MODEL / "onnx" / "encoder.onnx"), load_external_data=False)
print("encoder.onnx inputs:", [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print("encoder.onnx outputs:", [(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
print(enc.head); print(tg)
""")
code("""
# latency on this machine (CPU, ONNX Runtime): one-pass spotting for utterances of different lengths
from modules.mediapipe_pose import load_raw
raw = load_raw(ROOT / "cache" / "data_test_mp" / "พ่อดื่มน้ำ.npz")
for n in (60, 125, 250):
    sub = {k: (v[:n] if np.ndim(v) > 0 and len(v) == len(raw["t"]) else v) for k, v in raw.items()}
    t = time.perf_counter(); pipe.run_landmarks(sub); print(f"{n/25:.1f} s of video → {(time.perf_counter()-t)*1000:.0f} ms")
""")
md("""
### Web API (docker compose up --build → http://localhost:8080)

```python
import requests
r = requests.post("http://localhost:8080/api/v1/translate/video", files={"file": open("data_test/พ่อดื่มน้ำ.mp4", "rb")}, params={"vocab": "conversation"})
r.json()["sentence"], r.json()["segments"]
# browser path: POST /api/v1/translate {frames: [...MediaPipe landmarks...], width, height, vocab, llm, speak}
# speech:      POST /tts/v1/speak {text, emotion} → audio/wav
```
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
        errs = [c for c in nb.cells if c.cell_type == "code" and any(o.get("output_type") == "error" for o in c.get("outputs", []))]
        print(f"executed; cells with errors: {len(errs)}")


if __name__ == "__main__":
    main()
