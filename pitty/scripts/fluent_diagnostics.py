"""pitty: numbers behind "words recognised, sentences not" (Phase 6) → cache/analysis/fluent_diagnostics.json, cache/jack_test_letters.csv

1. chance control — share of reference words vs random words that reach rank 1 in SOME sliding window of a data_test video
2. tagger runs per word — data_test vs TSL51 tune vs synthetic sentences from held-out isolated clips (3-frame bridges)
3. three taggers (v7 / B control / A coart 0.5) on data_test — runs + begin peaks only (no word accuracy; data_test used ONCE
   with the user's permission, 2026-10-10, to decide whether the fluent-signing direction is worth pursuing)
4. face cues of the data_test videos (why "ไหม" never appears)
5. jack_test letters with fingerspelling on (auto / spell_only)
    python pitty/scripts/fluent_diagnostics.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import logging  # noqa: E402
logging.disable(logging.WARNING)
import onnxruntime as ort  # noqa: E402

from modules.encoder import PoseStore, to_iso  # noqa: E402
from modules.lexicon import DATA_TEST_REFS  # noqa: E402
from modules.mediapipe_pose import load_raw  # noqa: E402
from modules.pipeline import SignPipeline  # noqa: E402
from modules.schema import SH_L, SH_R, canonical_hands  # noqa: E402
from modules.sequence import _bridge, retarget  # noqa: E402

OUT = ROOT / "cache/analysis"; OUT.mkdir(parents=True, exist_ok=True)
pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False)
TAG0 = pipe.be.tagger
cidx = {c: i for i, c in enumerate(pipe.concepts)}
res = {}


def runs(p, smooth=5, thr=0.5):
    a = np.convolve(p, np.ones(smooth) / smooth, "same") > thr
    return int((a[1:] & ~a[:-1]).sum() + a[0])


def dt(stem):
    kp, sc, hw, pr, _ = pipe.standardise(load_raw(ROOT / "cache/data_test_mp" / f"{stem}.npz"))
    return to_iso(kp, hw), sc, pr


# 1. chance control
rng = np.random.RandomState(0)
cnt = dict(ref=[], rnd=[])
for stem, ref in DATA_TEST_REFS.items():
    kpi, sc, pr = dt(stem)
    prob, H = pipe._tag(kpi, sc, pr); T = len(prob)
    wins = [(a, a + L) for L in (14, 20, 28, 36) for a in range(0, max(1, T - L), 3)]
    E = pipe.span(H, wins); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    S = pipe.bank.scores(E); top1 = S.argmax(1)
    refc = [cidx[w] for w in ref if w in cidx and S[0, cidx[w]] > -1.5]
    others = [c for c in pipe.bank.concepts_present if c not in refc]
    for c in refc:
        cnt["ref"].append(int((top1 == c).any()))
    for c in rng.choice(others, 60, replace=False):
        cnt["rnd"].append(int((top1 == c).any()))
res["chance_control"] = dict(ref_n=len(cnt["ref"]), ref_rate=float(np.mean(cnt["ref"])), rnd_n=len(cnt["rnd"]), rnd_rate=float(np.mean(cnt["rnd"])))
print("1.", res["chance_control"])

# 2. runs per word
rows = []
for stem, ref in DATA_TEST_REFS.items():
    kpi, sc, pr = dt(stem); prob, _ = pipe._tag(kpi, sc, pr)
    rows.append(dict(source="data_test", item=stem, words=len(ref), runs=runs(prob[:, 1] + prob[:, 2], smooth=1)))
A = json.loads((ROOT / "artifacts/runs/seq-v7/tsl51_alignment.json").read_text(encoding="utf-8"))
for cid, a in A.items():
    if not cid.startswith("t51s:") or a["role"] != "tagger_train":
        continue
    z = np.load(ROOT / "cloud_s3/data_prep/pose_mp/tsl51_user_sentence" / f"{cid.split(':', 1)[1]}.npz")
    kp, sc = z["kp"].astype(np.float32), z["sc"].astype(np.float32); hw = tuple(int(v) for v in z["hw"])
    present = sc[:, [0, SH_L, SH_R]].mean(1) > 0.5
    sc = sc.copy(); sc[~present] = 0; kp = kp.copy(); kp[~present] = 0
    kp, sc, _ = canonical_hands(kp, sc, hw)
    prob, _ = pipe._tag(to_iso(kp, hw), sc, present)
    rows.append(dict(source="TSL51 tune", item=cid, words=len(a["glosses"]), runs=runs(prob[:, 1] + prob[:, 2], smooth=1)))
store = PoseStore(ROOT / "cloud_s3/data_prep/shards/pose_store.npz")
C = pd.read_parquet(ROOT / "cloud_s3/data_prep/shards/clips.parquet")
V = set(json.loads((ROOT / "models/vocab_conversation.json").read_text(encoding="utf-8"))["concepts"])
q = C[(C.kind == "isolated") & (C.view != "rtmw") & (C.role == "test") & C.concept.isin(V) & C.has_span & C.clip_id.isin(store.row)]


def clip(r, a, b):
    kp, sc, hw = store.get(store.row[r.clip_id], a, b); H, W = hw
    return retarget(kp * np.array([W / H, 1.0], np.float32), sc), sc


for sg, g in q.groupby("signer"):
    if len(g) < 4:
        continue
    for _ in range(6):
        picks = g.sample(min(len(g), rng.randint(3, 6)), random_state=rng.randint(1e9))
        KP, SC = [], []
        first = picks.iloc[0]
        if int(first.span_f0) > 1:
            kp, sc = clip(first, 0, int(first.span_f0)); KP.append(kp); SC.append(sc)
        for r in picks.itertuples():
            kp, sc = clip(r, int(r.span_f0), int(r.span_f1))
            if KP:
                b = _bridge(KP[-1][-1], kp[0], SC[-1][-1], sc[0], 3); KP.append(b[0]); SC.append(b[1])
            KP.append(kp); SC.append(sc)
        last = picks.iloc[-1]; L = store.length(store.row[last.clip_id])
        kp, sc = clip(last, int(last.span_f1), L)
        if len(kp):
            KP.append(kp); SC.append(sc)
        kp, sc = np.concatenate(KP), np.concatenate(SC).astype(np.float32)
        present = sc[:, [0, 5, 6]].mean(1) > 0.3
        prob, _ = pipe._tag(to_iso(kp, (1, 1)), sc, present)
        rows.append(dict(source=f"synthetic ({g.dataset.iloc[0]})", item=sg, words=len(picks), runs=runs(prob[:, 1] + prob[:, 2], smooth=1)))
R = pd.DataFrame(rows)
res["runs_per_word"] = R.to_dict("records")
print("2.", R.groupby("source").apply(lambda d: d.runs.sum() / d.words.sum()).round(2).to_dict())

# 3. three taggers on data_test (runs + begin peaks only)
TG = {"v7": ROOT / "models/onnx/tagger.onnx", "B_control": ROOT / "cache/taggers/tagger_coart0.onnx", "A_coart05": ROOT / "cache/taggers/tagger_coart05.onnx"}
rows = []
for stem, ref in DATA_TEST_REFS.items():
    kpi, sc, pr = dt(stem)
    for name, path in TG.items():
        pipe.be.tagger = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        prob, _ = pipe._tag(kpi, sc, pr)
        q1 = np.convolve(prob[:, 1], np.ones(3) / 3, "same") > 0.3
        rows.append(dict(video=stem, words=len(ref), tagger=name, runs=runs(prob[:, 1] + prob[:, 2]), begin_peaks=int((q1[1:] & ~q1[:-1]).sum() + q1[0])))
pipe.be.tagger = TAG0
res["tagger_compare"] = rows
print("3.", pd.DataFrame(rows).pivot_table(index="video", columns="tagger", values="begin_peaks").to_dict())

# 4. face cues
rows = []
for stem in DATA_TEST_REFS:
    r = pipe.run_landmarks(load_raw(ROOT / "cache/data_test_mp" / f"{stem}.npz"), title=stem)
    f = r.get("face") or {}; cu = f.get("cues") or {}
    rows.append(dict(video=stem, question_yesno_reported=f.get("question_yesno"), question_furrow=f.get("question_furrow"),
                     brow_raise_peak=cu.get("brow_raise_peak"), brow_raise_tail=cu.get("brow_raise_tail"), brow_furrow_rel=cu.get("brow_furrow_rel")))
res["face"] = rows
print("4.", rows)

# 5. letters with spelling on
rows = []
for f in sorted((ROOT / "cache/jack_test_mp").glob("*__en_alpha_*.npz")):
    L = f.stem.split("__")[0].split("_", 1)[1]
    for mode, kw in (("off (default)", {}), ("auto", dict(alphabet="en")), ("spell_only", dict(alphabet="en", spell_only=True))):
        r = pipe.run_landmarks(load_raw(f), title=f.stem, **kw)
        let = [s for s in r["segments"] if s.get("kind") == "letter"]
        words = [s["nearest"] for s in r["segments"] if s["status"] != "transition" and s.get("kind") != "letter"]
        out = "".join(s["nearest"] for s in let)
        rows.append(dict(letter=L, clip=f.stem.split("__")[1], mode=mode, output=out or " ".join(words), hit=L in out))
pd.DataFrame(rows).to_csv(ROOT / "cache/jack_test_letters.csv", index=False, encoding="utf-8-sig")
print("5.", pd.DataFrame(rows).groupby("mode").hit.mean().to_dict())
(OUT / "fluent_diagnostics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("saved")
