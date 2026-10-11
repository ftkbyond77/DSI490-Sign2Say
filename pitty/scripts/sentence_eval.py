"""Sentence-level eval (pitty): production SignPipeline on TSL51 sentence poses (cloud_s3/data_prep/pose_mp/tsl51_user_sentence),
bank = artifacts/eval/bank_eval.npz (training signers only). Usage: python pitty/scripts/sentence_eval.py <out.json> [alpha] [speed]"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import logging; logging.disable(logging.WARNING)
from modules.pipeline import SignPipeline
from modules.schema import canonical_hands, SH_L, SH_R
from modules.decode import wer
from modules.sequence import resample

out_f, alpha, speed = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.0, float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
delta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0     # depth prior: δ·log(signers per word) on the RECOGNITION table only
A = json.loads((ROOT / "artifacts/runs/seq-v7/tsl51_alignment.json").read_text(encoding="utf-8"))
import os
BANK = Path(os.environ.get("BANK_FILE", ROOT / "artifacts/eval/bank_eval.npz"))   # pitty: alternative bank (e.g. alias-merged)
pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False, bank_file=BANK)
if alpha:
    pipe.hub = np.load(ROOT / "cache/pitty_hub.npy").astype(np.float32); pipe.spot = dict(pipe.spot, csls_alpha=alpha)
if delta:
    from modules.decode import concept_prior
    be_ = np.load(ROOT / "artifacts/eval/bank_eval.npz")
    depth = pd.Series(be_["signer"]).groupby(be_["c"]).nunique().reindex(range(len(pipe.concepts))).fillna(0).values.astype(np.float32)
    pipe.word_prior = concept_prior(pipe.prior, depth, pipe.spot.get("freq_gamma", 0.0), delta)
import os
if os.environ.get("TAGGER_ONNX"):          # swap ONLY the tagger (encoder, span head, bank, decoder settings unchanged)
    import onnxruntime as ort
    pipe.be.tagger = ort.InferenceSession(os.environ["TAGGER_ONNX"], providers=["CPUExecutionProvider"])
if os.environ.get("SPOT_OVERRIDE"):         # decoder settings override (JSON, "post" merged) — models/spotting.json untouched
    ov = json.loads(os.environ["SPOT_OVERRIDE"])
    post = dict(pipe.spot.get("post", {}), **ov.pop("post", {}))
    pipe.spot = dict(pipe.spot, **ov, post=post)
if os.environ.get("FLAT_TAGGER"):           # stress test: the tagger sees no boundaries inside signing (what happens on fluent signing)
    _tag0 = pipe._tag
    def _flat(kpi, sc_, present):
        prob, H = _tag0(kpi, sc_, present)
        act = ((prob[:, 1] + prob[:, 2]) > 0.5) & present
        idx = np.flatnonzero(act)
        if len(idx):
            prob = prob.copy(); prob[idx[0]:idx[-1] + 1] = [0.0, 0.0, 1.0]
        return prob, H
    pipe._tag = _flat
ONLY = os.environ.get("SPLIT")              # "tune" → skip the test half entirely (test stays unseen)
cur = {}
def standardise(raw):
    kp, sc, hw = cur["kp"], cur["sc"], cur["hw"]
    present = sc[:, [0, SH_L, SH_R]].mean(1) > 0.5
    sc = sc.copy(); sc[~present] = 0.0; kp = kp.copy(); kp[~present] = 0.0
    kp, sc, m = canonical_hands(kp, sc, hw)
    return kp, sc, hw, present, m
pipe.standardise = standardise
res = []; t0 = time.time()
for cid, a in A.items():
    if not cid.startswith("t51s:"): continue
    if ONLY and ("tune" if a["role"] == "tagger_train" else "test") != ONLY: continue
    if os.environ.get("VARIATION") and a.get("variation") != os.environ["VARIATION"]: continue   # e.g. no_space only
    z = np.load(ROOT / "cloud_s3/data_prep/pose_mp/tsl51_user_sentence" / f"{cid.split(':', 1)[1]}.npz")
    kp, sc = z["kp"].astype(np.float32), z["sc"].astype(np.float32)
    if os.environ.get("HOLD_CUT"):           # pitty: cut still holds between signs (pitty/scripts/hold_cut.py) — "thr,keep,min_run"
        sys.path.insert(0, str(ROOT / "pitty" / "scripts")); from hold_cut import cut_holds
        t_, k_, m_ = os.environ["HOLD_CUT"].split(",")
        kp, sc = cut_holds(kp, sc, tuple(int(v) for v in z["hw"]), float(t_), int(k_), int(m_))
    if speed != 1.0: kp, sc = resample(kp, sc, speed)
    if os.environ.get("FPS_SIM"):            # pitty: live-camera capture at a low frame rate (browser MediaPipe), linearly re-sampled to 25 fps
        fps_ = float(os.environ["FPS_SIM"]); T_ = len(kp); rs_ = np.random.RandomState(abs(hash(cid)) % 2**31)
        tk = np.cumsum(rs_.uniform(0.7, 1.3, int(T_ / 25 * fps_) + 2) / fps_); tk = tk[tk * 25 < T_ - 1]       # jittered capture times (s)
        src = np.clip(tk * 25, 0, T_ - 1); grid = np.arange(T_, dtype=float)
        i0 = np.searchsorted(src, grid, side="right") - 1; i0 = np.clip(i0, 0, len(src) - 2); w = np.clip((grid - src[i0]) / np.maximum(src[i0 + 1] - src[i0], 1e-6), 0, 1)
        fa, fb = np.floor(src[i0]).astype(int), np.floor(src[i0 + 1]).astype(int)
        kp = (kp[fa] * (1 - w)[:, None, None] + kp[fb] * w[:, None, None]).astype(np.float32); sc = np.minimum(sc[fa], sc[fb])
    cur.update(kp=kp, sc=sc, hw=tuple(int(v) for v in z["hw"]))
    r = pipe.run_landmarks(None)
    segs = [s for s in r["segments"] if s["status"] != "transition"]
    top1 = [s["nearest"] for s in segs]; ref = a["glosses"]
    res.append(dict(clip=cid, split="tune" if a["role"] == "tagger_train" else "test", ref=ref, top1=top1,
                    top5=[[c["word"] for c in s["candidates"][:5]] for s in segs], WER=wer(ref, top1),
                    cands=[[[c["word"], round(float(c["score"]), 4)] for c in s["candidates"][:10]] for s in segs],
                    recall1=float(np.mean([w in top1 for w in ref])), recall5=float(np.mean([any(w in c for c in [[x["word"] for x in s["candidates"][:5]] for s in segs]) for w in ref])),
                    n_pred=len(top1), n_ref=len(ref)))
Path(out_f).write_text(json.dumps(dict(alpha=alpha, speed=speed, delta=delta, tagger=os.environ.get("TAGGER_ONNX", "models/onnx/tagger.onnx"), spot_override=os.environ.get("SPOT_OVERRIDE"), flat_tagger=bool(os.environ.get("FLAT_TAGGER")), results=res), ensure_ascii=False), encoding="utf-8")
df = pd.DataFrame(res); print(out_f, round(time.time() - t0), "s"); print(df.groupby("split")[["WER", "recall1", "recall5"]].mean().round(3).to_string())
