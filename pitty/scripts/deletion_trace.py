"""pitty: where do words go missing? TSL51 sentences (tune half only) through the production pipeline, recording the
segments after each stage; every reference word the final output misses (a deletion in the WER alignment) is put in one bucket:

  tagger      the tagger marks < 30 % of the word's frames as signing
  no_segment  signing is marked, but the decoder put no segment on the word
  shared      a segment covers the word, but it belongs to a neighbouring word (two signs became one segment)
  transition  the decoder had its own segment, the post-filter turned it into a 'transition' (short / null-like / weak)
  merged      it survived the post-filter, merge_repeats joined it into the neighbour (same word)
  other       its segment is in the output but the alignment paired it differently

  python pitty/scripts/deletion_trace.py <speed> <out.json>
bank = artifacts/eval/bank_eval.npz, tune grammar, decoder settings = models/spotting.json."""
import sys, json, copy, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import logging; logging.disable(logging.WARNING)
import modules.pipeline as pl
from modules.pipeline import SignPipeline
from modules.schema import canonical_hands, SH_L, SH_R
from modules.sequence import resample

speed, out_f = float(sys.argv[1]), sys.argv[2]
A = json.loads((ROOT / "artifacts/runs/seq-v7/tsl51_alignment.json").read_text(encoding="utf-8"))
pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False, bank_file=ROOT / "artifacts/eval/bank_eval.npz")
pipe.grammar = json.loads((ROOT / "artifacts/eval/grammar_tune.json").read_text(encoding="utf-8"))
cur, rec = {}, {}


def standardise(raw):
    kp, sc, hw = cur["kp"], cur["sc"], cur["hw"]
    present = sc[:, [0, SH_L, SH_R]].mean(1) > 0.5
    sc = sc.copy(); sc[~present] = 0.0; kp = kp.copy(); kp[~present] = 0.0
    kp, sc, m = canonical_hands(kp, sc, hw)
    return kp, sc, hw, present, m


pipe.standardise = standardise
_tag, _spot, _post, _merge = pipe._tag, pipe._spot, pl.postprocess, pl.merge_repeats


def tag(*a, **k):
    p, H = _tag(*a, **k); rec.setdefault("prob", p); return p, H


def spot(*a, **k):
    out = _spot(*a, **k); rec["spot"] = out[0]; rec["spot_bounds"] = None; return out


def post(segs, cfg):
    rec["pre_post"] = [(s["f0"], s["f1"]) for s in segs]          # final-frame bounds (after tempo mapping)
    r = _post(segs, cfg); rec["post"] = copy.deepcopy([{k: v for k, v in s.items() if k != "emb"} for s in segs]); return r


def merge(segs, **k):
    r = _merge(segs, **k); rec["merged"] = [dict(f0=s["f0"], f1=s["f1"], status=s["status"], nearest=s["nearest"]) for s in r]; return r


pipe._tag, pipe._spot = tag, spot
pl.postprocess, pl.merge_repeats = post, merge
from wer_ops import align

ov = lambda a, b: max(0, min(a[1], b[1]) - max(a[0], b[0]))
B = collections.Counter(); rows = []; tot = collections.Counter()
for cid, a in A.items():
    if not cid.startswith("t51s:") or a["role"] != "tagger_train":
        continue
    z = np.load(ROOT / "cloud_s3/data_prep/pose_mp/tsl51_user_sentence" / f"{cid.split(':', 1)[1]}.npz")
    kp, sc = z["kp"].astype(np.float32), z["sc"].astype(np.float32)
    if speed != 1.0:
        kp, sc = resample(kp, sc, speed)
    cur.update(kp=kp, sc=sc, hw=tuple(int(v) for v in z["hw"])); rec.clear()
    r = pipe.run_landmarks(None)
    segs = [s for s in r["segments"] if s["status"] != "transition"]; hyp = [s["nearest"] for s in segs]
    gt = [(g["concept"], g["f0"] / speed, g["f1"] / speed) for g in a["segments"]]
    if [g[0] for g in gt] != a["glosses"]:
        continue                                                   # alignment and gloss list disagree: skip rather than guess
    ops = align(a["glosses"], hyp)
    for op in ops:
        tot[op[0]] += 1
    prob = rec["prob"]; post_s = rec.get("post", [])
    for op in ops:
        if op[0] != "D":
            continue
        i = op[4]; w, f0, f1 = gt[i]; span = (f0, f1); L = max(1.0, f1 - f0)
        a0, a1 = int(f0), max(int(f0) + 1, int(np.ceil(f1)))
        act = float(((prob[a0:a1, 1] + prob[a0:a1, 2]) > pipe.spot["act_thr"]).mean()) if a1 <= len(prob) else 0.0
        own = lambda s: max(range(len(gt)), key=lambda j: ov((s["f0"], s["f1"]), gt[j][1:])) == i
        cov = [s for s in post_s if ov((s["f0"], s["f1"]), span) > 0.3 * L]
        if act < 0.3:
            b = "tagger"
        elif not cov:
            b = "no_segment"
        elif not any(own(s) for s in cov):
            b = "shared"
        elif all(s["status"] == "transition" for s in cov if own(s)):
            b = "transition"
        elif not any(own(s) and s["status"] != "transition" for s in rec["merged"] if ov((s["f0"], s["f1"]), span) > 0.3 * L):
            b = "merged"
        else:
            b = "other"
        B[b] += 1
        mine = [s for s in cov if own(s)]
        rows.append(dict(clip=cid, word=w, len_frames=round(L, 1), active=round(act, 2), bucket=b,
                         seg=[dict(f0=s["f0"], f1=s["f1"], status=s["status"], nearest=s["nearest"], p=round(s.get("p_correct", 0), 3),
                                   p_sign=round(s.get("p_sign", 0), 3), null_gap=round(s.get("null_gap", 0), 3)) for s in mine]))
print(f"speed {speed}: ops {dict(tot)}  deletions by stage {dict(B)}")
Path(out_f).write_text(json.dumps(dict(speed=speed, ops=dict(tot), buckets=dict(B), rows=rows), ensure_ascii=False, indent=1), encoding="utf-8")
