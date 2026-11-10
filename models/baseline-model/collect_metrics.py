"""Baseline (zero-shot Uni-Sign, conversation vocabulary = v10's) on TSL51 sentence_test + Jack + data_test -> results/baseline_slide_metrics.json
  python models/baseline-model/collect_metrics.py
Per clip: reference, top-1 sequence, segments (t0, t1, top-5), substitutions / deletions / insertions. Aggregates are pooled over reference words."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

import numpy as np

from baseline import Baseline, canon_fn, score_video
from evalsets import DATA_TEST, team
from modules.encoder import concept_scores, retrieval_metrics
from modules.mediapipe_pose import load_raw
from modules.schema import SH_L, SH_R

TSL51_VIDEOS = ROOT / "cloud_s3" / "raw_data" / "TSL51" / "videos" / "user_sentence"


def align(ref, hyp):
    """Levenshtein alignment -> (S, D, I)."""
    n, m = len(ref), len(hyp); d = np.zeros((n + 1, m + 1), int); d[:, 0] = range(n + 1); d[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, S, D, I = n, m, 0, 0, 0
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]; i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return int(S), int(D), int(I)


def tsl51_items():
    A = json.loads((ROOT / "artifacts/runs/seq-v7/tsl51_alignment.json").read_text(encoding="utf-8"))
    return {cid: dict(ref=list(a["glosses"]), gold=[(g["f0"] / 25, g["f1"] / 25) for g in a["segments"]],
                      pose=ROOT / "cloud_s3/data_prep/pose_mp/tsl51_user_sentence" / f"{cid.split(':', 1)[1]}.npz",
                      video=TSL51_VIDEOS / f"{cid.split(':', 1)[1]}.mp4")
            for cid, a in A.items() if cid.startswith("t51s:") and a["role"] == "sentence_test"}


def main():
    canon = canon_fn()
    b = Baseline("zeroshot", vocab="full")
    conv = {str(c) for c in json.loads((ROOT / "models/model-v10/vocab_conversation.json").read_text(encoding="utf-8"))["concepts"]}
    b.allowed[:] = [c in conv for c in b.concepts]                       # same conversation vocabulary as v10
    bv = {b.concepts[i] for i in set(b.inst_c) if b.allowed[i]}
    print("vocabulary:", int(b.allowed.sum()), "concepts")
    out = {"tsl51": {}, "jack": {}, "data_test": {}}; lat = []

    def keep(sname, k, segs, ref, **extra):
        s = score_video(segs, ref, canon, bv); s["S"], s["D"], s["I"] = align(s["ref"], s["hyp"])
        s["segments"] = [dict(t0=x["t0"], t1=x["t1"], top=[(w, round(sc, 3)) for w, sc in x["top"]]) for x in segs]; s.update(extra)
        out[sname][k] = s

    for k, (p, ref, v) in team("Jack").items():
        t = time.time(); segs = b.recognise_landmarks(load_raw(p)); lat.append(time.time() - t); keep("jack", k, segs, ref, video=str(v))
    for k, (p, ref) in DATA_TEST.items():
        t = time.time(); segs = b.recognise_landmarks(load_raw(p)); lat.append(time.time() - t); keep("data_test", k, segs, ref)
    for cid, it in tsl51_items().items():
        z = np.load(it["pose"]); kp, sc, hw = z["kp"].astype(np.float32), z["sc"].astype(np.float32), tuple(int(v) for v in z["hw"])
        present = sc[:, [0, SH_L, SH_R]].mean(1) > 0.5; sc = sc.copy(); sc[~present] = 0.0; kp = kp.copy(); kp[~present] = 0.0
        t = time.time(); segs = b.recognise(kp, sc, hw); lat.append(time.time() - t)
        keep("tsl51", cid, segs, it["ref"], video=str(it["video"]), gold=it["gold"])

    def agg(r):
        n = sum(x["n_ref"] for x in r.values()); N = len(r)
        return dict(n_videos=N, n_words=n, r1=sum(x["r1"] * x["n_ref"] for x in r.values()) / n, r5=sum(x["r5"] * x["n_ref"] for x in r.values()) / n,
                    wer=sum(x["S"] + x["D"] + x["I"] for x in r.values()) / n, sub=sum(x["S"] for x in r.values()) / n, dele=sum(x["D"] for x in r.values()) / n,
                    ins=sum(x["I"] for x in r.values()) / n, count_err=float(np.mean([x["count_err"] for x in r.values()])),
                    exact=float(np.mean([x["S"] + x["D"] + x["I"] == 0 for x in r.values()])))
    res = {k: agg(v) for k, v in out.items()}
    res["overall"] = agg({f"{s}/{k}": v for s, d in out.items() for k, v in d.items()})
    z = np.load(HERE / "bank.npz", allow_pickle=True); c, role, ds = z["concept"], z["role"], z["dataset"]; tr = role == "train"; n_c = len(z["concepts"]); Z = z["Z0"].astype(np.float32)
    iso = {}
    for d in ("ttrs", "tslone", "th_sl"):
        q = (role == "test") & (ds == d)
        iso[d] = {kk: float(v) for kk, v in retrieval_metrics(concept_scores(Z[q], Z[tr], c[tr], n_c), c[q]).items() if kk in ("n", "R1", "R5", "MRR")}
    json.dump(dict(vocab_size=int(b.allowed.sum()), agg=res, per_video=out, isolated=iso, n_params_M=90.3, enc_s_per_video_cpu=float(np.mean(lat))),
              open(HERE / "results" / "baseline_slide_metrics.json", "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)); print("latency/video", np.mean(lat))


if __name__ == "__main__":
    main()
