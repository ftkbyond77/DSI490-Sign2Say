"""v12 rung decisions (prompt-opt-v12 C5): several versions on the SAME cached utterances, per-utterance scores, paired bootstrap
(2,000 resamples by utterance AND by signer) of every version against a reference.

  python scripts/v12_paired.py --ref v10 --tags v12-B1,v12-B2 [--sets tune|devtest|a,b] [--out name]

Endpoints reported (C3): E1 pooled sentence top-1 WER (TSL51 1x / 1.5x / 2x + fluent), E2 new-signer isolated top-1 (nsig sets),
isolated WER, web TSL51 WER, Tail accepted share, idle accepted words / 10 s; matched-precision coverage comes from diagnose v12.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from decision import REP, Harness, cache_path, evaluate, load_set  # noqa: E402

SETS = dict(tune=["t51_tune_x1.0", "t51_tune_x1.5", "t51_tune_x2.0", "flu_tune", "iso_tune", "nsig_dev", "w_t51_tune", "tail_tune", "w_tail_tune", "idle_tune", "w_idle_tune"],
            devtest=["t51_test_x1.0", "t51_test_x1.5", "t51_test_x2.0", "flu_test", "iso_test", "w_t51_test", "tail_test", "w_tail_test", "idle_test", "w_idle_test"])
E1 = ("t51_", "flu_")


def per_item(h, name, rows):
    """→ list of dicts: id, signer, value (WER, or top-1 hit for nsig, or accepted flag for tail, accepted words for idle)"""
    out = []
    if name.startswith(("tail", "w_tail", "idle", "w_idle")):
        for r in rows:
            segs = h.back(r["st"], r["bank"])
            n_acc = sum(s["status"] == "accepted" for s in segs)
            v = float(n_acc > 0) if "tail" in name else float(n_acc)
            out.append(dict(id=r["id"], signer=str(r.get("signer", r["id"])), value=v, T=r["T"]))
        return out
    if name.startswith("nsig"):
        res, segs_all = evaluate(h, rows, keep_segs=True)
    else:
        res, segs_all = evaluate(h, rows), [[]] * len(rows)
    for r, x, sg in zip(rows, res, segs_all):
        v = x["wer"]
        if name.startswith("nsig"):
            live = [s for s in sg if s["status"] != "transition"]
            s = max(live, key=lambda s_: s_["f1"] - s_["f0"]) if live else None
            v = float(s is not None and s["nearest"] == r["ref"][0])
        out.append(dict(id=r["id"], signer=str(r.get("signer", r["id"].split("@")[0])), value=v))
    return out


def boot(d, groups_, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    d, g = np.asarray(d, float), np.asarray(groups_)
    bu = d[rng.randint(0, len(d), (n, len(d)))].mean(1)
    ug = np.unique(g); idx = {u: np.flatnonzero(g == u) for u in ug}
    bs = [np.concatenate([d[idx[u]] for u in ug[rng.randint(0, len(ug), len(ug))]]).mean() for _ in range(n)]
    return [float(np.percentile(bu, 2.5)), float(np.percentile(bu, 97.5))], [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="v10"); ap.add_argument("--tags", required=True); ap.add_argument("--sets", default="tune"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    names = [n for n in SETS.get(a.sets, a.sets.split(",")) if cache_path(n).exists()]
    tags = [a.ref] + a.tags.split(",")
    h = Harness()
    vals = {t: {} for t in tags}
    for t in tags:
        h.use(t)
        for n in names:
            vals[t][n] = per_item(h, n, load_set(n))
        print("scored", t, flush=True)
    res = {}
    for t in tags[1:]:
        R = {}
        for n in names:
            ref, new = vals[a.ref][n], vals[t][n]
            d = [y["value"] - x["value"] for x, y in zip(ref, new)]
            if n.startswith(("idle", "w_idle")):
                sec = sum(x["T"] for x in ref) / 25.0
                R[n] = dict(n=len(d), ref=10 * sum(x["value"] for x in ref) / sec, new=10 * sum(y["value"] for y in new) / sec, unit="accepted words / 10 s")
                continue
            ci_u, ci_s = boot(d, [x["signer"] for x in ref])
            R[n] = dict(n=len(d), ref=float(np.mean([x["value"] for x in ref])), new=float(np.mean([y["value"] for y in new])), diff=float(np.mean(d)),
                        ci_utt=ci_u, ci_signer=ci_s, better=int(sum((v < 0) if not n.startswith("nsig") else (v > 0) for v in d)),
                        worse=int(sum((v > 0) if not n.startswith("nsig") else (v < 0) for v in d)))
        pool = [n for n in names if n.startswith(E1) and not n.startswith("w_")]
        d = sum([[y["value"] - x["value"] for x, y in zip(vals[a.ref][n], vals[t][n])] for n in pool], [])
        g = sum([[x["signer"] for x in vals[a.ref][n]] for n in pool], [])
        if d:
            ci_u, ci_s = boot(d, g)
            R["E1_pooled_sentence_WER"] = dict(n=len(d), sets=pool, diff=float(np.mean(d)), ci_utt=ci_u, ci_signer=ci_s,
                                               ref=float(np.mean(sum([[x["value"] for x in vals[a.ref][n]] for n in pool], []))),
                                               new=float(np.mean(sum([[y["value"] for y in vals[t][n]] for n in pool], []))))
        res[t] = R
        print(f"== {t} vs {a.ref}", flush=True)
        for n, r in R.items():
            if "ci_utt" in r:
                print(f"  {n:24s} {r['ref']:.3f} -> {r['new']:.3f}  d {r['diff']:+.4f}  CI utt [{r['ci_utt'][0]:+.4f},{r['ci_utt'][1]:+.4f}]  signer [{r['ci_signer'][0]:+.4f},{r['ci_signer'][1]:+.4f}]  n {r['n']}", flush=True)
            else:
                print(f"  {n:24s} {r['ref']:.3f} -> {r['new']:.3f} ({r['unit']})", flush=True)
    out = REP / f"v12_paired_{a.out or a.sets}_{'_'.join(tags[1:])}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
