"""v12 live-mode rules (prompt-opt-v12 3E): the browser's cutting rule (modules/live_rule.py, TS parity tested) on web-rate TSL51 TUNE
sentences (w_t51_tune: dataset video re-extracted at browser rate, rest before / after the sentence) and on the saved live requests (LOG).

  python scripts/v12_live_rules.py sim [--set w_t51_tune] [--model v10]   # WER of the live cut per rule set vs the whole clip (paired CI)
  python scripts/v12_live_rules.py log                                     # LOG side effects: utterances hitting the 20 s cap per rule set

Rule sets: v10 (pre 500 / tail 400 / end 1500) · LM2 pre 1500 + tail 800 / 1000 / 1200 · LM2+LM1 (body-pose wrist keeps an utterance open
only 800 ms after the last real hand) · LM2+LM1+LM3 (hands back within 1.2 s → same utterance). Live words of a clip = the words of every
utterance the rule cut from it (split sentences count; a lost part is a deletion).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from modules.live_rule import frames_to_arrays, live_utts

RULES_ALL = {
    "v10": dict(end_ms=1500, pre=500, tail=400),
    "LM2_t800": dict(end_ms=1500, pre=1500, tail=800),
    "LM2_t1000": dict(end_ms=1500, pre=1500, tail=1000),
    "LM2_t1200": dict(end_ms=1500, pre=1500, tail=1200),
    "LM2_LM1": dict(end_ms=1500, pre=1500, tail=1200, wrist_grace_ms=800),
    "LM2_LM1_LM3": dict(end_ms=1500, pre=1500, tail=1200, wrist_grace_ms=800, continue_ms=1200),
    "v12": dict(end_ms=1500, pre=1500, tail=800, wrist_grace_ms=800, cap_cut=True),
}
RULES = RULES_ALL


def sub(raw, idx):
    T = len(raw["t"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == T and k != "hw" else v) for k, v in raw.items()}


def cmd_sim(a):
    import decision as D
    from modules.decode import wer
    from modules.pipeline import SignPipeline
    items = D.build_items(a.set)
    pipe = SignPipeline(model_dir=ROOT / "models" / f"model-{a.model}", backend="onnx", device="cpu", vocab="conversation", use_llm=False,
                        use_face=False, bank_file=ROOT / "artifacts/eval/bank_eval.npz")

    SEGS = []

    def words(raw):
        segs = [s for s in pipe.run_landmarks(raw)["segments"] if s["status"] != "transition"]
        SEGS.append([(s["nearest"], s["status"], float(s.get("p_correct", 0) or 0)) for s in segs])
        return [s["nearest"] for s in segs], [s["nearest"] for s in segs if s["status"] == "accepted"]
    res = {c: [] for c in ["full"] + list(RULES)}
    for n, it in enumerate(items):
        raw = dict(it["raw"]); raw["t"] = np.asarray(raw["t"], np.float64) - float(raw["t"][0])
        ref = it["ref"]
        top, shown = words(raw)
        res["full"].append(dict(id=it["id"], wer=wer(ref, top), disp=wer(ref, shown), n_utt=1, ref=ref, segs=SEGS[-1]))
        arr = dict(t=raw["t"], pose=np.asarray(raw["pose"], np.float32), lh=np.asarray(raw["lh"], np.float32), rh=np.asarray(raw["rh"], np.float32))
        for c, rule in RULES.items():
            utts, dropped = live_utts(arr, **rule)
            tp, sh, sg = [], [], []
            for u in utts:
                x, y = words(sub(raw, u)); tp += x; sh += y; sg += SEGS[-1]
            res[c].append(dict(id=it["id"], wer=wer(ref, tp), disp=wer(ref, sh), n_utt=len(utts), dropped=dropped, ref=ref, segs=sg))
        if n % 20 == 0:
            print(n, len(items), {c: round(float(np.mean([x["wer"] for x in v])), 3) for c, v in res.items()}, flush=True)
    rng = np.random.RandomState(0)
    summ = {}
    base = np.array([x["wer"] for x in res["full"]])
    for c, v in res.items():
        w = np.array([x["wer"] for x in v]); d = w - base
        bs = d[rng.randint(0, len(d), (2000, len(d)))].mean(1)
        summ[c] = dict(WER=float(w.mean()), displayed=float(np.mean([x["disp"] for x in v])), d_vs_full=float(d.mean()),
                       ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                       split=float(np.mean([x["n_utt"] > 1 for x in v])), missed=float(np.mean([x["n_utt"] == 0 for x in v])))
        if c != "full":
            dv = w - np.array([x["wer"] for x in res["v10"]]); bs2 = dv[rng.randint(0, len(dv), (2000, len(dv)))].mean(1)
            summ[c].update(d_vs_v10=float(dv.mean()), ci_vs_v10=[float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5))])
        print(f"{c:12s} WER {summ[c]['WER']:.3f} disp {summ[c]['displayed']:.3f} vs whole {summ[c]['d_vs_full']:+.3f} [{summ[c]['ci'][0]:+.3f},{summ[c]['ci'][1]:+.3f}]"
              f" split {summ[c]['split']:.2f} missed {summ[c]['missed']:.2f}" + (f" | vs v10 {summ[c]['d_vs_v10']:+.3f} [{summ[c]['ci_vs_v10'][0]:+.3f},{summ[c]['ci_vs_v10'][1]:+.3f}]" if c != "full" else ""), flush=True)
    out = ROOT / "artifacts" / "reports" / f"v12_live_rules_{a.set}_{a.model}.json"
    out.write_text(json.dumps(dict(summary=summ, rules=RULES, rows=res), ensure_ascii=False), encoding="utf-8")


def cmd_log(a):
    """saved live requests are what the v10 rule SENT: re-cutting each with the v12 rules shows how many runaway (capped) utterances
    would have ended earlier, and whether utterances that had accepted words keep their signing frames"""
    files = sorted((ROOT / "cache" / "live_requests").glob("*.json.gz"))
    out = {c: dict(n=0, capped_or_long=0, ends_earlier=0, kept_frames=[]) for c in RULES}
    for f in files:
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
        fr = d.get("frames") or []
        if len(fr) < 10:
            continue
        arr = frames_to_arrays(fr)
        dur = (arr["t"][-1] - arr["t"][0]) * 1000
        acc = any(s.get("status") == "accepted" for s in (d.get("result") or {}).get("segments", []))
        for c, rule in RULES.items():
            utts, _ = live_utts(arr, **dict(rule, pre=0))
            o = out[c]; o["n"] += 1
            long_ = dur >= 9000
            o["capped_or_long"] += int(long_)
            if long_ and utts and (arr["t"][utts[0][-1]] - arr["t"][0]) * 1000 < dur - 1000:
                o["ends_earlier"] += 1
            if acc:
                kept = sum(len(u) for u in utts) / len(fr)
                o["kept_frames"].append(kept)
    for c, o in out.items():
        k = np.array(o.pop("kept_frames")) if o.get("kept_frames") is not None else np.zeros(0)
        o["accepted_requests_frames_kept_median"] = float(np.median(k)) if len(k) else None
        o["accepted_requests_losing_half"] = float(np.mean(k < 0.5)) if len(k) else None
        print(c, o, flush=True)
    (ROOT / "artifacts" / "reports" / "v12_live_rules_log.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--set", default="w_t51_tune"); ap.add_argument("--model", default="v10")
    ap.add_argument("--rules", default="")
    a = ap.parse_args()
    if a.rules:
        RULES = {k: RULES_ALL[k] for k in a.rules.split(",")}
    dict(sim=cmd_sim, log=cmd_log)[a.cmd](a)
