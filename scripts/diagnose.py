"""Forward error diagnosis (v9 stage 1): where does each error come from — segmentation or recognition?

    python main.py diagnose ledger [--tag v8.1] [--sets a,b]   # error ledger + oracle bounds on the cached tune / test sets
                                                               #   → artifacts/reports/diagnosis_<tag>.json

Needs the front-pass cache of scripts/decision.py (`python main.py decision cache`). For every utterance with gold boundaries:
  * each reference word gets one class: correct · sub_top5 (right word 2nd–5th) · sub_out (not in top-5) · merged (one predicted
    segment covers it and a neighbour) · split (two predictions inside it) · missed (no prediction overlaps it)
  * extra = predicted segments that overlap no reference word
  * oracle segmentation: the same reader (span head + hand cue) on the gold spans → the recognition ceiling of this encoder / bank
  * oracle recognition: the predicted segments with every matched one named correctly → the segmentation floor of WER
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from decision import TEST_SETS, TUNE_SETS, Harness, _iou, load_set, score, REP, _p  # noqa: E402
from modules.decode import wer  # noqa: E402
from modules.utils import write_json  # noqa: E402


def ledger_row(segs, row):
    live = [s for s in segs if s["status"] != "transition"]
    ref, gold = row["ref"], row["gold"]
    cls = []
    for j, g in enumerate(gold):
        L = max(g[1] - g[0], 1)
        ov = [(i, min(s["f1"], g[1]) - max(s["f0"], g[0])) for i, s in enumerate(live)]
        ov = [(i, o) for i, o in ov if o > 0.3 * L or o > 0.5 * max(live[i]["f1"] - live[i]["f0"], 1)]
        if not ov:
            cls.append("missed"); continue
        if len(ov) >= 2:
            cls.append("split"); continue
        i = ov[0][0]; s = live[i]
        others = [k for k, h in enumerate(gold) if k != j and min(s["f1"], h[1]) - max(s["f0"], h[0]) > 0.5 * max(h[1] - h[0], 1)]
        if others:
            cls.append("merged"); continue
        top = [c["word"] for c in s["candidates"][:5]]
        cls.append("correct" if s["nearest"] == ref[j] else "sub_top5" if ref[j] in top else "sub_out")
    extra = sum(all(_iou((s["f0"], s["f1"]), g) < 0.1 for g in gold) for s in live)
    return cls, extra


def oracle_read(h, row, bank):
    """Gold spans read like any segment (h.pipe.hand_rescore: span head over the frame states + hand cue)."""
    st = {k: (v.astype(np.float32) if k in ("H", "H2") and v is not None else v) for k, v in row["st"].items()}
    if st.get("H") is None:
        return [None] * len(row["gold"])
    segs = [dict(f0=int(a), f1=int(max(b, a + 1)), candidates=[]) for a, b in row["gold"]]
    if h.pipe.spot.get("hand_w", 0) and getattr(h.banks[bank], "hand", None) is not None:
        segs = h.pipe.hand_rescore(segs, st, h.banks[bank])
    else:
        E = h.pipe.span(st["H"], [(s["f0"], s["f1"]) for s in segs])
        S = h.banks[bank].scores(E)
        for s, r in zip(segs, S):
            o = np.argsort(-r)[:5]
            s["nearest"] = h.pipe.concepts[int(o[0])]; s["candidates"] = [dict(word=h.pipe.concepts[int(k)]) for k in o]
    return segs


def cmd_ledger(a):
    h = Harness()
    h.use(a.tag)
    out = {}
    names = a.sets.split(",") if a.sets else TEST_SETS + TUNE_SETS
    for name in names:
        rows = load_set(name)
        if not rows[0].get("gold"):
            continue
        C = collections.Counter(); n_extra = 0
        conf = collections.Counter()
        o_hit1 = o_hit5 = o_n = 0; o_wer, f_wer, wers = [], [], []
        for row in rows:
            segs = h.back(row["st"], row["bank"])
            sc = score(segs, row, h.vocab)
            wers.append(sc["wer"])
            cls, extra = ledger_row(segs, row)
            C.update(cls); n_extra += extra
            live = [s for s in segs if s["status"] != "transition"]
            for s, t in zip(live, sc["true"]):
                if t is not None and s["nearest"] != t:
                    conf[(t, s["nearest"])] += 1
            # segmentation floor: matched predictions named correctly, everything else as is
            hyp = [t if t is not None else s["nearest"] for s, t in zip(live, sc["true"])]
            f_wer.append(wer(row["ref"], hyp))
            orc = oracle_read(h, row, row["bank"])
            if orc[0] is not None:
                hyp = []
                for s, w in zip(orc, row["ref"]):
                    top = [c["word"] for c in s["candidates"][:5]]
                    if w in h.vocab:
                        o_n += 1; o_hit1 += s["nearest"] == w; o_hit5 += w in top
                    hyp.append(s["nearest"])
                o_wer.append(wer(row["ref"], hyp))
        tot = sum(C.values())
        out[name] = dict(n_utt=len(rows), ref_words=tot, classes={k: round(v / tot, 4) for k, v in C.most_common()}, counts=dict(C),
                         extra_per_utt=n_extra / len(rows), WER=float(np.mean(wers)),
                         oracle_seg=dict(R1=o_hit1 / max(o_n, 1), R5=o_hit5 / max(o_n, 1), WER=float(np.mean(o_wer)) if o_wer else None),
                         oracle_recog_WER=float(np.mean(f_wer)),
                         top_confusions=[(f"{t}→{p}", n) for (t, p), n in conf.most_common(25)])
        r = out[name]
        _p("%-15s WER %.3f | oracle-seg WER %.3f R1 %.3f R5 %.3f | oracle-recog WER %.3f | %s extra/utt %.2f", name, r["WER"],
           r["oracle_seg"]["WER"] or -1, r["oracle_seg"]["R1"], r["oracle_seg"]["R5"], r["oracle_recog_WER"],
           " ".join(f"{k}={v:.3f}" for k, v in r["classes"].items()), r["extra_per_utt"])
        _p("    confusions: %s", ", ".join(f"{k}({n})" for k, n in r["top_confusions"][:12]))
    write_json(REP / f"diagnosis_{a.tag}.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["ledger"])
    ap.add_argument("--tag", default="v8.1")
    ap.add_argument("--sets", default="")
    a = ap.parse_args()
    dict(ledger=cmd_ledger)[a.step](a)


if __name__ == "__main__":
    main()
