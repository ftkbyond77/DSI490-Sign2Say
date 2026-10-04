"""Choose the fingerspelling settings (models/spotting.json "spell") on tuning data only, then report on held-out data.

  python scripts/tune_spelling.py

Negatives  TSL51 sentences (no spelling) at 1×/1.5×/2×: every letter inserted there is an error.  tune templates → choose, test → report
Positives  real spelling: 4 consecutive letter holds of an alphabet lesson (the continuous video between them).
           training channels → choose, held-out channels → report
Rule: among the settings that insert no letter into any tune sentence, take the one with the most correctly read letters (edit
distance).
"""
from __future__ import annotations

import copy
import itertools
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from modules.decode import wer
from modules.pipeline import SignPipeline
from modules.utils import ART, DATA, MODEL, get_logger, read_json, write_json
from train_fingerspell import std_raw

log = get_logger("spell_tune")


def prepare(p, kpi, sc):
    """Lexical result (letters off) + frame states, computed once per utterance."""
    p.standardise = lambda raw, kpi=kpi, sc=sc: (kpi, sc, (1, 1), sc[:, [0, 5, 6]].mean(1) > 0.5, False)
    r = p.run_landmarks({"bs": np.zeros((len(kpi), 52), np.float32), "face": np.zeros((len(kpi), 68, 2), np.float32),
                         "t": np.arange(len(kpi)) / 25.0, "hw": np.array([1, 1])}, alphabet="off")
    _, H = p._tag(kpi, sc, sc[:, [0, 5, 6]].mean(1) > 0.5)
    return dict(kpi=kpi, sc=sc, H=H, segs=[{k: v for k, v in s.items() if k != "emb"} for s in r["segments"]])


def spelled(p, u, cfg):
    p.spell_cfg = cfg
    segs = p._spell(u["kpi"], u["sc"], u["H"], copy.deepcopy(u["segs"]), "en")
    return segs


def main():
    p = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False, bank_file=ART / "eval" / "bank_eval.npz")
    D = pickle.load(open(ART / "runs" / "seq-v7" / "seq_tables.pkl", "rb"))
    neg = {sp: [] for sp in ("tune", "test")}
    for it in D["items"]:
        if it["kind"] == "real" and it["split"] in neg:
            neg[it["split"]].append(prepare(p, it["kp"].astype(np.float32), it["sc"].astype(np.float32)))
    log.info("negatives: %s utterances", {k: len(v) for k, v in neg.items()})
    it = pd.read_csv(DATA / "agent_data" / "en_alpha" / "items.csv").sort_values(["video", "f0"])
    pos = {"train": [], "test": []}
    for vid, g in it.groupby("video"):
        g = g.reset_index(drop=True)
        kpi, sc = std_raw(DATA / "agent_data" / "en_alpha" / "pose_mp_raw" / f"{vid}.npz")
        for i in range(0, len(g) - 3, 4):
            w = g.iloc[i:i + 4]
            a, b = max(0, int(w.f0.min()) - 8), min(len(kpi), int(w.f1.max()) + 8)
            if b - a > 400:
                continue
            u = prepare(p, kpi[a:b], sc[a:b]); u["truth"] = list(w.letter)
            pos[w.role.iloc[0]].append(u)
    log.info("positives: %s spans of 4 letters", {k: len(v) for k, v in pos.items()})
    base = dict(p.spell_cfg)
    grid = [dict(base, thr=t, single_min=s1, run_min=rm, accept=0.75, run_gap=rg) for t, s1, rm, rg in
            itertools.product([0.6, 0.65, 0.75], [0.9, 0.97, 1.01], [0.7, 0.8], [15, 30])]

    def score(cfg, which):
        false = sum(sum(s.get("kind") == "letter" for s in spelled(p, u, cfg)) for u in neg[which])
        ok = n = acc_wrong = acc = 0
        for u in pos["train" if which == "tune" else "test"]:
            got = [s["nearest"] for s in spelled(p, u, cfg) if s.get("kind") == "letter"]
            n += len(u["truth"]); ok += len(u["truth"]) - round(wer(u["truth"], got) * len(u["truth"]))
            a = [s for s in spelled(p, u, cfg) if s.get("kind") == "letter" and s["status"] == "accepted"]
            acc += len(a); acc_wrong += sum(s["nearest"] not in u["truth"] for s in a)
        return dict(false_letters_per_100=round(100 * false / max(1, len(neg[which])), 2), letters_read=round(ok / max(1, n), 3),
                    accepted_letters=acc, accepted_wrong=acc_wrong)
    rows = []
    for cfg in grid:
        sc_ = score(cfg, "tune")
        rows.append((cfg, sc_))
    ok = [r for r in rows if r[1]["false_letters_per_100"] == 0.0] or sorted(rows, key=lambda r: r[1]["false_letters_per_100"])[:1]
    best = max(ok, key=lambda r: (r[1]["letters_read"], -r[1]["false_letters_per_100"]))
    test = score(best[0], "test")
    log.info("chosen on tune: %s → tune %s | held-out test %s", {k: best[0][k] for k in ("thr", "single_min", "run_min", "accept")}, best[1], test)
    spot = read_json(MODEL / "spotting.json")
    spot["spell"] = {k: best[0][k] for k in ("thr", "accept", "single_min", "run_gap", "w_bank", "run_min")}
    write_json(MODEL / "spotting.json", spot)
    write_json(ART / "reports" / "spelling.json", dict(chosen=spot["spell"], tune=best[1], held_out=test,
                                                       grid=[dict(cfg={k: c[k] for k in ("thr", "single_min", "run_min", "accept")}, **s) for c, s in rows]))


if __name__ == "__main__":
    main()
