"""Figures for the v7 report (result_reporting/figures_v7/*.png) from artifacts/reports/*.json, the run history and the shards.

  python scripts/report_figures.py --run <encoder_run_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.utils import ART, PREP, read_json

OUT = ROOT / "result_reporting" / "figures_v7"
REP = ART / "reports"
plt.rcParams.update({"font.family": ["Tahoma", "Leelawadee UI", "DejaVu Sans"], "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
C7, C6, C0 = "#4f46e5", "#94a3b8", "#cbd5e1"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT / name, dpi=130); plt.close(fig)


def fig_data():
    c = pd.read_parquet(PREP / "shards" / "clips.parquet")
    p = c[c.view == "primary"]
    g = p.groupby("dataset").agg(clips=("clip_id", "size"), hours=("n_frames", lambda f: f.sum() / 25 / 3600), signers=("signer", "nunique"))
    g = g.sort_values("hours")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].barh(g.index, g.hours, color=C7); ax[0].set_xlabel("hours of pose (MediaPipe)"); ax[0].set_title("v7 training data — one extractor")
    for i, (h, n) in enumerate(zip(g.hours, g.signers)):
        ax[0].text(h, i, f"  {n} signers", va="center", fontsize=7)
    d = pd.read_csv(ROOT / "vocab" / "daily_conversation_depth.csv")
    v6 = pd.read_csv(ROOT / "cache" / "daily_depth_v6.csv") if (ROOT / "cache" / "daily_depth_v6.csv").exists() else None
    bins = [0, 1, 2, 3, 5, 10, 1000]; lab = ["0", "1", "2", "3-4", "5-9", "10+"]
    h7 = pd.cut(d.signers, bins, right=False, labels=lab).value_counts().reindex(lab)
    x = np.arange(len(lab))
    if v6 is not None:
        h6 = pd.cut(v6.signers, bins, right=False, labels=lab).value_counts().reindex(lab)
        ax[1].bar(x - 0.2, h6.values, 0.4, color=C6, label="v6")
    ax[1].bar(x + 0.2, h7.values, 0.4, color=C7, label="v7")
    ax[1].set_xticks(x, lab); ax[1].set_xlabel("signers per word"); ax[1].set_ylabel("daily-conversation words")
    ax[1].set_title("depth of the daily-conversation vocabulary"); ax[1].legend()
    save(fig, "data.png")


def fig_training(run):
    h = read_json(run / "train_history.json")
    s = [r["step"] for r in h]
    fig, ax = plt.subplots(figsize=(7, 3))
    for k, lab in (("E1_ttrs_val|open_vocab", "TTRS val"), ("ONE_val|open_vocab", "TSL-ONE-S val"), ("THSL_val|open_vocab", "th_sl val"),
                   ("AGENT_val|open_vocab", "YouTube val (new people)")):
        y = [r.get(k, {}).get("MRR") for r in h]
        if any(v is not None for v in y):
            ax.plot(s, y, marker="o", ms=3, label=lab)
    ax.plot(s, [r["val_score"] for r in h], color="k", lw=2, label="selection score")
    ax.set_xlabel("step"); ax.set_ylabel("MRR, held-out signers"); ax.legend(ncol=3); ax.set_title("encoder fine-tuning (Spot A100)")
    save(fig, "training_curves.png")


def fig_isolated():
    iso = read_json(REP / "isolated.json")
    comp = iso["comparison"]
    names = [k for k in comp if "daily" not in k]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    x = np.arange(len(names))
    for j, (enc, col) in enumerate((("zero_shot", C0), ("v6", C6), ("v7", C7))):
        y = [comp[n].get(enc, {}).get("R1", np.nan) for n in names]
        ax[0].bar(x + (j - 1) * 0.27, y, 0.27, color=col, label={"zero_shot": "zero-shot (ASL/CSL prior)", "v6": "v6 encoder", "v7": "v7"}[enc])
    ax[0].set_xticks(x, [n.split(" (")[0] for n in names], rotation=15); ax[0].set_ylabel("R@1 (same queries, same bank)")
    ax[0].set_title("isolated signs — held-out signers, MediaPipe"); ax[0].legend()
    d = pd.DataFrame(iso["recall_by_depth"]).dropna()
    ax[1].bar(d.depth, d.R1, color=C7); ax[1].set_xlabel("signers of the word in the bank"); ax[1].set_ylabel("R@1")
    for i, (r, n) in enumerate(zip(d.R1, d.n)):
        ax[1].text(i, r, f"n={n}", ha="center", va="bottom", fontsize=7)
    ax[1].set_title("recognition grows with signers per word")
    save(fig, "isolated.png")


def fig_continuous():
    sp = read_json(REP / "spotting.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    for k, vocab in enumerate(("conversation", "full")):
        r = sp[vocab]
        sets = [s for s in ("real_test_x1.0", "real_test_x1.5", "real_test_x2.0", "synth_test") if s in r]
        x = np.arange(len(sets))
        for j, (m, col) in enumerate((("seg_f1", "#16a34a"), ("exact_count_rate", "#0891b2"), ("word_recall_top1", C7), ("WER", "#dc2626"))):
            ax[k].bar(x + (j - 1.5) * 0.2, [r[s][m] for s in sets], 0.2, color=col, label=m)
        ax[k].set_xticks(x, [s.replace("real_test_", "TSL51 test ").replace("synth_test", "synthetic\nheld-out") for s in sets])
        ax[k].set_ylim(0, 1); ax[k].set_title(f"continuous signing — {vocab} vocabulary")
    ax[0].legend(ncol=2)
    save(fig, "continuous.png")


def fig_latency():
    inf = read_json(REP / "infer_final.json")
    v = inf["results"]["conversation"]["videos"]
    rows = []
    for k, x in v.items():
        t = x["timings_ms"]
        rows.append(dict(video=k[:14], dur=x["duration_s"], **{s: t.get(f"{s}_ms", 0) / 1000 for s in ("schema", "tagger", "spotting", "face", "language")}))
    d = pd.DataFrame(rows).set_index("video")
    fig, ax = plt.subplots(figsize=(8, 3))
    d[["schema", "tagger", "spotting", "face", "language"]].plot.barh(stacked=True, ax=ax, colormap="viridis")
    ax.set_xlabel("seconds on CPU (ONNX Runtime) after landmarks"); ax.set_title("latency per data_test video (signing length in labels)")
    ax.set_yticks(range(len(d)), [f"{i} ({r:.1f} s)" for i, r in zip(d.index, d.dur)])
    save(fig, "latency.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).exists() else ART / "runs" / a.run
    for f in (fig_data, lambda: fig_training(run), fig_isolated, fig_continuous, fig_latency):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print("figure failed:", getattr(f, "__name__", "lambda"), type(e).__name__, e)
    print("figures →", OUT, sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
