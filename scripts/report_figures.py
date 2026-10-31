"""Figures of the v10 report (result_reporting/figures_v10/*.png) from artifacts/reports/*.json and cache/v10/*.parquet. Figures whose
inputs do not exist yet are skipped. (v7–v9 figures stay in result_reporting/figures_v7 … figures_v9.)

  python scripts/report_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.utils import ART, read_json

OUT = ROOT / "result_reporting" / "figures_v10"
REP = ART / "reports"
plt.rcParams.update({"font.family": ["Leelawadee UI", "Tahoma", "DejaVu Sans"], "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
C9, C10, CG, CR = "#94a3b8", "#4f46e5", "#16a34a", "#dc2626"
SRC = {"th_sl": "th_sl", "ttrs": "TTRS", "tslone": "TSL-ONE-S", "tsl51": "TSL51", "agent": "YouTube"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT / name, dpi=130); plt.close(fig)
    print("figure", OUT / name)


def fig_diagnosis():
    f = ROOT / "cache" / "v10" / "diag_words.parquet"
    if not f.exists():
        return
    W = pd.read_parquet(f)
    W["err"] = W.cls != "correct"
    W = W[W.set.str.contains("tune")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    # (a) error vs dominant-hand presence inside the sign
    bins = [0, .5, .8, .95, 1.01]
    for k, (s, g) in enumerate(W.groupby("source")):
        e = g.groupby(pd.cut(g.dom_seen, bins, right=False), observed=False).err.agg(["mean", "size"])
        x = np.arange(len(e)) + (k - 2) * 0.16
        axs[0].bar(x, e["mean"].where(e["size"] >= 15), 0.16, label=f"{SRC.get(s, s)} (n={len(g)})")
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels(["<0.5", "0.5–0.8", "0.8–0.95", "≥0.95"])
    axs[0].set_xlabel("share of the sign's frames with the dominant hand (MediaPipe)"); axs[0].set_ylabel("P(error)")
    axs[0].set_title("(a) error vs hand presence — flat within a source"); axs[0].legend(fontsize=7)
    # (b) error vs bank depth (non-TSL51)
    z = np.load(ART / "eval" / "bank_eval.npz")
    concepts = read_json(ROOT / "models" / "model-v9" / "concepts.json")
    depth = pd.Series([concepts[int(c)] for c in z["c"]]).value_counts()
    g = W[W.source != "tsl51"].assign(n_bank=lambda d: d.word.map(depth).fillna(0))
    e = g.groupby(pd.cut(g.n_bank, [-1, 2, 5, 10, 20, 1e4]), observed=False).err.agg(["mean", "size"])
    axs[1].bar(range(len(e)), e["mean"], color=C10)
    for i, (m, n) in enumerate(zip(e["mean"], e["size"])):
        if n:
            axs[1].text(i, m + 0.02, f"n={n}", ha="center", fontsize=7)
    axs[1].set_xticks(range(len(e))); axs[1].set_xticklabels(["≤2", "3–5", "6–10", "11–20", ">20"])
    axs[1].set_xlabel("bank clips of the true word (training signers)"); axs[1].set_ylabel("P(error)")
    axs[1].set_title("(b) the source gap is bank coverage")
    # (c) confusion pairs: Uni-Sign vs landmark-encoder margin
    d = read_json(REP / "diagnosis_v10.json")["D_geometry"]
    rows = [(r["pair"], r["margin"], q["margin"]) for r, q in zip(d["unisign"]["pairs"], d["lmenc"]["pairs"]) if r["margin"] is not None][:12]
    y = np.arange(len(rows))
    axs[2].barh(y - 0.2, [r[1] for r in rows], 0.4, color=C9, label="Uni-Sign")
    axs[2].barh(y + 0.2, [r[2] for r in rows], 0.4, color=C10, label="Thai landmark encoder")
    axs[2].set_yticks(y); axs[2].set_yticklabels([r[0] for r in rows]); axs[2].invert_yaxis(); axs[2].axvline(0, color="k", lw=0.6)
    axs[2].set_xlabel("same-word cross-signer cos − cos to the confused word"); axs[2].set_title("(c) top TSL51/tune confusions: margin")
    axs[2].legend()
    save(fig, "diag_overview.png")


def fig_extractor():
    if not (REP / "pose_gate_v10.json").exists():
        return
    rep = read_json(REP / "pose_gate_v10.json")
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    if "proxies" in rep:
        b = rep["proxies"]["paired_rtmw_minus_mp"]
        keys = [("dom_seen", "hand seen"), ("bone_cv", "bone-length CV"), ("jitter", "jitter"), ("wrist_gap", "wrist gap"), ("swap", "L/R swap")]
        srcs = [s for s in ("th_sl", "ttrs", "tsl51") if s in b]
        for k, s in enumerate(srcs):
            x = np.arange(len(keys)) + (k - 1) * 0.26
            v = [b[s][kk]["diff"] for kk, _ in keys]
            lo = [b[s][kk]["diff"] - b[s][kk]["lo"] for kk, _ in keys]; hi = [b[s][kk]["hi"] - b[s][kk]["diff"] for kk, _ in keys]
            axs[0].bar(x, np.clip(v, -1, 2), 0.26, yerr=[lo, hi], label=SRC.get(s, s), capsize=2)
        axs[0].set_xticks(range(len(keys))); axs[0].set_xticklabels([n for _, n in keys]); axs[0].axhline(0, color="k", lw=0.6)
        axs[0].set_title("(a) RTMW − MediaPipe, same frames (95% CI)\nbone-length CV clipped at 2"); axs[0].legend()
    if "visual_judgement" in rep:
        v = rep["visual_judgement"]["counts"]
        lab = ["mp", "both", "rtmw", "neither", "unclear"]
        axs[1].bar(range(5), [v[k] for k in lab], color=[C9, CG, C10, CR, "#cbd5e1"])
        axs[1].set_xticks(range(5)); axs[1].set_xticklabels(["MediaPipe\nonly", "both\nright", "RTMW\nonly", "neither", "unclear"])
        axs[1].set_title(f"(b) blinded visual check, {rep['visual_judgement']['n']} disagreement frames")
    if "latency" in rep:
        L = rep["latency"]["per_frame"]
        names = [k for k in L if "detevery" not in k or k.startswith("rtmw_x_gpu_fp32")]
        axs[2].barh(range(len(names)), [L[k]["ms_p50"] for k in names], xerr=[[0] * len(names), [L[k]["ms_p95"] - L[k]["ms_p50"] for k in names]],
                    color=[C9 if k.startswith("mediapipe") else C10 for k in names])
        axs[2].set_yticks(range(len(names))); axs[2].set_yticklabels([k.replace("_", " ") for k in names]); axs[2].invert_yaxis()
        axs[2].axvline(40, color=CR, lw=0.8, ls="--"); axs[2].text(41, len(names) - 0.5, "25 fps", color=CR, fontsize=7)
        axs[2].set_xlabel("ms / frame (p50, bar to p95)"); axs[2].set_title(f"(c) per-frame cost — {rep['latency']['device']}")
    save(fig, "extractor_gate.png")
    if (REP / "xview_v10.json").exists():
        X = read_json(REP / "xview_v10.json")
        runs = [r for r in ("enc-x-mp", "enc-x-rtmw", "enc-x-mix", "enc-v8") if r in X]
        cells = ["q=mp|bank=mp", "q=rtmw|bank=rtmw", "q=mp|bank=rtmw", "q=rtmw|bank=mp"]
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
        for ax, ds in zip(axs, ("th_sl_val", "ttrs_val", "tsl51_test")):
            for k, r in enumerate(runs):
                v = [X[r].get(f"{ds}|{c}", {}).get("R1") or 0 for c in cells]
                ax.bar(np.arange(len(cells)) + (k - 1.5) * 0.2, v, 0.2, label=r)
            ax.set_xticks(range(len(cells))); ax.set_xticklabels(["MP→MP", "RTMW→RTMW", "MP→RTMW bank", "RTMW→MP bank"])
            ax.set_title(f"R@1 — {ds}"); ax.set_ylim(0, None)
        axs[0].legend(fontsize=7)
        save(fig, "extractor_recognition.png")


def fig_gate():
    """The recognition candidate v10x (extractor-averaged bank + re-tuned decision layer) vs v9 on the test sets — the release gate
    that kept it out of v10 (shipped v10 recognises exactly like v9)."""
    f = REP / "gate_v10x.json"
    if not f.exists():
        return
    g = read_json(f)["rows"]
    sets = [s for s in ("t51_test_x1.0", "t51_test_x1.5", "t51_test_x2.0", "iso_test", "flu_test") if s in g]
    metrics = [("WER", "WER ↓"), ("macro_F1", "macro-F1 ↑"), ("R1", "R@1 ↑"), ("acc_cover", "coverage ↑"), ("false_accept_rate", "false accept ↓"), ("ECE", "ECE ↓")]
    fig, axs = plt.subplots(1, len(metrics), figsize=(18, 3.4))
    for ax, (m, t) in zip(axs, metrics):
        o = [g[s][m]["old"] for s in sets if m in g[s]]; n = [g[s][m]["new"] for s in sets if m in g[s]]
        x = np.arange(len(o))
        ax.bar(x - 0.2, o, 0.4, color=C9, label="v9"); ax.bar(x + 0.2, n, 0.4, color=C10, label="v10x (opt-in)")
        ax.set_xticks(x); ax.set_xticklabels([s.replace("_test", "").replace("t51_x", "TSL51 ") for s in sets], rotation=35, ha="right"); ax.set_title(t)
    axs[0].legend()
    save(fig, "gate_v10x.png")


def fig_stream():
    f = REP / "stream_v10.json"
    if not f.exists():
        return
    s = read_json(f)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, n in zip(axs, ("t51_tune_x1.0", "flu_tune")):
        for r in s["grid"]:
            c = r["cfg"]
            ok = r["cfg"] == s["chosen"]
            ax.scatter(r[n]["lat_p50"], r[n]["WER"], s=30 + 25 * r[n]["flicker"], color=C10 if ok else C9, edgecolor="k" if ok else "none")
            ax.annotate(f"{c['step_s']}/{c['lookahead_s']}/{c['stable_n']}", (r[n]["lat_p50"], r[n]["WER"]), fontsize=6)
        ax.axhline(s["batch_tune"][n], color=CR, lw=0.8, ls="--", label="batch WER")
        ax.set_xlabel("word latency p50 (s)"); ax.set_ylabel("stream WER"); ax.set_title(f"{n}: step/look-ahead/stability (size = flicker)")
        ax.legend()
    save(fig, "stream_tradeoff.png")


def fig_face():
    f = REP / "face_model_v2.json"
    if not f.exists():
        return
    v = read_json(f)["variants"]
    names = list(v)
    keys = [("clips_acc", "RAVDESS acc ↑"), ("neutral_vs_emotion", "neutral vs emotion ↑"), ("false_alarm_cv", "false emotion on neutral signing ↓"),
            ("valence_agree_cv", "emotion-word valence ↑")]
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, (k, t) in zip(axs, keys):
        vals = [v[n]["tune"][k] for n in names]
        ax.barh(range(len(names)), vals, color=[C9 if n == "v9 recipe" else C10 for n in names])
        ax.axvline(v["v9 recipe"]["tune"][k], color=CR, lw=0.8, ls="--")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names if ax is axs[0] else []); ax.invert_yaxis(); ax.set_title(t)
        lo = min(vals); ax.set_xlim(max(0, lo - 0.1), max(vals) + 0.05)
    save(fig, "face_v2_variants.png")


def fig_speech():
    f = REP / "speech_v2.json"
    if not f.exists():
        return
    s = read_json(f)
    if "disfluency" not in s:
        return
    D = pd.DataFrame(s["disfluency"]["summary"])
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (k, t) in zip(axs, (("content_cer", "content CER (ASR) ↓"), ("guessed", "guessed words heard ↓"), ("dur_s", "duration (s)"))):
        for i, (vname, g) in enumerate(D.groupby("voice")):
            ax.bar(np.arange(len(g)) + (i - 0.5) * 0.4, g[k], 0.4, label=vname)
            ax.set_xticks(range(len(g))); ax.set_xticklabels(g["mode"])
        ax.set_title(t)
    axs[0].legend()
    save(fig, "speech_disfluency.png")


if __name__ == "__main__":
    for fn in (fig_diagnosis, fig_extractor, fig_gate, fig_stream, fig_face, fig_speech):
        try:
            fn()
        except Exception as e:                  # noqa: BLE001 — one figure must not stop the others
            print(f"{fn.__name__} skipped: {e}")
