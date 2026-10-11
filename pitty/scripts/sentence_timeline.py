"""pitty: where does a data_test sentence go wrong? (diagnosis only — nothing here is tuned on data_test)

Per video: tagger P(sign) / P(begin) per frame, hands-up frames, the pipeline's segments (word / transition), and for every
reference word the window where the bank scores it best (sliding windows over the span head) with its rank there.
→ cache/analysis/timeline_<video>.png + cache/analysis/sentence_timeline.json
    python pitty/scripts/sentence_timeline.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import logging  # noqa: E402
logging.disable(logging.WARNING)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from modules.face import hands_up  # noqa: E402
from modules.lexicon import DATA_TEST_REFS  # noqa: E402
from modules.mediapipe_pose import load_raw  # noqa: E402
from modules.pipeline import SignPipeline  # noqa: E402

for f in ("Tahoma", "Leelawadee UI", "TH Sarabun New"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break

OUT = ROOT / "cache/analysis"; OUT.mkdir(parents=True, exist_ok=True)
pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False)
import os, onnxruntime as ort                     # the timelines describe the v7 tagger (models/ may now hold a retrained one)
_TG = Path(os.environ.get("TAGGER_ONNX", ROOT / "cache/backup_models_v7/tagger.onnx"))
if _TG.exists():
    pipe.be.tagger = ort.InferenceSession(str(_TG), providers=["CPUExecutionProvider"])
cidx = {c: i for i, c in enumerate(pipe.concepts)}
cap = {}
_tag = pipe._tag


def tag_and_keep(kpi, sc, present):
    prob, H = _tag(kpi, sc, present)
    cap.setdefault("prob", prob); cap.setdefault("H", H)
    return prob, H


pipe._tag = tag_and_keep
report = {}
for stem, ref in DATA_TEST_REFS.items():
    cap.clear()
    raw = load_raw(ROOT / "cache/data_test_mp" / f"{stem}.npz")
    r = pipe.run_landmarks(raw, out_dir=None, title=stem)
    prob, H = cap["prob"], cap["H"]
    T = len(prob); t = np.arange(T) / 25.0
    up0 = hands_up(raw)                       # raw frames (source fps) → resample to the 25 fps timeline
    up = up0[np.clip(np.round(np.arange(T) * len(up0) / max(T, 1)).astype(int), 0, len(up0) - 1)] if len(up0) else np.zeros(T, bool)
    # sliding windows over the frame states → bank scores
    wins = [(a, a + L) for L in (14, 20, 28, 36) for a in range(0, max(1, T - L), 3)]
    E = pipe.span(H, wins); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    S = pipe.bank.scores(E)
    best = {}
    for w in ref:
        c = cidx.get(w)
        if c is None or S[0, c] < -1.5:
            best[w] = dict(in_vocab=False); continue
        i = int(np.argmax(S[:, c]))
        rank = int((S[i] > S[i, c]).sum()) + 1
        best[w] = dict(in_vocab=True, t0=round(wins[i][0] / 25, 2), t1=round(wins[i][1] / 25, 2), score=round(float(S[i, c]), 3), rank_there=rank,
                       top1_there=pipe.concepts[int(np.argmax(S[i]))])
    segs = [dict(t0=s["t0"], t1=s["t1"], status=s["status"], word=s.get("nearest"), p=round(s.get("p_correct", 0), 2)) for s in r["segments"]]
    first_up = float(t[up.argmax()]) if up.any() else None
    last_up = float(t[len(up) - 1 - up[::-1].argmax()]) if up.any() else None
    report[stem] = dict(ref=ref, duration=round(T / 25, 2), hands_up=[first_up, last_up], segments=segs, best_window=best)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.fill_between(t, 0, up * 0.15, color="#bbb", step="mid", label="hands up = มือยก")
    ax.plot(t, prob[:, 1] + prob[:, 2], color="#1f77b4", lw=1.4, label="P(sign) = กำลังทำท่า")
    ax.plot(t, prob[:, 2], color="#ff7f0e", lw=1, label="P(inside) = อยู่ในท่าเดิม")
    ax.plot(t, prob[:, 1], color="#2ca02c", lw=1, ls=":", label="P(begin) = เริ่มคำใหม่")
    for s in segs:
        col = {"accepted": "#2ca02c", "uncertain": "#e6b800", "unknown": "#d62728", "transition": "#999"}.get(s["status"], "#999")
        ax.axvspan(s["t0"], s["t1"], ymin=0.82, ymax=0.97, color=col, alpha=0.55)
        if s["status"] != "transition":
            ax.text((s["t0"] + s["t1"]) / 2, 1.13, s["word"], ha="center", fontsize=9)
    for k, (w, b) in enumerate(best.items()):
        if b.get("in_vocab"):
            y = 0.25 + 0.07 * (k % 6)
            ax.plot([b["t0"], b["t1"]], [y, y], color="#9467bd", lw=3)
            ax.text(b["t1"] + 0.05, y, f"{w} (#{b['rank_there']})", fontsize=8, va="center", color="#9467bd")
    ax.set_ylim(0, 1.25); ax.set_xlim(0, T / 25); ax.set_xlabel("seconds")
    ax.set_title(f"{stem}   ref: {' '.join(ref)}   |   top row = pipeline segments (green accepted / yellow uncertain / red unknown / grey transition)   purple = best window per ref word (#rank)", fontsize=9)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, 0.78))
    fig.tight_layout(); fig.savefig(OUT / f"timeline_{stem}.png", dpi=110); plt.close(fig)
    print(f"\n### {stem}  ref={ref}  dur {T / 25:.1f}s  hands up {first_up}–{last_up}s")
    for s in segs:
        print(f"   seg {s['t0']:5.2f}-{s['t1']:5.2f}  {s['status']:10s} {s['word']}")
    for w, b in best.items():
        print(f"   best window for {w}: {b}")
(OUT / "sentence_timeline.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
