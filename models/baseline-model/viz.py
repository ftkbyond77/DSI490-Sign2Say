"""Frame-by-frame plot of baseline predictions on a few videos (used by baseline-model.ipynb)."""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

FPS = 25
COL = dict(top1="#16a34a", top5="#f59e0b", wrong="#dc2626")


def _thai_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Leelawadee UI", "Tahoma", "Noto Sans Thai", "Angsana New"):
        if n in have:
            return n
    return "DejaVu Sans"


def _frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else np.zeros((90, 160, 3), np.uint8)


def plot_rows(items, results, n_frames=12, width=16):
    """items {name: (landmarks, ref, video)}; results {name: (segments, score_dict)} → one row per video: real frames along the time axis,
    below them each segment as a bar (green = top-1 is a reference word, amber = only in top-5, red = not in top-5) + the predicted word."""
    plt.rcParams["font.family"] = _thai_font()
    caps = {n: cv2.VideoCapture(str(v[2])) for n, v in items.items()}
    ar = {n: (lambda im: im.shape[0] / im.shape[1])(_frame(c, 0.5)) for n, c in caps.items()}
    fh = {n: (width * .93 / n_frames) * ar[n] for n in items}                       # frame height (inches) that keeps the aspect ratio
    fig = plt.figure(figsize=(width, sum(fh.values()) + 1.9 * len(items) + .6))
    gs = fig.add_gridspec(3 * len(items), 1, height_ratios=[x for n in items for x in (fh[n], 1.9, .6)], hspace=.05)
    for r, (name, (_, ref, video)) in enumerate(items.items()):
        segs, sc = results[name]; cap = caps[name]
        dur = max(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1), max(s["t1"] for s in segs))
        axf, ax = fig.add_subplot(gs[3 * r]), fig.add_subplot(gs[3 * r + 1])
        step = dur / n_frames
        for k in range(n_frames):
            axf.imshow(_frame(cap, (k + .5) * step), extent=(k * step, (k + 1) * step, 0, 1), aspect="auto")
        axf.set_xlim(0, dur); axf.set_ylim(0, 1); axf.axis("off")
        axf.set_title(f"{name.split('/')[-1]}   |   reference: {' '.join(sc['ref'])}   |   R@1 {sc['r1']:.2f} · R@5 {sc['r5']:.2f} · WER {sc['wer']:.2f}",
                      fontsize=10, loc="left")
        refc = set(sc["ref"])
        for k, s in enumerate(segs):
            words = [w for w, _ in s["top"]]; w1 = words[0]
            c = COL["top1"] if w1 in refc else COL["top5"] if refc & set(words) else COL["wrong"]
            lvl = .55 - .5 * (k % 2)
            ax.add_patch(plt.Rectangle((s["t0"], lvl), max(s["t1"] - s["t0"], .08), .4, color=c, alpha=.9))
            ax.text((s["t0"] + s["t1"]) / 2, lvl + .2, w1, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            for x in (s["t0"], s["t1"]):
                axf.axvline(x, color=c, lw=.9, ls=":")
        ax.set_xlim(0, dur); ax.set_ylim(0, 1.05); ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COL["top1"], label="top-1 is a reference word"), Patch(color=COL["top5"], label="reference word only in top-5"),
                        Patch(color=COL["wrong"], label="not in top-5")], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .0))
    return fig
