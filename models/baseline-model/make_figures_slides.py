"""Figures of the baseline slides -> models/baseline-model/figures/ (PNG 300 dpi + SVG).
  python models/baseline-model/make_figures_slides.py
needs results/baseline_slide_metrics.json (collect_metrics.py), results/slide_selection.json, results/v10_slide.json (run_v10_slide.py)

fig_tsl51_baseline / fig_jack_baseline   3 sentences each: frame · reference · baseline top-1   (slide 3)
fig_tsl51_vs_v10   / fig_jack_vs_v10     the same sentences, baseline vs v10 (slide 5)
fig_metrics                              baseline metrics on overall / TSL51 / Jack
fig_face_graph                           the 18 face points + graph edges on a real frame (slide 2)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
INK, MUTE, GRID = "#1f2937", "#6b7280", "#e5e7eb"
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#8E6BBF", grey="#9ca3af")
STAT = dict(ok=CB["green"], near=CB["orange"], miss=CB["red"])
have = {f.name for f in font_manager.fontManager.ttflist}
THAI = next((n for n in ("Leelawadee UI", "Tahoma", "Noto Sans Thai") if n in have), "DejaVu Sans")
plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK, "axes.linewidth": .8})
R = HERE / "results"
M = json.load(open(R / "baseline_slide_metrics.json", encoding="utf8"))
SEL = json.load(open(R / "selection.json" if (R / "selection.json").exists() else R / "slide_selection.json", encoding="utf8"))
V10 = json.load(open(R / "v10_slide.json", encoding="utf8"))


def save(fig, name, transparent=False):
    fc = "none" if transparent else "white"
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor=fc, transparent=transparent); fig.savefig(OUT / f"{name}.svg", facecolor=fc, transparent=transparent)
    plt.close(fig); print("wrote", name)


def pairs(ref, hyp):
    """Levenshtein alignment -> [(ref_idx | None, hyp_idx | None)]."""
    n, m = len(ref), len(hyp); d = np.zeros((n + 1, m + 1), int); d[:, 0] = range(n + 1); d[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, out = n, m, []
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    return out[::-1]


def wer(ref, hyp):
    return sum(1 for r, h in pairs(ref, hyp) if r is None or h is None or ref[r] != hyp[h]) / len(ref)


def frame(video, t, aspect, w=260):
    """Cover-crop (aspect = width / height) of the frame at t seconds."""
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read(); cap.release()
    if not ok:
        return np.full((int(w / aspect), w, 3), 235, np.uint8)
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB); h, ww = f.shape[:2]
    if ww / h > aspect:
        cw = int(h * aspect); x0 = (ww - cw) // 2; f = f[:, x0:x0 + cw]
    else:
        ch = int(ww / aspect); y0 = int((h - ch) * .35); f = f[y0:y0 + ch]
    return cv2.resize(f, (w, int(w / aspect)))


def base_segs(v):
    return [dict(t0=s["t0"], t1=s["t1"], word=s["top"][0][0], top5=[w for w, _ in s["top"]]) for s in v["segments"]]


def items_for(group):
    """[(title, video, ref, gold, {system: segs})] of the selected clips (group: 'tsl51' | 'jack')."""
    out = []
    for k in SEL[group]:
        v = M["per_video"][group][k]
        out.append(dict(key=k, video=v["video"], ref=v["ref"], gold=v.get("gold"), sys={"baseline": base_segs(v), "v10": V10[k]}))
    return out


def draw(name, group, systems, W, H, labels, fs=7.6, fh=.5, chip=.25):
    """One figure: per sentence a frame row, the reference row and one chip row per system, columns = reference words (+ extra words)."""
    items = items_for(group); nsys = len(systems)
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    L = .66; rh = .26 + .06 + fh + .05 + chip + .05 + nsys * (chip + .04) + .08
    ncol = max(len(it["ref"]) + sum(1 for s in systems for r, h in pairs(it["ref"], [x["word"] for x in it["sys"][s]]) if r is None) for it in items)
    cw = (W - L - .05) / max(ncol, 4)
    for r, it in enumerate(items):
        ref = it["ref"]; y0 = H - .04 - r * rh; yf = y0 - .28 - fh; yr = yf - .05 - chip; ys = [yr - .05 - (j + 1) * chip - j * .04 for j in range(nsys)]
        al = {s: pairs(ref, [x["word"] for x in it["sys"][s]]) for s in systems}
        cols = []                                                           # ("ref", i) | ("extra", system, hyp_idx)
        for i in range(-1, len(ref)):
            if i >= 0:
                cols.append(("ref", i))
            for s in systems:
                last = -1
                for ri, hi in al[s]:
                    if ri is not None:
                        last = ri
                    elif last == i:
                        cols.append(("extra", s, hi))
        times = []
        for c in cols:
            t = None
            if c[0] == "ref":
                if it["gold"]:
                    t = (it["gold"][c[1]][0] + it["gold"][c[1]][1]) / 2
                else:
                    for s in systems:
                        hi = next((h for ri, h in al[s] if ri == c[1] and h is not None), None)
                        if hi is not None:
                            sg = it["sys"][s][hi]; t = (sg["t0"] + sg["t1"]) / 2; break
            else:
                sg = it["sys"][c[1]][c[2]]; t = (sg["t0"] + sg["t1"]) / 2
            times.append(t)
        known = [i for i, t in enumerate(times) if t is not None]
        for i, t in enumerate(times):                                       # a missed word with no frame of its own: between its neighbours
            if t is None:
                lo = max([k for k in known if k < i], default=None); hi = min([k for k in known if k > i], default=None)
                times[i] = (times[lo] + times[hi]) / 2 if lo is not None and hi is not None else times[lo] + .8 if lo is not None else max(times[hi] - .8, .1)
        wers = {s: wer(ref, [x["word"] for x in it["sys"][s]]) for s in systems}
        ax.text(0, y0, "(" + "abc"[r] + ") " + " ".join(ref), fontsize=8.3, fontweight="bold", va="top", family=THAI)
        ax.text(W - .02, y0 - .02, "   ".join(f"{labels[s]} WER {wers[s]:.2f}" for s in systems), fontsize=7, ha="right", va="top", color=MUTE)
        for yy, lab in [(yf + fh / 2, "frame"), (yr + chip / 2, "reference")] + [(ys[j] + chip / 2, labels[s]) for j, s in enumerate(systems)]:
            ax.text(L - .07, yy, lab, ha="right", va="center", fontsize=6.6, color=MUTE)
        for c, (col, t) in enumerate(zip(cols, times)):
            x = L + c * cw + .03; w = cw - .06
            ax.imshow(frame(it["video"], t, w / fh), extent=(x, x + w, yf, yf + fh), aspect="auto", zorder=2)
            if col[0] == "ref":
                ax.add_patch(FancyBboxPatch((x, yr), w, chip, boxstyle="round,pad=0,rounding_size=.04", fc="#f3f4f6", ec=CB["grey"], lw=.7))
                ax.text(x + w / 2, yr + chip / 2, ref[col[1]], ha="center", va="center", fontsize=fs, family=THAI)
            for j, s in enumerate(systems):
                segs = it["sys"][s]
                if col[0] == "ref":
                    hi = next((h for ri, h in al[s] if ri == col[1]), None)
                    if hi is None:
                        ax.text(x + w / 2, ys[j] + chip / 2, "missed", ha="center", va="center", fontsize=6.8, color=CB["red"], style="italic"); continue
                    word = segs[hi]["word"]; st = "ok" if word == ref[col[1]] else "near" if ref[col[1]] in segs[hi]["top5"] else "miss"
                elif col[1] == s:
                    word, st = segs[col[2]]["word"], "miss"
                else:
                    continue
                ax.add_patch(FancyBboxPatch((x, ys[j]), w, chip, boxstyle="round,pad=0,rounding_size=.04", fc=STAT[st], ec="none"))
                ax.text(x + w / 2, ys[j] + chip / 2, word, ha="center", va="center", color="white", fontsize=fs, fontweight="bold", family=THAI)
    save(fig, name)


def fig_metrics():
    A = M["agg"]; sets = [("overall", "Overall"), ("tsl51", "TSL51"), ("jack", "Jack")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0)); x = np.arange(3); w = .26
    for j, (k, lab, c) in enumerate((("r1", "R@1 ↑", CB["blue"]), ("r5", "R@5 ↑", CB["green"]), ("wer", "WER ↓", CB["red"]))):
        bs = ax.bar(x + (j - 1) * w, [A[s][k] for s, _ in sets], w, color=c, label=lab)
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .015, f"{b.get_height():.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n({A[s]['n_words']} words)" for s, n in sets]); ax.set_ylim(0, 1.0); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8); fig.tight_layout(); save(fig, "fig_metrics")


def fig_face_graph():
    """What the face branch really reads: the 18 TEMPLATE points (jaw 9 · inner lip 8 · nose hub) that the data prep places on a frame with a
    per-frame similarity fitted to 5 anchors of MediaPipe Pose (nose, both eyes, both mouth corners) — the 478-point face mesh is not used.
    Frame = the most frontal one of the Jack clips (hands away from the face), shown in grayscale."""
    from evalsets import team
    from modules.mediapipe_pose import load_raw, to_standard
    from modules.parts import FACE_IDX
    from modules.schema import EAR_L, EAR_R, EYE_L, EYE_R, LH, MOUTH_L, MOUTH_R, NOSE, RH
    best = None
    for k, (p, _, video) in team("Jack").items():
        raw = load_raw(p); d = to_standard(raw); kp, sc = np.asarray(d["kp"]), np.asarray(d["sc"]); H, W = d["hw"]
        ri = np.abs(raw["t"][None, :] - (np.arange(len(kp)) / 25)[:, None]).argmin(1)              # raw frame of each 25 fps frame
        for t in range(3, len(kp) - 3):
            pose = raw["pose"][ri[t]]; A = pose[[0, 2, 5, 9, 10]]                                  # nose, eyes, mouth corners (MediaPipe Pose)
            if A[:, 3].min() < .5:
                continue
            iso = kp[t] * [W / H, 1]; hands = [i for i in list(LH) + list(RH) if sc[t, i] > 0]
            if hands and np.linalg.norm(iso[hands] - iso[NOSE], axis=1).min() < .3:
                continue
            eye = abs(A[2, 0] - A[1, 0]) * W / H
            score = (abs(A[0, 0] - (A[1, 0] + A[2, 0]) / 2) + abs(A[1, 1] - A[2, 1]) + abs(A[3, 0] + A[4, 0] - 2 * A[0, 0])) * (W / H) / eye
            if best is None or score < best[0]:
                best = (score, k, t, video, kp[t].copy(), A[:, :2].copy())
    _, k, t, video, kp, an = best
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_MSEC, t / 25 * 1000); ok, f = cap.read(); cap.release()
    fh, fw = f.shape[:2]; px = kp * [fw, fh]; an = an * [fw, fh]                                   # coordinates are stored as x/W, y/H
    q = px[FACE_IDX]
    c = q.mean(0) - [0, .1 * max(np.ptp(q[:, 0]), np.ptp(q[:, 1]))]; r = max(np.ptp(q[:, 0]), np.ptp(q[:, 1])) * 1.08
    x0, x1, y0, y1 = int(c[0] - r), int(c[0] + r), int(c[1] - r), int(c[1] + r)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[max(y0, 0):y1, max(x0, 0):x1]
    fig, ax = plt.subplots(figsize=(2.6, 2.6)); ax.imshow(g, cmap="gray", extent=(max(x0, 0), max(x0, 0) + g.shape[1], max(y0, 0) + g.shape[0], max(y0, 0)), vmin=20, vmax=235, zorder=0)
    edges = [(i, i + 1) for i in range(8)] + [(i, i + 1) for i in range(9, 16)] + [(16, 9)]
    for i in range(17):
        ax.plot(q[[17, i], 0], q[[17, i], 1], color="white", lw=.9, alpha=.8, zorder=2)
    for a_, b_ in edges:
        ax.plot(q[[a_, b_], 0], q[[a_, b_], 1], color="black", lw=4.2, zorder=3, solid_capstyle="round"); ax.plot(q[[a_, b_], 0], q[[a_, b_], 1], color="white", lw=2.0, zorder=3, solid_capstyle="round")
    ax.scatter(q[:9, 0], q[:9, 1], s=85, c=CB["blue"], zorder=5, edgecolor="white", lw=1.4)
    ax.scatter(q[9:17, 0], q[9:17, 1], s=60, c=CB["red"], zorder=5, edgecolor="white", lw=1.2)
    ax.scatter(q[17:, 0], q[17:, 1], s=170, c=CB["green"], zorder=6, edgecolor="white", lw=1.8)
    ax.scatter(an[:, 0], an[:, 1], s=120, facecolors="none", edgecolors=CB["orange"], lw=2.2, marker="D", zorder=6)
    ax.set_xlim(max(x0, 0), max(x0, 0) + g.shape[1]); ax.set_ylim(max(y0, 0) + g.shape[0], max(y0, 0)); ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1); save(fig, "fig_face_graph"); print("frontal frame:", k, "t =", t / 25, "s")


if __name__ == "__main__":
    fig_face_graph()
    for g, nm in (("tsl51", "fig_tsl51"), ("jack", "fig_jack")):
        draw(f"{nm}_baseline", g, ["baseline"], 6.15, 4.65, dict(baseline="baseline"))
        draw(f"{nm}_vs_v10", g, ["baseline", "v10"], 6.15, 5.15, dict(baseline="baseline", v10="v10"), fh=.4, chip=.24)
    fig_metrics()
