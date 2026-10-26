"""pitty: labelled timeline of one TSL51 sentence (like pitty/scripts/my_eval.py timeline_png, for clips that have landmarks only).

One column per detected sign: the skeleton at the middle of the sign · time + P(correct) coloured by status (green confirmed,
orange unsure, red unknown) · the top-3 words as bars · the true word under it (✓ / ✗, matched by overlap with the hand-marked
word boundaries, IoU ≥ 0.3). True words no segment matched are listed as missing.

    from sentence_strip import strip
    fig = strip(result_row, speed, accept_thr, title)       # result_row = one entry of a sent_eval.py results file
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POSE = ROOT / "cloud_s3/data_prep/pose_mp/tsl51_user_sentence"
ALIGN = ROOT / "artifacts/runs/seq-v7/tsl51_alignment.json"
COL = {"accepted": "#16a34a", "uncertain": "#f59e0b", "unknown": "#dc2626"}
BODY = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (0, 5), (0, 6)]
HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
_A = None


def _iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def _skeleton(ax, kp, sc):
    ok = sc > 0.3
    for a, b in BODY:
        if ok[a] and ok[b]:
            ax.plot(kp[[a, b], 0], kp[[a, b], 1], color="#64748b", lw=2)
    for base, col in ((91, "#2563eb"), (112, "#db2777")):
        for a, b in HAND:
            if ok[base + a] and ok[base + b]:
                ax.plot(kp[[base + a, base + b], 0], kp[[base + a, base + b], 1], color=col, lw=1.2)
    if ok[0]:
        ax.add_patch(__import__("matplotlib").patches.Circle(kp[0], 0.045, fill=False, color="#64748b", lw=2))
    ax.set_xlim(0.15, 0.85); ax.set_ylim(1.0, 0.05); ax.set_aspect("equal"); ax.axis("off")


def strip(r, speed, accept_thr, title):
    """→ matplotlib figure for one sentence result."""
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Tahoma", "Leelawadee UI", "sans-serif"]
    global _A
    _A = _A or json.loads(ALIGN.read_text(encoding="utf-8"))
    cid = r["clip"]; z = np.load(POSE / f"{cid.split(':', 1)[1]}.npz")
    kp, sc = z["kp"].astype(np.float32), z["sc"].astype(np.float32)
    gt = [(g["concept"], g["f0"] / speed, g["f1"] / speed) for g in _A[cid]["segments"]]
    used = set()
    n = max(1, len(r["top1"]))
    fig = plt.figure(figsize=(2.1 * max(n, 3), 4.4))
    for i, (seg, p, cands, st) in enumerate(zip(r["seg_f"], r["p"], r["cands"], r["status"])):
        status = "accepted" if p >= accept_thr else ("unknown" if st == "unknown" else "uncertain")   # same calibration, threshold of this view
        col = COL[status]
        m = max(range(len(gt)), key=lambda j: _iou(seg, gt[j][1:]))
        truth = gt[m][0] if _iou(seg, gt[m][1:]) >= 0.3 else None
        if truth:
            used.add(m)
        mid = min(len(kp) - 1, int(round((seg[0] + seg[1]) / 2 * speed)))
        ax = fig.add_axes([i / max(n, 3) + 0.01, 0.45, 1 / max(n, 3) - 0.02, 0.42])
        _skeleton(ax, kp[mid], sc[mid])
        ax.set_title(f"{i + 1}. {seg[0] / 25:.1f}–{seg[1] / 25:.1f}s  p={p:.2f}", fontsize=9.5, color=col, fontweight="bold")
        bx = fig.add_axes([i / max(n, 3) + 0.01, 0.1, 1 / max(n, 3) - 0.02, 0.32]); bx.axis("off"); bx.set_xlim(0, 1); bx.set_ylim(0, 3)
        for j, (w, s_) in enumerate(cands[:3]):
            y = 2.55 - j
            bx.barh(y, max(float(s_), 0.02), height=0.75, color=col if j == 0 else "#cbd5e1")
            bx.text(0.03, y, f"{w}  {float(s_):.2f}", va="center", fontsize=9.5, color="white" if j == 0 else "#0f172a",
                    fontweight="bold" if j == 0 else "normal")
        ok = truth == r["top1"][i]
        lab = "ไม่มีท่าจริง (คำเกิน)" if truth is None else f"จริง: {truth} {'✓' if ok else '✗'}"
        fig.text(i / max(n, 3) + 0.5 / max(n, 3), 0.04, lab, ha="center", fontsize=10.5, fontweight="bold",
                 color="#16a34a" if ok else "#dc2626")
    miss = [gt[j][0] for j in range(len(gt)) if j not in used]
    sub = f"เฉลย: {' '.join(g[0] for g in gt)}   ·   ระบบอ่านได้: {' '.join(r['top1'])}" + (f"   ·   คำที่หาย: {' '.join(miss)}" if miss else "")
    fig.suptitle(f"{title}\n{sub}", fontsize=11, x=0.01, ha="left", y=1.02)
    return fig
