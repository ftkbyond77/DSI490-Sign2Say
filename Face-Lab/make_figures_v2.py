"""Figures for face-report-v2 (numbers from results/*.json).   python make_figures_v2.py → figures/v2_*.png"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / "results"; F = Path(__file__).resolve().parent / "figures"
J = lambda n: json.loads((R / n).read_text(encoding="utf-8"))
rh, pr, ck, en = J("rav_head_24actors.json"), J("rav_prior.json"), J("ravhead_check.json"), J("ensemble24.json")
card = json.loads((R.parents[1] / "models/face-model/model_card.json").read_text(encoding="utf-8"))["cv"]
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a8a7a0"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "Tahoma", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})

# 1. clips macro-F1, RAVDESS 24 actors
rows = [("ensemble เดิม (บนเว็บก่อนหน้านี้)", en["ensemble"]["clips_macro_f1"], GRAY),
        ("ensemble + ตัวจับขมวดคิ้ว", en["ensemble_brow"]["clips_macro_f1"], GRAY),
        ("blendshape ของเพื่อน (ตัวที่ใช้จริง)", round(card["clips_macro_f1"], 3), ORANGE),
        ("blendshape ของเพื่อน (RAVDESS อย่างเดียว)", rh["blendshape_lr (friend recipe)"]["clips_macro_f1"], ORANGE),
        ("หัวใหม่ + ภาษามือหน้าปกติ + prior 0.5 (ใช้จริง)", pr["neutral_prior_0.5"]["clips_macro_f1"], BLUE),
        ("หัวใหม่ + blendshape (fusion, ไม่ได้ใช้)", rh["fusion (picture head + blendshape)"]["clips_macro_f1"], GRAY)]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for i, (n, v, c) in enumerate(rows):
    ax.barh(i, v, color=c, height=0.55); ax.text(v + 0.008, i, f"{v:.3f}", va="center", color=INK, fontsize=9)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9); ax.invert_yaxis()
ax.set_xlim(0, 0.85); ax.set_xlabel("macro-F1 ระดับคลิป (RAVDESS 24 คน, leave-actors-out)"); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.axvline(1 / 7, color=INK2, lw=1, ls=":"); ax.text(1 / 7 + 0.005, len(rows) - 0.45, "สุ่ม 1/7", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(F / "v2_f1.png", dpi=200); plt.close(fig)

# 2. per-class recall
emo = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
th = ["ปกติ", "ยิ้ม/สุข", "เศร้า", "โกรธ", "กลัว", "รังเกียจ", "ประหลาดใจ"]
ser = [("หัวใหม่ (ใช้จริง)", pr["neutral_prior_0.5"]["per_class_recall"], BLUE),
       ("blendshape ของเพื่อน", rh["blendshape_lr (friend recipe)"]["per_class_recall"], ORANGE),
       ("ensemble เดิม + คิ้ว", en["ensemble_brow"]["per_class_recall"], AQUA)]
fig, ax = plt.subplots(figsize=(7.2, 3.0)); w = 0.26
for k, (n, r, c) in enumerate(ser):
    xs = [i + (k - 1) * (w + 0.02) for i in range(len(emo))]
    ax.bar(xs, [r[e] for e in emo], w, color=c, label=n)
    for x, e in zip(xs, emo):
        ax.text(x, r[e] + 0.015, f"{r[e]:.2f}".lstrip("0"), ha="center", fontsize=6.5, color=INK2)
ax.set_xticks(range(len(emo))); ax.set_xticklabels(th); ax.set_ylim(0, 1.08); ax.set_ylabel("recall ระดับคลิป")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
fig.tight_layout(); fig.savefig(F / "v2_recall.png", dpi=200); plt.close(fig)

# 3. trade-off: RAVDESS F1 vs false emotion on the team's neutral signing frames (app path)
pts = [("ensemble เดิม", ck["ensemble (deployed)"]["wrong"], en["ensemble"]["clips_macro_f1"], AQUA),
       ("หัวใหม่ prior 0", ck["ravhead neutral_prior 0.0"]["wrong"], pr["neutral_prior_0.0"]["clips_macro_f1"], BLUE),
       ("หัวใหม่ prior 0.5 (ใช้จริง)", ck["ravhead neutral_prior 0.5"]["wrong"], pr["neutral_prior_0.5"]["clips_macro_f1"], BLUE),
       ("หัวใหม่ prior 1.0", ck["ravhead neutral_prior 1.0"]["wrong"], pr["neutral_prior_1.0"]["clips_macro_f1"], BLUE)]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.plot([p[1] * 100 for p in pts[1:]], [p[2] for p in pts[1:]], color=BLUE, lw=2, zorder=1)
for n, x, y, c in pts:
    ax.scatter(x * 100, y, s=70, color=c, edgecolor=SURF, linewidth=2, zorder=2)
    ax.annotate(n, (x * 100, y), xytext=(8, -12 if "1.0" in n else 6), textcoords="offset points", fontsize=8.5, color=INK)
ax.set_xlabel("เดาเป็นอารมณ์ผิด บนเฟรมหน้าปกติของคลิปทีม (%)  ← ดีกว่า"); ax.set_ylabel("macro-F1 RAVDESS  ↑ ดีกว่า")
ax.set_xlim(0, 11); ax.set_ylim(0.4, 0.8); ax.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(F / "v2_tradeoff.png", dpi=200); plt.close(fig)
print("ok")
