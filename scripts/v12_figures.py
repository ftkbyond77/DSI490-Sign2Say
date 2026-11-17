"""v12 report figures → result_reporting/figures_v12/v12_*.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "result_reporting" / "figures_v12"; FIG.mkdir(parents=True, exist_ok=True)
REP = ROOT / "artifacts" / "reports"
for f in ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/LeelawUI.ttf"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name(); break

# 1) primary reader re-fine-tune: selection score per evaluation vs the enc-v7 control
z = json.load(open(ROOT / "artifacts/runs/v12-P1/metrics_zero_shot.json"))
ks = ["E1_ttrs_val|open_vocab", "ONE_val|open_vocab", "THSL_val|open_vocab", "AGENT_val|open_vocab", "NEWSIG_dev|daily_vocab"]
ctrl = float(np.mean([z[k]["MRR"] for k in ks]))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for run, c in (("v12-P1", "#c0392b"), ("v12-P1b", "#2471a3")):
    h = json.load(open(ROOT / f"artifacts/runs/{run}/train_history.json"))
    ax[0].plot([x["step"] for x in h], [x["val_score"] for x in h], "o-", color=c, label=run)
    ax[1].plot([x["step"] for x in h], [x["AGENT_val|open_vocab"]["R1"] for x in h], "o-", color=c, label=run)
ax[0].axhline(ctrl, color="k", ls="--", label="enc-v7 (ขั้น 0)"); ax[0].set_title("คะแนนคัดเลือก (MRR เฉลี่ย 4 ชุด held-out + NEWSIG-dev)")
ax[1].axhline(z["AGENT_val|open_vocab"]["R1"], color="k", ls="--", label="enc-v7"); ax[1].set_title("R@1 YouTube val (คนที่ไม่เคยเห็น)")
for a in ax:
    a.set_xlabel("step"); a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "v12_primary_reader.png", dpi=120); plt.close()

# 2) live mode: whole clip vs v10 rule vs v12 rule (TUNE: v10 model; TEST: both models)
fig, ax = plt.subplots(figsize=(8, 3.6))
tune = json.load(open(REP / "v12_live_rules_w_t51_tune.json", encoding="utf-8"))["summary"] if (REP / "v12_live_rules_w_t51_tune.json").exists() else None
t10 = json.load(open(REP / "v12_live_rules_w_t51_test_v10.json", encoding="utf-8"))["summary"]
t12 = json.load(open(REP / "v12_live_rules_w_t51_test_v12.json", encoding="utf-8"))["summary"]
bars = []
if tune:
    bars += [("TUNE: คลิปเต็ม", tune["full"]["WER"]), ("TUNE: กติกา v10", tune["v10"]["WER"]), ("TUNE: กติกา v12", tune["LM2_LM1"]["WER"])]
bars += [("TEST: คลิปเต็ม (v10)", t10["full"]["WER"]), ("TEST: v10 + กติกา v10", t10["v10"]["WER"]), ("TEST: v12 + กติกา v12", t12["v12"]["WER"])]
cols = ["#7f8c8d", "#c0392b", "#27ae60"] * 2
ax.barh([b[0] for b in bars][::-1], [b[1] for b in bars][::-1], color=cols[:len(bars)][::-1])
for i, (_, v) in enumerate(bars[::-1]):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlabel("WER top-1 ของประโยค (ประโยค TSL51 แบบเว็บ)"); ax.set_title("โหมดสด: ตัดประโยคด้วยกติกาเดียวกับเบราว์เซอร์"); ax.set_xlim(0.3, 0.42)
plt.tight_layout(); plt.savefig(FIG / "v12_live_mode.png", dpi=120); plt.close()

# 3) ladder: pooled sentence WER delta vs v10 (TUNE, paired CI)
rows = []
for f, tag, label in (("v12_paired_tune_v12-B1_v12-B2.json", "v12-B1", "B1 คลังผู้ทำท่าใหม่"), ("v12_paired_tune_v12-B1_v12-B2.json", "v12-B2", "B1+B2"),
                      ("v12_paired_tune_v12-B1b_v12-B2b.json", "v12-B2b", "B1b+B2 (จำกัด 2/คน)"), ("v12_paired_tune_v12-J.json", "v12-J", "B2 อ่าน 5 crop (ทุกตัวอ่าน)"),
                      ("v12_paired_tune_v12-J3.json", "v12-J3", "B2 อ่าน 3 crop"), ("v12_paired_tune_v12-Jp.json", "v12-Jp", "B2 อ่าน 5 crop (ไม่รวม lmenc) = v12")):
    p = REP / f
    if p.exists():
        r = json.load(open(p, encoding="utf-8"))[tag]["E1_pooled_sentence_WER"]
        rows.append((label, r["diff"], r["ci_utt"]))
fig, ax = plt.subplots(figsize=(8, 3.6))
for i, (lab, d, ci) in enumerate(rows[::-1]):
    ax.errorbar(d, i, xerr=[[d - ci[0]], [ci[1] - d]], fmt="o", color="#2471a3" if ci[1] < 0 else "#7f8c8d", capsize=4)
ax.axvline(0, color="k", lw=1); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlabel("Δ WER ประโยครวม เทียบ v10 (TUNE, 618 ประโยค; ติดลบ = ดีขึ้น; CI 95 %)"); ax.set_title("บันได v12: ผลของแต่ละขั้น")
plt.tight_layout(); plt.savefig(FIG / "v12_ladder.png", dpi=120); plt.close()
print("figures ->", FIG)
