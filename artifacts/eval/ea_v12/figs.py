"""Figures for Error-Analysis-for-v12 / result-v12 (from the measured JSON files)."""
import json, sys, glob, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["Leelawadee UI", "Tahoma", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BL, OR, GY, GR = "#2b5cab", "#d9772b", "#8a8f98", "#2e8b57"


def boot(R, B):
    rng = np.random.RandomState(0); n = len(R)
    N = np.array([x["n_ref"] for x in R], float); e = np.array([x["e_top"] for x in R], float); eb = np.array([x["e_top"] for x in B], float)
    d = [e[i].sum() / N[i].sum() - eb[i].sum() / N[i].sum() for i in (rng.randint(0, n, n) for _ in range(2000))]
    return e.sum() / N.sum(), e.sum() / N.sum() - eb.sum() / N.sum(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def load(pat):
    f = sorted(glob.glob(str(S / pat)))[0]; r = json.load(open(f, encoding="utf-8")); n = min(len(v) for v in r.values())
    return {k: v[:n] for k, v in r.items()}


# 1. nuisance audit on 37 real clips
a = load("live_sim_real_full-u12-*.json"); b = load("live_sim_real_full-drop10-*.json")
lab = [("full", a, "คลิปเต็ม 25–30 fps\n(อ้างอิง)"), ("u12", a, "12–15 fps\nสม่ำเสมอ"), ("jit", a, "จังหวะเฟรมจริง\nของโหมดสด"), ("nostall", a, "จังหวะจริง\nตัดช่วงค้าง 200 ms"),
       ("drop10", b, "มือหาย\n+10 %"), ("drop20", b, "มือหาย\n+20 %"), ("drop35", b, "มือหาย\n+35 %"), ("zoom130", b, "ซูม 1.3×"), ("zoom160", b, "ซูม 1.6×"),
       ("trim1500", a, "ตัดแบบโหมดสด\n(จบ 1.5 วิ)"), ("live1500", a, "ตัด + จังหวะจริง")]
fig, ax = plt.subplots(figsize=(11, 3.8))
for i, (c, r, name) in enumerate(lab):
    w, d, lo, hi = boot(r[c], r["full"])
    ax.bar(i, w, color=GY if c == "full" else BL, width=0.65)
    ax.errorbar(i, w, yerr=[[d - lo], [hi - d]] if c != "full" else None, color="k", capsize=3, lw=1)
    ax.text(i, w + 0.035, f"{w:.3f}", ha="center", fontsize=9)
ax.set_xticks(range(len(lab))); ax.set_xticklabels([x[2] for x in lab], fontsize=8.5); ax.set_ylim(0, 0.75); ax.set_ylabel("WER (top-1)")
ax.set_title("ตรวจปัจจัยของโหมดสดทีละตัวบนคลิปจริงที่มีเฉลย 37 คลิป (v10, สายจริง) — ไม่มีตัวใดทำให้ WER เปลี่ยนเกินช่วงความเชื่อมั่น", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "ea12_nuisance_audit.png", dpi=170); plt.close(fig)

# 2. live cutting on sentences
t = load("live_sim_w_t51_tune_full-trim900-trim1500-stream.json"); t2 = load("live_sim_w_t51_tune_full-trim1500-trim1500_1000_800-*.json")
rows = [("full", t, "คลิปเต็ม\n(= กด start/stop)", GY), ("trim900", t, "โหมดสด\nก่อน 0.5 / หลัง 0.4\nจบ 0.9 วิ", OR), ("trim1500", t, "โหมดสด (ปัจจุบัน)\nก่อน 0.5 / หลัง 0.4\nจบ 1.5 วิ", OR),
        ("trim1500_1000_800", t2, "ก่อน 1.0 / หลัง 0.8", BL), ("trim1500_1500_1200", t2, "ก่อน 1.5 / หลัง 1.2\n(ข้อเสนอ)", GR)]
fig, ax = plt.subplots(figsize=(8.6, 3.9))
for i, (c, r, name, col) in enumerate(rows):
    w, d, lo, hi = boot(r[c], r["full"])
    ax.bar(i, w, color=col, width=0.6)
    if c != "full":
        ax.errorbar(i, w, yerr=[[d - lo], [hi - d]], color="k", capsize=3, lw=1); ax.text(i, w + 0.045, f"{w:.3f}\nΔ {d:+.3f}", ha="center", fontsize=9)
    else:
        ax.text(i, w + 0.02, f"{w:.3f}", ha="center", fontsize=9)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([x[2] for x in rows], fontsize=8.5); ax.set_ylim(0, 0.52); ax.set_ylabel("WER (top-1)")
ax.set_title("ประโยค TSL51 ที่อัตราเฟรมแบบเว็บ (n 126): การตัดประโยคแบบโหมดสดทำให้ WER เพิ่ม — เก็บเฟรมก่อน/หลังให้ยาวขึ้นแล้วหาย", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "ea12_live_cut.png", dpi=170); plt.close(fig)

# 3. frame intervals + saved request profile
dts = np.concatenate(json.load(open(S / "live_dts.json")))
rows_ = [x for x in json.load(open(S / "live_rows.json")) if "cat" in x]
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
axs[0].hist(np.clip(dts, 0, 320), bins=64, color=BL); axs[0].set_xlabel("ระยะห่างระหว่างเฟรม (ms)"); axs[0].set_ylabel("จำนวนเฟรม")
axs[0].set_title("จังหวะเฟรมจริงของโหมดสด\n(มัธยฐาน 64 ms + ค้าง ~200 ms ทุกเฟรมที่ 8)", fontsize=9.5)
L = [x for x in rows_ if x["cat"] == "live"]
axs[1].hist([x["dur"] for x in L], bins=np.arange(0, 22, 1), color=OR); axs[1].set_xlabel("ความยาว utterance (วินาที)")
axs[1].set_title(f"โหมดสด {len(L)} รายการ: {np.mean([x['dur'] > 19 for x in L]) * 100:.0f} % ชนเพดาน 20 วิ", fontsize=9.5)
for cat, col, nm in (("live", OR, "โหมดสด"), ("upload", GY, "อัปโหลดวิดีโอ"), ("cam-manual", GR, "กล้อง + กด start/stop")):
    P = np.array([p for x in rows_ if x["cat"] == cat for p in x["p"]])
    if len(P):
        axs[2].hist(P, bins=20, range=(0, 1), histtype="step", lw=1.8, color=col, density=True, label=f"{nm} (n {len(P)})")
axs[2].axvline(0.88, color="k", ls=":", lw=1); axs[2].set_xlabel("P(correct) ของแต่ละคำ"); axs[2].legend(fontsize=8, frameon=False)
axs[2].set_title("ความมั่นใจของคำในคำขอที่บันทึกไว้\n(เส้นประ = เกณฑ์ยอมรับ 0.88; ไม่มีเฉลย)", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "ea12_live_requests.png", dpi=170); plt.close(fig)

# 4. new-data coverage
import csv
g = list(csv.DictReader(open(S / "vocab_gain.csv", encoding="utf-8-sig")))
bins = [("≤1", lambda s: s <= 1), ("2", lambda s: s == 2), ("3–4", lambda s: 3 <= s <= 4), ("5–9", lambda s: 5 <= s <= 9), ("≥10", lambda s: s >= 10)]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
bot = np.zeros(len(bins))
for gain, col, nm in ((0, GY, "ไม่มีในข้อมูลใหม่"), (1, BL, "ได้เพิ่ม 1 แหล่ง"), (2, GR, "ได้เพิ่ม 2 แหล่ง")):
    v = np.array([sum(1 for r in g if f(float(r["signers"])) and int(r["gain"]) == gain) for _, f in bins])
    axs[0].bar(range(len(bins)), v, bottom=bot, color=col, label=nm, width=0.6)
    for i, x in enumerate(v):
        if x:
            axs[0].text(i, bot[i] + x / 2, str(x), ha="center", va="center", color="w", fontsize=9)
    bot += v
axs[0].set_xticks(range(len(bins))); axs[0].set_xticklabels([b[0] for b in bins]); axs[0].set_xlabel("จำนวนผู้ทำท่าของคำนั้นในคลังเดิม"); axs[0].set_ylabel("จำนวนคำ (จาก 328 คำสนทนา)")
axs[0].legend(frameon=False, fontsize=8.5); axs[0].set_title("ข้อมูลใหม่เติมผู้ทำท่าให้คำที่ยังบาง", fontsize=10)
kinds = ["คำ (ราชบัณฑิต)", "ประโยค (ราชบัณฑิต)", "ลักษณนาม (ราชบัณฑิต)", "คำ (e-TSL)", "วลี (e-TSL)", "ประโยค (e-TSL)", "ตัวสะกด (e-TSL)"]
vals = [200, 725, 324, 1802, 241, 23, 130]
axs[1].barh(range(len(kinds))[::-1], vals, color=[OR, OR, OR, BL, BL, BL, BL]); axs[1].set_yticks(range(len(kinds))[::-1]); axs[1].set_yticklabels(kinds, fontsize=9)
for i, v in zip(range(len(kinds))[::-1], vals):
    axs[1].text(v + 20, i, str(v), va="center", fontsize=9)
axs[1].set_xlabel("จำนวนคลิป"); axs[1].set_title("ข้อมูลใหม่ 3,445 คลิป แยกตามชนิด", fontsize=10); axs[1].set_xlim(0, 2100)
fig.tight_layout(); fig.savefig(OUT / "ea12_new_data.png", dpi=170); plt.close(fig)

# 5. OOD (if available)
fs = {m: S / f"ood_res_model-{m}.json" for m in ("v10", "v11")}
if all(f.exists() for f in fs.values()):
    res = {m: json.load(open(f, encoding="utf-8")) for m, f in fs.items()}
    groups = [("ราชบัณฑิต\nคำในคลัง", lambda r: r["src"] == "rid" and r["kind"] == "in"), ("e-TSL\nคำในคลัง", lambda r: r["src"] == "etsl" and r["kind"] == "in")]
    mets = [("top-1 ถูก", lambda r: r["word"] in r["top"]), ("ถูกใน 5 อันดับ", lambda r: r["word"] in r["top"] or any(r["word"] in c for c in r["cands"])),
            ("ถูกและแสดง", lambda r: any(w == r["word"] and s == "accepted" for w, s in zip(r["top"], r["status"])))]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for k, (gn, gf) in enumerate(groups):
        ax = axs[k]
        for j, (m, col) in enumerate((("v10", GY), ("v11", BL))):
            R = [r for r in res[m] if gf(r)]
            v = [np.mean([mf(r) for r in R]) for _, mf in mets]
            ax.bar(np.arange(3) + (j - 0.5) * 0.36, v, width=0.34, color=col, label=m)
            for i, x in enumerate(v):
                ax.text(i + (j - 0.5) * 0.36, x + 0.02, f"{x:.2f}", ha="center", fontsize=8.5)
        ax.set_xticks(range(3)); ax.set_xticklabels([x[0] for x in mets], fontsize=9); ax.set_ylim(0, 1); ax.set_title(f"{gn.replace(chr(10), ' ')} (n {len(R)})", fontsize=10); ax.legend(frameon=False, fontsize=8.5)
    ax = axs[2]
    for j, (m, col) in enumerate((("v10", GY), ("v11", BL))):
        R = [r for r in res[m] if r["kind"] == "oov"]
        v = [np.mean([any(s == "accepted" for s in r["status"]) for r in R])]
        ax.bar([j], v, color=col, width=0.5); ax.text(j, v[0] + 0.01, f"{v[0]:.3f}", ha="center", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["v10", "v11"]); ax.set_ylim(0, max(0.3, ax.get_ylim()[1])); ax.set_title(f"e-TSL คำนอกคลัง (n {len(R)}): คลิปที่ขึ้นเป็นคำ", fontsize=10)
    fig.suptitle("v10 / v11 กับผู้ทำท่าใหม่ที่ไม่เคยเห็น (ข้อมูลใหม่, ยังไม่ได้เทรน)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "ea12_new_signers.png", dpi=170); plt.close(fig)
print("figures", sorted(p.name for p in OUT.glob("ea12_*.png")))
