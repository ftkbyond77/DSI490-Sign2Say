"""v11 report figures (prompt-opt-v11 C6 figure list) from the saved JSON reports → result_reporting/figures_v11/*.png

  python main.py v11-report figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

REP = ROOT / "artifacts" / "reports"
OUT = ROOT / "result_reporting" / "figures_v11"
INK, MUTED, GRID = "#1f2328", "#59636e", "#d1d9e0"
C_OLD, C_NEW, C_ALT = "#8c959f", "#0969da", "#bf8700"


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Tahoma", "Leelawadee UI", "TH Sarabun New"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f; break
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    return plt


def load(name):
    f = REP / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def bars(ax, labels, series, colors, fmt="{:.3f}"):
    x = np.arange(len(labels)); w = 0.8 / len(series)
    for i, ((name, vals), c) in enumerate(zip(series.items(), colors)):
        b = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=name, color=c)
        for r, v in zip(b, vals):
            if v is not None and np.isfinite(v):
                ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(frameon=False, fontsize=8)


def cmd_figures(a):
    plt = _setup()
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    # 1. WER decomposition per BLIND set (subs / dels / ins), v10 vs v11
    b10, b11 = load("diag_v11_blind_v10.json"), load("diag_v11_blind_v11.json")
    if b10 and b11:
        sets = [k for k in b10["sets"] if "WER" in b10["sets"][k]]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        for ax, part, title in zip(axes, ("subs", "dels", "ins"), ("แทนที่ (substitution)", "ตกหล่น (deletion)", "แทรก (insertion)")):
            bars(ax, [s.replace("_blind", "") for s in sets], {"v10": [b10["sets"][s][part] for s in sets], "v11": [b11["sets"][s][part] for s in sets]},
                 [C_OLD, C_NEW], "{:.2f}")
            ax.set_title(title, fontsize=9); ax.tick_params(axis="x", labelsize=7)
        fig.suptitle("BLIND-v11: องค์ประกอบของ WER ต่อชุด (ต่อคำอ้างอิง)"); fig.tight_layout(); fig.savefig(OUT / "wer_decomposition_blind.png", dpi=150); plt.close(fig)
        made.append("wer_decomposition_blind.png")
        # 3. per-signer gold-span R@1 on BLIND
        ps10, ps11 = b10["per_signer"]["signers"], b11["per_signer"]["signers"]
        keys = sorted(set(ps10) & set(ps11))
        if keys:
            fig, ax = plt.subplots(figsize=(10, 3.2))
            bars(ax, [k.split("|")[1][:14] for k in keys], {"v10": [ps10[k]["gold_R1"] for k in keys], "v11": [ps11[k]["gold_R1"] for k in keys]}, [C_OLD, C_NEW], "{:.2f}")
            ax.set_ylabel("R@1 บนช่วงคำที่ถูกต้อง"); ax.set_title("BLIND-v11: R@1 ต่อผู้ใช้ภาษามือที่ไม่เคยเห็น (≥ 8 คำ)"); ax.tick_params(axis="x", labelsize=7, rotation=20)
            fig.tight_layout(); fig.savefig(OUT / "per_signer_blind.png", dpi=150); plt.close(fig)
            made.append("per_signer_blind.png")
    # 2. error vs effective signers (TUNE, gold spans) from the encoder run table and the bank depth: from diag per-word records is not saved,
    #    so use the encoder protocol table (TUNE) of v10 lmenc vs v11-final
    enc = load("v11_encoder_tune.json")
    if enc:
        runs = [r for r in ("enc-v8", "v11-R0", "v11-R1", "v11-R2", "v11-R3", "v11-R4", "v11-R5", "v11-R6", "v11-R8", "v11-final") if r in enc]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        axes[0].bar(range(len(runs)), [enc[r]["tune_MRR"] for r in runs], color=[C_OLD if r == "enc-v8" else C_ALT if r == "v11-final" else C_NEW for r in runs])
        axes[0].set_xticks(range(len(runs))); axes[0].set_xticklabels([r.replace("v11-", "") .replace("enc-v8", "v10") for r in runs], fontsize=7)
        axes[0].set_ylim(0.3, 0.5); axes[0].set_title("TUNE MRR (ชุดผู้ใช้ที่ไม่เคยเห็น 4 ชุด)")
        axes[1].bar(range(len(runs)), [enc[r]["confusion_margin"]["mean"] for r in runs], color=[C_OLD if r == "enc-v8" else C_ALT if r == "v11-final" else C_NEW for r in runs])
        axes[1].set_xticks(range(len(runs))); axes[1].set_xticklabels([r.replace("v11-", "").replace("enc-v8", "v10") for r in runs], fontsize=7)
        axes[1].set_title("ระยะห่างในกลุ่มคำที่สับสน (margin)")
        fig.tight_layout(); fig.savefig(OUT / "encoder_runs.png", dpi=150); plt.close(fig)
        made.append("encoder_runs.png")
    # 4. reader ablation (TUNE): lm_w 0 / 0.15 / 0.3 / 0.45 on S0
    rows = []
    for tag, lab in (("v10", "v10"), ("v11-A2-lmw0.0", "ไม่มี LM"), ("v11-A2-lmw0.15", "v11 0.15"), ("v11-A2-lamnone", "v11 0.30"), ("v11-A2-lmw0.45", "v11 0.45"),
                     ("v11-A2-lam0.08", "v11 0.30 + bank v11")):
        d = load(f"diag_v11_{tag}.json")
        if d:
            rows.append((lab, [d["sets"][s]["WER"] for s in ("t51_tune_x1.0", "t51_tune_x2.0", "iso_tune", "flu_tune")]))
    if rows:
        fig, ax = plt.subplots(figsize=(10, 3.3))
        bars(ax, ["TSL51 1×", "TSL51 2×", "iso", "fluent"], {l: v for l, v in rows}, [C_OLD, "#d0d7de", "#80ccff", C_NEW, "#0550ae", C_ALT], "{:.3f}")
        ax.set_ylim(0.2, 0.55); ax.set_ylabel("WER (TUNE)"); ax.set_title("Reader ablation: น้ำหนักของ landmark encoder ใน fusion")
        fig.tight_layout(); fig.savefig(OUT / "reader_ablation.png", dpi=150); plt.close(fig)
        made.append("reader_ablation.png")
    # 5. segmentation S0–S3 (TUNE WER)
    rows = []
    for tag, lab in (("v10", "v10"), ("v11-S0", "S0 (Path A)"), ("v11-S1", "S1"), ("v11-S2", "S2"), ("v11-S3", "S3")):
        d = load(f"diag_v11_{tag}.json")
        if d:
            rows.append((lab, [d["sets"][s]["WER"] for s in ("t51_tune_x1.0", "t51_tune_x2.0", "iso_tune", "flu_tune")]))
    seg = load("v11_segtune_v11-S1.json")
    if seg:
        best = min(seg["rows"], key=lambda r: r["obj"])
        rows = [(l, v) if l != "S1" else ("S1 (tuned)", [best["t51_tune_x1.0"], best["t51_tune_x2.0"], best["iso_tune"], best["flu_tune"]]) for l, v in rows]
    if rows:
        fig, ax = plt.subplots(figsize=(10, 3.3))
        bars(ax, ["TSL51 1×", "TSL51 2×", "iso", "fluent"], {l: v for l, v in rows}, [C_OLD, C_NEW, C_ALT, "#cf222e", "#8250df"], "{:.2f}")
        ax.set_ylabel("WER (TUNE)"); ax.set_title("Segmentation S0–S3 → Path decision")
        fig.tight_layout(); fig.savefig(OUT / "segmentation_S0_S3.png", dpi=150); plt.close(fig)
        made.append("segmentation_S0_S3.png")
    # 6. risk–coverage (BLIND) per main group for v10 and v11, open-set marked
    if b10 and b11:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        for d, c, lab in ((b10, C_OLD, "v10"), (b11, C_NEW, "v11")):
            pts = {}
            for g, v in d["risk_coverage"].items():
                if v["n"] >= 30:
                    for p_ in v["curve"]:
                        pts.setdefault(round(p_["coverage"], 1), []).append(p_["risk"])
            xs = sorted(pts); ax.plot(xs, [np.mean(pts[x]) for x in xs], "-o", color=c, ms=3, label=lab)
            if d.get("openset"):
                ax.scatter([1.0], [d["openset"]["false_accept"]], marker="x", color=c, s=40)
        ax.set_xlabel("coverage (สัดส่วนคำที่ตอบ)"); ax.set_ylabel("risk (ความผิด)"); ax.set_title("Risk–coverage (BLIND, เฉลี่ยกลุ่ม n ≥ 30); × = open-set FA")
        ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "risk_coverage_blind.png", dpi=150); plt.close(fig)
        made.append("risk_coverage_blind.png")
    # 7. latency (dev-test report): model ms per second of video, v10 vs v11 + API p50 / p90
    rep = load("decision_v11.json"); sysc = load("v11_system_compare.json")
    if rep and sysc:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        L = rep["latency"]
        axes[0].bar(["v10", "v11"], [L["v10"]["model_ms_per_video_second"], L["v11"]["model_ms_per_video_second"]], color=[C_OLD, C_NEW])
        axes[0].set_title("ms ต่อวินาทีวิดีโอ (โมเดล)")
        lo, ln = sysc["latency"]["old"], sysc["latency"]["new"]
        bars(axes[1], ["p50", "p90"], {"v10": [lo["p50"], lo["p90"]], "v11": [ln["p50"], ln["p90"]]}, [C_OLD, C_NEW], "{:.0f}")
        axes[1].set_title("API /v1/translate (ms, 25 utt × 3)")
        fig.tight_layout(); fig.savefig(OUT / "latency.png", dpi=150); plt.close(fig)
        made.append("latency.png")
    print("figures:", made)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("figures")
    a = ap.parse_args()
    dict(figures=cmd_figures)[a.cmd](a)


if __name__ == "__main__":
    main()
