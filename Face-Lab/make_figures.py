"""Figures for Face-lab.ipynb and the report -> Face-Lab/figures/*.png.   python make_figures.py [arch|eval|all]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
C = dict(sign="#2563eb", face="#0d9488", fuse="#7c3aed", speech="#ea580c", data="#64748b", bad="#dc2626")


def box(ax, x, y, w, h, text, col, fs=9, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=col, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.25)


def arrow(ax, a, b, col="#334155", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=col, lw=1.4, linestyle=ls, shrinkA=0, shrinkB=0))


def architecture():
    """System view: two independent branches from the same camera frame, joined only AFTER the sign decision (late fusion)."""
    fig, ax = plt.subplots(figsize=(13, 6.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.6); ax.axis("off")
    box(ax, 0.2, 2.7, 1.7, 1.2, "Camera frame\n(browser)", C["data"])
    # sign branch (unchanged)
    ax.text(2.6, 6.2, "SIGN BRANCH  (unchanged, v10)", color=C["sign"], fontsize=10, weight="bold")
    box(ax, 2.6, 4.9, 1.9, 1.0, "MediaPipe Holistic\n133 landmarks\n+ 52 blendshapes", C["sign"], 8.5)
    box(ax, 4.9, 4.9, 1.9, 1.0, "Sign model v10\nspotting + words", C["sign"], 8.5)
    box(ax, 7.2, 4.9, 1.9, 1.0, "Sign result\nwords · sentence\n(never edited)", C["sign"], 8.5)
    # face branch
    ax.text(2.6, 2.05, "FACE BRANCH  (new, independent)", color=C["face"], fontsize=10, weight="bold")
    box(ax, 2.6, 0.7, 1.9, 1.2, "Face box from the\nsame 68 face points\n(read only) → crop\n48×48 gray", C["face"], 8)
    box(ax, 4.9, 0.7, 1.9, 1.2, "Quality gate\nsize · light · blur\n· hand over face", C["face"], 8.5)
    box(ax, 7.2, 0.7, 1.9, 1.2, "FER CNN (ONNX)\n7 classes +\ncalibration", C["face"], 8.5)
    box(ax, 9.5, 0.7, 1.7, 1.2, "unknown rule\nEMA +\nhysteresis", C["face"], 8.5)
    # fusion
    box(ax, 9.5, 2.9, 1.7, 1.1, "LATE FUSION\ntimestamps +\nconfidence", C["fuse"], 9)
    box(ax, 11.5, 2.9, 1.4, 1.1, "TTS v4\ntext + emotion\n+ intensity", C["speech"], 8.5)
    box(ax, 11.5, 4.9, 1.4, 1.0, "Screen\nwords · cues\n· expression", C["data"], 8.5)
    for a, b in [((1.9, 3.5), (2.6, 5.4)), ((1.9, 3.1), (2.6, 1.3)), ((4.5, 5.4), (4.9, 5.4)), ((6.8, 5.4), (7.2, 5.4)), ((4.5, 1.3), (4.9, 1.3)),
                 ((6.8, 1.3), (7.2, 1.3)), ((9.1, 1.3), (9.5, 1.3)), ((10.35, 1.9), (10.35, 2.9)), ((9.1, 5.1), (9.9, 4.0)), ((11.2, 3.45), (11.5, 3.45)),
                 ((9.1, 5.4), (11.5, 5.4))]:
        arrow(ax, a, b)
    arrow(ax, (10.9, 4.0), (11.9, 4.9))
    # forbidden path
    arrow(ax, (9.5, 1.55), (6.0, 4.9), C["bad"], "-|>", "--")
    ax.plot([7.6, 8.0], [3.0, 3.4], color=C["bad"], lw=3); ax.plot([7.6, 8.0], [3.4, 3.0], color=C["bad"], lw=3)
    ax.text(5.0, 3.45, "no path back: the face result never changes\nlandmarks, sign input or sign output", color=C["bad"], fontsize=8.5, ha="center")
    ax.text(0.2, 0.15, "Face crop = pixels of the frame, not landmarks. A face-branch failure, timeout or `unknown` leaves the sign result byte-identical.",
            fontsize=8.5, color="#334155")
    fig.tight_layout(); fig.savefig(FIG / "architecture.png", dpi=150); plt.close(fig)


def model_graph(arch="efficientnet_b0", res=96, params="4.0 M"):
    """The classifier itself: what happens to one 48×48 crop."""
    fig, ax = plt.subplots(figsize=(13, 2.8)); ax.set_xlim(0, 13); ax.set_ylim(0, 2.8); ax.axis("off")
    steps = [("crop 48×48\ngray, 0–255", C["data"]), (f"bilinear\n48 → {res}", C["face"]), ("÷255, gray→3ch\nImageNet mean/std", C["face"]),
             (f"{arch}\npretrained backbone\n(ImageNet-1K)", C["sign"]), ("global\naverage pool", C["sign"]), ("head: dropout\n+ Linear → 7", C["fuse"]),
             ("÷ temperature T\nsoftmax", C["fuse"]), ("probabilities\n+ margin", C["speech"])]
    w = 1.45
    for i, (t, c) in enumerate(steps):
        box(ax, 0.1 + i * 1.6, 0.9, w, 1.2, t, c, 8)
        if i:
            arrow(ax, (0.1 + i * 1.6 - 0.15, 1.5), (0.1 + i * 1.6, 1.5))
    ax.text(0.1, 0.35, f"all of this (except T / softmax) is inside the exported ONNX graph  ·  {params} parameters  ·  input name `face`, output `logits`",
            fontsize=9, color="#334155")
    fig.tight_layout(); fig.savefig(FIG / "model_graph.png", dpi=150); plt.close(fig)


def smoothing_diagram():
    """EMA + hysteresis on a toy sequence: why the label stops flickering."""
    import sys as _s
    _s.path.insert(0, str(ROOT.parent))
    from modules.face_fer import ExpressionSmoother
    rng = np.random.RandomState(3)
    T = np.arange(0, 12, 0.18)
    truth = np.where(T < 5, "neutral", np.where(T < 9, "happy", "neutral"))
    raw, sm, s = [], [], ExpressionSmoother()
    for t, tr in zip(T, truth):
        p = rng.dirichlet(np.ones(7) * 0.6) * 0.45
        p[CLASSES.index(tr)] += 0.55
        p /= p.sum()
        r = dict(t=float(t), valid=True, probs={c: float(p[i]) for i, c in enumerate(CLASSES)}, expression=CLASSES[int(p.argmax())])
        raw.append(r["expression"]); sm.append(s.update(r)["expression"])
    fig, ax = plt.subplots(figsize=(10, 2.8))
    idx = {c: i for i, c in enumerate(CLASSES)}
    ax.step(T, [idx[x] for x in raw], where="mid", color="#94a3b8", lw=1, label="raw argmax per frame")
    ax.step(T, [idx[x] for x in sm], where="mid", color=C["face"], lw=2.4, label="after EMA + hysteresis")
    ax.step(T, [idx[x] for x in truth], where="mid", color="#111827", lw=1, ls=":", label="what the face really did")
    ax.set_yticks(range(7)); ax.set_yticklabels(CLASSES); ax.set_xlabel("seconds"); ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    ax.set_title("")
    fig.tight_layout(); fig.savefig(FIG / "smoothing.png", dpi=150); plt.close(fig)
    ch = lambda x: sum(a != b for a, b in zip(x[:-1], x[1:]))      # noqa: E731
    return dict(raw_changes=ch(raw), smoothed_changes=ch(sm))


def evaluation():
    f = json.load(open(RES / "final_eval.json"))
    cm = np.array(f["test"]["confusion"], float)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, m, title in ((axs[0], cm, "counts"), (axs[1], cm / cm.sum(1, keepdims=True), "recall (row-normalised)")):
        ax.imshow(m, cmap="Blues"); ax.set_xticks(range(7)); ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticks(range(7)); ax.set_yticklabels(CLASSES)
        for i in range(7):
            for j in range(7):
                v = m[i, j]; ax.text(j, i, f"{v:.2f}" if m.max() <= 1 else f"{int(v)}", ha="center", va="center", fontsize=8, color="white" if v > m.max() * .55 else "black")
        ax.set_title(f"Test confusion — {title}"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(FIG / "confusion.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    rel = f["test_calibration"]["reliability"]
    axs[0].plot([0, 1], [0, 1], "k--", lw=1); axs[0].plot([r["conf"] for r in rel], [r["acc"] for r in rel], "o-", color=C["face"], label=f"after T (ECE {f['test_calibration']['ece']:.3f})")
    rr = f["test_calibration_raw"]["reliability"]; axs[0].plot([r["conf"] for r in rr], [r["acc"] for r in rr], "s-", color="#94a3b8", label=f"raw (ECE {f['test_calibration_raw']['ece']:.3f})")
    axs[0].set_xlabel("confidence"); axs[0].set_ylabel("accuracy"); axs[0].set_title("Reliability (test)"); axs[0].legend(fontsize=8)
    sel = f["test_selective"]
    axs[1].plot([s["coverage"] for s in sel], [s["accuracy"] for s in sel], "o-", color=C["face"], label="accuracy"); axs[1].plot([s["coverage"] for s in sel], [s["macro_f1"] for s in sel], "s-", color=C["fuse"], label="Macro-F1")
    axs[1].axvline(f["test_with_unknown"]["coverage"], color=C["bad"], ls=":", label=f"chosen rule (coverage {f['test_with_unknown']['coverage']:.2f})")
    axs[1].set_xlabel("coverage (share answered, rest = unknown)"); axs[1].set_title("Say `unknown` instead of guessing"); axs[1].legend(fontsize=8)
    pc = f["test"]["per_class"]; x = np.arange(7)
    axs[2].bar(x - 0.2, [pc[c]["recall"] for c in CLASSES], 0.4, color=C["face"], label="recall"); axs[2].bar(x + 0.2, [pc[c]["precision"] for c in CLASSES], 0.4, color=C["sign"], label="precision")
    axs[2].set_xticks(x); axs[2].set_xticklabels(CLASSES, rotation=45, ha="right"); axs[2].set_title("Per class (test)"); axs[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "calibration_selective.png", dpi=150); plt.close(fig)

    # model comparison (pilot + main runs on validation) + latency
    runs = [(t, json.load(open(p))) for t in ("pilot_mobilenet_v3_large", "pilot_efficientnet_b0", "main_mobilenet_v3_large", "main_efficientnet_b0", "main_resnet18") if (p := RES / f"{t}.json").exists()]
    if runs:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for t, r in runs:
            axs[0].plot([l["epoch"] for l in r["log"]], [l["val_macro_f1"] for l in r["log"]], label=f"{t.replace('_', ' ')}  (best {r['best_val_macro_f1']:.3f})",
                        ls="--" if t.startswith("pilot") else "-", color=C["sign"] if "mobilenet" in t else ("#ea580c" if "resnet" in t else C["face"]))
        axs[0].set_xlabel("epoch"); axs[0].set_ylabel("validation Macro-F1"); axs[0].legend(fontsize=7); axs[0].set_title("Model selection runs")
        lat = f["efficiency"]
        names = ["ONNX CPU 1 thr", "ONNX CPU 2 thr", "ONNX CPU 4 thr", "ONNX CUDA", "torch GPU fp16"]
        vals = [lat["onnx_cpu_1thread_b1"]["median_ms"], lat["onnx_cpu_2thread_b1"]["median_ms"], lat["onnx_cpu_4thread_b1"]["median_ms"],
                lat["onnx_cuda_b1"]["median_ms"] if isinstance(lat.get("onnx_cuda_b1"), dict) else 0, lat["torch_gpu_fp16_b1"]["median_ms"]]
        axs[1].barh(names, vals, color=C["face"]); axs[1].set_xlabel("ms per face (batch 1, median)"); axs[1].set_title("Inference latency")
        for i, v in enumerate(vals):
            axs[1].text(v, i, f" {v:.1f}", va="center", fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "selection_latency.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("arch", "all"):
        architecture(); smoothing_diagram(); model_graph()
    if what in ("eval", "all"):
        evaluation()
    print("ok")
