"""Architecture figures of the baseline → models/baseline-model/figures/ (PNG 300 dpi + SVG).

  python models/baseline-model/make_figures.py

fig1  Uni-Sign pose branch end to end (what the original checkpoint is) — numbers are counted from the real modules
fig2  one body-part branch in detail (the ST-GCN block structure, tensor shapes)
fig3  the baseline as used in this project (offline bank + online recognition), zero-shot vs adapted, and what v8–v10 add
fig4  the 4 keypoint parts on a real MediaPipe frame of data_test
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
C = dict(inp="#dbeafe", gcn="#fde68a", fuse="#fed7aa", mt5="#bbf7d0", bank="#e9d5ff", off="#e5e7eb", head="#fecaca", ink="#111827", mute="#6b7280")
PART = dict(body="#2563eb", left="#16a34a", right="#dc2626", face_all="#9333ea")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, text, fc, ec=None, fs=8.5, dashed=False, bold=False, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec or C["ink"], lw=1.1,
                                ls="--" if dashed else "-", alpha=.55 if dashed else 1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or (C["mute"] if dashed else C["ink"]),
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, dashed=False, color=None):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=1.2, color=color or C["ink"], ls="--" if dashed else "-", shrinkA=0, shrinkB=0))


def canvas(w, h, title, sub=None):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.text(.15, h - .25, title, fontsize=13, fontweight="bold", va="top")
    if sub:
        ax.text(.15, h - .62, sub, fontsize=8.5, color=C["mute"], va="top")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", name)


def counts():
    from modules.unisign import UniSignPose
    import torch
    p = UniSignPose(); n = lambda m: sum(x.numel() for x in m.parameters())
    sd = torch.load(HERE / "unisign_wlasl_enc.pt", map_location="cpu") if (HERE / "unisign_wlasl_enc.pt").exists() else {}
    mt5 = sum(v.numel() for k, v in sd.items() if k.startswith("mt5")) or 84_954_242
    return dict(pose=n(p), mt5=mt5, stgcn=sum(n(p.gcn_modules[m]) + n(p.fusion_gcn_modules[m]) for m in ("body", "left", "face_all")), proj=n(p.pose_proj))


# ----------------------------------------------------------------------------- fig1
def fig1():
    n = counts()
    fig, ax = canvas(15, 7.2, "Uni-Sign pose branch — the original encoder used as the baseline",
                     "Li et al., 'Uni-Sign: Toward Unified Sign Language Understanding at Scale', ICLR 2025 · checkpoint wlasl_pose_only_islr.pth · "
                     "parameter counts measured on the loaded modules\nPre-training: CSL-News (1,985 h), fine-tuned on WLASL-2000 (isolated ASL signs). The checkpoint has no Thai output and no classifier head.")
    box(ax, .3, 2.2, 1.9, 2.0, "Video\n↓\nwhole-body\nkeypoints\n133 × (x, y, score)\n/ frame\n(RTMW in the paper,\nMediaPipe here)", C["inp"], bold=False)
    arrow(ax, (2.2, 3.2), (2.75, 3.2))
    ys = dict(body=4.45, left=3.35, right=2.25, face_all=1.15)
    lab = dict(body="body · 9 pts", left="left hand · 21 pts", right="right hand · 21 pts", face_all="face · 18 pts")
    for m, y in ys.items():
        box(ax, 2.75, y, 1.75, .85, f"{lab[m]}\n[T, V, 3]", C["inp"], ec=PART[m])
        arrow(ax, (4.5, y + .42), (5.0, y + .42))
        box(ax, 5.0, y, 1.55, .85, "Linear 3→64", C["gcn"])
        arrow(ax, (6.55, y + .42), (7.0, y + .42))
        box(ax, 7.0, y, 2.2, .85, "Spatial ST-GCN\n64 → 128 → 256", C["gcn"])
        arrow(ax, (9.2, y + .42), (9.6, y + .42))
        box(ax, 9.6, y, 1.85, .85, "Temporal ST-GCN\n3 × 256 · k=5", C["fuse"])
        arrow(ax, (11.45, y + .42), (11.9, 3.3))
    ax.text(8.1, 5.65, "one branch per part  (left / right hand share weights)", ha="center", color=C["mute"], fontsize=8.5)
    ax.text(8.1, 6.0, "hand / face branches also receive the body branch's wrist / nose feature (detached) between the two ST-GCN stages", ha="center", color=C["mute"], fontsize=8)
    box(ax, 11.9, 2.55, 1.35, 1.5, "mean over joints\n→ concat 4×256\n+ part_para\n→ Linear\n1024 → 768", C["head"], fs=7.5)
    arrow(ax, (13.25, 3.3), (13.6, 3.3))
    box(ax, 13.6, 2.2, 1.2, 2.2, "per-frame\ntokens\n[T, 768]", C["inp"])
    # second row: mT5
    box(ax, 5.0, -.2, 5.0, .9, f"mT5-base ENCODER · 12 layers · 768-d · 12 heads · gated-GELU\n{n['mt5'] / 1e6:.1f} M params  (prefix: task text embeddings, prepended)", C["mt5"], fs=8)
    box(ax, 10.4, -.2, 3.1, .9, "mT5 DECODER + LM head\n(text generation: ASL gloss / CSL sentence)\nnot used by the baseline", C["off"], dashed=True, fs=7.5)
    ax.plot([14.2, 14.2, 7.5], [2.2, 1.0, 1.0], color=C["ink"], lw=1.2); arrow(ax, (7.5, 1.0), (7.5, .7))
    ax.text(11.0, .78, "frame tokens → encoder", fontsize=8, color=C["mute"])
    arrow(ax, (10.0, .25), (10.4, .25), dashed=True)
    box(ax, .3, -.2, 4.3, .9, f"Pose branch (ST-GCN ×3 parts + proj) = {n['pose'] / 1e6:.2f} M params\nEncoder total used here ≈ {(n['pose'] + n['mt5']) / 1e6:.1f} M params", C["bank"], fs=8)
    ax.set_xlim(0, 15); ax.set_ylim(-.5, 7.2)
    save(fig, "fig1_unisign_pose_architecture")


# ----------------------------------------------------------------------------- fig2
def fig2():
    fig, ax = canvas(14.4, 5.4, "One part branch in detail (example: right hand, 21 joints)",
                     "Graph = hand skeleton, 'distance' partitioning with max_hop = 1 → K = 3 adjacency matrices; adaptive graph weights; BatchNorm + ReLU + residual in every block")
    steps = [("input", "[B, T, 21, 3]\n(x, y, score)", C["inp"]), ("Linear 3→64", "[B, 64, T, 21]", C["gcn"]),
             ("GCN block", "64 → 64\nspatial only (1×K)", C["gcn"]), ("GCN block", "64 → 128", C["gcn"]), ("GCN block", "128 → 256", C["gcn"]),
             ("+ body feature", "body wrist (hands)"+"\nor nose (face)\nfeature, detached", C["off"]), ("ST-GCN block ×3", "256 → 256\nGCN K=3 +\ntemporal k=5", C["fuse"]),
             ("mean over joints", "[B, T, 256]", C["head"])]
    x = .3
    for i, (a, b, c) in enumerate(steps):
        box(ax, x, 2.0, 1.55, 1.7, f"{a}\n\n{b}", c, fs=7.6, dashed=(i == 5))
        if i < len(steps) - 1:
            arrow(ax, (x + 1.55, 2.85), (x + 1.72, 2.85))
        x += 1.72
    ax.annotate("", xy=(.3, 1.6), xytext=(8.5, 1.6), arrowprops=dict(arrowstyle="<->", color=C["mute"]))
    ax.text(4.4, 1.25, "temporal length T never changes (25 fps in, 25 fps out) — no pooling over time inside the GCN", ha="center", color=C["mute"], fontsize=8)
    ax.text(.3, .55, "Ablation-relevant facts: spatial blocks have no temporal kernel; all temporal modelling is in the 3 fusion blocks (k = 5; 3 stacked blocks → 13 frames ≈ 0.5 s)\n"
                     "and in the mT5 encoder (full self-attention over all frames). 4 parts → 4 × 256 = 1024 channels → Linear → 768.", fontsize=8.2, color=C["ink"], va="top")
    save(fig, "fig2_part_branch_stgcn")


# ----------------------------------------------------------------------------- fig3
def fig3():
    fig, ax = canvas(15, 8.2, "Baseline as used in this project — and what v8–v10 add on top",
                     "Same MediaPipe landmarks, same Thai clip bank, same scoring as v10 — only the recognition stack differs")
    # offline bank
    ax.text(.3, 7.0, "OFFLINE — build the bank (build.py)", fontweight="bold", fontsize=10)
    box(ax, .3, 5.6, 2.3, 1.0, "~19k isolated Thai clips\n9.5k concepts\ntrain / val / test signers", C["inp"], fs=8)
    arrow(ax, (2.6, 6.1), (3.1, 6.1))
    box(ax, 3.1, 5.6, 2.6, 1.0, "Uni-Sign encoder\n(frozen)  ·  mean-pool\n→ Z0  [N, 768]", C["mt5"], fs=8)
    arrow(ax, (5.7, 6.1), (6.2, 6.1))
    box(ax, 6.2, 5.6, 2.6, 1.0, "bank_zeroshot = Z0", C["bank"], fs=8.5, bold=True)
    ax.plot([4.4,4.4],[5.6,4.85],color=C["ink"],lw=1.2); arrow(ax, (4.4, 4.85), (6.2, 4.85))
    box(ax, 6.2, 4.35, 2.6, 1.0, "light adapter (trained)\nresidual MLP 768-1024-768\nCosFace · train signers only", C["head"], fs=8)
    arrow(ax, (8.8, 4.85), (9.3, 4.85))
    box(ax, 9.3, 4.35, 2.6, 1.0, "bank_adapted = Z1", C["bank"], fs=8.5, bold=True)
    ax.text(12.2, 5.3, "adapter picked on VAL signers;\nTEST signers + data_test\nnever used for any choice", fontsize=8, color=C["mute"], va="center")
    # online
    ax.text(.3, 3.85, "ONLINE — recognise a video", fontweight="bold", fontsize=10)
    row = [("video", C["inp"]), ("MediaPipe\nHolistic\n(shared)", C["inp"]), ("Standard\nSchema 25 fps\n(shared)", C["inp"]),
           ("hands-raised runs\n(data-prep rule)\ncut every\n≈ median sign length", C["fuse"]), ("Uni-Sign\nencoder\nper segment", C["mt5"]),
           ("cosine vs bank\ntop-2 mean per\nconcept", C["bank"]), ("top-1 word\n+ top-5 list\nper segment", C["inp"])]
    x = .3
    for i, (t, c) in enumerate(row):
        box(ax, x, 2.35, 1.85, 1.25, t, c, fs=7.8)
        if i < len(row) - 1:
            arrow(ax, (x + 1.85, 2.97), (x + 2.07, 2.97))
        x += 2.07
    ax.text(.3, 2.05, "zero-shot: encoder + Z0   ·   adapted: encoder + adapter + Z1   ·   vocabulary: full (9.5k concepts) or the daily-conversation list (≈330)",
            fontsize=8.3, color=C["mute"])
    # v10 additions
    ax.text(.3, 1.45, "NOT in the baseline (added by v8–v10, see models/model-v10/)", fontweight="bold", fontsize=10, color=C["mute"])
    adds = ["BiGRU frame tagger\n(sign / non-sign)", "span head +\nsemi-Markov spotting", "split-sign merge\nmodel", "fine-tuned encoder\n(CosFace+SupCon)", "Thai landmark\nencoder (2nd opinion)",
            "hand-shape cue", "fingerspelling", "calibrated P(correct)\nAccept / ? / Reject", "grammar re-rank +\nsentence layer"]
    for i, t in enumerate(adds):
        box(ax, .3 + i * 1.62, .15, 1.5, 1.0, t, C["off"], dashed=True, fs=6.6)
    save(fig, "fig3_baseline_pipeline")


# ----------------------------------------------------------------------------- fig4
def fig4():
    from modules.mediapipe_pose import load_raw, to_standard
    from modules.parts import BODY_IDX, FACE_IDX
    z = load_raw(ROOT / "cache" / "data_test_mp" / "ฉันรักเพื่อน.npz")
    d = to_standard(z); kp, sc = np.asarray(d["kp"]), np.asarray(d["sc"])
    H, W = d["hw"]
    from modules.schema import activity
    t = int(np.argmax(activity(kp, sc, d["hw"])["height"] * (sc[:, 91:133].mean(1) > .9)))                            # a frame where the hands are clearly seen
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw=dict(width_ratios=[1, 1]))
    parts = dict(body=BODY_IDX, left=list(range(91, 112)), right=list(range(112, 133)), face_all=FACE_IDX)
    names = dict(body="body (9)", left="left hand (21)", right="right hand (21)", face_all="face (18)")
    ax = axs[0]
    for m, ix in parts.items():
        p = kp[t, ix] * np.array([W / H, 1]); ok = sc[t, ix] > .3
        ax.scatter(p[ok, 0], p[ok, 1], s=22, c=PART[m], label=names[m], zorder=3)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=8); ax.set_title(f"4 input parts on a real frame (data_test, t = {t / 25:.1f} s)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax2 = axs[1]; ax2.axis("off")
    ax2.text(0, 1, "What the encoder reads per frame", fontweight="bold", va="top", fontsize=11)
    rows = [("body", "9 joints: nose, ears, shoulders, elbows, wrists", "[T, 9, 3]"), ("left hand", "21 joints, relative to the wrist", "[T, 21, 3]"),
            ("right hand", "21 joints, relative to the wrist", "[T, 21, 3]"), ("face", "18 points: 9 jaw line, 8 mouth, 1 nose", "[T, 18, 3]")]
    for i, (a, b, c) in enumerate(rows):
        col = list(PART.values())[i]
        ax2.add_patch(plt.Rectangle((0, .78 - i * .17), .035, .12, color=col, transform=ax2.transAxes))
        ax2.text(.06, .86 - i * .17, a, fontweight="bold", va="center", fontsize=9.5); ax2.text(.06, .8 - i * .17, f"{b}   {c}", va="center", fontsize=8.2, color=C["mute"])
    ax2.text(0, .06, "(x, y, score); body box → [-1, 1]; hands / face relative to their root\n÷ body scale — identical to Uni-Sign's load_part_kp. All other\nof the 133 COCO-WholeBody slots are ignored (64 of 133 never read).",
             fontsize=8.2, va="bottom")
    save(fig, "fig4_input_keypoint_parts")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        f()
