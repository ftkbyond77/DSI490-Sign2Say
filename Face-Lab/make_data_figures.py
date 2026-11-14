"""What the data looks like, for face-report-v2: example face crops (the 96x96 gray picture the models see) from RAVDESS, the dictionary
signing clips (neutral) and the team's signer clips, the 68 points + brow height on one actor, and the data counts.
  python make_data_figures.py → figures/v2_data_*.png, results/data_summary.json   (thaislm env: cv2 + mediapipe)"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = Path(__file__).resolve().parent; ROOT = H.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(H))
from modules.face_fer import crop48, crop_box   # noqa: E402
import real_check                              # noqa: E402
from compare24 import EMO, spans               # noqa: E402
from sign_neutral import SKIP                  # noqa: E402

F = H / "figures"; DATA = H / "cache" / "ravdess24"
TH = {"neutral": "ปกติ", "happy": "สุข", "sad": "เศร้า", "angry": "โกรธ", "fear": "กลัว", "disgust": "รังเกียจ", "surprise": "ประหลาดใจ"}
ORDER = [1, 3, 4, 5, 6, 7, 8]                  # RAVDESS codes: neutral, happy, sad, angry, fear, disgust, surprise (calm = 2 → neutral)
INK2, SURF = "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "Tahoma", "font.size": 9, "figure.facecolor": SURF, "text.color": "#0b0b0b"})


def grid(imgs, titles, path, ncol, size=1.05, row_labels=None):
    nrow = int(np.ceil(len(imgs) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * size + (0.6 if row_labels else 0), nrow * (size + 0.22)))
    axs = np.atleast_2d(axs)
    for i, ax in enumerate(axs.flat):
        ax.axis("off")
        if i < len(imgs):
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255); ax.set_title(titles[i], fontsize=8, color=INK2, pad=2)
    if row_labels:
        for r, lab in enumerate(row_labels):
            axs[r, 0].text(-12, 48, lab, ha="right", va="center", fontsize=8.5, color=INK2)
    fig.tight_layout(pad=0.3); fig.savefig(path, dpi=200); plt.close(fig)


def mid_crop(z, ci):
    sel = np.where((z["crop_clip"] == ci) & (z["crop_q"] >= 0.25))[0]
    return sel[len(sel) * 3 // 5] if len(sel) else None


def ravdess():
    imgs, titles, rows = [], [], []
    for a in (1, 2):                            # actor 1 (male), actor 2 (female) — RAVDESS: odd = male, even = female
        z = np.load(DATA / f"actor_{a:02d}.npz"); rows.append(f"นักแสดง {a}")
        for code in ORDER:
            ci = [i for i in range(len(z["name"])) if z["emotion"][i] == code and z["intensity"][i] == (1 if code == 1 else 2)][0]
            imgs.append(z["crop_g96"][mid_crop(z, ci)]); titles.append(TH[EMO[code]])
    grid(imgs, titles, F / "v2_data_ravdess.png", 7, row_labels=rows)


def points():
    """actor 1: neutral vs strong angry — the crop with the 68 points and the brow height (inner brow → eye line)."""
    z = np.load(DATA / "actor_01.npz"); W, Hh = 1280, 720
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 2.5))
    for ax, code, lab in zip(axs, (1, 5), ("ปกติ", "โกรธ (ขมวดคิ้ว)")):
        ci = [i for i in range(len(z["name"])) if z["emotion"][i] == code and z["intensity"][i] == (1 if code == 1 else 2)][0]
        j = mid_crop(z, ci); a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1])
        k = a0 + int(np.argmin(np.abs(z["t"][a0:a1] - z["crop_t"][j])))
        face = z["face"][k].astype(np.float64); x0, y0, s = crop_box(face, W, Hh); sc = 96 / s
        p = np.c_[(face[:, 0] * W - x0) * sc, (face[:, 1] * Hh - y0) * sc]
        ax.imshow(z["crop_g96"][j], cmap="gray", vmin=0, vmax=255); ax.axis("off")
        ax.scatter(p[:, 0], p[:, 1], s=3, color="#2a78d6", linewidths=0)
        ax.scatter(p[17:27, 0], p[17:27, 1], s=5, color="#eb6834", linewidths=0)
        el, er = p[36:42].mean(0), p[42:48].mean(0)
        ax.plot([el[0], er[0]], [el[1], er[1]], color="#1baf7a", lw=1)
        for b, e in ((21, el), (22, er)):
            ax.plot([p[b, 0], p[b, 0]], [p[b, 1], e[1]], color="#eb6834", lw=1.2)
        ax.set_xlim(0, 96); ax.set_ylim(96, 0); ax.set_title(lab, fontsize=8.5, color=INK2, pad=4)
    fig.tight_layout(pad=0.6); fig.savefig(F / "v2_data_points.png", dpi=220); plt.close(fig)


def signing(n=8):
    """middle frame of n dictionary-signing clips (data_test/jack_test, the neutral samples) → the same 96 px crop."""
    from modules.mediapipe_pose import Holistic
    clips = sorted(p for p in glob.glob(str(ROOT / "data_test" / "jack_test" / "*" / "*.mp4")) if not Path(p).parent.name.startswith(SKIP))
    pick = [clips[i] for i in np.linspace(0, len(clips) - 1, n).astype(int)]
    imgs = []
    for p in pick:
        cap = cv2.VideoCapture(p); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); hol = Holistic(); got = None
        for f in (N // 2, N // 3, 2 * N // 3, N // 4):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
            if not ok:
                continue
            if fr.shape[0] > 720:
                fr = cv2.resize(fr, (fr.shape[1] * 720 // fr.shape[0], 720), interpolation=cv2.INTER_AREA)
            _, _, _, face, _ = hol.frame(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), f * 33.0)
            if np.abs(face).sum() > 0:
                got = crop48(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), crop_box(face, fr.shape[1], fr.shape[0]), 96); break
        cap.release(); hol.close()
        if got is not None:
            imgs.append(got)
    grid(imgs, [f"คลิป {i + 1}" for i in range(len(imgs))], F / "v2_data_signing.png", len(imgs))


def team():
    S = np.load(real_check.CACHE, allow_pickle=True)
    jl = real_check.judged(S["clip"], S["t"], S["q"]); rng = np.random.RandomState(1)
    gray = np.stack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in S["col"]])
    pick = [(6, 4, "ปกติ"), (3, 3, "ยิ้ม"), (-2, 2, "กำกวม (ไม่นับ)"), (-1, 2, "มือบัง / ไม่ชัด")]
    imgs, titles = [], []
    for code, n, lab in pick:
        idx = np.where(jl == code)[0]
        if code == 6:                                    # one neutral frame per clip, spread over clips
            idx = np.array([rng.choice(np.where((jl == 6) & (S["clip"] == c))[0]) for c in rng.permutation(np.unique(S["clip"][jl == 6]))])
        for i in idx[:n] if code != 6 else idx[:n]:
            imgs.append(gray[i]); titles.append(lab)
    grid(imgs, titles, F / "v2_data_team.png", 11)
    return {k: int((jl == v).sum()) for k, v in (("neutral", 6), ("smile", 3), ("ambiguous", -2), ("covered_or_poor", -1))}, int(len(np.unique(S["clip"])))


def counts(team_counts, team_clips):
    per, crops, kept, samples = {e: 0 for e in TH}, 0, 0, 0
    for f in sorted(DATA.glob("actor_*.npz")):
        z = np.load(f); crops += len(z["crop_q"]); kept += int((z["crop_q"] >= 0.25).sum())
        for ci in range(len(z["name"])):
            per[EMO[int(z["emotion"][ci])]] += 1; samples += len(spans(int(z["off"][ci + 1] - z["off"][ci]), float(z["fps"])))
    s = np.load(H / "cache" / "sign_neutral.npz")
    out = dict(ravdess=dict(actors=24, clips=sum(per.values()), clips_per_class=per, samples_clip_plus_windows=samples, crops_5fps=crops, crops_quality_ok=kept),
               signing_neutral=dict(clips=int(len(s["names"])), crops_5fps=int(len(s["q"])), crops_quality_ok=int((s["q"] >= 0.25).sum()), samples=120),
               team=dict(clips=team_clips, frames_every_0_5s=int(sum(team_counts.values())), **team_counts))
    (H / "results" / "data_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    import sys as _s
    if "--points" in _s.argv:
        points(); raise SystemExit
    ravdess(); points(); tc = team(); signing(); counts(*tc)
