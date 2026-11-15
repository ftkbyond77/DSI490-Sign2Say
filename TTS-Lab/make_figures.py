"""Figures for TTS-Lab/figures/ (every number comes from files in data/).   python make_figures.py [arch|data|onset|emotion|latency|results]"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).parent; FIG = HERE / "figures"; DATA = HERE / "data"; RUN = DATA / "run"; FIG.mkdir(exist_ok=True)
C = dict(frozen="#c9d6e8", tuned="#f6c28b", new="#9fd8b0", io="#eeeeee", ink="#222222", a="#5b7db1", b="#e07b00", c="#6aa84f")
EMO = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]


def box(ax, x, y, w, h, t, c, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec=C["ink"], lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=C["ink"])


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.2, color=C["ink"]))


def arch():
    fig, ax = plt.subplots(figsize=(12.5, 6.4)); ax.set_xlim(0, 12.5); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.set_title("Speech-model-v4-TTS: ThonburianTTS (F5-TTS 337M) → v3 → v4 — what is trained and what is added", fontsize=12, weight="bold")
    box(ax, 0.2, 4.6, 2.2, 0.95, "[emotion tag] + Thai text\n(char tokens, '…' = gap)\nintensity = tag-embedding mix", C["new"], 8.5)
    box(ax, 0.2, 2.9, 2.2, 0.95, "Reference clip (3-4 s)\n+ 0.3 s silence tail\nyouthful male / female", C["new"], 8.5)
    box(ax, 3.1, 2.9, 1.6, 0.95, "Mel 100ch\n(Vocos, hop 256)", C["frozen"])
    box(ax, 3.1, 4.6, 1.6, 0.95, "Text embed +\nConvNeXt x4\n(+7 tag rows)", C["tuned"], 8.5)
    box(ax, 5.4, 3.0, 3.0, 2.55, "DiT flow-matching\n22 layers · dim 1024 · 16 heads\n(prompt-format training: loss on the\nnew text only, first 12 frames x2)", C["tuned"], 9)
    box(ax, 9.1, 3.9, 1.5, 0.95, "ODE solve\nEuler, EPSS\n16 steps", C["new"], 8.5)
    box(ax, 10.8, 3.9, 1.4, 0.95, "Vocos\nvocoder", C["frozen"])
    box(ax, 10.8, 2.4, 1.4, 0.95, "24 kHz\nwaveform", C["io"])
    for a, b in [((2.4, 5.05), (3.1, 5.05)), ((4.7, 5.05), (5.4, 4.7)), ((2.4, 3.35), (3.1, 3.35)), ((4.7, 3.35), (5.4, 3.6)), ((8.4, 4.35), (9.1, 4.35)),
                 ((10.6, 4.35), (10.8, 4.35)), ((11.5, 3.9), (11.5, 3.35))]:
        arrow(ax, a, b)
    box(ax, 0.2, 0.25, 12.0, 1.9, "Fine-tuning (one A100 Spot run, ~50 min, AdamW 1.5e-5, tag rows 3e-4, bf16, EMA):  29,162 clips = 34.7 h of S+ / S / A+ Thai speech (30,000 labelled; DNSMOS per grade, text = Whisper-large ∩ Typhoon)\n"
        "sampling: masculine/feminine hours balanced · youthful speakers (estimated age) up-weighted · 4 default-voice speakers held out\n"
        "samples: real (random-span infill) · pair [prompt clip | target clip, same speaker] · gap 'A … B' · emotion: natural pseudo-labels (SER ∩ prosody) + prosody-transformed clips (PSOLA, labels by construction)", C["new"], 8)
    for i, (t, c) in enumerate([("trained (all weights)", C["tuned"]), ("frozen / unchanged", C["frozen"]), ("added by v4", C["new"])]):
        ax.add_patch(plt.Rectangle((0.3 + i * 3, 5.95), 0.3, 0.25, fc=c, ec=C["ink"])); ax.text(0.7 + i * 3, 6.07, t, va="center", fontsize=9)
    fig.savefig(FIG / "architecture.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def data():
    m = pd.read_csv(DATA / "manifest.csv"); m = m[(m.reason.isna()) & (m.labelled == True)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8)); ax = ax.ravel()
    h = m.groupby(["grade_avg", "voice"]).duration.sum().div(3600).unstack().reindex(["S+", "S", "A+"])
    h.plot.bar(stacked=True, ax=ax[0], rot=0, color=[C["b"], C["a"], "#bbbbbb"]); ax[0].set(title=f"Hours per grade and voice presentation ({m.duration.sum() / 3600:.1f} h, {len(m)} clips)", ylabel="h")
    s = m.drop_duplicates("speaker_id").dropna(subset=["spk_age"])
    for v, c in (("masculine", C["a"]), ("feminine", C["b"])):
        ax[1].hist(s[s.voice == v].spk_age, np.arange(10, 70, 2), alpha=0.6, color=c, label=v)
    ax[1].axvspan(18, 23, color=C["c"], alpha=0.2); ax[1].legend(); ax[1].set(title=f"Estimated (perceived) speaker age — {len(s)} speakers; green = 18-23", xlabel="years")
    e = m.emo.value_counts().reindex(EMO).fillna(0); ax[2].bar(e.index, e.values, color=C["a"]); ax[2].set(title="Natural pseudo-labels (SER ∩ prosody agree) — clips"); ax[2].tick_params(axis="x", rotation=30)
    ax[3].hist(m.lead_ms.clip(upper=600), 60, color=C["a"]); ax[3].set(title="Leading silence of the dataset clips (ms) — most clips start at speech onset", xlabel="ms")
    sp = m.drop_duplicates("speaker_id"); ax[4].hist(sp.f0.dropna(), 60, color=C["a"]); [ax[4].axvline(v, c="r", ls="--") for v in (150, 180)]; ax[4].set(title="Per-speaker median F0 (Hz): ≤150 masculine · ≥180 feminine")
    sl = m.split.value_counts(); ax[5].bar(sl.index, sl.values, color=C["a"]); ax[5].set(title="Speaker-disjoint splits (clips)")
    fig.tight_layout(); fig.savefig(FIG / "data.png", dpi=125); plt.close(fig)


def onset():
    A = DATA / "ablation"; fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # A: reference tail experiment (v3, the four original v3 voices, short sentences)
    rows = []
    for k, n in (("tail005", "tail 0.05 s (F5 default)"), ("tail030", "tail 0.30 s"), ("tail060", "tail 0.60 s")):
        g = pd.read_csv(A / f"{k}.csv"); rows.append(dict(cond=n, pathumma=g.fw_ok_pathumma.mean(), turbo=g.fw_ok_turbo.mean()))
    d = pd.DataFrame(rows).set_index("cond"); d.plot.bar(ax=ax[0], rot=15, color=[C["a"], C["b"]]); ax[0].set(ylim=(0.5, 1), title="v3, 4 old voices, 8 short sentences x3 seeds:\nfirst-word hit vs silence after the reference")
    # B: first-word hit by voice: old refs / new refs (v3) / v4
    names, vals = [], []
    g = pd.read_csv(A / "v3_old_refs_nfe32.csv")
    for v, n in (("masculine_1", "v3 male (old ref)"), ("feminine_1", "v3 female (old ref)")):
        names.append(n); vals.append(g[g.voice == v].fw_ok_pathumma.mean())
    for tag, f in (("v3 + new refs", RUN / "gen_v3y_n32" / "score.csv"), ("v4", RUN / "gen_v4" / "score.csv")):
        if f.exists():
            g = pd.read_csv(f); g = g[g.kind.str.startswith(("onset", "short"))]
            for v in ("male", "female"):
                names.append(f"{tag} {v}"); vals.append(g[g.voice == v].fw_pathumma.mean())
    ax[1].bar(range(len(vals)), vals, color=[C["a"]] * 2 + [C["b"]] * (len(vals) - 2)); ax[1].set_xticks(range(len(vals)), names, rotation=30, ha="right", fontsize=8)
    ax[1].set(ylim=(0.5, 1.02), title="First-word hit (Pathumma ASR), onset + short sets")
    # C: the reported sentence ("ฉันชอบแมว"): first word said, by voice and silence after the reference (3 seeds, both ASR must hear it)
    vs = ["masculine_1", "masculine_2", "feminine_1", "feminine_2"]
    for j, (k, n, c) in enumerate((("tail005", "0.05 s (F5 default)", C["a"]), ("tail030", "0.30 s", C["b"]), ("tail060", "0.60 s", C["c"]))):
        g = pd.read_csv(A / f"{k}.csv"); g = g[g.text == "ฉันชอบแมว"]
        ax[2].bar(np.arange(4) + (j - 1) * 0.27, [((g[g.voice == v].fw_ok_pathumma + g[g.voice == v].fw_ok_turbo) / 2).mean() for v in vs], 0.27, color=c, label=f"tail {n}")
    ax[2].set_xticks(range(4), vs, rotation=20, fontsize=8); ax[2].legend(fontsize=7)
    ax[2].set(ylim=(0, 1.05), title='"ฉันชอบแมว", 3 seeds: first word heard by both ASRs\n(the reported failure: v3 male = masculine_1)')
    fig.tight_layout(); fig.savefig(FIG / "onset.png", dpi=130); plt.close(fig)


def emotion():
    r = json.loads((DATA / "results.json").read_text(encoding="utf-8")); e = r["emotion"]
    dsp = DATA / "run" / "gen_v4_dsp"
    fig, ax = plt.subplots(2, 3, figsize=(16, 9)); ax = ax.ravel()
    mats = [("model tag only, intensity 1.0", np.array(e["confusion_intensity_1.0"]["matrix"]))]
    if (dsp / "score.csv").exists() and (dsp / "affect.csv").exists():
        d = pd.read_csv(dsp / "score.csv").merge(pd.read_csv(dsp / "affect.csv"), on="file"); M = np.zeros((7, 7))
        x = d[(d.intensity == 1.0) | (d.emotion == "neutral")]
        for i, emo in enumerate(EMO):
            p_ = x[x.emotion == emo][[f"ser_{c}" for c in EMO]].values
            if len(p_):
                M[i] = np.bincount(p_.argmax(1), minlength=7) / len(p_)
        mats.append(("model + PSOLA layer, intensity 1.0", M))
    for k, (ttl, M) in enumerate(mats):
        ax[k].imshow(M, vmin=0, vmax=1, cmap="Blues"); ax[k].set_xticks(range(7), EMO, rotation=40, ha="right"); ax[k].set_yticks(range(7), EMO)
        for a_ in range(7):
            for b_ in range(7):
                ax[k].text(b_, a_, f"{M[a_, b_]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[a_, b_] > .5 else "black")
        ax[k].set(xlabel="emotion heard (independent SER)", ylabel="requested", title=f"{ttl}\naccuracy {np.trace(M) / 7:.2f} (chance 0.14)")
    sh = e["prosody_shift_vs_neutral"]; T = {"happy": 1.5, "sad": -1.5, "angry": 1.0, "fear": 3.0, "disgust": -1.0, "surprise": 3.5}
    emos = list(T); xs = np.arange(len(emos))
    ax[2].bar(xs - 0.27, [T[k] for k in emos], 0.27, color="#999999", label="template (intensity 1)")
    ax[2].bar(xs, [sh[k]["1.0"]["f0_semitones"] for k in emos], 0.27, color=C["a"], label="model tag only")
    if (dsp / "score.csv").exists():
        d["pres"] = d.voice; base = d[d.emotion == "neutral"].groupby("voice").f0_mean.mean()
        vals = [float(np.mean([12 * np.log2(g_.f0_mean.mean() / base[v]) for v, g_ in d[(d.emotion == k) & (d.intensity == 1.0)].groupby("voice")])) for k in emos]
        ax[2].bar(xs + 0.27, vals, 0.27, color=C["b"], label="model + PSOLA layer")
    ax[2].set_xticks(xs, emos, rotation=30); ax[2].axhline(0, color="k", lw=0.6); ax[2].legend(fontsize=8); ax[2].set(title="Mean F0 shift vs neutral (semitones), intensity 1.0")
    for j, (col, ttl) in enumerate((("energy_db", "Level shift vs neutral (dB), model tag only"), ("rate_ratio", "Speaking-rate ratio vs neutral, model tag only"))):
        for it, c in ((0.5, C["a"]), (1.0, C["b"])):
            ax[3 + j].bar(np.arange(6) + (0.2 if it == 1.0 else -0.2), [sh[k][str(it)][col] - (1.0 if col == "rate_ratio" else 0) for k in emos], 0.4, color=c, label=f"intensity {it}")
        ax[3 + j].set_xticks(range(6), emos, rotation=30); ax[3 + j].axhline(0, color="k", lw=0.6); ax[3 + j].set_title(ttl + (" (0 = same)" if col == "rate_ratio" else "")); ax[3 + j].legend(fontsize=8)
    ax[5].bar(list(e["cer_by_emotion"]), list(e["cer_by_emotion"].values()), color=C["a"]); ax[5].tick_params(axis="x", rotation=30); ax[5].set(title="Mean CER (turbo, clipped) by requested emotion: intelligibility kept")
    fig.tight_layout(); fig.savefig(FIG / "emotion.png", dpi=125); plt.close(fig)


def latency():
    r = json.loads((DATA / "results.json").read_text(encoding="utf-8")); st = json.loads((DATA / "stack_summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6)); A = DATA / "ablation"
    rows = []
    for k, n in (("L32", "32 steps, long ref"), ("L16", "16 steps, long ref"), ("S16", "16 steps, short ref"), ("S12", "12 steps"), ("S8", "8 steps")):
        g = pd.read_csv(A / f"speed_{k}.csv"); t = json.loads((A / f"speed_{k}.time.json").read_text())
        rows.append(dict(cfg=n, s=t["per_clip_s"], fw=g.fw_ok_pathumma.mean(), cer=g.cer_turbo.clip(upper=1).mean()))
    d = pd.DataFrame(rows)
    ax[0].plot(d.s, d.fw, "o-", color=C["a"], label="first-word hit (Pathumma)"); ax[0].plot(d.s, 1 - d.cer, "s--", color=C["b"], label="1 - mean CER (turbo)")
    for _, q in d.iterrows():
        ax[0].annotate(q.cfg, (q.s, q.fw), textcoords="offset points", xytext=(3, -12), fontsize=7)
    ax[0].set(xlabel="seconds per short clip, RTX 2050", title="v3 weights: quality vs speed (steps x reference length)"); ax[0].legend(fontsize=8)
    sw = r["step_sweep_v4"]; ks = ["32", "16 (default)", "12"]
    ax[1].plot([int(k.split()[0]) for k in ks], [sw[k]["fw_pathumma"] for k in ks], "o-", label="first-word hit (Pathumma)")
    ax[1].plot([int(k.split()[0]) for k in ks], [1 - sw[k]["cer_mean"] for k in ks], "s--", label="1 - mean CER")
    ax[1].plot([int(k.split()[0]) for k in ks], [sw[k]["sim"] for k in ks], "^:", label="speaker similarity"); ax[1].invert_xaxis()
    ax2 = ax[1].twinx(); ax2.plot([int(k.split()[0]) for k in ks], [sw[k]["gen_s_median"] for k in ks], "k-", lw=0.8); ax2.set_ylabel("median s per clip (black)")
    ax[1].set(xlabel="ODE steps", title="v4 (shipped): quality vs steps, RTX 2050"); ax[1].legend(fontsize=7)
    for i, (k, c) in enumerate((("v3_tts_latency", C["a"]), ("v4_tts_latency", C["b"]))):
        ax[2].bar(np.arange(3) + (i - 0.5) * 0.4, [st[k][s]["p50"] for s in ("1s", "3s", "6s")], 0.4, color=c, label=k[:2] + " (p50)")
    ax[2].set_xticks(range(3), ["~1 s", "~3 s", "~6 s"]); ax[2].legend(); ax[2].set(ylabel="seconds", title="Docker + gateway, RTX 2050, warm")
    fig.tight_layout(); fig.savefig(FIG / "latency.png", dpi=130); plt.close(fig)


def results():
    lg = RUN / "log.json"
    if not lg.exists():
        return
    L = json.loads(lg.read_text())["log"]; fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot([x["step"] for x in L], [x["val"] for x in L], "o-"); ax.set(title="Validation flow-matching loss (EMA; real + pair + gap items of held-out speakers)", xlabel="update")
    fig.tight_layout(); fig.savefig(FIG / "results.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    for n, f in (("arch", arch), ("data", data), ("onset", onset), ("emotion", emotion), ("latency", latency), ("results", results)):
        if len(sys.argv) < 2 or sys.argv[1] == n:
            f(); print("made", n)
