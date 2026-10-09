"""All evaluations of result_v1.md in one run → data/eval/summary.json + result_reporting/figures/*.png.

  python main.py eval all            (needs: models/ built; the docker stack on :8081 for latency / ASR / renders)
Each block is independent: python main.py eval tsl51 | conversation | backtranslate | library | latency | renders
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .. import paths

SUMMARY = paths.WORK / "eval" / "summary.json"
FIG = paths.REPORT / "figures"
# reference palette (dataviz skill, light mode): categorical slots in fixed order + ink tokens
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _load():
    return json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}


def _save(d):
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _fig(w=7.2, h=3.2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Tahoma", "Leelawadee UI", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURF)
    return fig, ax


def tsl51(d):
    from ..text.translate import Mapper, translate_thai
    pairs = json.loads((paths.PREP / "params" / "tsl51_thai_sentences.json").read_text(encoding="utf-8"))
    m = Mapper()
    # the grammar alone (no recorded-phrase shortcut): what the rules produce
    ok = sum([u.gloss for u in translate_thai(th, m, phrasebook=False).units] == gold.split() for gold, th in pairs.items())
    no_topic = sum([u.gloss for u in translate_thai(th, m, topicalize=False, phrasebook=False).units] == gold.split()
                   for gold, th in pairs.items())
    d["tsl51"] = dict(exact=ok, n=len(pairs), exact_no_topicalisation=no_topic, word_for_word_baseline=_thai_order_baseline(pairs, m))
    return d


def _thai_order_baseline(pairs, m):
    """signs in Thai word order (no TSL grammar) — what a word-for-word system would produce."""
    from ..text.translate import tokenize, normalize
    ok = 0
    for gold, th in pairs.items():
        toks = [t for t in tokenize(normalize(th).replace(" ", ""), m) if t in m.entries]
        ok += toks == gold.split()
    return ok


def conversation(d):
    os.environ.setdefault("T2S_MT_DIR", str(paths.WORK / "speech_models" / "nllb-600m"))
    from .conversation import run
    df, s = run()
    d["conversation"] = {lang: dict(sentences=int(r.sentences), all_signed=round(float(r.all_signed), 3), units=int(r.units),
                                    sign_rate=round(float(r.sign_rate), 3), spelled=int(r.spelled), text_cards=int(r.text_cards),
                                    recorded_phrases=int(r.phrases))
                         for lang, r in s.iterrows()}
    return d


def backtranslate(d, n=1000):
    from .backtranslate import run
    o, df = run(n=n)
    o["by_tier"] = {t: dict(n=int(len(g)), top1_avatar=float((g.rank_avatar <= 1).mean()), top1_human=float((g.rank_human <= 1).mean()))
                    for t, g in df.groupby("tier")}
    d["backtranslate"] = o
    fig, ax = _fig(6.4, 3.0)
    ks = ["top1", "top5", "top10"]
    x = np.arange(len(ks))
    w = 0.32
    ax.bar(x - w / 2, [o["human"][k] * 100 for k in ks], w * 0.9, color=C1, label="real signer's landmarks")
    ax.bar(x + w / 2, [o["avatar"][k] * 100 for k in ks], w * 0.9, color=C2, label="avatar (retargeted)")
    for i, k in enumerate(ks):
        ax.text(i + w / 2, o["avatar"][k] * 100 + 1.5, f"{o['avatar'][k] * 100:.0f}%", ha="center", fontsize=8, color=INK2)
        ax.text(i - w / 2, o["human"][k] * 100 + 1.5, f"{o['human'][k] * 100:.0f}%", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, ["top-1", "top-5", "top-10"])
    ax.set_ylim(0, 100)
    _style(ax, f"Forward recogniser reads the sign (n = {o['n']}, leave-source-out)", ylabel="% correct")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "backtranslate.png")
    return d


def library(d):
    idx = json.loads((paths.MODELS / "motion" / "index.json").read_text(encoding="utf-8"))
    c = pd.DataFrame([dict(key=k, tier=v.get("tier"), dataset=v.get("dataset"), wrist=v["fidelity"]["wrist"], elbow=v["fidelity"]["elbow"],
                           finger=v["fidelity"]["fingertip"], n=v["n"], loc=v["desc"]["location"], two=v["desc"]["two_handed"])
                      for k, v in idx["clips"].items()])
    words = c[~c.key.str.startswith("phrase:")]
    d["library"] = dict(signs=int(len(words)), phrases=int((c.key.str.startswith("phrase:")).sum()), frames=int(idx["frames"]),
                        by_source=words.dataset.value_counts().to_dict(), tiers=words.tier.value_counts().to_dict(),
                        wrist_median=float(words.wrist.median()), wrist_p90=float(words.wrist.quantile(.9)),
                        elbow_median=float(words.elbow.median()), finger_median=float(words.finger.median()),
                        finger_p90=float(words.finger.quantile(.9)), two_handed=float(words.two.mean()),
                        location=words["loc"].value_counts(normalize=True).round(3).to_dict(),
                        q_bin_mb=round((paths.MODELS / "motion" / "q.bin").stat().st_size / 2 ** 20, 1))
    fig, ax = _fig(6.4, 2.8)
    bins = np.linspace(0, 0.3, 31)
    ax.hist(words.wrist.clip(upper=0.3), bins=bins, color=C1, rwidth=0.85)
    ax.axvline(words.wrist.median(), color=INK2, linewidth=1)
    ax.text(words.wrist.median() + 0.004, ax.get_ylim()[1] * 0.9, f"median {words.wrist.median():.3f}", fontsize=8, color=INK2)
    _style(ax, f"Retargeting error per sign — wrist, avatar vs signer (n = {len(words):,})", "mean 2D error (shoulder widths)", "signs")
    fig.tight_layout()
    fig.savefig(FIG / "fidelity_wrist.png")
    return d


def data_funnel(d):
    t = pd.read_parquet(paths.WORK / "audit" / "clips_quality.parquet")
    sel = pd.read_parquet(paths.WORK / "select" / "exemplars.parquet")
    rows = []
    for ds in ["ttrs", "th_sl", "tsl51_user_sign", "youtube_agent"]:
        g = t[t.dataset == ds]
        rows.append(dict(source=ds, clips=len(g), tier_a=int((g.tier == "A").sum()), tier_b=int((g.tier == "B").sum()),
                         rejected=int(g.tier.isna().sum()), selected=int((sel.dataset == ds).sum())))
    h = paths.HARVEST / "qc.csv"
    if h.exists():
        q = pd.read_csv(h)
        dec = pd.read_csv(paths.PKG / "data" / "harvest_decisions.csv")
        rows.append(dict(source="youtube_harvest (2026-10-10)", clips=len(q), tier_a=int((q.tier == "A").sum()), tier_b=int((q.tier == "B").sum()),
                         rejected=int(q.tier.isna().sum()) + int((dec.decision == "reject").sum()),
                         selected=int(sel.dataset.isin(["youtube_harvest", "youtube_phrase"]).sum())))
    d["data"] = rows
    reasons = (t.assign(fail=t["fail"].fillna("")).fail.str.split("|").explode())
    d["fail_reasons"] = reasons[reasons != ""].value_counts().to_dict()
    return d


def latency(d, base="http://localhost:8081", n=30):
    import urllib.request
    texts = ["คุณชื่ออะไร", "วันนี้คุณไปไหน", "พรุ่งนี้คุณจะไปโรงเรียนไหม", "ฉันอยากไปเที่ยวเชียงใหม่กับครอบครัว", "ฉันไม่ชอบกินผัก",
             "ห้องน้ำอยู่ที่ไหน", "ขอบคุณมากครับ", "แม่ของฉันเป็นครูที่โรงเรียน", "ช่วยพูดช้าๆหน่อยได้ไหม", "ฉันหิวข้าว"]

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                                                                           "Accept-Encoding": "gzip"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
        return (time.time() - t0) * 1000, len(raw)
    out = {}
    for name, items in (("thai", [{"text": t, "speed": 1.0 + 0.01 * i} for i in range(n // len(texts) + 1) for t in texts][:n]),
                        ("english", [{"text": t, "speed": 1.0 + 0.013 * i} for i, t in enumerate(
                            ["Where is the hospital?", "What is your name?", "I like to eat eggs.", "Thank you very much.",
                             "Where did you go today?", "I am hungry.", "I don't understand.", "See you tomorrow."])])):
        ms, sz = zip(*[post("/api/translate", b) for b in items])
        out[name] = dict(p50_ms=round(float(np.median(ms)), 1), p95_ms=round(float(np.percentile(ms, 95)), 1),
                         gzip_kb_median=round(float(np.median(sz)) / 1024, 1), n=len(ms))
    d["latency"] = out
    return d


def speech(d):
    from .asr import run as asr_run
    d["asr"] = asr_run()
    return d


def figures(d):
    """coverage before/after the conversational work, and the tempo calibration."""
    import pandas as pd
    # coverage: first measurement of the session (logged in result_v1.md §4.4) vs final
    before = {"th": (0.924, 0.792), "en": (0.890, 0.717)}
    after = {k: (d["conversation"][k]["sign_rate"], d["conversation"][k]["all_signed"]) for k in ("th", "en")}
    fig, ax = _fig(6.6, 3.0)
    labels = ["Thai: words signed", "Thai: sentences fully signed", "English: words signed", "English: sentences fully signed"]
    b = [before["th"][0], before["th"][1], before["en"][0], before["en"][1]]
    a = [after["th"][0], after["th"][1], after["en"][0], after["en"][1]]
    x = np.arange(4); w = 0.34
    ax.bar(x - w / 2, np.array(b) * 100, w * 0.9, color=C1, label="first run (dictionary + grammar only)")
    ax.bar(x + w / 2, np.array(a) * 100, w * 0.9, color=C2, label="final (curated variants, harvest, phrasebook)")
    for i in range(4):
        ax.text(i + w / 2, a[i] * 100 + 1, f"{a[i] * 100:.0f}%", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [l.replace(": ", ":\n") for l in labels], fontsize=8)
    ax.set_ylim(50, 102)
    _style(ax, "Everyday sentences → real recorded signs (96 Thai, 53 English)", ylabel="%")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "coverage.png")
    # tempo
    t = pd.read_csv(paths.WORK / "eval" / "phrase_tempo.csv")
    conv = json.loads((paths.MODELS / "timing.json").read_text(encoding="utf-8"))
    sp, tr = conv.get("stroke_speed", 1.0), conv.get("c0", 0.45)
    t["conversational_s"] = t.strokes_s / sp + (t.n - 1) * tr
    t = t.sort_values("real_s")
    fig, ax = _fig(7.0, 6.4)
    y = np.arange(len(t))
    ax.plot(t.generated_s, y, "o", ms=4.5, color=C1, label="generated, citation speed (first version)")
    ax.plot(t.conversational_s, y, "o", ms=4.5, color=C2, label="generated, conversational timing (final)")
    ax.plot(t.real_s, y, "o", ms=4.5, color=C3, label="real signer, same sentence")
    ax.set_yticks(y, t.phrase, fontsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.grid(axis="y", visible=False)
    _style(ax, "Sentence duration: generated vs a real signer (28 everyday phrases)", xlabel="seconds of signing")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.45, 1.02), ncol=1)
    ax.set_title("Sentence duration: generated vs a real signer (28 everyday phrases)", loc="left", fontsize=11, color=INK, pad=62)
    fig.tight_layout(); fig.savefig(FIG / "tempo.png")
    d["tempo"] = dict(n=len(t), median_real=float(t.real_s.median()), median_citation=float(t.generated_s.median()),
                      median_conversational=float(t.conversational_s.median()))
    return d


def run(what="all"):
    d = _load()
    steps = dict(data=data_funnel, tsl51=tsl51, conversation=conversation, library=library, backtranslate=backtranslate,
                 latency=latency, asr=speech, figures=figures)
    for k, f in steps.items():
        if what in ("all", k):
            t0 = time.time()
            try:
                d = f(d)
                print(f"[eval] {k}: {time.time() - t0:.0f} s")
            except Exception as e:  # noqa: BLE001 — one failing block (e.g. docker down) must not lose the others
                print(f"[eval] {k} FAILED: {type(e).__name__}: {e}")
            _save(d)
    return d
