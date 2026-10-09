"""Why did my sign fail? A diagnostic report for one recording (prompt_10 Phase E: make failures explainable, never relabel them).

  python main.py diagnose <video.mp4 | cache/live_requests/<file>.json.gz> [--intended "ฉัน รัก เพื่อน"] [--vocab conversation|daily]
  python main.py diagnose cache/live_requests --all                  # every saved live utterance → one summary table

Per recording (written to result_reporting/diagnose/<name>/):
  report.md     intended words (what you meant — never treated as a verified label), what the system heard, and for every segment:
                top-5 words with scores, calibrated P(correct), the 3 nearest reference clips (dataset · signer · similarity) and a
                failure category (below)
  tracking.png  hand / body / face presence over time, raised-hand frames whose hand was lost, the decoded segments
  result.json   everything above, machine-readable
Failure categories (evidence-based; one or more per intended word):
  landmark      the hands were lost in > 30 % of the raised-hand frames of the utterance (extractor, light, blur, framing)
  segmentation  the number of signs found differs from the intended number of words (signs merged or split)
  not_in_vocab  the intended word is not in the vocabulary (no reference data: the model cannot know it)
  confusable    the intended word is in some segment's top-5 but not first (a similar sign won; references exist)
  sign_differs  the intended word is in the vocabulary but in no segment's top-5 (the attempt does not look like any reference —
                possibly a different variant, or an incorrect attempt: ask a signer before using it for training)
  ok            the intended word was the accepted / first reading
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from modules.utils import get_logger, write_json

log = get_logger("diagnose")


def load_input(path, pipe):
    from modules.mediapipe_pose import from_browser
    p = Path(path)
    if p.suffix == ".gz":
        rec = json.load(gzip.open(p, "rt", encoding="utf-8"))
        return from_browser(rec["frames"], (rec["height"], rec["width"])), rec
    return pipe.landmarks_from_video(p), None


def nearest_refs(pipe, emb, k=3):
    b = np.load(pipe.md / "bank.npz", allow_pickle=False)
    Z = b["Z"].astype(np.float32)
    s = Z @ (emb / (np.linalg.norm(emb) + 1e-9))
    o = np.argsort(-s)[:k]
    return [dict(word=pipe.concepts[int(b["c"][i])], dataset=str(b["dataset"][i]), signer=str(b["signer"][i]), sim=round(float(s[i]), 3)) for i in o]


def plot(raw, r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from modules.mediapipe_pose import wrist_raised
    t = raw["t"]
    T = len(t)
    rows = dict(body=raw["pose"][:, 11, 3] > 0.5, face=np.abs(raw["face"]).sum((1, 2)) > 0,
                left=np.abs(raw["lh"]).sum((1, 2)) > 0, right=np.abs(raw["rh"]).sum((1, 2)) > 0)
    lost = {k: np.array([wrist_raised(raw["pose"][i], wi) and not rows[k][i] for i in range(T)]) for k, wi in (("left", 15), ("right", 16))}
    fig, ax = plt.subplots(figsize=(12, 3.2))
    for y, (k, v) in enumerate(rows.items()):
        ax.scatter(t[v], np.full(v.sum(), y), s=6, c="#2a9d8f", marker="|")
        if k in lost:
            ax.scatter(t[lost[k]], np.full(lost[k].sum(), y), s=14, c="#e63946", marker="x", label=f"{k} hand lost while raised" if y == 2 else None)
    for s in r["segments"]:
        col = {"accepted": "#2a9d8f", "uncertain": "#e9c46a", "unknown": "#adb5bd", "transition": "#dee2e6"}[s["status"]]
        ax.axvspan(s["t0"], s["t1"], ymin=0.86, ymax=0.98, color=col)
        if s["status"] != "transition":
            ax.text((s["t0"] + s["t1"]) / 2, 4.1, s["nearest"] if s["status"] == "accepted" else f"[?]≈{s['nearest']}", ha="center", fontsize=8,
                    fontname="Tahoma")
    ax.set_yticks(range(4)); ax.set_yticklabels(list(rows)); ax.set_ylim(-0.6, 4.6); ax.set_xlabel("seconds")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def diagnose_one(pipe, path, intended, out_root, canon):
    raw, rec = load_input(path, pipe)
    r = pipe.run_landmarks(raw, title=Path(path).name)
    name = Path(path).name.split(".")[0]
    out = out_root / name; out.mkdir(parents=True, exist_ok=True)
    segs = [s for s in r["segments"] if s["status"] != "transition"]
    vocab = {pipe.concepts[c] for c in pipe.bank.concepts_present}
    ht = r.get("hand_tracking") or {}
    worst = min([h["hit"] for h in (ht.get("left", {}), ht.get("right", {})) if h.get("hit") is not None and h.get("n", 0) >= 10] or [1.0])
    rows = []
    for s in segs:
        rows.append(dict(t0=s["t0"], t1=s["t1"], status=s["status"], p_correct=round(s.get("p_correct", 0.0), 3),
                         top5=[(c["word"], round(c["score"], 3)) for c in s["candidates"][:5]],
                         nearest_refs=nearest_refs(pipe, np.asarray(s["emb"], np.float32)) if "emb" in s else []))
    words = [canon(w) for w in (intended.split() if intended else [])]
    per_word = []
    for w in words:
        cats = []
        if worst < 0.7:
            cats.append("landmark")
        if len(segs) != len(words):
            cats.append("segmentation")
        top1 = [s["nearest"] for s in segs]
        top5 = [c["word"] for s in segs for c in s["candidates"][:5]]
        if w not in vocab:
            cats.append("not_in_vocab")
        elif w in top1:
            cats.append("ok" if not cats else "ok_despite_" + "+".join(cats))
        elif w in top5:
            cats.append("confusable")
        else:
            cats.append("sign_differs")
        per_word.append(dict(word=w, in_vocab=w in vocab, categories=cats))
    res = dict(input=str(path), intended=intended or None, heard=[s["nearest"] if s["status"] == "accepted" else f"[?]≈{s['nearest']}" for s in segs],
               sentence=r.get("sentence"), hand_tracking=ht, segments=rows, per_word=per_word, timings_ms=r["timings_ms"])
    write_json(out / "result.json", res)
    plot(raw, r, out / "tracking.png")
    md = [f"# Diagnosis — {name}", "", f"* intended (yours, not a verified label): **{intended or '–'}**", f"* heard: {' · '.join(res['heard']) or '–'}",
          f"* sentence: {res['sentence'] or '(withheld: not confident)'}",
          f"* hand tracking (raised-hand frames with all 21 points): left {ht.get('left', {}).get('hit')}, right {ht.get('right', {}).get('hit')}, "
          f"longest gap {max(ht.get('left', {}).get('max_gap', 0), ht.get('right', {}).get('max_gap', 0))} frames, {ht.get('fps')} fps", "",
          "![tracking](tracking.png)", "", "| # | time (s) | status | P(correct) | top-5 | nearest reference clips |", "|---|---|---|---|---|---|"]
    for i, x in enumerate(rows, 1):
        md.append(f"| {i} | {x['t0']:.2f}–{x['t1']:.2f} | {x['status']} | {x['p_correct']:.2f} | " + ", ".join(f"{w} {s:.2f}" for w, s in x["top5"]) + " | "
                  + "; ".join(f"{n['word']} ({n['dataset']}, {n['signer']}, {n['sim']:.2f})" for n in x["nearest_refs"]) + " |")
    if per_word:
        md += ["", "| intended word | in vocabulary | diagnosis |", "|---|---|---|"] + [f"| {p['word']} | {p['in_vocab']} | {', '.join(p['categories'])} |" for p in per_word]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--intended", default="")
    ap.add_argument("--vocab", default="conversation")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "result_reporting" / "diagnose"))
    a = ap.parse_args()
    from modules.pipeline import SignPipeline
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval import _canon
    pipe = SignPipeline(backend="onnx", device="cpu", vocab=a.vocab, use_llm=False)
    canon = _canon()
    out = Path(a.out)
    files = sorted(Path(a.input).glob("*.json.gz")) if a.all else [Path(a.input)]
    summary = []
    for f in files:
        res = diagnose_one(pipe, f, a.intended, out, canon)
        summary.append(dict(input=f.name, heard=" ".join(res["heard"]), left=res["hand_tracking"].get("left", {}).get("hit"),
                            right=res["hand_tracking"].get("right", {}).get("hit"), fps=res["hand_tracking"].get("fps")))
        log.info("%s → %s | hands L %s R %s", f.name, summary[-1]["heard"], summary[-1]["left"], summary[-1]["right"])
    if len(summary) > 1:
        write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()
