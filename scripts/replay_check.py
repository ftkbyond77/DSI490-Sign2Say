"""Replay saved live requests (cache/live_requests, landmarks only) through a model folder offline and score them against the words the
user says they signed. A real-world check reported next to the held-out tests — never used to choose settings.

  python scripts/replay_check.py --models models [--models2 cache/models_v8_backup] [--vocab conversation]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# recordings whose intended words the user confirmed (2026-10-09 webcam session: "สวัสดี ฉัน ชื่อ J", then single words)
LABELS = {
    "20261009_014043_845": ["สวัสดี", "ฉัน", "ชื่อ", "J"],
    "20261009_014121_227": ["สวัสดี", "ฉัน", "ชื่อ", "J"],
    "20261009_014145_707": ["สวัสดี", "ฉัน", "ชื่อ", "J"],
    "20261009_014907_828": ["ชื่อ"],
}


def run(models, vocab):
    from modules.pipeline import SignPipeline
    pipe = SignPipeline(model_dir=models, backend="onnx", device="cpu", vocab=vocab, use_llm=False, use_face=False)
    rows = []
    for name, words in LABELS.items():
        f = ROOT / "cache" / "live_requests" / f"{name}.json.gz"
        if not f.exists():
            continue
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
        raw = pipe.landmarks_from_browser(d["frames"], (d["height"], d["width"]))
        r = pipe.run_landmarks(raw, alphabet=d.get("alphabet", "en"))
        segs = [s for s in r["segments"] if s["status"] != "transition"]
        got = [s["nearest"] for s in segs]
        top1 = sum(w in got for w in words)
        top5 = sum(any(w in [c["word"] for c in s["candidates"][:5]] for s in segs) for w in words)
        p_true = [round(max([s.get("p_correct", 0) for s in segs if s["nearest"] == w] or [0]), 2) for w in words]
        acc = [s["nearest"] for s in segs if s["status"] == "accepted"]
        prob = [s["nearest"] for s in segs if s.get("probable") and s["status"] != "accepted"]
        rows.append(dict(name=name, words=words, read=got, top1=top1, top5=top5, n=len(words), p_true=p_true, accepted=acc, probable=prob,
                         wrong_accepted=[w for w in acc if w not in words], wrong_probable=[w for w in prob if w not in words],
                         sentence=r.get("sentence", ""), tentative=(r.get("fusion") or {}).get("tentative_sentence_th", ""),
                         hands=r.get("hand_tracking")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--vocab", default="conversation")
    a = ap.parse_args()
    rows = run(a.models, a.vocab)
    n = sum(r["n"] for r in rows)
    for r in rows:
        print(json.dumps({k: r[k] for k in ("name", "read", "p_true", "accepted", "probable", "sentence", "tentative")}, ensure_ascii=False))
    print(json.dumps(dict(models=a.models, vocab=a.vocab, top1=f"{sum(r['top1'] for r in rows)}/{n}", top5=f"{sum(r['top5'] for r in rows)}/{n}",
                          accepted_correct=sum(len(r["accepted"]) - len(r["wrong_accepted"]) for r in rows),
                          accepted_wrong=sum(len(r["wrong_accepted"]) for r in rows),
                          probable_correct=sum(len(r["probable"]) - len(r["wrong_probable"]) for r in rows),
                          probable_wrong=sum(len(r["wrong_probable"]) for r in rows)), ensure_ascii=False))


if __name__ == "__main__":
    main()
