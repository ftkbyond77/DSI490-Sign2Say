"""Replay real recorded utterances (the web app's diagnostic saves, cache/live_requests/*.json.gz: landmarks only) through the running
service and compare with what was shown at recording time — a real-world regression test with the user's own camera, room and signing.

  python scripts/replay_live.py cache/replay_src [--url http://localhost:8080] [--vocab conversation] [--alphabet en]

Per utterance: duration, fps, hands seen, what the recording showed (old) vs what the service shows now (new), and the latency.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def shown(res):
    segs = [s for s in res.get("segments") or [] if s["status"] != "transition"]
    words = " ".join(s["word"] if s["status"] == "accepted" else "[?]" for s in segs)
    return words, res.get("sentence") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--vocab", default="conversation")
    ap.add_argument("--alphabet", default="en")
    ap.add_argument("--llm", default="true")
    a = ap.parse_args()
    rows = []
    for f in sorted(Path(a.src).glob("*.json.gz")):
        rec = json.load(gzip.open(f, "rt", encoding="utf-8"))
        fr = rec["frames"]
        t = np.array([x["t"] for x in fr]); dur = (t[-1] - t[0]) / 1000 if len(t) > 1 else 0.0
        body = dict(frames=fr, width=rec["width"], height=rec["height"], vocab=a.vocab, llm=a.llm == "true", speak=False, alphabet=a.alphabet)
        req = urllib.request.Request(f"{a.url}/api/v1/translate", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        t0 = time.perf_counter()
        new = json.loads(urllib.request.urlopen(req, timeout=120).read())
        ms = (time.perf_counter() - t0) * 1000
        old_res = rec["result"]
        ow = " ".join((s["word"] if s["status"] == "accepted" else s["word"] + "?") for s in old_res["segments"] if s["status"] != "transition")
        nw, ns = shown(new)
        rows.append(dict(file=f.name[:15], dur=dur, fps=(len(fr) - 1) / max(dur, 1e-3), old=ow, old_sentence=old_res.get("sentence") or "", new=nw, new_sentence=ns,
                         ms=ms, old_ms=(old_res.get("timings_ms") or {}).get("total_ms"), n_conf=sum(s["status"] == "accepted" for s in new["segments"])))
        r = rows[-1]
        print(f'{r["file"]} {dur:4.1f}s | old: {ow[:60]:60s} "{r["old_sentence"]}" | new: {nw:30s} "{ns}" | {ms:5.0f} ms (old {r["old_ms"]})')
    old_ms = [r["old_ms"] for r in rows if r["old_ms"]]
    print(f"\n{len(rows)} utterances · new latency median {np.median([r['ms'] for r in rows]):.0f} ms (old {np.median(old_ms):.0f} ms) · "
          f"utterances with a shown sentence: old {sum(bool(r['old_sentence']) for r in rows)}, new {sum(bool(r['new_sentence']) for r in rows)} · "
          f"confident words shown: {sum(r['n_conf'] for r in rows)}")


if __name__ == "__main__":
    main()
