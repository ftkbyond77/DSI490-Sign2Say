"""Speech input, measured through the deployed stack (v2, result_v2.md §7). Same audio for every configuration:

  dev      the 96 Thai everyday sentences of tests/conversation.tsv, VachanaTTS female voice (v1's set, used to tune v2)
  test-A   the 73 held-out Thai sentences of tests/language_heldout.tsv, VachanaTTS MALE voice (never used to tune)
  test-B   20 long utterances (3–5 test-A clips joined with 0.35–0.8 s pauses, 5–12 s)
  FLEURS   100 utterances of Google FLEURS Thai (dev split): REAL human read speech, long sentences
CER after removing spaces / punctuation and spelling variants (เท่าไหร่ = เท่าไร), digits read as Thai words.
Synthetic speech is clean and read-style; FLEURS is real but read, not conversational — a microphone in a room is harder still.

Through the API (docker stack on :8081): /api/asr (one request) and /api/asr/stream (the audio is sent every second while it is
"spoken", in real time, then final=1) → accuracy, the latency the person waits after they stop, and how many short-clip
hallucinations the reliability check stops.
"""
from __future__ import annotations

import io
import json
import re
import time
import wave

import numpy as np
import pandas as pd

from .. import paths

VAR = {"เท่าไหร่": "เท่าไร", "เมื่อไหร่": "เมื่อไร"}


def _num(d):
    try:
        from pythainlp.util import num_to_thaiword
        return num_to_thaiword(int(d))
    except Exception:  # noqa: BLE001
        return d


def norm(s):
    for a, b in VAR.items():
        s = s.replace(a, b)
    s = re.sub(r"\d+", lambda m: _num(m.group()), s)
    return re.sub(r"[\s\.\,\?\!\"'“”\-ๆ…]+", "", s)


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def sets():
    import csv
    out = {}
    rows = [r for r in csv.DictReader(open(paths.HERE / "tests" / "conversation.tsv", encoding="utf-8"), delimiter="\t") if r["lang"] == "th"]
    out["dev"] = [(f"th_{i:03d}", paths.WORK / "eval" / "asr_th" / f"th_{i:03d}.wav", r["text"]) for i, r in enumerate(rows)]
    meta = json.loads((paths.WORK / "eval" / "asr_test" / "meta.json").read_text(encoding="utf-8"))
    out["test_a"] = [(m["id"], paths.WORK / "eval" / "asr_test" / f"{m['id']}.wav", m["text"]) for m in meta["short"]]
    out["test_b"] = [(m["id"], paths.WORK / "eval" / "asr_test" / f"{m['id']}.wav", m["text"]) for m in meta["long"]]
    return out


def _wav_prefix(f, seconds):
    """the first `seconds` of a wav file as wav bytes (what a recorder would have produced by then)."""
    with wave.open(str(f)) as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        k = min(n, int(sr * seconds))
        data = w.readframes(k)
    b = io.BytesIO()
    with wave.open(b, "wb") as o:
        o.setnchannels(ch); o.setsampwidth(sw); o.setframerate(sr); o.writeframes(data)
    return b.getvalue(), n / sr


def via_api(items, base="http://localhost:8081", stream=False):
    import httpx
    rows = []
    with httpx.Client(timeout=120) as c:
        for iid, f, ref in items:
            if not stream:
                t0 = time.time()
                r = c.post(base + "/api/asr", files={"audio": (f.name, f.read_bytes(), "audio/wav")}, data={"lang": "th"}).json()
                wait = time.time() - t0
                partials = 0
            else:
                _, dur = _wav_prefix(f, 1e9)
                sess, partials, t_start = "", 0, time.time()
                for s in np.arange(1.0, dur, 1.0):                   # one request per second of speech, in real time
                    while time.time() - t_start < s:
                        time.sleep(0.01)
                    data, _ = _wav_prefix(f, s)
                    p = c.post(base + "/api/asr/stream", files={"audio": ("a.wav", data, "audio/wav")},
                               data={"lang": "th", "session": sess, "final": "0"}).json()
                    sess = p["session"]; partials += bool(p.get("committed") or p.get("tentative"))
                while time.time() - t_start < dur:
                    time.sleep(0.01)
                t0 = time.time()
                r = c.post(base + "/api/asr/stream", files={"audio": ("a.wav", f.read_bytes(), "audio/wav")},
                           data={"lang": "th", "session": sess, "final": "1"}).json()
                wait = time.time() - t0
            a, b = norm(ref), norm(r.get("text", ""))
            rows.append(dict(id=iid, ref=ref, hyp=r.get("text", ""), cer=round(lev(a, b) / max(len(a), 1), 3), err=lev(a, b), n=len(a),
                             wait_ms=round(wait * 1000), reliable=r.get("reliable"), flags="; ".join(r.get("flags") or []),
                             suggestions=len(r.get("suggestions") or []), partials=partials, mode=r.get("mode")))
    d = pd.DataFrame(rows)
    bad = d.cer > 0.3
    return d, dict(cer=round(float(d.err.sum() / d.n.sum()), 4), exact=int((d.cer == 0).sum()), n=len(d),
                   wait_p50_ms=int(d.wait_ms.median()), wait_p95_ms=int(d.wait_ms.quantile(.95)),
                   wrong_flagged=int((bad & ~d.reliable.astype(bool)).sum()), wrong=int(bad.sum()),
                   right_flagged=int((~bad & ~d.reliable.astype(bool)).sum()),
                   cer_if_flagged_retyped=round(float(d[d.reliable.astype(bool)].err.sum() / d.n.sum()), 4))
