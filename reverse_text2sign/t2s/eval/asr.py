"""Speech input check (system level): the 96 Thai / 53 English everyday sentences of tests/conversation.tsv spoken by a TTS voice
→ reverse-speech /asr → character error rate (Thai, spelling variants normalised) / word error rate (English).
Thai audio = the forward part's VachanaTTS (inside the compose network); English = Windows SAPI "Zira" (data/eval/asr_en/*.wav).
Synthetic speech is clean and read-style — real microphones, accents and noise are harder: this is an upper bound, not field accuracy.
"""
from __future__ import annotations

import json
import re
import subprocess

import pandas as pd

from .. import paths

TH_SCRIPT = r'''
import csv, json, re, time, httpx
rows = [r for r in csv.DictReader(open("/tmp/conv.tsv", encoding="utf-8"), delimiter="\t") if r["lang"] == "th"]
out = []
for r in rows:
    wav = httpx.post("http://tts:8001/v1/speak", json={"text": r["text"]}, timeout=60).content
    a = httpx.post("http://reverse-speech:8002/asr", files={"audio": ("s.wav", wav, "audio/wav")}, data={"lang": "th"}, timeout=120).json()
    out.append(dict(ref=r["text"], hyp=a["text"], ms=a["ms"], sec=a["seconds"], vad=a.get("vad")))
print(json.dumps(out, ensure_ascii=False))
'''
VAR = {"เท่าไหร่": "เท่าไร", "เมื่อไหร่": "เมื่อไร"}


def _norm_th(s):
    for a, b in VAR.items():
        s = s.replace(a, b)
    s = re.sub(r"\d+", lambda m: _num(m.group()), s)
    return re.sub(r"[\s\.\,\?\!\"'“”\-]+", "", s)


def _num(d):
    try:
        from pythainlp.util import num_to_thaiword
        return num_to_thaiword(int(d))
    except Exception:  # noqa: BLE001
        return d


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def thai():
    root = paths.REPO
    tsv = paths.HERE / "tests" / "conversation.tsv"
    subprocess.run(["docker", "compose", "cp", str(tsv), "reverse-api:/tmp/conv.tsv"], cwd=root, check=True, capture_output=True)
    r = subprocess.run(["docker", "compose", "exec", "-T", "reverse-api", "python", "-"], input=TH_SCRIPT, cwd=root, text=True,
                       capture_output=True, encoding="utf-8", check=True)
    rows = json.loads(r.stdout.strip().splitlines()[-1])
    E = N = 0
    for x in rows:
        a, b = _norm_th(x["ref"]), _norm_th(x["hyp"])
        x["cer"] = round(_lev(a, b) / max(len(a), 1), 3); E += _lev(a, b); N += len(a)
    d = pd.DataFrame(rows)
    return dict(cer=round(E / N, 4), exact=int((d.cer == 0).sum()), n=len(d), median_ms=int(d.ms.median()),
                p95_ms=int(d.ms.quantile(.95)), empty=int((d.hyp.str.strip() == "").sum())), d


def english(base="http://localhost:8081"):
    import httpx
    t = pd.read_csv(paths.HERE / "tests" / "conversation.tsv", sep="\t")
    t = t[t.lang == "en"].reset_index(drop=True)
    rows = []
    for i, r in t.iterrows():
        f = paths.WORK / "eval" / "asr_en" / f"en_{i:03d}.wav"
        a = httpx.post(base + "/api/asr", files={"audio": (f.name, f.read_bytes(), "audio/wav")}, data={"lang": "en"}, timeout=120).json()
        ref = re.sub(r"[^a-z' ]", " ", r.text.lower()).split()
        hyp = re.sub(r"[^a-z' ]", " ", a["text"].lower()).split()
        rows.append(dict(ref=r.text, hyp=a["text"], wer=_lev(ref, hyp) / max(len(ref), 1), errs=_lev(ref, hyp), n=len(ref), ms=a["ms"]))
    d = pd.DataFrame(rows)
    return dict(wer=round(d.errs.sum() / d.n.sum(), 4), exact=int((d.errs == 0).sum()), n=len(d), median_ms=int(d.ms.median())), d


def run():
    out = {}
    th, dth = thai()
    en, den = english()
    out.update(thai=th, english=en)
    (paths.WORK / "eval").mkdir(parents=True, exist_ok=True)
    dth.to_csv(paths.WORK / "eval" / "asr_th.csv", index=False, encoding="utf-8-sig")
    den.to_csv(paths.WORK / "eval" / "asr_en.csv", index=False, encoding="utf-8-sig")
    return out
