"""System performance through the deployed stack (v2, result_v2.md §8): latency per stage for short and long sentences, under
concurrency, memory per container, first-load bytes. Every request uses a sentence the API has not cached yet."""
from __future__ import annotations

import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import paths


def _texts():
    conv = [r["text"] for r in csv.DictReader(open(paths.HERE / "tests" / "conversation.tsv", encoding="utf-8"), delimiter="\t")
            if r["lang"] == "th"]
    held = [r for r in csv.DictReader(open(paths.HERE / "tests" / "language_heldout.tsv", encoding="utf-8"), delimiter="\t")]
    return conv, [r["text"] for r in held if r["category"] == "long"], [r["text"] for r in held if r["category"] == "en"]


def _one(c, base, text, grammar="tsl51", lang="auto"):
    t0 = time.time()
    r = c.post(base + "/api/translate", json={"text": text, "grammar": grammar, "lang": lang})
    wall = time.time() - t0
    d = r.json()
    C = d["ir"]["confidence"]
    return dict(wall_ms=wall * 1000, language_ms=C.get("t_language_ms"), motion_ms=C.get("t_motion_ms"),
                total_ms=C.get("t_total_ms"), kb=len(r.content) / 1024, seconds=C.get("duration_s"))


def _stats(rows, k):
    x = np.array([r[k] for r in rows if r.get(k) is not None], float)
    return dict(p50=round(float(np.median(x)), 1), p95=round(float(np.percentile(x, 95)), 1))


def run(base="http://localhost:8081", grammar_tag="thai"):
    import httpx
    conv, longs, en = _texts()
    out = {}
    with httpx.Client(timeout=60) as c:
        # the API keeps an LRU cache: the Normal-mode variant of each sentence has not been requested yet
        short = [_one(c, base, t, grammar_tag) for t in conv[:60]]
        long_ = [_one(c, base, t, grammar_tag) for t in longs]
        eng = [_one(c, base, t, grammar_tag, "en") for t in en]
        out["short"] = {k: _stats(short, k) for k in ("wall_ms", "language_ms", "motion_ms", "kb", "seconds")}
        out["long"] = {k: _stats(long_, k) for k in ("wall_ms", "language_ms", "motion_ms", "kb", "seconds")}
        out["english"] = {k: _stats(eng, k) for k in ("wall_ms", "language_ms", "motion_ms")}

    def worker(ts):
        with httpx.Client(timeout=60) as c:
            return [_one(c, base, t, "tsl51") for t in ts]
    texts = [t + suffix for suffix in ("", "ครับ", "ค่ะ") for t in conv[60:96]]     # politeness particles: same signs, new cache keys
    for n in (1, 4, 8):
        chunks = [texts[i::n] for i in range(n)][:n]
        t0 = time.time()
        with ThreadPoolExecutor(n) as ex:
            res = [r for part in ex.map(worker, [ch[:12] for ch in chunks]) for r in part]
        out[f"concurrent_{n}"] = dict(requests=len(res), wall_p50=_stats(res, "wall_ms")["p50"], wall_p95=_stats(res, "wall_ms")["p95"],
                                      throughput_rps=round(len(res) / (time.time() - t0), 1))
        texts = [t + "นะ" for t in texts]
    stats = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.MemUsage}}|{{.CPUPerc}}"], capture_output=True,
                           text=True).stdout
    out["memory"] = {l.split("|")[0]: l.split("|")[1] for l in stats.splitlines() if "reverse" in l}
    sizes = {}
    with httpx.Client(timeout=60) as c:
        html = c.get(base + "/").text
        for a in __import__("re").findall(r'(?:src|href)="(/assets/[^"]+)"', html):
            sizes[a] = len(c.get(base + a, headers={"Accept-Encoding": "gzip"}).content)
    out["first_load_app_kb"] = round(sum(sizes.values()) / 1024, 1)
    for tag, path in (("avatar_default", "/avatar.vrm"),):
        r = subprocess.run(["curl", "-s", "-o", "NUL" if __import__("os").name == "nt" else "/dev/null", "-w", "%{size_download}",
                            "--compressed", base + path], capture_output=True, text=True).stdout
        out[f"{tag}_wire_kb"] = round(float(r or 0) / 1024, 1)
    (paths.WORK / "eval").mkdir(parents=True, exist_ok=True)
    (paths.WORK / "eval" / "perf.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out
