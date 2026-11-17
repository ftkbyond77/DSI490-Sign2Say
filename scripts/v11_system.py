"""v11 system tests (prompt-opt-v11 §4C) — the model as a product, through the real FastAPI app (in-process client, no network).

  python scripts/v11_system.py run --model models/model-v11 --tag v11     # one process per model version → artifacts/reports/v11_system_<tag>.json
  python scripts/v11_system.py compare --old v10 --new v11                 # → artifacts/reports/v11_system_compare.json + pass / fail per test

Runs with an interpreter that has the API's web dependencies (fastapi, httpx, python-multipart) + this repo's runtime (ONNX Runtime,
MediaPipe). THAISLM_SAVE_REQUESTS is forced empty: nothing is written to cache/live_requests. Speech is off (speak=False); llm=False.

Tests: contract (health / info / vocab / translate / translate/video / live WebSocket: response keys) · replay of the owner's saved web
requests (cache/replay_src + every 15th of cache/live_requests; evaluation only) · streaming (live words vs final words, flicker, word
delay) · latency (25 utterances × 3 rounds, alternating order is done by `compare` across two processes run back to back) and load
(4 concurrent requests) · robustness (empty, no hands, one hand, 8 fps, 60 s, left-handed, out-of-frame hands, corrupt frame).
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "webapp" / "api"))
REP = ROOT / "artifacts" / "reports"


def requests_pool(n_live=40):
    rs = sorted((ROOT / "cache" / "replay_src").glob("*.json.gz"))
    live = sorted((ROOT / "cache" / "live_requests").glob("*.json.gz"))
    rs += live[::max(1, len(live) // n_live)][:n_live]
    out = []
    for f in rs:
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
        if len(d.get("frames", [])) >= 10:
            out.append((f.name, d))
    return out


def body(d, **kw):
    t0 = d["frames"][0]["t"] if d["frames"] else 0
    return dict(frames=[{**f, "t": f["t"] - t0} for f in d["frames"]], width=d["width"], height=d["height"], vocab=d.get("vocab", "conversation"),
                llm=False, speak=False, user_bank=[], alphabet=d.get("alphabet", "en"), **kw)


def words(res):
    segs = [s for s in res.get("segments", []) if s["status"] != "transition"]
    return dict(words=[s["word"] for s in segs], status=[s["status"] for s in segs], shown=[s["word"] for s in segs if s["status"] == "accepted"],
                unsure=sum(s["status"] == "uncertain" for s in segs), decision=[s.get("decision") for s in segs])


def mutate(d, kind):
    """robustness variants of one saved request"""
    d = copy.deepcopy(d)
    fr = d["frames"]
    if kind == "empty":
        d["frames"] = []
    elif kind == "no_hands":
        for f in fr:
            for k in ("lh", "rh", "left_hand", "right_hand", "hands"):
                if k in f:
                    f[k] = None if not isinstance(f[k], list) else []
    elif kind == "one_hand":
        for f in fr:
            for k in ("lh", "left_hand"):
                if k in f:
                    f[k] = None if not isinstance(f[k], list) else []
    elif kind == "fps8":
        d["frames"] = fr[::3]
    elif kind == "long60":
        span = fr[-1]["t"] - fr[0]["t"] + 40
        reps = max(1, int(60000 // max(span, 1)))
        d["frames"] = [dict(f, t=f["t"] + i * span) for i in range(reps) for f in fr]
    elif kind == "left_handed":
        def mx(v):
            if isinstance(v, dict) and "x" in v:
                return dict(v, x=1.0 - v["x"])
            if isinstance(v, list):
                return [mx(x) for x in v]
            if isinstance(v, dict):
                return {k: mx(x) for k, x in v.items()}
            return v
        d["frames"] = [mx(f) for f in fr]
    elif kind == "out_of_frame":
        def sh(v):
            if isinstance(v, dict) and "y" in v:
                return dict(v, y=v["y"] + 0.6)
            if isinstance(v, list):
                return [sh(x) for x in v]
            if isinstance(v, dict):
                return {k: sh(x) for k, x in v.items()}
            return v
        d["frames"] = [dict(f, **{k: sh(f[k]) for k in f if k in ("lh", "rh", "left_hand", "right_hand", "hands")}) for f in fr]
    elif kind == "corrupt":
        k = len(fr) // 2
        fr[k] = {"t": fr[k]["t"], "garbage": True}
    return d


def cmd_run(a):
    os.environ["THAISLM_MODELS"] = str(Path(a.model).resolve())
    os.environ["THAISLM_SAVE_REQUESTS"] = ""
    os.environ.setdefault("ORT_DEVICE", "cpu")
    import logging
    logging.disable(logging.WARNING)
    from fastapi.testclient import TestClient
    import app as api
    c = TestClient(api.app)
    out = dict(tag=a.tag, model=a.model, contract={}, replay={}, robustness={}, latency={}, load={}, streaming={})
    pool = requests_pool()
    # ---- contract
    for ep in ("/v1/health", "/v1/info", "/v1/vocab"):
        r = c.get(ep); out["contract"][ep] = dict(status=r.status_code, keys=sorted(r.json().keys()) if r.headers.get("content-type", "").startswith("application/json") and isinstance(r.json(), dict) else None)
    r = c.post("/v1/translate", json=body(pool[0][1]))
    j = r.json()
    out["contract"]["/v1/translate"] = dict(status=r.status_code, keys=sorted(j.keys()), segment_keys=sorted({k for s in j.get("segments", []) for k in s}))
    vid = ROOT / "data_test" / "iphone_test"
    vids = sorted(p for p in (ROOT / "cache" / "iphone_test_mp4").glob("*.mp4"))
    if vids:
        with open(vids[0], "rb") as fh:
            r = c.post("/v1/translate/video?llm=false&speak=false", files=dict(file=(vids[0].name, fh, "video/mp4")))
        jj = r.json() if r.status_code == 200 else {}
        out["contract"]["/v1/translate/video"] = dict(status=r.status_code, keys=sorted(jj.keys()), segment_keys=sorted({k for s in jj.get("segments", []) for k in s}))
    # ---- live WebSocket (+ streaming quality on 10 requests): partial / confirmed vs the final result of /v1/translate
    st = []
    for name, d in pool[:10]:
        b = body(d)
        conf, partial_seen, t_first = [], [], None
        with c.websocket_connect("/v1/live") as ws:
            ws.send_json(dict(hw=[d["height"], d["width"]], vocab="conversation", alphabet=b["alphabet"]))
            for f in b["frames"]:
                ws.send_json(f)
            ws.send_json(dict(end=True))
            while True:
                m = ws.receive_json()
                conf += m.get("confirmed", []); partial_seen.append(tuple(x["word"] for x in m.get("partial", [])))
                if m.get("final"):
                    break
        fin = words(c.post("/v1/translate", json=b).json())
        live_words = [x["word"] for x in conf if x["status"] != "transition"]
        flick = sum(1 for i in range(1, len(partial_seen)) if partial_seen[i] and partial_seen[i - 1] and partial_seen[i][:1] != partial_seen[i - 1][:1])
        from collections import Counter
        ov = sum((Counter(live_words) & Counter(fin["words"])).values())
        agree = 2 * ov / max(len(live_words) + len(fin["words"]), 1) if (live_words or fin["words"]) else 1.0     # word-level F1 live vs final
        st.append(dict(name=name, live=live_words, final=fin["words"], agree=agree, exact=float(live_words == fin["words"]), flicker=flick,
                       delay=[x.get("latency") for x in conf]))
    out["contract"]["/v1/live"] = dict(ok=bool(st), message_keys=["partial", "confirmed", "final"])
    out["streaming"] = dict(n=len(st), agreement=sum(x["agree"] for x in st) / max(len(st), 1), flicker=sum(x["flicker"] for x in st) / max(len(st), 1),
                            delay_mean=float(sum(sum(v for v in x["delay"] if v is not None) for x in st) / max(sum(len(x["delay"]) for x in st), 1)), items=st)
    # ---- replay of saved web requests (evaluation only)
    for name, d in pool:
        t = time.perf_counter(); res = c.post("/v1/translate", json=body(d)).json(); ms = (time.perf_counter() - t) * 1000
        out["replay"][name] = dict(ms=ms, **words(res), dur_s=(d["frames"][-1]["t"] - d["frames"][0]["t"]) / 1000)
    # ---- latency (25 utterances × 3 rounds) and load (4 concurrent)
    lat = []
    for _ in range(3):
        for name, d in pool[:25]:
            t = time.perf_counter(); c.post("/v1/translate", json=body(d)); lat.append((time.perf_counter() - t) * 1000)
    s_ = sorted(lat)
    out["latency"] = dict(n=len(lat), p50=s_[len(s_) // 2], p90=s_[int(0.9 * len(s_))], max=s_[-1])
    errs = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(4) as ex:
        rs = list(ex.map(lambda nd: c.post("/v1/translate", json=body(nd[1])).status_code, pool[:16]))
    errs = sum(x != 200 for x in rs)
    out["load"] = dict(requests=len(rs), concurrent=4, errors=errs, wall_s=time.perf_counter() - t0)
    # ---- robustness
    base = pool[0][1]
    for kind in ("empty", "no_hands", "one_hand", "fps8", "long60", "left_handed", "out_of_frame", "corrupt"):
        try:
            r = c.post("/v1/translate", json=body(mutate(base, kind)))
            jj = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            w = words(jj) if r.status_code == 200 else {}
            out["robustness"][kind] = dict(status=r.status_code, crash=r.status_code >= 500, words=w.get("words"), shown=w.get("shown"))
        except Exception as e:  # noqa: BLE001
            out["robustness"][kind] = dict(status=None, crash=True, error=repr(e)[:200])
    (REP / f"v11_system_{a.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(a.tag, "latency", out["latency"], "load", out["load"], "streaming", {k: v for k, v in out["streaming"].items() if k != "items"})
    print("robustness", {k: (v.get("status"), v.get("crash")) for k, v in out["robustness"].items()})


def cmd_compare(a):
    o = json.loads((REP / f"v11_system_{a.old}.json").read_text(encoding="utf-8"))
    n = json.loads((REP / f"v11_system_{a.new}.json").read_text(encoding="utf-8"))
    res = {}
    # contract: old keys still present, new keys only additive
    cmp_ = {}
    for ep, ov in o["contract"].items():
        nv = n["contract"].get(ep, {})
        lost = sorted(set(ov.get("keys") or []) - set(nv.get("keys") or [])) + sorted(set(ov.get("segment_keys") or []) - set(nv.get("segment_keys") or []))
        added = sorted(set(nv.get("segment_keys") or []) - set(ov.get("segment_keys") or []))
        cmp_[ep] = dict(status_old=ov.get("status"), status_new=nv.get("status"), lost=lost, added=added)
    res["contract"] = dict(rows=cmp_, passed=all(not v["lost"] and v["status_new"] in (200, None) for v in cmp_.values()))
    # replay: shown words, [?], idle (words shown in requests that show nothing in v10 are not judged — no reference); differences listed
    diff = [dict(name=k, old=o["replay"][k]["shown"], new=n["replay"].get(k, {}).get("shown"), old_unsure=o["replay"][k]["unsure"],
                 new_unsure=n["replay"].get(k, {}).get("unsure")) for k in o["replay"] if o["replay"][k]["words"] != n["replay"].get(k, {}).get("words")]
    tot = lambda r, key: sum(len(v[key]) if isinstance(v[key], list) else v[key] for v in r.values())          # noqa: E731
    res["replay"] = dict(n=len(o["replay"]), changed=len(diff), shown_old=tot(o["replay"], "shown"), shown_new=tot(n["replay"], "shown"),
                         unsure_old=tot(o["replay"], "unsure"), unsure_new=tot(n["replay"], "unsure"), diffs=diff)
    res["latency"] = dict(old=o["latency"], new=n["latency"], p50_ratio=n["latency"]["p50"] / o["latency"]["p50"], p90_ratio=n["latency"]["p90"] / o["latency"]["p90"],
                          passed=n["latency"]["p50"] <= 1.05 * o["latency"]["p50"] and n["latency"]["p90"] <= 1.05 * o["latency"]["p90"])
    res["load"] = dict(old=o["load"], new=n["load"], passed=n["load"]["errors"] == 0)
    res["streaming"] = dict(old={k: v for k, v in o["streaming"].items() if k != "items"}, new={k: v for k, v in n["streaming"].items() if k != "items"},
                            passed=n["streaming"]["agreement"] >= o["streaming"]["agreement"] - 1e-9 and n["streaming"]["flicker"] <= o["streaming"]["flicker"] + 1e-9)
    res["robustness"] = dict(new=n["robustness"], passed=not any(v.get("crash") for v in n["robustness"].values()))
    (REP / "v11_system_compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(f"{k:11s} {'PASS' if v.get('passed', True) else 'FAIL'}", {kk: vv for kk, vv in v.items() if kk not in ("diffs", "rows", "new", "old")} if k not in ("contract",) else v["rows"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--tag", required=True)
    c = sub.add_parser("compare"); c.add_argument("--old", default="v10"); c.add_argument("--new", default="v11")
    a = ap.parse_args()
    dict(run=cmd_run, compare=cmd_compare)[a.cmd](a)


if __name__ == "__main__":
    main()
