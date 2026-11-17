"""Hand-loss statistics inside real live utterances (saved requests) — the dropout pattern a training corruption should replay."""
import gzip, json, glob, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
S = sys.argv[1]
rows = {x["fn"]: x for x in json.load(open(S + "/live_rows.json"))}
out = {"live": [], "cam-manual": [], "upload": []}; runs = {"live": [], "upload": []}; scale = {"live": [], "upload": []}
for fn in sorted(glob.glob("cache/live_requests/*.json.gz")):
    k = fn[-27:]
    if k not in rows or "cat" not in rows[k]:
        continue
    cat = rows[k]["cat"]; r = json.load(gzip.open(fn, "rt", encoding="utf-8")); fr = r["frames"]
    if rows[k]["dur"] > 9 and cat == "live":
        continue                                         # 20 s cap utterances analysed apart
    t = np.array([f["t"] for f in fr]) / 1000.0
    lh = np.array([len(f["lh"]) == 21 for f in fr]); rh = np.array([len(f["rh"]) == 21 for f in fr]); a = lh | rh
    idx = np.where(a)[0]
    if len(idx) < 5:
        continue
    inner = a[idx[0]:idx[-1] + 1]; ti = t[idx[0]:idx[-1] + 1]
    # missing runs (seconds)
    rl = []; i = 0
    while i < len(inner):
        if not inner[i]:
            j = i
            while j < len(inner) and not inner[j]:
                j += 1
            rl.append(ti[min(j, len(ti) - 1)] - ti[i - 1] if i > 0 else 0); i = j
        else:
            i += 1
    out[cat].append(dict(inner_missing=float(1 - inner.mean()), flips=float(np.abs(np.diff(a.astype(int))).sum() / max(t[-1] - t[0], 1e-6)), both=float((lh & rh).mean()),
                         one_only=float((lh ^ rh).mean())))
    if cat in runs:
        runs[cat] += rl
        for f in fr:                                     # hand size / shoulder width (framing)
            p = f["pose"]
            if len(p) < 13:
                continue
            sw = np.hypot((p[11][0] - p[12][0]) * r["width"], (p[11][1] - p[12][1]) * r["height"])
            for h in (f["lh"], f["rh"]):
                if len(h) == 21 and sw > 1:
                    h = np.array(h)[:, :2] * [r["width"], r["height"]]
                    scale[cat].append(float(np.hypot(*(h.max(0) - h.min(0))) / sw))
for c, v in out.items():
    if v:
        print(f"{c:10s} n={len(v):4d} missing-inside-sign={np.mean([x['inner_missing'] for x in v]):.3f} flips/s={np.mean([x['flips'] for x in v]):.2f} both={np.mean([x['both'] for x in v]):.3f} one-only={np.mean([x['one_only'] for x in v]):.3f}")
for c, v in runs.items():
    v = np.array(v)
    if len(v):
        print(c, "missing runs n", len(v), "sec pct 50/75/90/95", np.percentile(v, [50, 75, 90, 95]).round(2), "share >0.3s", float((v > 0.3).mean().round(3)))
for c, v in scale.items():
    v = np.array(v)
    if len(v):
        print(c, "hand diag / shoulder width pct 5/25/50/75/95/99", np.percentile(v, [5, 25, 50, 75, 95, 99]).round(2))
json.dump(dict(runs={k: list(map(float, v)) for k, v in runs.items()}), open(S + "/live_handloss.json", "w"))
