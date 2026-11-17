import json, glob, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
rng = np.random.RandomState(0)
out = {}
for f in sorted(glob.glob(sys.argv[1] + "/live_sim_*.json")):
    r = json.load(open(f, encoding="utf-8")); n = min(len(v) for v in r.values())
    if n < 30:
        continue
    base = "full" if "full" in r else list(r)[0]
    print("\n", f.split("live_sim_")[-1][:70], "n", n, "base", base)
    for c, R in r.items():
        R = R[:n]; B = r[base][:n]
        N = np.array([x["n_ref"] for x in R], float); e = np.array([x["e_top"] for x in R], float); eb = np.array([x["e_top"] for x in B], float)
        es = np.array([x["e_shown"] for x in R], float)
        d = []
        for _ in range(2000):
            i = rng.randint(0, n, n); d.append(e[i].sum() / N[i].sum() - eb[i].sum() / N[i].sum())
        lo, hi = np.percentile(d, [2.5, 97.5])
        dele = sum(max(0, x["n_ref"] - x["n_hyp"]) for x in R) / N.sum()
        ex = ""
        if "n_utt" in R[0]:
            ex = f" utt/clip={np.mean([x['n_utt'] for x in R]):.2f} split={np.mean([x['n_utt'] > 1 for x in R]):.2f} none={np.mean([x['n_utt'] == 0 for x in R]):.2f} kept={np.mean([x['kept_frac'] for x in R]):.2f}"
        if "lat" in R[0]:
            ex = f" delay={np.mean([x['lat'] for x in R if x.get('lat') is not None]):.2f}s changes={np.mean([x['flicker'] for x in R]):.2f} proc={np.mean([x['proc'] for x in R if x.get('proc')]):.0f}ms"
        print(f"  {c:20s} top={e.sum()/N.sum():.3f} d={e.sum()/N.sum()-eb.sum()/N.sum():+.3f} [{lo:+.3f},{hi:+.3f}] shown={es.sum()/N.sum():.3f} hyp/ref={sum(x['n_hyp'] for x in R)/N.sum():.2f} shown/ref={sum(x['n_shown'] for x in R)/N.sum():.2f} prec={sum(x['shown_ok'] for x in R)/max(1,sum(x['n_shown'] for x in R)):.2f}" + ex)
        out[f.split("live_sim_")[-1][:30] + "|" + c] = dict(top=e.sum() / N.sum(), d=e.sum() / N.sum() - eb.sum() / N.sum(), lo=lo, hi=hi, shown=es.sum() / N.sum())
json.dump(out, open(sys.argv[1] + "/sim_ci.json", "w"), default=float)
