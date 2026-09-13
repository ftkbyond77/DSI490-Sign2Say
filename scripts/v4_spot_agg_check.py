import sys, pickle
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import v4_build as vb
from modules.pipeline_v4 import load_bank
from modules.spotting import decode
full = load_bank(); keep = [m['signer'] != 'S_ttrs_00' for m in full.meta]
vocab = sorted(set(w for w, k in zip(full.words, keep) if k)); vs = set(vocab)
data = pickle.load(open('artifacts/v4/spot_eval_cache_S_ttrs_00_120.pkl', 'rb'))
def aggregate(table, a, b, mode, slack=2, ml=10):
    acc = {}
    for w, (x, y) in enumerate(table['wins']):
        if x >= a - slack and y <= b + slack and y - x >= ml:
            for r, (j, s) in enumerate(zip(table['top_idx'][w], table['top_sc'][w])):
                if mode == 'vote': acc[j] = acc.get(j, 0) + (5 - r)
                elif mode == 'sum': acc[j] = acc.get(j, 0) + s
                elif mode == 'max': acc[j] = max(acc.get(j, -9), s)
    return [vocab[j] for j, _ in sorted(acc.items(), key=lambda x: -x[1])[:5]]
for split in ('tune', 'test'):
    res = {m: [0, 0] for m in ('single', 'vote', 'sum', 'max')}; n = 0
    for r in data[split]:
        segs = decode(r['table'], r['act'], vocab, 9.0, 1.0, min_len=14)
        for j, (g, w) in enumerate(zip(r['gt'], r['words'])):
            if w not in vs: continue
            n += 1
            ov = [s for s in segs if vb._iou((s['f0'], s['f1']), g) > 0]
            c = {'single': [[x['word'] for x in s['candidates']] for s in ov]}
            for m in ('vote', 'sum', 'max'):
                c[m] = [aggregate(r['table'], s['f0'], s['f1'], m) for s in ov]
            for m, lists in c.items():
                res[m][0] += any(l and l[0] == w for l in lists); res[m][1] += any(w in l for l in lists)
    print(split, 'n in-vocab', n, {m: (round(v[0]/n, 3), round(v[1]/n, 3)) for m, v in res.items()})
