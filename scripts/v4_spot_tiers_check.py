import sys, pickle, json
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import v4_build as vb
from modules.pipeline_v4 import load_bank
from modules.spotting import decode
full = load_bank(); keep = [m['signer'] != 'S_ttrs_00' for m in full.meta]
vocab = sorted(set(w for w, k in zip(full.words, keep) if k)); vs = set(vocab)
data = pickle.load(open('artifacts/v4/spot_eval_cache_S_ttrs_00_120.pkl', 'rb'))
cal = json.load(open('artifacts/v4/calibration.json', encoding='utf-8'))
for split in ('tune','test'):
    rows=[]
    for r in data[split]:
        segs = decode(r['table'], r['act'], vocab, 9.0, 1.0, min_len=14)
        for s in segs:
            j, v = max(((j, vb._iou((s['f0'], s['f1']), g)) for j, g in enumerate(r['gt'])), key=lambda x: x[1])
            w = r['words'][j]
            rows.append(dict(z=s['z'], score=s['score'], margin=s['margin'], inv=w in vs, correct=s['nearest']==w, top5=w in [c['word'] for c in s['candidates']]))
    d = pd.DataFrame(rows); iv = d[d.inv]
    print(split, 'n', len(d), 'in-vocab', len(iv), 'top1 acc in-vocab', iv.correct.mean().round(3), 'top5', iv.top5.mean().round(3))
    acc = (d.score >= cal['accept']['tau_score']) & (d.margin >= cal['accept']['tau_margin'])
    unc = ~acc & (d.score >= cal['uncertain']['tau_score'])
    for name, m in (('accepted', acc), ('uncertain', unc), ('unknown', ~acc & ~unc)):
        print('  E1-calib', name, 'share', m.mean().round(3), 'in-vocab prec', d[m & d.inv].correct.mean().round(3), 'top5', d[m & d.inv].top5.mean().round(3), 'n_inv', int((m & d.inv).sum()), 'oov share', (~d[m].inv).mean().round(2))
    for q in (0.5, 0.7, 0.9):
        t = d.margin.quantile(q); m = d.margin >= t
        print('  margin>=q%.1f'%q, 'in-vocab prec', d[m & d.inv].correct.mean().round(3), 'n_inv', int((m & d.inv).sum()))
