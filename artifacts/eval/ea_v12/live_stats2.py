import gzip,json,glob,sys,numpy as np,collections
sys.stdout.reconfigure(encoding='utf-8')
rows=[x for x in json.load(open(sys.argv[1])) if x['fps']]
r=json.load(gzip.open(sorted(glob.glob('cache/live_requests/*.json.gz'))[-1],'rt',encoding='utf-8'))
print('seg keys',list((r['result'].get('segments') or [{}])[0].keys())); print('timings',r['result'].get('timings_ms'))
# reload p
pk={}
for fn in sorted(glob.glob('cache/live_requests/*.json.gz')):
    r=json.load(gzip.open(fn,'rt',encoding='utf-8'))
    segs=[s for s in (r['result'].get('segments') or []) if s.get('status')!='transition']
    pk[fn[-27:]]=dict(p=[s.get('p',s.get('p_correct',0)) or 0 for s in segs],st=[s['status'] for s in segs],sdur=[s.get('t1',0)-s.get('t0',0) for s in segs],
        hf=[s.get('hand_frac',1) for s in segs],w=[s.get('word') for s in segs])
for x in rows:
    x.update(pk[x['fn']]); x['h_live']=(x['li']<=750) and (x['lo']<=520)
    x['cat']='upload' if x['jit']<1.25 else ('live' if (x['live'] or x['h_live']) else 'cam-manual')
print(collections.Counter(x['cat'] for x in rows)); print(collections.Counter(s for x in rows for s in x['st']))
def summ(name,rs):
    if not rs: print(name,'n=0'); return
    P=np.array([p for r in rs for p in r['p']]); D=np.array([d for r in rs for d in r['sdur']]); ns=sum(r['nseg'] for r in rs)
    print(f"{name:26s} n={len(rs):4d} dur={np.median([r['dur'] for r in rs]):5.1f} fps med={np.median([r['fps'] for r in rs]):5.1f} p10={np.percentile([r['fps'] for r in rs],10):5.1f} "
      f"li={np.nanmedian([r['li'] for r in rs]):5.0f} lo={np.nanmedian([r['lo'] for r in rs]):5.0f} hand={np.mean([r['handfrac'] for r in rs]):.2f} seg/utt={ns/len(rs):.2f} seg/s={ns/sum(r['dur'] for r in rs):.2f} "
      f"acc={sum(r['acc'] for r in rs)/max(1,ns):.3f} P med={np.median(P) if len(P) else 0:.3f} P>=.88={np.mean(P>=.88) if len(P) else 0:.3f} P>=.6={np.mean(P>=.6) if len(P) else 0:.3f} segdur={np.median(D) if len(D) else 0:.2f} none_acc_utt={np.mean([r['acc']==0 for r in rs]):.2f} gap>.6={np.mean([r['maxgap']>600 for r in rs]):.2f}")
for c in ('upload','cam-manual','live'):
    summ(c,[x for x in rows if x['cat']==c])
    for m in ('2026-10','2026-11'): summ(f'  {c} {m}',[x for x in rows if x['cat']==c and x['date'].startswith(m)])
print('--- v10 only (model_version v10)')
for c in ('upload','cam-manual','live'): summ(c+' v10',[x for x in rows if x['cat']==c and x['mv']=='v10'])
print('--- live by fps bucket (v10)')
L=[x for x in rows if x['cat']=='live' and x['mv']=='v10']
for a,b in ((0,11),(11,12.5),(12.5,14),(14,99)): summ(f'fps {a}-{b}',[x for x in L if a<=x['fps']<b])
print('--- live by duration'); 
for a,b in ((0,2.5),(2.5,5),(5,9),(9,99)): summ(f'dur {a}-{b}',[x for x in L if a<=x['dur']<b])
print('--- live by handfrac')
for a,b in ((0,.4),(.4,.7),(.7,1.01)): summ(f'hand {a}-{b}',[x for x in L if a<=x['handfrac']<b])
print('--- cam-manual v10 by fps'); M=[x for x in rows if x['cat']=='cam-manual' and x['mv']=='v10']
for a,b in ((0,12.5),(12.5,16),(16,99)): summ(f'fps {a}-{b}',[x for x in M if a<=x['fps']<b])
json.dump(rows,open(sys.argv[1],'w'),default=float)
