import gzip,json,glob,sys,numpy as np
sys.stdout.reconfigure(encoding='utf-8')
def up(f,k):
    p=f['pose']
    if len(p)<17: return False
    ls,rs=p[11],p[12]; sw=np.hypot(ls[0]-rs[0],ls[1]-rs[1]) or 0.2; mid=(ls[1]+rs[1])/2
    def h(hand): return len(hand)==21 and np.mean([q[1] for q in hand])<mid+k*sw
    if k==0.8: return h(f['lh']) or h(f['rh'])
    def w(i): return p[i][3]>0.5 and p[i][1]<mid+1.3*sw
    return h(f['lh']) or h(f['rh']) or w(15) or w(16)
rows=[]
for fn in sorted(glob.glob('cache/live_requests/*.json.gz')):
    r=json.load(gzip.open(fn,'rt',encoding='utf-8'))
    fr=r['frames']
    if len(fr)<5: continue
    t=np.array([f['t'] for f in fr]); t=t-t[0]
    raised=[up(f,0.8) for f in fr]; upp=[up(f,1.5) for f in fr]
    hands=[len(f['lh'])==21 or len(f['rh'])==21 for f in fr]
    li=t[raised.index(True)] if any(raised) else np.nan
    lo=t[-1]-t[len(upp)-1-upp[::-1].index(True)] if any(upp) else np.nan
    segs=[s for s in (r['result'].get('segments') or []) if s.get('status')!='transition']
    # down-gaps inside utterance (between first raised and last up) 
    gaps=[];g0=None
    for i,u in enumerate(upp):
        if not u and g0 is None: g0=t[i]
        if u and g0 is not None:
            if any(raised[:i]): gaps.append(t[i]-g0)
            g0=None
    dt=np.diff(t)
    rows.append(dict(fn=fn[-27:],date=r['time'][:10],live=('perf_mode' in r),mode=r.get('perf_mode'),n=len(fr),dur=t[-1]/1000,fps=r.get('fps_actual'),
        li=li,lo=lo,handfrac=float(np.mean(hands)),nseg=len(segs),acc=sum(s['status']=='accepted' for s in segs),
        unk=sum(s['status']!='accepted' for s in segs),p=[s.get('p_correct',0) for s in segs],maxgap=max(gaps) if gaps else 0,
        jit=float(np.percentile(dt,95)/max(np.median(dt),1e-6)),mv=r['result'].get('model_version'),
        tm=r['result'].get('timings_ms')))
json.dump(rows,open(sys.argv[1],'w'),default=float)
import collections
print('n',len(rows),'perf_mode-tagged',sum(r['live'] for r in rows))
print('dates',collections.Counter(r['date'][:7] for r in rows))
# heuristic live: lead-in<=0.75s and lead-out<=0.5s
for r in rows: r['h_live']= (r['li']<=750) and (r['lo']<=520)
print('tagged live: heuristic agreement', np.mean([r['h_live'] for r in rows if r['live']]))
print('untagged heuristic-live frac', np.mean([r['h_live'] for r in rows if not r['live']]))
def summ(name,rs):
    if not rs: return
    P=np.concatenate([r['p'] for r in rs]) if any(r['p'] for r in rs) else np.array([0])
    print(f"{name:28s} n={len(rs):4d} dur={np.median([r['dur'] for r in rs]):5.1f}s fps={np.median([r['fps'] for r in rs if r['fps']]):5.1f} "
          f"p10fps={np.percentile([r['fps'] for r in rs if r['fps']],10):5.1f} leadin={np.nanmedian([r['li'] for r in rs]):6.0f} leadout={np.nanmedian([r['lo'] for r in rs]):6.0f} "
          f"handfrac={np.mean([r['handfrac'] for r in rs]):.2f} seg/utt={np.mean([r['nseg'] for r in rs]):.2f} acc_rate={sum(r['acc'] for r in rs)/max(1,sum(r['nseg'] for r in rs)):.3f} "
          f"meanP={P.mean():.3f} empty={np.mean([r['nseg']==0 for r in rs]):.2f} maxgap>0.6s={np.mean([r['maxgap']>600 for r in rs]):.2f} jit={np.median([r['jit'] for r in rs]):.2f}")
summ('ALL',rows)
summ('tagged live',[r for r in rows if r['live']])
summ('heur live',[r for r in rows if r['h_live']])
summ('heur manual',[r for r in rows if not r['h_live']])
for m in ('2026-10','2026-11'):
    summ(m+' heur live',[r for r in rows if r['h_live'] and r['date'].startswith(m)])
    summ(m+' heur manual',[r for r in rows if not r['h_live'] and r['date'].startswith(m)])
print(collections.Counter(r['mv'] for r in rows))
print(collections.Counter(r['mode'] for r in rows))
