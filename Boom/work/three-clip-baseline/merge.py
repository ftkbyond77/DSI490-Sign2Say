import sys,os,json,gzip
from pathlib import Path
os.environ['ORT_THREADS']='2'
R=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say');sys.path.insert(0,str(R))
import numpy as np
from modules.pipeline import SignPipeline
from modules.mediapipe_pose import from_browser
from modules.parts import prep,to_iso
from modules.runtime import table_from_Z
from modules.decode import active_from_prob
from modules.segment import window_list
O=Path('outputs/merge-experiment');p=SignPipeline(model_dir=R/'models',backend='onnx',device='cpu',vocab='conversation',use_llm=False)
d=json.load(gzip.open('outputs/three-clip-baseline/clip2_browser.json.gz','rt'));raw=from_browser(d['frames'],(d['height'],d['width']));kp,sc,hw,present,mirrored=p.standardise(raw);kpi=to_iso(kp,hw);prob,H=p._tag(kpi,sc,present);assert p.tempo_factor(prob)==1
spans=[(36,68),(68,112),(36,112)];student=p.span(H,spans);teacher=p.be.embed([prep(kpi[a:b],sc[a:b]) for a,b in spans],bs=1)
act,_=active_from_prob(prob,p.spot['act_thr'],present);wins=window_list(len(kpi),p.spot['lens'],p.spot['stride'],act=act,min_active=p.spot.get('min_inside',.6));table=table_from_Z(p.span(H,wins),wins,p.bank,p.null_proto);W=np.asarray(wins);L=W[:,1]-W[:,0];target=p.concepts.index('ภาษามือ');rows=[]
for i,(a,b) in enumerate(spans):
 for method,z in [('span_head',student[i:i+1]),('exact_encoder',teacher[i:i+1])]:
  s=p.bank.scores(z)[0];ids=np.argsort(-s)[:5];rows.append(dict(frames=[a,b],seconds=[a/25,b/25],method=method,top5=[dict(word=p.concepts[int(j)],cosine=float(s[j])) for j in ids],target_rank=int((s>s[target]).sum()+1),target_score=float(s[target]),p_correct=None))
 # Same candidate aggregation rule as decode, with a manually specified span instead of DP-selected boundaries.
 mask=(W[:,0]>=a-2)&(W[:,1]<=b+2)&(L>=p.spot['min_len']);ids=table['top_idx'][mask,:5].ravel();vals=table['top_sc'][mask,:5].ravel();uni,inv=np.unique(ids,return_inverse=True);acc=np.zeros(len(uni));np.add.at(acc,inv,vals);best=np.full(len(uni),-9.);np.maximum.at(best,inv,vals);order=np.argsort(-acc);top=order[:5];pos=np.flatnonzero(uni==target)
 rows.append(dict(frames=[a,b],seconds=[a/25,b/25],method='temporal_ensemble_forced_span',windows=int(mask.sum()),top5=[dict(word=p.concepts[int(uni[j])],peak_cosine=float(best[j]),support=float(acc[j]/mask.sum())) for j in top],target_rank=int(np.flatnonzero(order==pos[0])[0])+1 if len(pos) else None,p_correct=None))
result=dict(note='Development-only forced-boundary diagnostic, not a general merge-rule fix. No threshold or production changes. Clip3 untouched. No recalibrated p for changed spans.',results=rows,mirrored=bool(mirrored))
(O/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
