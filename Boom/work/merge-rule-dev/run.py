import sys,json,gzip,os
from pathlib import Path
os.environ['ORT_THREADS']='2'
R=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say');O=Path('outputs/merge-rule-dev');sys.path[:0]=[str(O.resolve()),str(R)]
import numpy as np
from merge_rule import assess,MergeConfig
from modules.pipeline import SignPipeline
from modules.mediapipe_pose import from_browser
from modules.parts import to_iso
from modules.decode import active_from_prob
from modules.segment import window_list
from modules.runtime import table_from_Z
(O/'frozen_config.json').write_text(json.dumps(MergeConfig().__dict__,indent=2))
pipe=SignPipeline(model_dir=R/'models',backend='onnx',device='cpu',vocab='conversation',use_llm=False)
rows=[]
for ident in ['clip1','clip2']:
 # Only development caches are allowed; held-out raw media never opened.
 d=json.load(gzip.open(Path('outputs/three-clip-baseline')/(ident+'_browser.json.gz'),'rt'));raw=from_browser(d['frames'],(d['height'],d['width']));h,w=raw['hw'];scale=np.array([w/h,1.]);shoulders=raw['pose'][:,[11,12],:2]*scale;origin=shoulders.mean(1);width=np.linalg.norm(shoulders[:,0]-shoulders[:,1],axis=-1);good=(raw['pose'][:,[11,12],3]>.5).all(1)&(width>.02)
 hands=np.stack([raw['lh'][:,:,:2],raw['rh'][:,:,:2]],1);obs=(np.abs(hands).sum((2,3))>0)&good[:,None];xy=(hands*scale-origin[:,None,None,:])/np.maximum(width[:,None,None,None],.02)
 base=json.load(open(Path('outputs/three-clip-baseline')/(ident+'_baseline.json')))['segments']
 kp,sc,hw,present,mirrored=pipe.standardise(raw);kpi=to_iso(kp,hw);prob,H=pipe._tag(kpi,sc,present);assert pipe.tempo_factor(prob)==1.
 act,_=active_from_prob(prob,pipe.spot['act_thr'],present);wins=window_list(len(kpi),pipe.spot['lens'],pipe.spot['stride'],act=act,min_active=pipe.spot.get('min_inside',.6));tab=table_from_Z(pipe.span(H,wins),wins,pipe.bank,pipe.null_proto);W=np.asarray(wins);L=W[:,1]-W[:,0]
 for i,(a,b) in enumerate(zip(base,base[1:])):
  verdict=assess(a,b,raw['t'],xy,obs);r=dict(clip=ident,pair=[i+1,i+2],span=[a['t0'],b['t1']],assessment=verdict,reference_status='unverified sign boundary; no human time-aligned gloss labels')
  sweep=[]
  for speed in [.1,.15,.2]:
   for ratio in [.15,.25,.4]:
    conf=MergeConfig(min_speed=speed,min_boundary_speed_ratio=ratio);sweep.append(dict(min_speed=speed,min_ratio=ratio,merge=assess(a,b,raw['t'],xy,obs,conf)['merge']))
  r['sensitivity']=sweep
  if verdict['merge']:
   aa=round(a['t0']*25);bb=round(b['t1']*25);mask=(W[:,0]>=aa-2)&(W[:,1]<=bb+2)&(L>=pipe.spot['min_len']);ids=tab['top_idx'][mask,:5].ravel();vals=tab['top_sc'][mask,:5].ravel();uni,inv=np.unique(ids,return_inverse=True);sums=np.zeros(len(uni));np.add.at(sums,inv,vals);order=np.argsort(-sums)[:5];r['merged_top5']=[dict(word=pipe.concepts[int(uni[j])],support=float(sums[j]/mask.sum())) for j in order];r['merged_p']=None
  rows.append(r)
(O/'real_dev_pairs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(json.dumps(rows,ensure_ascii=False,indent=2))
