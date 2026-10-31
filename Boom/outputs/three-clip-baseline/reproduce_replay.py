import os,sys,json,gzip,time
from pathlib import Path
os.environ['ORT_THREADS']='2';os.environ['MPLCONFIGDIR']=str(Path('work/three-clip-baseline/mpl').resolve())
R=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say');sys.path.insert(0,str(R))
import numpy as np
from modules.pipeline import SignPipeline
from modules.mediapipe_pose import from_browser
O=Path('outputs/three-clip-baseline');md=O/'models_add_talk';md.mkdir(exist_ok=True)
for p in (R/'models').iterdir():
 dest=md/p.name
 if p.name=='vocab_conversation.json':
  d=json.load(open(p));assert 'คุย' not in d['concepts'];d['concepts'].append('คุย');dest.write_text(json.dumps(d,ensure_ascii=False,indent=2))
 elif not dest.exists():dest.symlink_to(p.resolve(),target_is_directory=p.is_dir())
summary=[]
for tag,model in [('baseline',R/'models'),('add_talk_only',md)]:
 pipe=SignPipeline(model_dir=model,backend='onnx',device='cpu',vocab='conversation',use_llm=False)
 for ident in ['clip1','clip2']:
  data=json.load(gzip.open(O/(ident+'_browser.json.gz'),'rt'));assert data['n_user_signs']==0
  raw=from_browser(data['frames'],(data['height'],data['width']));start=time.time()
  result=pipe.run_landmarks(raw,title=ident,tts=False,alphabet='off')
  clean=dict(clip=ident,variant=tag,threshold=pipe.cal['tau_accept'],sentence=result['sentence'],mirrored=result['mirrored_to_right_hand'],seconds=time.time()-start,segments=[dict(t0=s['t0'],t1=s['t1'],word=s['nearest'],p=float(s['p_correct']),status=s['status'],top5=[c['word'] for c in s['candidates'][:5]]) for s in result['segments']],coverage={k:float((np.abs(raw[k]).sum((1,2))>0).mean()) for k in ['lh','rh']})
  (O/(ident+'_'+tag+'.json')).write_text(json.dumps(clean,ensure_ascii=False,indent=2));summary.append(clean);print(json.dumps(clean,ensure_ascii=False),flush=True)
(O/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
