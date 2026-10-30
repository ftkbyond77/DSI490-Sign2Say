import os,sys,json,time
from pathlib import Path
os.environ['MP_HOLISTIC_MODEL']='/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say/models/mediapipe/holistic_landmarker.task'
os.environ['MPLCONFIGDIR']=str(Path('work/three-clip-baseline/mpl').resolve())
sys.path.insert(0,'/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say')
from mediapipe.tasks.python import BaseOptions
original=BaseOptions.__init__
def cpu(self,*a,**kw):kw['delegate']=BaseOptions.Delegate.CPU;original(self,*a,**kw)
BaseOptions.__init__=cpu
import numpy as np
from modules.mediapipe_pose import extract
O=Path('outputs/three-clip-baseline');manifest=json.load(open(O/'split.json'))
for r in manifest['clips']:
 if r['split']!='development':continue
 print('EXTRACT',r['id'],flush=True);start=time.time()
 raw=extract(r['file'],normalize=True,model_path=os.environ['MP_HOLISTIC_MODEL']);assert raw is not None
 np.savez_compressed(O/(r['id']+'_raw.npz'),**raw)
 quality={k:float((np.abs(raw[k]).sum((1,2))>0).mean()) for k in ['lh','rh','face']};quality.update(seconds=time.time()-start,frames=len(raw['t']))
 (O/(r['id']+'_quality.json')).write_text(json.dumps(quality,indent=2));print(quality,flush=True)
