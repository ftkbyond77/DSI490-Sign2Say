from pathlib import Path
import json,hashlib,cv2
R=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say');O=Path('outputs/three-clip-baseline');files=sorted((R/'คุยกันภาษามือ').glob('*.MOV'));assert len(files)==3
rows=[]
for i,p in enumerate(files):
 cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4));role='development' if i<2 else 'heldout';r=dict(id=f'clip{i+1}',file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fps=fps,frames=n,width=w,height=h,duration_s=n/fps,split=role)
 if role=='development':
  ims=[]
  for fraction in [.15,.35,.55,.75]:
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(n*fraction));ok,im=cap.read()
   if ok:ims.append(cv2.resize(im,(int(240*w/h),240)))
  if ims:cv2.imwrite(str(O/(r['id']+'_frames.jpg')),cv2.hconcat(ims))
 cap.release();rows.append(r)
manifest=dict(selection_rule='lexicographic filename order: first two development, third heldout; fixed before any inference',same_signer_user_reported=True,reference_text_user_reported='สวัสดีค่ะ คุยกันภาษามือ',reference_gloss_verified=False,clips=rows)
(O/'split.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
f={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((R/'models').glob('*.json'))+[R/'models/onnx/encoder.onnx',R/'models/onnx/tagger.onnx',R/'models/bank.npz']}
(O/'model_fingerprints.json').write_text(json.dumps(f,indent=2));print(json.dumps(manifest,ensure_ascii=False,indent=2))
