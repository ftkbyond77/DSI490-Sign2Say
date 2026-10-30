import os,sys,json,gzip
from pathlib import Path
os.environ['ORT_THREADS']='2'
R=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say');sys.path.insert(0,str(R))
import numpy as np
from modules.pipeline import SignPipeline
from modules.mediapipe_pose import from_browser
O=Path('outputs/segment-diagnostics');B=Path('outputs/three-clip-baseline');pipe=SignPipeline(model_dir=R/'models',backend='onnx',device='cpu',vocab='conversation',use_llm=False)
rows=[]
for ident in ['clip1','clip2']:
 d=json.load(gzip.open(B/(ident+'_browser.json.gz'),'rt'));raw=from_browser(d['frames'],(d['height'],d['width']));r=pipe.run_landmarks(raw,tts=False,alphabet='off');prior=json.load(open(B/(ident+'_baseline.json')))
 assert [s['nearest'] for s in r['segments']]==[s['word'] for s in prior['segments']]
 for i,s in enumerate(r['segments'],1):
  selected=(raw['t']>=s['f0']/25)&(raw['t']<s['f1']/25);n=int(selected.sum()); observed={}
  for name in ['lh','rh']:
   valid=np.abs(raw[name][:,:,:2]).sum((1,2))>0;k=int((selected&valid).sum());observed[name]=dict(detected=k,total=n,rate=k/n if n else None)
  z=s['emb'][None];norm=np.linalg.norm(pipe.bank.Z,axis=1);assert np.allclose(norm,1,atol=.003);assert np.allclose(np.linalg.norm(z,axis=1),1,atol=.001)
  direct=pipe.bank.scores(z)[0];cand=[]
  for rank,c in enumerate(s['candidates'][:5],1):
   cand.append(dict(rank=rank,word=c['word'],production_score=c['score'],temporal_support=c['support'],whole_segment_cosine_top2_mean=float(direct[pipe.concepts.index(c['word'])]),candidate_p=None))
  rows.append(dict(clip=ident,segment=i,start_frame_25fps=s['f0'],end_frame_exclusive_25fps=s['f1'],start_seconds=s['t0'],end_seconds=s['t1'],segment_p=s['p_correct'],status=s['status'],observed_before_interpolation_reflection=observed,mirrored=r['mirrored_to_right_hand'],tempo_factor=r['timings_ms'].get('tempo_factor'),top5=cand))
assert len(rows)==5
(O/'segments.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(json.dumps(rows,ensure_ascii=False,indent=2))
lines=['# รายละเอียด 5 ช่วง — baseline เดิม','','เฟรม 0-based ที่ 25 fps ช่วง [start,end) ไม่ใช่เลขเฟรมต้นฉบับ ~30fps; ช่วงเหล่านี้เป็นผล tagger + window/DP decoder + postprocessing ไม่ใช่ขอบเขตจาก tagger ล้วน','','Observed rate: จำนวน browser frames ที่พบ xy มือไม่เป็นศูนย์ / browser frames ที่ timestamp อยู่ในช่วง ก่อน interpolation และ reflection; ซ้าย/ขวาตาม extractor เดิม ไม่ใช่ด้านซ้าย/ขวาของภาพ และไม่ใช่ความแม่นยำของ landmark','','ระบบมี p_correct ต่อ segment เท่านั้น ไม่มี p แยกให้แต่ละอันดับ จึงไม่สร้าง softmax แทน p หรือทำซ้ำค่า p ให้ทั้ง 5 คำ','','production_score คือ score ที่เว็บใช้: ใน config นี้ exact_rescore=false, prior/CSLS เป็นศูนย์ เป็นค่าสูงสุดของคะแนน concept จาก windows ที่ร่วม ensemble (แต่ละ window ใช้ mean ของสอง cosine สูงสุดตาม Bank.scores ซึ่งมี fallback เมื่อมีตัวอย่างเดียว/คะแนนเท่ากัน) อันดับเรียงจาก temporal_support ไม่ใช่ score สูงสุดเดี่ยว; whole-segment cosine เป็น diagnostic คำนวณใหม่จาก span embedding ของทั้งช่วง โดยใช้กติกา Bank.scores เดิม ไม่ได้ใช้เปลี่ยนอันดับเว็บ','']
for r in rows:
 a=r['observed_before_interpolation_reflection']['lh'];b=r['observed_before_interpolation_reflection']['rh']
 lines += [f"## {r['clip']} ช่วง {r['segment']}: {r['start_seconds']:.2f}–{r['end_seconds']:.2f} s",'',f"เฟรม [{r['start_frame_25fps']},{r['end_frame_exclusive_25fps']}) • segment p={r['segment_p']:.6f} • มือซ้าย {a['detected']}/{a['total']} ({a['rate']:.1%}) • มือขวา {b['detected']}/{b['total']} ({b['rate']:.1%})",'','|อันดับ|คำ|Production score|Temporal support|Whole-segment cosine|p รายอันดับ|','|---:|---|---:|---:|---:|---|']
 for c in r['top5']:lines.append(f"|{c['rank']}|{c['word']}|{c['production_score']:.4f}|{c['temporal_support']:.4f}|{c['whole_segment_cosine_top2_mean']:.4f}|ไม่มีในโมเดล|")
 lines.append('')
(O/'REPORT-th.md').write_text('\n'.join(lines))
