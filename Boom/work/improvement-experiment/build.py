from pathlib import Path
import json,hashlib,shutil,sys
import numpy as np,pandas as pd
R=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say'); W=Path('/Users/thawaphorn/Documents/Codex/2026-10-07/new-chat'); O=Path('outputs/improvement-experiment').resolve();sys.path.insert(0,str(R))
old='ทั้งหมด (ภาษามือโรงเรียนเศรษฐเสถียรในพระราชูปถัมภ์)';new='ทั้งหมด'
concepts=json.load(open(R/'models/concepts.json'));assert old in concepts and new not in concepts
idx=concepts.index(old);after=concepts.copy();after[idx]=new
assert len(set(after))==len(after)
base=O/'baseline';base.mkdir(exist_ok=True);exp=O/'models';exp.mkdir(exist_ok=True)
fingerprints={}
def remap(x):
 if isinstance(x,str):return new if x==old else x
 if isinstance(x,list):return [remap(v) for v in x]
 if isinstance(x,dict):
  out={remap(k):remap(v) for k,v in x.items()};assert len(out)==len(x);return out
 return x
for p in (R/'models').iterdir():
 if p.is_file() and p.suffix=='.json':
  fingerprints[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,base/p.name)
  d=json.load(open(p))
  if p.name in ('concepts.json','vocab_conversation.json'): d=remap(d)
  if p.name=='vocab_conversation.json':
   assert new not in d['concepts'];d['concepts'].append(new)
  (exp/p.name).write_text(json.dumps(d,ensure_ascii=False,indent=2))
 else:
  dest=exp/p.name
  if not dest.exists():dest.symlink_to(p.resolve(),target_is_directory=p.is_dir())
for name in ['bank.npz','encoder.pt']:
 p=R/'models'/name;fingerprints[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
manifest=pd.read_parquet(W/'work/training-data/manifest.parquet');rows=manifest.concept.eq(old);assert rows.any();fixed=manifest.copy();fixed.loc[rows,'concept']=new;fixed.loc[rows,'lemma']=new
# Original annotation is preserved in gloss; no unverified synonym merge.
fixed.to_parquet(O/'manifest.experimental.parquet',index=False)
make=manifest[manifest.concept.eq('ทำ')];make[['clip_id','source_file','pose_file','exclude_reason']].to_json(O/'pose_recovery_queue.json',orient='records',force_ascii=False,indent=2)
clip=Path('/Users/thawaphorn/Pee4/Term1/DSI490/data/videos/14265_ทำ.mp4')
(O/'pose_candidate.json').write_text(json.dumps(dict(path=str(clip),bytes=clip.stat().st_size,sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),expected_manifest_source='th_sl_00936.mp4',expected_inventory_bytes=286460,identity_verified=False,action='do not overwrite or silently mark recovered'),ensure_ascii=False,indent=2))
(O/'baseline_fingerprints.json').write_text(json.dumps(fingerprints,indent=2))
summary=dict(old=old,new=new,concept_id=idx,manifest_rows=int(rows.sum()),concept_count_before=len(concepts),concept_count_after=len(after),conversation_before=len(json.load(open(R/'models/vocab_conversation.json'))['concepts']),conversation_after=len(json.load(open(exp/'vocab_conversation.json'))['concepts']),production_modified=False,weights_changed=False,threshold_changed=False,language_role_conflict={'annotated':'ADJ','canonical':'OBJ'},deployment_status='candidate_only_requires_role_review')
# Structural safety checks: all other concept IDs are invariant; each bank row still points to an existing concept.
b=np.load(R/'models/bank.npz');assert b['c'].max()<len(after);assert all(a==b for i,(a,b) in enumerate(zip(concepts,after)) if i!=idx)
summary['renamed_bank_rows']=int((b['c']==idx).sum());summary['structural_checks']='passed'
(O/'mapping_change.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
# Validation-only retrieval before/after with frozen embeddings and cross-signer exclusion.
from modules.runtime import Bank
q=pd.read_csv(W/'outputs/full-audit/evaluation_queries.csv');Z=np.load(W/'outputs/full-audit/production_query_embeddings.npy');q['embedding_row']=np.arange(len(q));q=q[q.role.eq('val')].copy()
conv=json.load(open(R/'models/vocab_conversation.json'))['concepts'];allowed0=[i for i,c in enumerate(concepts) if c in conv];allowed1=allowed0+[idx]
q=q[q.concept.isin(set(conv)|{old,new})];known=lambda x:str(x).strip().lower() not in ['','none','nan','null','unknown','<na>'];q=q[q.signer.map(known)]
f=np.load(R/'artifacts/eval/bank_eval.npz');good=np.array([known(x) for x in f['signer']]);results=[]
for signer,qs in q.groupby('signer'):
 keep=good&(f['signer'].astype(str)!=str(signer))&~np.isin(f['clip'],qs.clip_id)
 for tag,allowed in [('before',allowed0),('after',allowed1)]:
  kk=keep&np.isin(f['c'],allowed); bank=Bank(f['Z'][kk],f['c'][kk],len(concepts))
  for start in range(0,len(qs),64):
   ch=qs.iloc[start:start+64];scores=bank.scores(Z[ch.embedding_row])
   for j,r in enumerate(ch.itertuples()):results.append(dict(phase=tag,clip_id=r.clip_id,expected=r.concept,predicted=after[int(scores[j].argmax())],correct=int(scores[j].argmax())==int(r.target)))
(O/'validation_mapping_predictions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));metrics={}
for phase in ['before','after']:
 a=[r for r in results if r['phase']==phase];metrics[phase]=dict(n=len(a),correct=sum(r['correct'] for r in a))
metrics['new_concept_validation_queries']=int(q.target.eq(idx).sum());metrics['scope']='cached embedding retrieval only; validation role, no test selection'
(O/'validation_mapping_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2));print(metrics)
