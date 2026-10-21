import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
R=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say'); W=Path('/Users/thawaphorn/Documents/Codex/2026-10-07/new-chat');O=Path('outputs/word-audit');sys.path.insert(0,str(R))
from modules.runtime import Bank
words='สวัสดี ขอบคุณ ชอบ ฉัน แม่ กาแฟ ชา ธนาคาร ซื้อ จ่าย'.split()
concepts=json.load(open(R/'models/concepts.json')); conv=set(json.load(open(R/'models/vocab_conversation.json'))['concepts']); mapping=json.load(open(W/'work/training-data/concepts.json'))['concept']
bank=np.load(R/'models/bank.npz'); print('bank fields',bank.files,flush=True)
inv=pd.read_csv(W/'outputs/full-audit/bank_inventory.csv');m=pd.read_parquet(W/'work/training-data/manifest.parquet')
missing=pd.read_csv(W/'outputs/conversation-audit/details/daily_missing.csv');lex=[]
for word in list(missing.concept)+words:
 c=mapping.get(word,word); d=m[m.concept.eq(c)]; n=int(np.isin(bank['c'],[concepts.index(c)]).sum()) if c in concepts else 0
 lex.append(dict(word=word,mapped_concept=c,in_model=c in concepts,in_conversation=c in conv,bank_examples=n,glosses=sorted(d.gloss.dropna().unique().tolist()),roles=d.role.value_counts().to_dict(),excluded_reasons=d.exclude_reason.dropna().value_counts().to_dict()))
(O/'lexicon.json').write_text(json.dumps(lex,ensure_ascii=False,indent=2))
q=pd.read_csv(W/'outputs/full-audit/evaluation_queries.csv');z=np.load(W/'outputs/full-audit/production_query_embeddings.npy');q['embedding_row']=np.arange(len(q));q=q[q.concept.isin(conv)&q.role.eq('test')].copy()
known=lambda x:str(x).strip().lower() not in ['','none','nan','null','unknown','<na>']
q=q[q.signer.map(known)]; b=np.load(R/'artifacts/eval/bank_eval.npz');ids=[i for i,c in enumerate(concepts) if c in conv];base=np.array([known(x) for x in b['signer']])&np.isin(b['c'],ids);rows=[]
for signer,qq in q.groupby('signer'):
 keep=base&(b['signer'].astype(str)!=str(signer))&~np.isin(b['clip'],qq.clip_id)
 bk=Bank(b['Z'][keep],b['c'][keep],len(concepts))
 for start in range(0,len(qq),64):
  batch=qq.iloc[start:start+64];scores=bk.scores(z[batch.embedding_row])
  for j,row in enumerate(batch.itertuples()):
   pred=int(scores[j].argmax()); target=int(row.target);covered=bool(target>=0 and scores[j,target]>-1.5)
   rows.append(dict(clip_id=row.clip_id,expected=row.concept,predicted=concepts[pred],correct=pred==target,covered=covered,signer=str(signer),rank=int((scores[j]>scores[j,target]).sum()+1) if covered else None))
(O/'all_test_predictions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
wrong=[r for r in rows if not r['correct']];(O/'top1_errors.json').write_text(json.dumps(wrong,ensure_ascii=False,indent=2))
summary=[]
for w in words:
 d=inv[inv.concept.eq(w)];test=[r for r in rows if r['expected']==w];summary.append(dict(word=w,group='many_examples' if w in words[:5] else 'dictionary_only',bank_examples=len(d),signers=int(d.signer.nunique()),sources=d.source_group.value_counts().to_dict(),test_queries=len(test),correct=sum(r['correct'] for r in test),top1=sum(r['correct'] for r in test)/len(test) if test else None))
(O/'groups.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2));print('all test',len(rows),'wrong',len(wrong))
# Locate raw dictionary clips for same-input browser tests; these are NOT held-out accuracy.
paths=list((W/'work/training-data/raw').rglob('*.mp4')); choices=[]
for w in words:
 d=m[m.concept.eq(w)&m.dataset.isin(['th_sl','ttrs'])]
 found=None
 for r in d.itertuples():
  hits=[p for p in paths if p.name==Path(r.source_file).name]
  if hits:found=dict(word=w,clip_id=r.clip_id,path=str(hits[0]),role=r.role);break
 choices.append(found or dict(word=w,path=None))
(O/'browser_inputs.json').write_text(json.dumps(choices,ensure_ascii=False,indent=2));print('browser inputs',choices)
