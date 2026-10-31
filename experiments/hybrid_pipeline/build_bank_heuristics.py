"""Join bank clip IDs to pose store; emit explicit coverage and mask provenance."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from features import stream,profile,FEATURES
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.repo));from modules.encoder import PoseStore
    a.out.mkdir(parents=True,exist_ok=True)
    bank=np.load(a.repo/'models/bank.npz',allow_pickle=False)
    concepts=json.loads((a.repo/'models/concepts.json').read_text());vocab=json.loads((a.repo/'models/vocab_conversation.json').read_text())['concepts']
    meta=pd.read_parquet(a.data/'clips.parquet').set_index('clip_id')
    if not meta.index.is_unique:raise ValueError('Ambiguous clip metadata')
    store=PoseStore(a.data/'pose_store.npz');records=[];seen=set();excluded={}
    for clip,c in zip(bank['clip'],bank['c']):
        clip=str(clip);word=concepts[int(c)]
        if word not in vocab or (clip,word) in seen:continue
        seen.add((clip,word))
        if clip not in meta.index or clip not in store.row:excluded['missing_pose_or_metadata']=excluded.get('missing_pose_or_metadata',0)+1;continue
        m=meta.loc[clip]
        # Use only training references, so validation/test labels cannot tune the priors.
        if m['role']!='train':excluded[str(m['role'])]=excluded.get(str(m['role']),0)+1;continue
        i=store.row[clip];n=store.length(i)
        lo=int(m['span_f0']) if bool(m['has_span']) else 0;hi=int(m['span_f1']) if bool(m['has_span']) else n
        if not 0<=lo<hi<=n:excluded['invalid_span']=excluded.get('invalid_span',0)+1;continue
        kp,sc,hw=store.get(i,lo,hi);f=profile(stream(kp,sc,hw,np.arange(len(kp))/25))
        records.append(dict(concept=word,clip_id=clip,**{k:f[k] for k in FEATURES}))
    df=pd.DataFrame(records);rows=[]
    for word in vocab:
        sub=df[df.concept==word];row=dict(concept=word,n_clips=len(sub),mask_provenance='prepared_pose_scores_not_raw_detection',split='train')
        for f in FEATURES:
            x=sub[f].dropna();row.update({f+'_count':len(x),f+'_mean':x.mean(),f+'_median':x.median(),f+'_std':x.std(ddof=0)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.out/'bank_heuristics_326.csv',index=False)
    summary=dict(vocab=len(vocab),reference_clips=len(records),concepts_with_references=sum(r['n_clips']>0 for r in rows),excluded=excluded,mask_provenance='prepared scores may include interpolation; not raw observed rate',split_policy='training references only')
    (a.out/'heuristic_coverage.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
