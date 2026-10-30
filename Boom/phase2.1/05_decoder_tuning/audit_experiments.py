"""Grammar provenance/CV and real bank-neighbour diagnostics; production files are read-only."""
from __future__ import annotations
import argparse,copy,itertools,os
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import decoder_experiment as de
ex=de.ex
OUT=de.OUT/'audit_v72';OUT.mkdir(exist_ok=True)
PAIRS=[('ไป','กลับ'),('เที่ยว','หนึ่ง'),('ภาษามือ','วิ่ง'),('กิน','ดู'),('ข้าว','ไม่มี'),('ยาย','ที่ไหน'),('ยาย','ญาติ'),('คุณ','หนึ่ง'),('คุณ','ป้า')]
def score(ref,hyp):
    """Ordered Levenshtein error counts."""
    c=Counter(op for _,_,op in ex.edit_alignment(ref,hyp));return dict(WER=(c['S']+c['D']+c['I'])/len(ref),S=c['S'],D=c['D'],I=c['I'],n_ref=len(ref))
def grid():
    """Requested 60 parameter combinations, including redundant zero-weight controls."""
    return {f'w{w:.1f}_n{n}_g{g:.2f}':dict(weight=w,topn=n,min_gap=g) for w,n,g in itertools.product([0,.3,.5,.7,1.],[2,3,4],[.03,.05,.08,.12])}
def audit():
    """Reconstruct source list, then exclude each evaluated role-template from grammar learning."""
    from modules.grammar import learn_patterns,rerank
    from modules.pipeline import merge_repeats
    role=ex.read(ex.ROOT/'models/roles.json');pat=ex.read(ex.ROOT/'models/grammar.json')
    s=pd.read_parquet(ex.ROOT/'cloud_s3/data_prep/shards/clips.parquet');s=s[(s.kind=='sentence')&(s.view=='primary')].copy()
    s['words']=s.concept.map(lambda x:x.split('|'));s['text']=s.words.map(' '.join);s['template']=s.words.map(lambda ws:' '.join(role.get(w,'OBJ') for w in ws))
    uniq=sorted(s.text.unique());rebuilt=learn_patterns([x.split() for x in uniq],role)
    matched=all(rebuilt[k]==pat[k] for k in ['bigram','unigram','templates','n','roles'])
    s[['clip_id','source','signer','role','text','template']].to_csv(OUT/'grammar_source_recordings.csv',index=False)
    exact={x['tsl_order'] for x in pat.get('examples',[])};templates={x['template'] for x in pat['templates']}
    coverage=[dict(split=r,n_recordings=len(g),same_word_sequence=int(g.text.isin(exact).sum()),same_role_template=int(g.template.isin(templates).sum()),recordings_in_reconstructed_input=len(g) if matched else None) for r,g in s.groupby('role')]
    pd.DataFrame(coverage).to_csv(OUT/'grammar_overlap.csv',index=False)
    ex.save(OUT/'provenance.json',dict(grammar_n=pat['n'],reconstructed_n=len(uniq),exact_reconstruction=matched,recording_provenance='Original grammar omits clip IDs; this reconstructs current script/manifest, not a historical execution log',test_metadata_only=True,source_script='scripts/eval.py:cmd_grammar',tune_test_clip_id_overlap=len(set(s[s.role=='tagger_train'].clip_id)&set(s[s.role=='sentence_test'].clip_id)),tune_test_word_sequence_overlap=len(set(s[s.role=='tagger_train'].text)&set(s[s.role=='sentence_test'].text)),tune_test_role_template_overlap=len(set(s[s.role=='tagger_train'].template)&set(s[s.role=='sentence_test'].template))))
    tune=s[s.role=='tagger_train'];folds={t:learn_patterns([x.split() for x in sorted(tune[tune.template!=t].text.unique())],role) for t in tune.template.unique()}
    ex.save(OUT/'grammar_tune_only.json',learn_patterns([x.split() for x in sorted(tune.text.unique())],role))
    cache=ex.read(Path(ex.read(de.OUT/'latest_grammar.json')['path'])/'pregrammar.json');configs=grid();metrics=[];changes=[]
    for row in cache:
        template=' '.join(role.get(w,'OBJ') for w in row['reference'])
        for version,grammar in [('original',pat),('template_cv',folds[template])]:
            for name,cfg in configs.items():
                ss=copy.deepcopy(row['segments']);before=[x['nearest'] for x in ss];chosen=rerank(ss,role,grammar,**cfg) if cfg['weight'] else before
                for j,(seg,new) in enumerate(zip(ss,chosen)):
                    if new!=seg['nearest']:
                        cs={x['word']:x['score'] for x in seg['candidates']};changes.append(dict(clip=row['clip'],version=version,config=name,segment=j,t0=seg['t0'],t1=seg['t1'],old=seg['nearest'],new=new,visual_drop=cs.get(seg['nearest'],0)-cs.get(new,0),new_in_reference=new in row['reference'],motion_verdict='pending human review'))
                        seg['nearest']=new;seg['candidates']=sorted(seg['candidates'],key=lambda c:c['word']!=new)
                hyp=[x['nearest'] for x in merge_repeats(ss)]
                metrics.append(dict(clip=row['clip'],template=template,version=version,config=name,**score(row['reference'],hyp)))
    df=pd.DataFrame(metrics);df.to_csv(OUT/'grammar_cv_metrics.csv',index=False);pd.DataFrame(changes).to_csv(OUT/'grammar_cv_changes.csv',index=False)
    summary=df.groupby(['version','config']).agg(macro_WER=('WER','mean'),S=('S','sum'),D=('D','sum'),I=('I','sum')).reset_index();summary.to_csv(OUT/'grammar_cv_summary.csv',index=False)
    winner=summary[summary.version=='template_cv'].sort_values(['macro_WER','D','S','config']).iloc[0]
    ex.save(OUT/'selection.json',dict(config=str(winner.config),params=configs[winner.config],criterion='lowest out-of-fold tune macro WER; tie fewer D, S, config name',cv_is_selection_estimate=True,test_previously_reported_by_user=True,production_enable=False))
    print('PROVENANCE',matched,len(uniq));print(pd.DataFrame(coverage).to_string(index=False));print('CV BEST',winner.to_dict(),flush=True)
def bank_audit():
    """Use original encoder on tune auto-aligned spans; human review must establish label noise."""
    os.environ['ORT_THREADS']='2'
    from modules.runtime import OnnxBackend
    from modules.parts import prep,to_iso
    from modules.schema import canonical_hands
    concepts=ex.read(ex.ROOT/'models/concepts.json');words={w for p in PAIRS for w in p};meta=pd.read_parquet(ex.ROOT/'cloud_s3/data_prep/shards/clips.parquet').drop_duplicates('clip_id').set_index('clip_id')
    align=ex.read(ex.ROOT/'artifacts/runs/seq-v7/tsl51_alignment.json');inventories=[];banks={}
    for name,path in [('eval',ex.ROOT/'artifacts/eval/bank_eval.npz'),('production',ex.ROOT/'models/bank.npz')]:
        z=np.load(path,allow_pickle=False);banks[name]={k:z[k] for k in z.files};z=banks[name]
        for i,c in enumerate(z['c']):
            # Keep complete metadata for nearest neighbours outside the requested pairs.
            cid=str(z['clip'][i]);base=cid.split('@')[0];m=meta.loc[base] if base in meta.index else None
            source=str(m.source) if m is not None else '';f0=int(cid.split('@')[-1]) if '@' in cid else 0
            duration=int(m.n_frames) if m is not None and '@' not in cid else next((s['f1']-s['f0'] for s in align.get(base,{}).get('segments',[]) if s['f0']==f0),None)
            inventories.append(dict(bank=name,index=i,concept=concepts[int(c)],clip=cid,dataset=str(z['dataset'][i]),signer=str(z['signer'][i]),frames=duration,source=source,source_exists=Path(source).exists() if source else False,f0=f0))
    inv=pd.DataFrame(inventories);inv[inv.concept.isin(words)].to_csv(OUT/'bank_inventory.csv',index=False)
    parts=[];queries=[]
    for cid,a in align.items():
        if not cid.startswith('t51s:') or a['role']!='tagger_train':continue
        with np.load(ex.ROOT/'cloud_s3/data_prep/pose_mp/tsl51_user_sentence'/f'{cid.split(":",1)[1]}.npz') as z:kp=z['kp'].copy();sc=z['sc'].copy();hw=z['hw']
        kp,sc,_=canonical_hands(kp,sc,hw);kp=to_iso(kp,hw)
        for seg in a['segments']:
            if seg['concept'] not in words:continue
            f0,f1=int(seg['f0']),int(seg['f1'])
            if f1-f0<2:continue
            parts.append(prep(kp[f0:f1],sc[f0:f1]));queries.append(dict(clip=cid,truth=seg['concept'],f0=f0,f1=f1,alignment_score=seg['score'],boundary_type='automatic; not human verified'))
    be=OnnxBackend(ex.ROOT/'models','cpu');Z=be.embed(parts,bs=16);Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-9)
    rows=[];neighbours=[]
    for name,b in banks.items():
        bz=b['Z'].copy();bz/=np.maximum(np.linalg.norm(bz,axis=1,keepdims=True),1e-9);sim=Z@bz.T;labels=np.array([concepts[int(c)] for c in b['c']])
        for qi,q in enumerate(queries):
            allowed=set(ex.read(ex.ROOT/'models/vocab_conversation.json')['concepts']); candidates=np.flatnonzero(np.isin(labels,list(allowed))); nn=int(candidates[sim[qi,candidates].argmax()]);neighbours.append(dict(**q,bank=name,nn_index=nn,nn_label=labels[nn],nn_clip=str(b['clip'][nn]),nn_dataset=str(b['dataset'][nn]),nn_signer=str(b['signer'][nn]),nn_cos=float(sim[qi,nn]),same_source_recording=str(b['clip'][nn]).split('@')[0]==q['clip'],bank_scope='conversation vocab',evaluation_caveat='production contains sentence-derived exemplars; eval bank is primary'))
            for true,wrong in PAIRS:
                if q['truth']!=true:continue
                tm=labels==true;wm=labels==wrong
                rows.append(dict(**q,bank=name,pair=f'{true} → {wrong}',true_mean=float(sim[qi,tm].mean()) if tm.any() else None,wrong_mean=float(sim[qi,wm].mean()) if wm.any() else None,true_max=float(sim[qi,tm].max()) if tm.any() else None,wrong_max=float(sim[qi,wm].max()) if wm.any() else None))
    pd.DataFrame(rows).to_csv(OUT/'pair_cosines.csv',index=False);nn=pd.DataFrame(neighbours);nn.to_csv(OUT/'nearest_neighbours.csv',index=False)
    nn[nn.truth!=nn.nn_label].merge(inv,left_on=['bank','nn_index'],right_on=['bank','index'],how='left',suffixes=('','_bank')).to_csv(OUT/'visual_review_queue.csv',index=False)
    cache=ex.read(Path(ex.read(de.OUT/'latest_tsl51.json')['path'])/'predictions.json');counts=Counter()
    for r in cache:
        if r['mode']!='baseline':continue
        hyp=[s['nearest'] for s in r['result']['segments'] if s['status']!='transition']
        for i,j,op in ex.edit_alignment(r['ref'],hyp):
            if op=='S':counts[(r['ref'][i],hyp[j])]+=1
    summary=[];df=pd.DataFrame(rows)
    for a,b in PAIRS:
        g=df[(df.bank=='eval')&(df.pair==f'{a} → {b}')]
        summary.append(dict(pair=f'{a} → {b}',tune_pipeline_confusions=counts[(a,b)],n_queries=len(g),true_mean=g.true_mean.mean(),wrong_mean=g.wrong_mean.mean(),wrong_max_beats_true=int((g.wrong_max>g.true_max).sum()),cause='unresolved: variant/label/similar pose/automatic boundaries',evidence='pair_cosines + nearest_neighbours + visual_review_queue'))
    pd.DataFrame(summary).to_csv(OUT/'pair_summary.csv',index=False);print(pd.DataFrame(summary).round(3).to_string(index=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grammar-audit',action='store_true');p.add_argument('--bank-audit',action='store_true');a=p.parse_args()
    if a.grammar_audit:audit()
    if a.bank_audit:bank_audit()
