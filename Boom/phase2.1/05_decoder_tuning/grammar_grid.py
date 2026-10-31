"""Real tune-only grammar grid; capture pre-grammar inputs and replay production functions."""
from __future__ import annotations
import copy, itertools, os, time
from collections import Counter
import numpy as np
import pandas as pd
import decoder_experiment as de
ex=de.ex

def run():
    """Evaluate twelve settings on identical real visual candidates, never test poses."""
    os.environ['THAISLM_SYMBOLIC_WEIGHT']='0';os.environ['ORT_THREADS']='2'
    from modules.pipeline import SignPipeline, merge_repeats
    from modules.schema import canonical_hands, SH_L, SH_R
    import modules.grammar as grammar
    original=grammar.rerank
    dest=de.OUT/('grammar_tune_'+time.strftime('%Y%m%d_%H%M%S'));dest.mkdir()
    pipe=SignPipeline(backend='onnx',device='cpu',vocab='conversation',use_llm=False,use_face=False,bank_file=ex.ROOT/'artifacts/eval/bank_eval.npz')
    pipe.symbolic_weight=0;pipe.spot['grammar_rerank_weight']=.5
    entries=[(k,v) for k,v in ex.read(ex.ROOT/'artifacts/runs/seq-v7/tsl51_alignment.json').items() if k.startswith('t51s:') and v['role']=='tagger_train']
    assert len(entries)==126
    configs={'off':None}
    for gap,topn,w in itertools.product([.03,.06,.10],[3,5],[.5,1.]):configs[f'g{gap:.2f}_n{topn}_w{w:.1f}']=dict(min_gap=gap,topn=topn,weight=w)
    baseline='g0.03_n3_w0.5'
    ex.save(dest/'manifest.json',dict(split='tune',configs=configs,baseline=baseline,spot=pipe.spot,grammar_source='existing models/grammar.json; TSL51-derived; no claim of unseen-template generalisation',hashes={str(p):ex.digest(ex.ROOT/p) for p in ['models/grammar.json','models/roles.json','modules/grammar.py','models/spotting.json']},test_opened=False,selection='candidate requires lower mean WER than baseline, no increase in total S or D; manual motion review before test'))
    raw={};captured=[]
    def standardise(_):
        kp,sc,hw=raw['kp'].copy(),raw['sc'].copy(),raw['hw'];present=sc[:,[0,SH_L,SH_R]].mean(1)>.5
        kp[~present]=0;sc[~present]=0;kp,sc,mirrored=canonical_hands(kp,sc,hw)
        return kp,sc,hw,present,mirrored
    def capture(segs,role,pat,**kw):
        captured[:]=copy.deepcopy(segs)
        return original(segs,role,pat,**kw)
    pipe.standardise=standardise;grammar.rerank=capture
    metrics=[];changes=[];cache=[];predictions=[]
    try:
        for idx,(cid,a) in enumerate(entries):
            p=ex.ROOT/'cloud_s3/data_prep/pose_mp/tsl51_user_sentence'/f'{cid.split(":",1)[1]}.npz'
            with np.load(p) as z:raw.update(kp=z['kp'].astype(np.float32),sc=z['sc'].astype(np.float32),hw=tuple(map(int,z['hw'])))
            captured.clear();start=time.perf_counter();actual=pipe.run_landmarks(None,alphabet='off');elapsed=time.perf_counter()-start
            cache.append(dict(clip=cid,reference=a['glosses'],segments=copy.deepcopy(captured)))
            for name,cfg in configs.items():
                ss=copy.deepcopy(captured);before=[s['nearest'] for s in ss]
                start=time.perf_counter();chosen=original(ss,pipe.roles,pipe.grammar,**cfg) if cfg else before
                # Match production candidate reordering, then existing repeat collapse.
                for i,(s,new) in enumerate(zip(ss,chosen)):
                    if new!=s['nearest']:
                        old=s['nearest'];cs={c['word']:c['score'] for c in s['candidates']}
                        roles=[pipe.roles.get(w,'OBJ') for w in before];alt=roles.copy();alt[i]=pipe.roles.get(new,'OBJ')
                        changes.append(dict(clip=cid,mode=name,segment=i,t0=s['t0'],t1=s['t1'],old=old,new=new,old_cos=cs.get(old),new_cos=cs.get(new),visual_drop=cs.get(old,0)-cs.get(new,0),isolated_role_logprob_gain=grammar.role_logprob(alt,pipe.grammar)-grammar.role_logprob(roles,pipe.grammar),new_in_reference=new in a['glosses'],manual_motion_verdict='pending',note='Reference membership does not establish temporal correctness'))
                        s['nearest']=new;s['candidates']=sorted(s['candidates'],key=lambda c:c['word']!=new)
                ss=merge_repeats(ss);hyp=[s['nearest'] for s in ss];counts=Counter(op for _,_,op in ex.edit_alignment(a['glosses'],hyp))
                if name==baseline:assert hyp==[s['nearest'] for s in actual['segments'] if s['status']!='transition'],f'Replay mismatch: {cid}'
                metrics.append(dict(clip=cid,mode=name,WER=(counts['S']+counts['D']+counts['I'])/len(a['glosses']),S=counts['S'],D=counts['D'],I=counts['I'],n_ref=len(a['glosses']),n_pred=len(hyp),collapsed=len(captured)-len(ss),grammar_s=time.perf_counter()-start,base_pipeline_s=elapsed))
                predictions.append(dict(clip=cid,mode=name,ref=a['glosses'],hyp=hyp,segments=ss))
            if idx%10==0:print(f'Captured/replayed {idx+1}/126',flush=True)
            pd.DataFrame(metrics).to_csv(dest/'progress.csv',index=False)
    finally:grammar.rerank=original
    df=pd.DataFrame(metrics);df.to_csv(dest/'metrics.csv',index=False);pd.DataFrame(changes).to_csv(dest/'changed_segments.csv',index=False)
    ex.save(dest/'pregrammar.json',cache);ex.save(dest/'predictions.json',predictions)
    summary=df.groupby('mode').agg(macro_WER=('WER','mean'),S=('S','sum'),D=('D','sum'),I=('I','sum'),n_ref=('n_ref','sum'),collapsed=('collapsed','sum')).reset_index();summary['micro_WER']=(summary.S+summary.D+summary.I)/summary.n_ref
    b=summary.set_index('mode').loc[baseline];summary['delta_WER']=summary.macro_WER-b.macro_WER;summary['delta_S']=summary.S-b.S;summary['delta_D']=summary.D-b.D
    summary['eligible_for_review']=(summary.delta_WER<0)&(summary.delta_S<=0)&(summary.delta_D<=0)
    summary.to_csv(dest/'summary.csv',index=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,6));summary.plot.barh(x='mode',y=['S','D','I'],ax=axes[0]);summary.plot.barh(x='mode',y='macro_WER',ax=axes[1]);fig.tight_layout();fig.savefig(dest/'grid.png',dpi=150);plt.close(fig)
    (dest/'REPORT.md').write_text('# Grammar grid — TSL51 tune only\n\n'+summary.round(4).to_string(index=False)+'\n\nAll baseline replays checked against full pipeline. No production changes. S/D are Levenshtein alignment counts; changes do not necessarily mean changed segmentation. Visual-score loss is a review flag, not proof of incorrect hand motion. Manual segment review pending; test remains unopened. Boom must be reported separately.\n')
    ex.save(de.OUT/'latest_grammar.json',{'path':str(dest)});print(summary.round(4).to_string(index=False),flush=True)
if __name__=='__main__':run()
