"""Explicit opt-in full-pipeline experiments; immutable production configuration and banks."""
from __future__ import annotations
import argparse,copy,hashlib,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import audit_experiments as audit
ex=audit.ex;OUT=audit.OUT

def run(dataset: str, speed: float=1.) -> None:
    """Run composer-inclusive cases with identical cached neural/decoder outputs per clip.

    Cache is in-memory, one clip only, keyed by input bytes. Postprocess, grammar,
    repeat collapse, embedding export and composer execute afresh for every case.
    Latency with caches is explicitly not end-to-end inference latency.
    """
    os.environ['ORT_THREADS']='2';os.environ['THAISLM_SYMBOLIC_WEIGHT']='0'
    import modules.pipeline as pipeline
    import modules.grammar as grammar
    from modules.schema import canonical_hands,SH_L,SH_R
    from modules.segment import resample
    selection=ex.read(OUT/'selection.json');selected=selection['params'];clean=ex.read(OUT/'grammar_tune_only.json')
    dest=OUT/f'{dataset}_{speed:g}x';dest.mkdir(exist_ok=True)
    if dataset=='test':
        lock=dest/'TEST_STARTED.json'
        if lock.exists():raise RuntimeError('Test already started: no automatic rerun or retuning permitted')
        ex.save(lock,dict(selection=selection,warning='previously inspected test according to user; this is a frozen reproduction, not a virgin holdout',started=time.strftime('%Y-%m-%d %H:%M:%S')))
    bank=ex.ROOT/('models/bank.npz' if dataset=='boom' else 'artifacts/eval/bank_eval.npz')
    fingerprints={str(p):ex.digest(ex.ROOT/p) for p in ['models/bank.npz','artifacts/eval/bank_eval.npz','models/spotting.json','models/grammar.json','models/onnx/tagger.onnx','models/onnx/encoder.onnx']}
    pipe=pipeline.SignPipeline(backend='onnx',device='cpu',vocab='conversation',use_llm=False,use_face=False,bank_file=bank)
    base=copy.deepcopy(pipe.spot);cal=copy.deepcopy(pipe.cal);oldgrammar=copy.deepcopy(pipe.grammar)
    names=['grammar_off','grammar_original','grammar_clean_selected','p_sign055','accept070','hand030','filters_all']
    if dataset=='test':names=['grammar_off','grammar_original','grammar_clean_selected','filters_all']
    manifest=dict(dataset=dataset,speed=speed,configs=names,hashes=fingerprints,bank=str(bank),grammar_params=selected,neural_cache='per clip only; exact input bytes',timing='amortised cached pipeline, not independent E2E latency',hand_definition='any hand with mean standard-schema landmark confidence >= .5; standardisation may interpolate; not raw observed rate',production_enable=False)
    ex.save(dest/'manifest.json',manifest)
    # Exact memoisation: no parameter affecting neural/spot computation changes among cases.
    memo={};orig_tag=pipe._tag;orig_spot=pipe._spot
    def memo_call(label,fn,args):
        h=hashlib.sha256(label.encode())
        for a in args:
            if isinstance(a,np.ndarray):h.update(str(a.shape).encode());h.update(a.tobytes())
        key=h.hexdigest()
        if key not in memo:memo[key]=copy.deepcopy(fn(*args))
        return copy.deepcopy(memo[key])
    pipe._tag=lambda *a:memo_call('tag',orig_tag,a)
    pipe._spot=lambda *a:memo_call('spot',orig_spot,a)
    state={};orig_standard=pipe.standardise
    def standard(raw):
        if dataset=='boom':answer=orig_standard(raw)
        else:
            kp,sc,hw=state['kp'].copy(),state['sc'].copy(),state['hw'];present=sc[:,[0,SH_L,SH_R]].mean(1)>.5
            kp[~present]=0;sc[~present]=0;kp,sc,mirrored=canonical_hands(kp,sc,hw);answer=(kp,sc,hw,present,mirrored)
        state['sc_now']=answer[1];return answer
    pipe.standardise=standard
    orig_post=pipeline.postprocess;orig_rerank=grammar.rerank
    active={'mode':''}
    def post(segs,cfg):
        orig_post(segs,cfg)
        if active['mode'] in ('hand030','filters_all'):
            sc=state['sc_now'];seen=(sc[:,91:112].mean(1)>=.5)|(sc[:,112:133].mean(1)>=.5)
            for s in segs:
                span=seen[s['f0']:s['f1']];s['experimental_hand_frac']=float(span.mean()) if len(span) else None
                if s['status']!='accepted' and len(span) and span.mean()<.3:s['status']='transition';s['hand_filter_experiment']=True
    def rerank(segs,roles,pat,**kw):
        return orig_rerank(segs,roles,pat,**(selected if active['mode']=='grammar_clean_selected' else kw))
    pipeline.postprocess=post;grammar.rerank=rerank
    alignment=ex.read(ex.ROOT/'artifacts/runs/seq-v7/tsl51_alignment.json')
    if dataset=='boom':entries=[(v.name,None) for v in ex.videos()]
    else:entries=[(cid,a) for cid,a in alignment.items() if cid.startswith('t51s:') and a['role']==('tagger_train' if dataset=='tune' else 'sentence_test')]
    rows=[];predictions=[]
    try:
        for idx,(cid,a) in enumerate(entries):
            memo.clear();raw=None
            if dataset=='boom':
                import importlib.metadata
                video=ex.ROOT/'Boom/boom datatest'/cid;model=ex.ROOT/'models/mediapipe/holistic_landmarker.task'
                key=ex.digest(video)+ex.digest(model)+ex.digest(ex.ROOT/'modules/mediapipe_pose.py')+importlib.metadata.version('mediapipe')
                cache=OUT.parent/'raw_cache'/(hashlib.sha256(key.encode()).hexdigest()+'.npz')
                if not cache.exists():raise FileNotFoundError(f'Expected real extraction cache: {cache}')
                with np.load(cache) as z:raw={k:z[k] for k in z.files}
                ref=ex.tokenize(video.stem)
            else:
                path=ex.ROOT/'cloud_s3/data_prep/pose_mp/tsl51_user_sentence'/f'{cid.split(":",1)[1]}.npz'
                with np.load(path) as z:kp=z['kp'].astype(np.float32);sc=z['sc'].astype(np.float32);hw=tuple(map(int,z['hw']))
                if speed!=1:kp,sc=resample(kp,sc,speed)
                state.update(kp=kp,sc=sc,hw=hw);ref=a['glosses']
            for mode in names:
                active['mode']=mode;pipe.spot=copy.deepcopy(base);pipe.cal=copy.deepcopy(cal);pipe.grammar=oldgrammar
                if mode=='grammar_off':pipe.spot['grammar_rerank_weight']=0
                elif mode=='grammar_clean_selected':
                    pipe.spot['grammar_rerank_weight']=selected['weight'];pipe.grammar=clean
                    if dataset=='tune':
                        # Held-out role-template grammar, matching CV rather than full-tune leakage.
                        role=pipe.roles;t=' '.join(role.get(w,'OBJ') for w in ref)
                        seqs=sorted({tuple(x['glosses']) for x in alignment.values() if x['role']=='tagger_train' and ' '.join(role.get(w,'OBJ') for w in x['glosses'])!=t})
                        pipe.grammar=grammar.learn_patterns(seqs,role)
                if mode in ('p_sign055','filters_all'):pipe.spot['post']['p_sign_min']=.55
                if mode in ('accept070','filters_all'):pipe.cal['tau_accept']=.7
                start=time.perf_counter();r=pipe.run_landmarks(raw,alphabet='off');elapsed=time.perf_counter()-start
                segs=[s for s in r['segments'] if s['status']!='transition'];hyp=[s['nearest'] for s in segs]
                if dataset=='boom':hyp=ex.tokenize(''.join(hyp))
                counts=audit.score(ref,hyp);accepted=[s for s in segs if s['status']=='accepted']
                correct=None
                if dataset!='boom':
                    matches={j for i,j,op in ex.edit_alignment(ref,hyp) if op=='C'};correct=sum(j in matches for j,s in enumerate(segs) if s['status']=='accepted')
                rows.append(dict(clip=cid,mode=mode,**counts,n_pred=len(hyp),count_error=abs(len(hyp)-len(ref)),accepted=len(accepted),accepted_correct_ordered=correct,pipeline_cached_s=elapsed,ref=' | '.join(ref),hyp=' | '.join(hyp)))
                for s in r['segments']:s.pop('emb',None)
                predictions.append(dict(clip=cid,mode=mode,ref=ref,result=r))
            pd.DataFrame(rows).to_csv(dest/'progress.csv',index=False)
            if idx%10==0:print(dataset,speed,idx+1,'/',len(entries),flush=True)
    finally:pipeline.postprocess=orig_post;grammar.rerank=orig_rerank
    df=pd.DataFrame(rows);df.to_csv(dest/'metrics.csv',index=False);ex.save(dest/'predictions.json',predictions)
    summary=df.groupby('mode').agg(macro_WER=('WER','mean'),S=('S','sum'),D=('D','sum'),I=('I','sum'),accepted=('accepted','sum'),accepted_correct=('accepted_correct_ordered','sum'),count_error=('count_error','mean')).reset_index()
    if dataset=='boom':summary['accepted_correct']=np.nan
    summary['accepted_precision_ordered']=summary.accepted_correct/summary.accepted.replace(0,np.nan)
    summary.to_csv(dest/'summary.csv',index=False)
    basewer=df[df['mode']=='grammar_original'].set_index('clip').WER;stats=[];rng=np.random.default_rng(20261019)
    for mode,g in df.groupby('mode'):
        delta=(g.set_index('clip').WER-basewer).to_numpy();ci=np.quantile(rng.choice(delta,(5000,len(delta))).mean(1),[.025,.975])
        stats.append(dict(mode=mode,delta=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),improved=int((delta< -1e-9).sum()),unchanged=int((abs(delta)<=1e-9).sum()),worsened=int((delta>1e-9).sum()),conclusion='ไม่มีนัยสำคัญ' if ci[0]<=0<=ci[1] else ('WER ลดลง' if ci[1]<0 else 'WER เพิ่มขึ้น'),note='clip bootstrap; not independent signers'))
    pd.DataFrame(stats).to_csv(dest/'paired_ci.csv',index=False)
    assert all(ex.digest(ex.ROOT/p)==v for p,v in fingerprints.items()),'Production inputs changed'
    print(summary.round(4).to_string(index=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['tune','boom','test']);p.add_argument('--speed',type=float,default=1);p.add_argument('--run',action='store_true');p.add_argument('--final-test',action='store_true');a=p.parse_args()
    if a.run:
        if not a.dataset:p.error('dataset required')
        if a.dataset=='test' and not a.final_test:p.error('test requires explicit --final-test after freeze')
        run(a.dataset,a.speed)
