"""Real TSL51 tune evaluation, mirroring pitty/scripts/sentence_eval.py preprocessing.

Test split is not opened. Reference alignment spans are not human verified and
therefore are not used as ground truth Segment F1.
"""
from __future__ import annotations
import copy,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import decoder_experiment as de
ex=de.ex


def run() -> Path:
    """Compare baseline and proposed combined override on all 126 tune sequences."""
    os.environ['THAISLM_SYMBOLIC_WEIGHT']='0'
    os.environ.setdefault('ORT_THREADS','2')
    from modules.pipeline import SignPipeline
    from modules.schema import canonical_hands,SH_L,SH_R
    from modules.decode import wer
    alignment=ex.ROOT/'artifacts/runs/seq-v7/tsl51_alignment.json'
    bank=ex.ROOT/'artifacts/eval/bank_eval.npz'
    entries=[(cid,a) for cid,a in ex.read(alignment).items() if cid.startswith('t51s:') and a['role']=='tagger_train']
    if len(entries)!=126:raise ValueError(f'Expected 126 tune clips, got {len(entries)}')
    pipe=SignPipeline(backend='onnx',device='cpu',vocab='conversation',use_llm=False,use_face=False,bank_file=bank)
    base=copy.deepcopy(pipe.spot);pipe.symbolic_weight=0
    names=['baseline','combined_without_csls']
    configs={name:de.resolved(base,de.CASES[name]) for name in names}
    dest=de.OUT/('tsl51_tune_'+time.strftime('%Y%m%d_%H%M%S'));dest.mkdir(parents=True)
    manifest=dict(dataset='TSL51',split='tune',n_clips=len(entries),configs=configs,
        reference='ordered glosses from tsl51_alignment.json',bank=str(bank),bank_hash=ex.digest(bank),
        alignment_hash=ex.digest(alignment),tagger_hash=ex.digest(ex.ROOT/'models/onnx/tagger.onnx'),
        config_hash=ex.digest(ex.ROOT/'models/spotting.json'),threads=os.environ['ORT_THREADS'],
        preprocessing='same as pitty/scripts/sentence_eval.py, speed=1, no hold cut',
        untouched_test=True,metrics_note='Recall@1/5 presence-based like pitty, not temporal alignment. F1 unavailable without verified spans.')
    ex.save(dest/'manifest.json',manifest)
    cur={}
    def standardise(raw):
        kp,sc,hw=cur['kp'],cur['sc'],cur['hw']
        present=sc[:,[0,SH_L,SH_R]].mean(1)>.5
        kp=kp.copy();sc=sc.copy();kp[~present]=0.;sc[~present]=0.
        kp,sc,mirrored=canonical_hands(kp,sc,hw)
        return kp,sc,hw,present,mirrored
    pipe.standardise=standardise
    rows=[];predictions=[];started=time.perf_counter()
    for index,(cid,a) in enumerate(entries):
        path=ex.ROOT/'cloud_s3/data_prep/pose_mp/tsl51_user_sentence'/f'{cid.split(":",1)[1]}.npz'
        with np.load(path,allow_pickle=False) as z:
            cur.update(kp=z['kp'].astype(np.float32),sc=z['sc'].astype(np.float32),hw=tuple(int(v) for v in z['hw']))
        for name in (names if index%2==0 else names[::-1]):
            pipe.spot=copy.deepcopy(configs[name]);t=time.perf_counter()
            result=pipe.run_landmarks(None,alphabet='off');elapsed=time.perf_counter()-t
            for segment in result['segments']:segment.pop('emb',None)
            segments=[s for s in result['segments'] if s['status']!='transition']
            hyp=[s['nearest'] for s in segments];ref=a['glosses']
            top5=[[c['word'] for c in s['candidates'][:5]] for s in segments]
            pairs=ex.edit_alignment(ref,hyp);counts=Counter(op for _,_,op in pairs)
            measured=wer(ref,hyp)
            assert abs(measured-(counts['S']+counts['D']+counts['I'])/len(ref))<1e-6
            rows.append(dict(clip=cid,mode=name,WER=measured,S=counts['S'],D=counts['D'],I=counts['I'],
                recall1=float(np.mean([w in hyp for w in ref])),recall5=float(np.mean([any(w in c for c in top5) for w in ref])),
                n_ref=len(ref),n_pred=len(hyp),count_error=abs(len(hyp)-len(ref)),inference_s=elapsed,
                accepted_count=sum(s['status']=='accepted' for s in segments),ref=' | '.join(ref),hyp=' | '.join(hyp)))
            predictions.append(dict(clip=cid,mode=name,ref=ref,result=result))
        # Save progress atomically every clip; never publish partial as latest-complete.
        pd.DataFrame(rows).to_csv(dest/'progress.csv',index=False)
        if index%10==0 or index==len(entries)-1:
            print(f'TSL51 tune {index+1}/126; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(dest/'metrics.csv',index=False);ex.save(dest/'predictions.json',predictions)
    summaries=[]
    for name,g in frame.groupby('mode'):
        summaries.append(dict(mode=name,n_clips=len(g),macro_WER=g.WER.mean(),micro_WER=(g.S.sum()+g.D.sum()+g.I.sum())/g.n_ref.sum(),
            recall1=g.recall1.mean(),recall5=g.recall5.mean(),S=int(g.S.sum()),D=int(g.D.sum()),I=int(g.I.sum()),
            count_error=g.count_error.mean(),mean_inference_s=g.inference_s.mean()))
    summary=pd.DataFrame(summaries);summary.to_csv(dest/'summary.csv',index=False)
    paired=frame.pivot(index='clip',columns='mode',values='WER');paired['delta']=paired.combined_without_csls-paired.baseline
    paired.to_csv(dest/'paired_wer.csv')
    rng=np.random.default_rng(20261019);delta=paired.delta.to_numpy();boots=rng.choice(delta,(5000,len(delta)),replace=True).mean(1)
    stats=dict(improved=int((delta< -1e-9).sum()),unchanged=int((np.abs(delta)<=1e-9).sum()),worsened=int((delta>1e-9).sum()),
        mean_delta=float(delta.mean()),ci95=[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],
        caveat='clip-level bootstrap, not signer-clustered; tune evaluation only')
    ex.save(dest/'paired_stats.json',stats)
    (dest/'REPORT.md').write_text('# TSL51 tune — baseline vs combined without CSLS\n\n'+
        '126 real cached pose sequences; evaluation bank; test split not opened. Presence Recall follows pitty. Segment F1 and accepted correctness unavailable without human temporal labels.\n\n```\n'+
        summary.round(3).to_string(index=False)+'\n```\n\n```json\n'+json.dumps(stats,indent=2)+'\n```\n',encoding='utf-8')
    assert ex.digest(ex.ROOT/'models/spotting.json')==manifest['config_hash']
    ex.save(de.OUT/'latest_tsl51.json',{'path':str(dest)})
    print(summary.round(3).to_string(index=False),flush=True);print('SAVED',dest,flush=True)
    return dest


if __name__=='__main__':run()
