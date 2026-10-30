"""Frozen test evaluation. No tuning or production mutation."""
from __future__ import annotations
import copy, hashlib, json, os, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
os.environ.update(MPLBACKEND='Agg',ORT_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',THAISLM_SYMBOLIC_WEIGHT='0',MPLCONFIGDIR=str(HERE/'.mplconfig'))
sys.path[:0]=[str(ROOT/'Boom/phase2.1/02_hybrid_score/runtime_packages'),str(ROOT)]
import numpy as np
import pandas as pd

def save(p:Path,x:object)->None:
    """Atomically persist genuine results."""
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)));t.replace(p)
def digest(p:Path)->str:
    """Hash actual model/code artifacts."""
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def align(ref:list,hyp:list)->list:
    """Deterministic minimum-edit alignment, diagonal preferred on ties."""
    n,m=len(ref),len(hyp);d=np.zeros((n+1,m+1),int);d[:,0]=np.arange(n+1);d[0,:]=np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):d[i,j]=min(d[i-1,j-1]+(ref[i-1]!=hyp[j-1]),d[i-1,j]+1,d[i,j-1]+1)
    out=[];i,j=n,m
    while i or j:
        if i and j and d[i,j]==d[i-1,j-1]+(ref[i-1]!=hyp[j-1]):out.append((i-1,j-1,'C' if ref[i-1]==hyp[j-1] else 'S'));i-=1;j-=1
        elif i and d[i,j]==d[i-1,j]+1:out.append((i-1,None,'D'));i-=1
        else:out.append((None,j-1,'I'));j-=1
    return out[::-1]
def summarize(dest:Path,rows:list)->None:
    """Aggregate test metrics without selecting settings."""
    df=pd.DataFrame(rows);df.to_csv(dest/'metrics.csv',index=False)
    summary=df.groupby(['speed','case']).agg(clips=('clip','size'),WER=('WER','mean'),S=('S','sum'),D=('D','sum'),I=('I','sum'),n_ref=('n_ref','sum'),recall1=('recall1','mean'),recall5=('recall5','mean'),count_error=('count_error','mean'),accepted=('accepted','sum'),correct=('accepted_correct','sum'),inference_s=('inference_s','mean')).reset_index()
    summary['accepted_precision']=summary.correct/summary.accepted.replace(0,np.nan);summary['corpus_WER']=(summary.S+summary.D+summary.I)/summary.n_ref
    summary.to_csv(dest/'summary.csv',index=False)
    base=summary[summary['case']=='baseline'].set_index('speed').WER
    summary['delta_WER']=summary.apply(lambda r:r.WER-base.loc[r.speed],axis=1)
    gates=[]
    for name,g in summary.groupby('case'):
        full=len(g)==3 and (g.clips==126).all()
        gates.append(dict(case=name,complete=bool(full),mean_delta=g.delta_WER.mean(),worst_delta=g.delta_WER.max(),minimum_precision=g.accepted_precision.min(),pass_gate=bool(full and g.accepted_precision.notna().all() and (g.accepted_precision>=.9).all() and g.delta_WER.max()<=.003 and g.delta_WER.mean()<-.002)))
    pd.DataFrame(gates).drop(columns=['pass_gate']).to_csv(dest/'descriptive_checks.csv',index=False)
    (dest/'REPORT.md').write_text('# Boom Phase 2.2 — frozen test\n\nReal pipeline, current baseline; no selection on these test results. Grammar held fixed with previously identified reference leakage: exploratory evidence, no production promotion.\n\n'+summary.round(4).to_string(index=False)+'\n\n'+pd.DataFrame(gates).drop(columns=['pass_gate']).round(4).to_string(index=False)+'\n\nLatency excludes landmark extraction and model load; no inference memoization. Recall@5 uses the top-1 edit alignment (not presence anywhere). Segment F1 is unavailable in this run.\n')
def run(plan_path: Path | None = None)->None:
    """Execute single-variable tune trials at three speeds, recording all segments."""
    from modules.pipeline import SignPipeline
    from modules.schema import canonical_hands,SH_L,SH_R
    from modules.segment import resample
    lock=json.loads((HERE/'TEST_STARTED.json').read_text());dest=Path(lock['output']).resolve()
    if (dest/'DONE.json').exists():raise RuntimeError('Test already completed')
    save(HERE/'latest_test.json',{'path':str(dest)})
    files=['models/config.json','models/spotting.json','models/segment_calibration.json','models/grammar.json','models/concepts.json','models/vocab_conversation.json','models/onnx/encoder.onnx','models/onnx/tagger.onnx','artifacts/eval/bank_eval.npz','modules/pipeline.py','modules/decode.py']
    hashes={f:digest(ROOT/f) for f in files}
    frozen=json.loads((HERE/'FROZEN_TEST.json').read_text())
    assert hashes==frozen['hashes'], 'Inputs differ from frozen tune artifacts'
    assert lock['frozen_sha256']==digest(HERE/'FROZEN_TEST.json'), 'Frozen config changed'
    plan=json.loads((plan_path or HERE/'FROZEN_TEST.json').read_text());cases=plan['trials']
    pipe=SignPipeline(backend='onnx',device='cpu',vocab='conversation',use_llm=False,use_face=False,bank_file=ROOT/'artifacts/eval/bank_eval.npz')
    base=copy.deepcopy(pipe.spot);cal=copy.deepcopy(pipe.cal);origcal=json.loads((ROOT/'models/segment_calibration.json').read_text());origcal=origcal.get('conversation',origcal)
    state={}
    def standard(raw):
        kp,sc,hw=state['kp'].copy(),state['sc'].copy(),state['hw'];present=sc[:,[0,SH_L,SH_R]].mean(1)>.5;kp[~present]=0;sc[~present]=0;kp,sc,m=canonical_hands(kp,sc,hw);return kp,sc,hw,present,m
    pipe.standardise=standard
    allalign=json.loads((ROOT/'artifacts/runs/seq-v7/tsl51_alignment.json').read_text())
    entries=[(cid,a) for cid,a in allalign.items() if cid.startswith('t51s:') and a['role']=='sentence_test']
    assert len(entries)==126, 'Unexpected test size'
    manifest=dict(hashes=hashes,cases=cases,n_test=len(entries),speeds=[1,1.5,2],bank='artifacts/eval/bank_eval.npz',baseline_cal=cal,baseline_spot=base,grammar='unchanged; prior leakage known; exploratory only',test_run=True,production_enable=False,latency='warm pipeline from standardized poses; extraction excluded; no memoization')
    previous=[json.loads(line) for line in (dest/'predictions.jsonl').read_text().splitlines()]
    completed={(r['speed'],r['clip'],r['case']) for r in previous}
    assert len(completed)==len(previous), 'Duplicate test results'
    rows=[{k:v for k,v in r.items() if k not in ('reference','result')} for r in previous]
    save(dest/'RESUMED.json',dict(reason='Original process lost write permission after directory rename',existing_rows=len(rows),rerun_completed_cases=False))
    with (dest/'predictions.jsonl').open('a') as out:
        for speed in [1.,1.5,2.]:
            for idx,(cid,a) in enumerate(entries):
                path=ROOT/'cloud_s3/data_prep/pose_mp/tsl51_user_sentence'/f'{cid.split(":",1)[1]}.npz'
                with np.load(path) as z:kp=z['kp'].astype(np.float32);sc=z['sc'].astype(np.float32);hw=tuple(map(int,z['hw']))
                if speed!=1:kp,sc=resample(kp,sc,speed)
                state.update(kp=kp,sc=sc,hw=hw);ref=a['glosses']
                for name,patch in cases.items():
                    if (speed,cid,name) in completed:continue
                    pipe.spot=copy.deepcopy(base);pipe.cal=copy.deepcopy(cal)
                    for key,value in patch.items():
                        if key=='accept_min':pipe.cal['tau_accept']=max(float(origcal.get('tau_accept',0)),value)
                        elif key.startswith('post.'):pipe.spot['post'][key.split('.',1)[1]]=value
                        else:pipe.spot[key]=value
                    start=time.perf_counter();r=pipe.run_landmarks(None,alphabet='off');elapsed=time.perf_counter()-start
                    segs=[s for s in r['segments'] if s['status']!='transition'];hyp=[s['nearest'] for s in segs];al=align(ref,hyp)
                    counts={op:sum(x[2]==op for x in al) for op in ['S','D','I','C']};correct={j for i,j,op in al if op=='C'}
                    r5=sum(i is not None and j is not None and ref[i] in [c['word'] for c in segs[j]['candidates'][:5]] for i,j,op in al)
                    row=dict(speed=speed,clip=cid,case=name,WER=(counts['S']+counts['D']+counts['I'])/len(ref),**counts,n_ref=len(ref),n_pred=len(hyp),recall1=counts['C']/len(ref),recall5=r5/len(ref),count_error=abs(len(hyp)-len(ref)),accepted=sum(s['status']=='accepted' for s in segs),accepted_correct=sum(s['status']=='accepted' and j in correct for j,s in enumerate(segs)),inference_s=elapsed,effective_tau_accept=pipe.cal['tau_accept'])
                    rows.append(row)
                    for s in r['segments']:s.pop('emb',None)
                    out.write(json.dumps(dict(**row,reference=ref,result=r),ensure_ascii=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n');out.flush()
                if idx%10==0:print(f'speed={speed} clip={idx+1}/{len(entries)} rows={len(rows)}',flush=True);summarize(dest,rows)
            summarize(dest,rows)
    assert all(digest(ROOT/f)==h for f,h in hashes.items()),'Input files changed during run'
    save(dest/'DONE.json',dict(rows=len(rows),immutable_inputs_verified=True));print('DONE',dest,flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path)
    run(parser.parse_args().plan)
