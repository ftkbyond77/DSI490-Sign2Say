"""Explicit dev-cache allowlist. Never opens source videos or held-out cache."""
import argparse,gzip,json,os,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from features import stream,profile
from segment_merger import assess
from reranker import rerank,rerank_detailed,load_manual
from manual_features import detailed_profile

def raw_stream(raw):
    n=len(raw['t']);kp=np.zeros((n,133,2));sc=np.zeros((n,133))
    for dst,src in [(0,0),(5,11),(6,12)]:kp[:,dst]=raw['pose'][:,src,:2];sc[:,dst]=raw['pose'][:,src,3]
    for dst,key in [(91,'lh'),(112,'rh')]:
        kp[:,dst]=raw[key][:,0,:2];sc[:,dst]=(np.abs(raw[key]).sum((1,2))>0)
    return stream(kp,sc,raw['hw'],raw['t'])
def gate(s):
    observed=s['observed'].mean(0)
    valid=s['observed'][1:]&s['observed'][:-1]
    jumps=(np.linalg.norm(np.diff(s['xy'],axis=0),axis=-1)>.4)&valid
    rate=jumps.sum(0)/np.maximum(valid.sum(0),1)
    return dict(block_flip=bool((observed<.5).any() or (rate>.5).any()),observed_rate=observed.tolist(),jump_rate=rate.tolist())
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    os.environ['ORT_THREADS']='2';sys.path.insert(0,str(a.repo))
    from modules.pipeline import SignPipeline
    from modules.schema import SH_L,SH_R,canonical_hands
    from modules.mediapipe_pose import from_browser,to_standard
    from modules.parts import to_iso
    from modules.decode import active_from_prob
    from modules.segment import window_list
    from modules.runtime import table_from_Z
    class GatedPipeline(SignPipeline):
        def standardise(self,raw):
            d=to_standard(raw,gap_s=float(self.spot_all.get('hand_gap_s',.2)))
            kp,sc,hw=d['kp'].copy(),d['sc'].copy(),tuple(d['hw'])
            present=sc[:,[0,SH_L,SH_R]].mean(1)>.5;kp[~present]=0;sc[~present]=0
            if gate(raw_stream(raw))['block_flip']:return kp,sc,hw,present,False
            kp,sc,mirrored=canonical_hands(kp,sc,hw);return kp,sc,hw,present,mirrored
    def fingerprints():
        return {str(p.relative_to(a.repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.repo/'models').iterdir() if p.is_file() and p.suffix in ('.onnx','.npz','.json')}
    before=fingerprints();rows=[]
    manual_path=Path(__file__).with_name('manual_heuristics_tsl_detailed.json')
    manual=load_manual(manual_path)
    manual_hash=hashlib.sha256(manual_path.read_bytes()).hexdigest()
    table=pd.read_csv(a.out/'bank_heuristics_326.csv').set_index('concept').to_dict('index')
    pipe=GatedPipeline(model_dir=a.repo/'models',backend='onnx',device='cpu',vocab='conversation',use_llm=False)
    manifest=json.loads((a.cache/'split.json').read_text())
    for ident in ('clip1','clip2'):
        assert next(x for x in manifest['clips'] if x['id']==ident)['split']=='development'
        d=json.load(gzip.open(a.cache/(ident+'_browser.json.gz'),'rt'));raw=from_browser(d['frames'],(d['height'],d['width']));s=raw_stream(raw)
        result=pipe.run_landmarks(raw,title=ident,tts=False,alphabet='off')
        kp,sc,hw,present,_=pipe.standardise(raw);prob,H=pipe._tag(to_iso(kp,hw),sc,present)
        if pipe.tempo_factor(prob)!=1:raise ValueError('Time mapping requires tempo_factor=1 in this evaluator')
        act,_=active_from_prob(prob,pipe.spot['act_thr'],present)
        wins=window_list(len(kp),pipe.spot['lens'],pipe.spot['stride'],act=act,min_active=pipe.spot.get('min_inside',.6))
        tab=table_from_Z(pipe.span(H,wins),wins,pipe.bank,pipe.null_proto);W=np.asarray(wins)
        def support(a0,b0):
            mask=(W[:,0]>=round(a0*25)-2)&(W[:,1]<=round(b0*25)+2)&((W[:,1]-W[:,0])>=pipe.spot['min_len'])
            if not mask.any():return []
            ids=tab['top_idx'][mask,:5].ravel();vals=tab['top_sc'][mask,:5].ravel();u,inv=np.unique(ids,return_inverse=True);sums=np.zeros(len(u));np.add.at(sums,inv,vals)
            return [dict(word=pipe.concepts[int(u[j])],support=float(sums[j]/mask.sum()),cosine=float(vals[ids==u[j]].max())) for j in np.argsort(-sums)[:5]]
        segs=[dict(t0=x['t0'],t1=x['t1'],top5=[c['word'] for c in x['candidates'][:5]],nearest=x['nearest'],original_p=x['p_correct']) for x in result['segments']]
        proposals=[dict(pair=[i,i+1],**assess(x,y,s)) for i,(x,y) in enumerate(zip(segs,segs[1:]))]
        merged=[];decisions=[]
        for seg in segs:
            if merged:
                verdict=assess(merged[-1],seg,s);decisions.append(verdict)
                if verdict['merge']:
                    start=merged[-1]['t0'];cs=support(start,seg['t1'])
                    if cs:merged[-1]=dict(t0=start,t1=seg['t1'],top5=[c['word'] for c in cs],merged=True);continue
            merged.append(dict(seg))
        def score_segments(items):
            out=[]
            for seg in items:
                q=profile(s,seg['t0'],seg['t1']);cs=support(seg['t0'],seg['t1']);rr=rerank(cs,q,table)
                detail=detailed_profile(s,seg['t0'],seg['t1'])
                updated=rerank_detailed(cs,q,table,manual,detail)
                out.append(dict(segment=seg,profile=q,detailed_profile=detail,temporal_top5=cs,reranked_top5=rr,detailed_top5=updated))
            return out
        rows.append(dict(clip=ident,baseline=json.loads((a.cache/(ident+'_baseline.json')).read_text()),quality_gate=gate(s),gated_segments=segs,adjacent_proposals=proposals,merge_decisions=decisions,without_merge=score_segments(segs),combined=score_segments(merged)))
    assert before==fingerprints(),'Production files changed'
    (a.out/'dev_results_manual.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False))
    (a.out/'verification_manual.json').write_text(json.dumps(dict(production_unchanged=True,model_hashes=before,clips_read=['clip1','clip2'],heldout_run=False,alpha=.1,manual_weight=.5,manual_rules_sha256=manual_hash),indent=2))
    print(json.dumps([dict(clip=r['clip'],gate=r['quality_gate'],merges=r['adjacent_proposals'],final=[x['detailed_top5'][0]['word'] for x in r['combined'] if x['reranked_top5']]) for r in rows],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
