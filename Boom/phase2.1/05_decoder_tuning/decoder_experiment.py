"""Controlled decoder ablations using actual SignPipeline and shared real landmarks."""
from __future__ import annotations
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'02_hybrid_score'))
import experiment_analysis as ex
OUT=HERE/'outputs'
CASES={
    'baseline':{},
    'activity060':{'act_thr':.60},
    'min_len16':{'min_len':16},
    'exact_rescore':{'exact_rescore':True},
    'post_frames12':{'post':{'min_frames':12}},
    'post_psign070':{'post':{'p_sign_min':.70}},
    'grammar070':{'grammar_rerank_weight':.70},
    'stride2':{'stride':2},
    'csls015':{'csls_alpha':.15},
    'combined_without_csls':{'act_thr':.60,'min_len':16,'exact_rescore':True,'grammar_rerank_weight':.70,
                             'post':{'min_frames':12,'p_sign_min':.70}},
}


def resolved(base: dict,patch: dict) -> dict:
    """Deep-copy settings and merge the post block without dropping existing keys."""
    result=copy.deepcopy(base)
    for key,value in patch.items():
        if key=='post':result['post'].update(value)
        else:result[key]=value
    return result


def plan() -> pd.DataFrame:
    """Show exact experiment changes and known prerequisite gaps."""
    base=ex.read(ROOT/'models/spotting.json')['conversation']
    return pd.DataFrame([dict(case=name,changes=json.dumps(patch,ensure_ascii=False),
        status='blocked: models/hub.npy missing' if patch.get('csls_alpha') and not (ROOT/'models/hub.npy').exists() else 'ready',
        effective_config=json.dumps(resolved(base,patch),ensure_ascii=False)) for name,patch in CASES.items()])


def run(names: list[str],video_names: list[str]) -> None:
    """Run configured ablations; write no production files and no inferred frame truth."""
    os.environ['MP_HOLISTIC_MODEL']=str(ROOT/'models/mediapipe/holistic_landmarker.task')
    os.environ['THAISLM_SYMBOLIC_WEIGHT']='0'
    from modules.pipeline import SignPipeline
    pipe=SignPipeline(backend='onnx',device='cpu',vocab='conversation',use_llm=False,use_face=False)
    pipe.symbolic_weight=0
    base=copy.deepcopy(pipe.spot)
    stamp=time.strftime('%Y%m%d_%H%M%S')
    dest=OUT/stamp;dest.mkdir(parents=True,exist_ok=False)
    snapshot={str(p.relative_to(ROOT)):ex.digest(p) for p in [ROOT/'models/spotting.json',ROOT/'models/bank.npz',
        ROOT/'models/onnx/tagger.onnx',ROOT/'modules/pipeline.py',ROOT/'modules/decode.py']}
    model=ROOT/'models/mediapipe/holistic_landmarker.task'
    configs={name:resolved(base,CASES[name]) for name in names}
    blocked={name:'missing hub; do not silently run alpha with no effect' for name in names if configs[name].get('csls_alpha') and pipe.hub is None}
    manifest=dict(created=stamp,scope='development diagnostics; not held-out test',configs=configs,blocked=blocked,
        symbolic_weight=0,alphabet='off',use_llm=False,hashes=snapshot,
        mediapipe=importlib.metadata.version('mediapipe'),python=sys.executable)
    ex.save(dest/'manifest.json',manifest)
    rows=[]
    for video_name in video_names:
        video=ROOT/'Boom/boom datatest'/video_name
        if video not in ex.videos():raise ValueError('Only the authorized six Boom clips may be used')
        key=ex.digest(video)+ex.digest(model)+ex.digest(ROOT/'modules/mediapipe_pose.py')+manifest['mediapipe']
        import hashlib
        cache=OUT/'raw_cache'/(hashlib.sha256(key.encode()).hexdigest()+'.npz')
        if cache.exists():
            with np.load(cache,allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
            extraction_s=None;cached=True
        else:
            t=time.perf_counter();raw=pipe.landmarks_from_video(video);extraction_s=time.perf_counter()-t;cached=False
            if raw is None:raise RuntimeError(f'No landmarks: {video}')
            cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,**raw)
        modes={}
        for name in names:
            if name in blocked:print('SKIP',name,blocked[name],flush=True);continue
            pipe.spot=copy.deepcopy(configs[name])
            if configs[name].get('exact_rescore') and pipe.span is None:raise RuntimeError('Exact rescore prerequisite: missing span head')
            print('RUN',video_name,name,flush=True)
            t=time.perf_counter();result=pipe.run(video,raw=copy.deepcopy(raw),alphabet='off');elapsed=time.perf_counter()-t
            for segment in result['segments']:segment.pop('emb',None)
            modes[name]=dict(weight=0,inference_s=elapsed,result=result)
            print('DONE',name,round(elapsed,2),'seconds',len(result['segments']),'segments',flush=True)
            ex.save(dest/'partial_current_clip.json',dict(video=video_name,modes=modes))
        rows.append(dict(video=video_name,extraction_s=extraction_s if extraction_s is not None else 0.,
                         extraction_cached=cached,modes=modes))
        ex.save(dest/'results.json',rows)
    metrics,segments,candidates=ex.tables(rows)
    metrics['latency_s']=np.nan # not comparable end-to-end when extraction is cached
    metrics['real_time_factor']=np.nan
    for name,df in [('metrics',metrics),('segments',segments),('top5',candidates)]:df.to_csv(dest/f'{name}.csv',index=False)
    # Guard against any unexpected production mutation during the experiment.
    assert all(ex.digest(ROOT/k)==v for k,v in snapshot.items()),'Production inputs changed during run'
    ex.save(OUT/'latest.json',{'path':str(dest)})
    (dest/'REPORT.md').write_text('# Decoder ablation — development diagnostics\n\n'+
        'No production configs changed. CSLS is blocked without a verified hub. WER uses filename text tokenization; segment F1 needs timed labels.\n\n'+
        '```\n'+metrics[['video','mode','WER','substitutions','deletions','insertions','n_pred','inference_s']].round(3).to_string(index=False)+'\n```\n',encoding='utf-8')
    print('SAVED',dest,flush=True)


def launch(names: list[str],video_names: list[str]) -> None:
    """Stream native logs while preserving a copy for troubleshooting."""
    OUT.mkdir(parents=True,exist_ok=True)
    args=[str(ex.PYTHON),'-u',str(Path(__file__)),'--cases',*names,'--videos',*video_names]
    with (OUT/'run.log').open('w') as log:
        child=subprocess.Popen(args,cwd=ROOT,env=ex.subprocess_environment(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=child.wait()
        except BaseException:child.terminate();child.wait();raise
    if code:raise RuntimeError(f'Experiment exited {code}; see {OUT}/run.log')


def results():
    """Load the last completed sweep, never incomplete artifacts."""
    directory=Path(ex.read(OUT/'latest.json')['path'])
    return directory,pd.read_csv(directory/'metrics.csv'),pd.read_csv(directory/'segments.csv'),pd.read_csv(directory/'top5.csv')


def compare() -> None:
    """Compare errors, latency and segment boundaries with the actual baseline."""
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    directory,m,s,c=results()
    ex.show_table(m[['video','mode','WER','token_accuracy','substitutions','deletions','insertions','n_pred','count_error','inference_s','prediction']],
        '**อ่านผล:** WER ต่ำลงต้องตรวจ S/D/I ด้วย เพราะลดคำแทรกอาจเพิ่มคำตกหล่น; inference_s ไม่รวม extraction/model loading และเป็น single run')
    base=m[m['mode']=='baseline'][['video','WER','deletions','insertions','n_pred','inference_s']]
    delta=m.merge(base,on='video',suffixes=('','_baseline'))
    for field in ['WER','deletions','insertions','n_pred','inference_s']:delta['delta_'+field]=delta[field]-delta[field+'_baseline']
    ex.show_table(delta[['video','mode']+[c for c in delta if c.startswith('delta_')]],'**Delta = variant − baseline:** ไม่ใช้ผลคลิปเดียวเลือก production config; ชุดนี้เป็น development หลังตรวจปัญหาแล้ว')
    for field in ['WER','deletions','insertions','inference_s']:
        ax=m.pivot(index='video',columns='mode',values=field).plot.bar(figsize=(12,4),title=field,rot=20)
        ax.figure.tight_layout();ax.figure.savefig(directory/f'{field}.png');plt.show()
        display(Markdown(f'**{field}:** เทียบแต่ละ setting บนคลิปเดียวกันและ landmarks ชุดเดียวกัน'))
    ex.show_table(s[['video','mode','segment','t0','t1','word','status','p_correct']],
        '**ขอบเขตช่วง:** ดูว่าทั้ง ฉัน/แอบ/ชอบ/คุณ ยังอยู่ครบหรือไม่; p_correct จาก calibrator เดิมอาจเปลี่ยนความน่าเชื่อถือหลังปรับ config')
    ex.show_table(c,'**ผู้สมัครจริง:** คะแนน cosine ไม่ใช่ probability; กรณีเปลี่ยนขอบเขตไม่ควรจับคู่ด้วย segment index อย่างเดียว')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cases',nargs='+',choices=list(CASES),default=['baseline','combined_without_csls'])
    parser.add_argument('--videos',nargs='+',default=['ฉันแอบชอบคุณ.MOV']);args=parser.parse_args();run(args.cases,args.videos)


def sweep_summary() -> pd.DataFrame:
    """Summarize actual paired development outcomes without claiming test generalization."""
    directory,m,s,c=results()
    baseline=m[m['mode']=='baseline'].set_index('video')
    rows=[]
    for name,g in m.groupby('mode',sort=False):
        aligned=g.set_index('video').join(baseline[['WER','inference_s']],rsuffix='_baseline')
        delta=aligned.WER-aligned.WER_baseline
        errors=g[['substitutions','deletions','insertions']].sum().sum()
        rows.append(dict(mode=name,clips=len(g),macro_WER=g.WER.mean(),
            micro_WER=errors/g.n_ref.sum() if g.n_ref.sum() else np.nan,
            delta_macro_WER=delta.mean(),improved=int((delta < -1e-9).sum()),
            unchanged=int((delta.abs()<=1e-9).sum()),worsened=int((delta>1e-9).sum()),
            substitutions=int(g.substitutions.sum()),deletions=int(g.deletions.sum()),insertions=int(g.insertions.sum()),
            total_predicted_segments=int(g.n_pred.sum()),mean_count_error=g.count_error.mean(),
            mean_inference_s=g.inference_s.mean(),median_inference_s=g.inference_s.median()))
    summary=pd.DataFrame(rows)
    summary.to_csv(directory/'summary.csv',index=False)
    return summary


def finish_report() -> Path:
    """Write paired summary and clip-level evidence into one Markdown artifact."""
    directory,m,s,c=results();summary=sweep_summary()
    manifest=ex.read(directory/'manifest.json')
    baseline=m[m['mode']=='baseline'].set_index('video')
    report=['# Decoder tuning — real six-video development sweep',
        'These videos were inspected to propose changes. Results are development diagnostics, not an independent held-out evaluation.',
        'WER compares Thai filename sentences against concatenated retrieval output using the same newmm tokenizer. WER can exceed 1.0 when insertion errors are numerous. Timed gloss boundaries are not verified.',
        'All variants share extracted landmarks per clip. Inference latency excludes extraction/model load and is measured once in fixed order; do not treat timing as a rigorous benchmark.',
        'Production config/model/bank hashes were verified unchanged. No configuration is automatically promoted.',
        '## Aggregate results\n```\n'+summary.round(3).to_string(index=False)+'\n```',
        '## Per-video results\n```\n'+m[['video','mode','WER','substitutions','deletions','insertions','n_pred','inference_s','prediction']].round(3).to_string(index=False)+'\n```',
        '## Skipped cases\n```json\n'+json.dumps(manifest['blocked'],ensure_ascii=False,indent=2)+'\n```',
        '## Interpretation limits\nOne-factor ablations isolate configuration changes, but both pre/post-processing and recalibration assumptions can affect the result. The combined case omits CSLS explicitly. Grammar weights may improve text agreement without validating visual recognition. Candidate changes are not a substitute for human boundary annotation.',
        '## Vocabulary constraint\nThe conversation vocabulary contains ฉัน, ชอบ, คุณ but not แอบ or แอบชอบ. A closed-set decoder cannot output that missing word. Vocabulary/bank changes require a separate controlled experiment.',
        '## Next decision\nReview the winning and losing clips and annotate the four words ฉัน/แอบ/ชอบ/คุณ in time. Confirm configurations on separate development data and evaluate the final fixed configuration on a new held-out set. Do not infer that a lower count alone is better segmentation.']
    path=directory/'REPORT.md';path.write_text('\n\n'.join(report),encoding='utf-8')
    return path


def launch_tsl51() -> None:
    """Run only TSL51 tune; preserve native log and leave held-out test untouched."""
    env=ex.subprocess_environment();env['ORT_THREADS']='2'
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'tsl51.log').open('w') as log:
        child=subprocess.Popen([str(ex.PYTHON),'-u',str(HERE/'tsl51_experiment.py')],cwd=ROOT,
            env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=child.wait()
        except BaseException:child.terminate();child.wait();raise
    if code:raise RuntimeError(f'TSL51 failed with {code}; see {OUT}/tsl51.log')


def show_tsl51() -> None:
    """Display actual TSL51 tune outcomes separately from Boom text-token metrics."""
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    directory=Path(ex.read(OUT/'latest_tsl51.json')['path'])
    summary=pd.read_csv(directory/'summary.csv');m=pd.read_csv(directory/'metrics.csv')
    ex.show_table(summary,'**TSL51 tune:** 126 ประโยคจริง, gloss labels เดิม, evaluation bank. Recall เป็น presence-based ตาม pitty; ไม่เทียบตัวเลขตรงกับ WER ข้อความของ Boom')
    ex.show_table(pd.DataFrame([ex.read(directory/'paired_stats.json')]),
        '**จับคู่คลิป:** delta = combined − baseline; CI เป็น bootstrap ระดับคลิป ไม่ใช่ระดับผู้ทำท่า และเป็นผลบน tune')
    paired=pd.read_csv(directory/'paired_wer.csv')
    for label,df in [('ดีขึ้นมากสุด',paired.sort_values('delta').head(10)),('แย่ลงมากสุด',paired.sort_values('delta',ascending=False).head(10))]:
        ex.show_table(df,label+' — เปิดคู่คำอ้างอิง/คำทำนายด้านล่างเพื่อตรวจ ไม่เลือกเฉพาะตัวอย่างที่ดี')
    ax=paired.delta.hist(bins=20,figsize=(10,4));ax.set(xlabel='Combined WER − baseline WER',ylabel='clips',title='TSL51 tune paired WER changes')
    ax.figure.tight_layout();ax.figure.savefig(directory/'paired_delta_hist.png');plt.show()
    display(Markdown('**กราฟ:** ซ้ายศูนย์=WER ลดลง ขวาศูนย์=แย่ลง ไม่ได้พิสูจน์ขอบเขต segment ถูก เพราะยังไม่มี human timed labels'))
    ex.show_table(m[['clip','mode','WER','S','D','I','recall1','recall5','n_pred','n_ref','inference_s','ref','hyp']],
        '**ผลรายประโยคครบ:** reference gloss เป็นลำดับจริงใน dataset; พิกัด alignment เดิมไม่ถูกใช้เป็น human ground truth เพื่อสร้าง F1')
