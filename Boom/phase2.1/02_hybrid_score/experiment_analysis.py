"""Real-data Hybrid experiment: provenance, inference, metrics and visual audit.

Sentence ground truth is the filename, explicitly confirmed by the user.
Temporal boundaries remain unannotated; tokenization is displayed for review.
All inference uses modules.pipeline.SignPipeline; symbolic logic is not reimplemented.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = ROOT / 'Boom/venv-sign2say/bin/python'
OUT = HERE / 'boom_datatest/scientific'
RUNTIME_PACKAGES = HERE / 'runtime_packages'
if (RUNTIME_PACKAGES/'mediapipe').is_dir():
    sys.path.insert(0, str(RUNTIME_PACKAGES))
sys.path.insert(0, str(ROOT))


def read(path: Path):
    """Read UTF-8 JSON without silently substituting absent data."""
    return json.loads(path.read_text(encoding='utf-8'))


def save(path: Path, value):
    """Persist JSON atomically, supporting NumPy values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)), encoding='utf-8')
    tmp.replace(path)


def digest(path: Path) -> str:
    """SHA256 provenance for exact code/data/model identity."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def setup():
    """Configure Thai figures and readable tables in the notebook kernel."""
    os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.mplconfig'))
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Tahoma','Leelawadee UI','DejaVu Sans'],
                         'figure.dpi':110, 'savefig.dpi':180})
    # macOS may lack the requested Windows fonts; retain Thai rendering if available.
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = [name for name in ['Tahoma','Leelawadee UI','DejaVu Sans'] if name in available]
    if not available.intersection({'Tahoma','Leelawadee UI'}):
        for name in ['Thonburi','Arial Unicode MS','Noto Sans Thai']:
            if name in available:
                plt.rcParams['font.family'] = [name,'DejaVu Sans']; break
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')


def subprocess_environment() -> dict[str, str]:
    """Use a noninteractive backend in child processes, preserving notebook plots."""
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    if (RUNTIME_PACKAGES/'mediapipe').is_dir():
        env['PYTHONPATH'] = str(RUNTIME_PACKAGES) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def environment() -> pd.DataFrame:
    """Inspect actual project interpreter and import production dependencies."""
    code = """import sys,json,platform,torch,onnxruntime,mediapipe
from modules.pipeline import SignPipeline
from modules.symbolic import observe,score
print(json.dumps({'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__,'onnxruntime':onnxruntime.__version__,'providers':onnxruntime.get_available_providers(),'mediapipe':mediapipe.__version__,'project_imports':'OK'}))"""
    if not PYTHON.exists(): raise FileNotFoundError(PYTHON)
    p = subprocess.run([str(PYTHON), '-c', code], cwd=ROOT, text=True, capture_output=True, env=subprocess_environment())
    if p.returncode: raise RuntimeError(f'Project environment preflight failed:\n{p.stderr}')
    values = json.loads(p.stdout.strip().splitlines()[-1])
    values['notebook_kernel'] = sys.executable
    return pd.DataFrame([{'component':k,'value':str(v)} for k,v in values.items()])


def videos() -> list[Path]:
    """Discover exactly the six authorized real videos, without traversing held-out folders."""
    files = sorted(p for p in (ROOT/'Boom/boom datatest').iterdir() if p.suffix.lower() in {'.mov','.mp4'})
    if len(files) != 6: raise ValueError(f'Expected six videos, found {len(files)}')
    return files


def video_worker(request: dict) -> list[dict]:
    """Isolate OpenCV/NumPy ABI from the user's Jupyter kernel."""
    result = subprocess.run([str(PYTHON), str(HERE/'video_io.py')],
        input=json.dumps(request), text=True, capture_output=True, cwd=ROOT,
        env=subprocess_environment())
    if result.returncode:
        raise RuntimeError(f'Project video worker failed:\n{result.stderr}')
    return json.loads(result.stdout)


def inventory() -> pd.DataFrame:
    """Read metadata with project OpenCV, never importing cv2 into the notebook."""
    return pd.DataFrame(video_worker({'operation':'inventory',
        'videos':[{'path':str(p)} for p in videos()]}))


def build_and_audit():
    """Call real dictionary builder and validate every attribute variant."""
    from modules.symbolic import build_dictionary
    dictionary=build_dictionary(ROOT/'Boom/book-catalog/reviewed_entries.json',read(ROOT/'models/concepts.json'))
    if not dictionary: raise ValueError('Dictionary builder produced no mapped entries')
    save(HERE/'dictionary_attributes.json',dictionary)
    rows=[]
    allowed={'hands':{1,2},'location_zone':{'mouth','chest','face','low','arm'},
             'relation':{'together','apart','alternate'}}
    for word,variants in dictionary.items():
        if not isinstance(variants,list) or not variants: raise ValueError(word)
        for i,v in enumerate(variants):
            for k,vals in allowed.items():
                if v.get(k) is not None and v[k] not in vals: raise ValueError((word,k,v[k]))
            if not isinstance(v.get('motion'),list): raise ValueError((word,'motion'))
            rows.append(dict(word=word,variant=i,**v))
    df=pd.DataFrame(rows)
    fields=pd.DataFrame([dict(attribute=k,known=int(df[k].map(lambda x: x is not None and x!=[]).sum()),
                              total=len(df)) for k in ['hands','location_zone','relation','motion']])
    fields['known_fraction']=fields.known/fields.total
    distribution=[]
    for k in ['hands','location_zone','relation','motion']:
        vals=df[k].explode() if k=='motion' else df[k]
        for value,n in vals.fillna('UNKNOWN').value_counts().items():
            distribution.append(dict(attribute=k,value=str(value),variants=n))
    return df,fields,pd.DataFrame(distribution)


def coverage(split='test') -> pd.DataFrame:
    """Exact runtime-key coverage on real TSL51 reference occurrences, by frozen split."""
    dictionary=read(HERE/'dictionary_attributes.json')
    align=read(ROOT/'artifacts/runs/seq-v7/tsl51_alignment.json')
    counts=Counter(w for cid,a in align.items() if cid.startswith('t51s:')
        and ('tune' if a['role']=='tagger_train' else 'test')==split for w in a['glosses'])
    return pd.DataFrame([dict(word=w,occurrences=n,covered=w in dictionary,
                              variants=len(dictionary.get(w,[])),split=split) for w,n in counts.items()]).sort_values('occurrences',ascending=False)


def tokenize(text: str) -> list[str]:
    """Deterministic Thai word tokenization for text-to-text evaluation."""
    from pythainlp.tokenize import word_tokenize
    return [w.strip() for w in word_tokenize(text, engine='newmm') if w.strip()]


def annotation_template() -> Path:
    """Initialize user-confirmed filename sentences; temporal truth is still absent."""
    p=HERE/'ground_truth.json'
    if not p.exists():
        save(p, {'instructions':'Filename sentences confirmed by user. glosses are proposed sign-unit tokenization, editable after viewing. segments require human t0/t1/word annotation. sentence_text remains exact filename.',
                 'videos':{v.name:{'verified':True,'sentence_text':v.stem,
                    'source':'filename confirmed by user','glosses':tokenize(v.stem),
                    'gloss_order_verified':False,'segments':None,'annotator':'user (sentence); newmm (tokens)'} for v in videos()}})
    return p


def launch_inference() -> None:
    """Stream native stderr/stdout and persist a complete crash log for diagnosis."""
    from collections import deque
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT/'inference.log'
    tail = deque(maxlen=45)
    with log_path.open('w', encoding='utf-8') as log:
        child = subprocess.Popen([str(PYTHON), '-u', str(HERE/'experiment_analysis.py'), 'run'],
            cwd=ROOT, env=subprocess_environment(), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in child.stdout:
                print(line, end='', flush=True); log.write(line); log.flush(); tail.append(line)
            status = child.wait()
        except BaseException:
            child.terminate(); child.wait(); raise
    if status:
        raise RuntimeError(f'Inference exited {status}. Full log: {log_path}\n' + ''.join(tail))


def run_experiment():
    """Run actual pipeline twice per video using identical extracted landmarks.

Extraction is timed once and shared to prevent detector variability confounding
fusion. Latency reports inference and extraction separately, excluding model load.
"""
    os.environ['MP_HOLISTIC_MODEL']=str(ROOT/'models/mediapipe/holistic_landmarker.task')
    os.environ['THAISLM_SYMBOLIC_DICT']=str(HERE/'dictionary_attributes.json')
    os.environ['THAISLM_SYMBOLIC_WEIGHT']='0'
    from modules.pipeline import SignPipeline
    pipe=SignPipeline(backend='onnx',device='cpu',vocab='conversation',use_llm=False,use_face=False)
    from modules.symbolic import load_dictionary
    pipe.symbolic_dict=load_dictionary(HERE/'dictionary_attributes.json',pipe.concepts)
    if not pipe.symbolic_dict: raise ValueError('Symbolic dictionary empty')
    provenance={str(p.relative_to(ROOT)):digest(p) for p in [HERE/'dictionary_attributes.json',
        ROOT/'modules/pipeline.py',ROOT/'modules/symbolic.py',ROOT/'models/spotting.json',
        ROOT/'models/bank.npz',ROOT/'models/concepts.json',ROOT/'models/segment_calibration.json',
        *sorted((ROOT/'models/onnx').glob('*.onnx'))]}
    manifest=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),python=sys.executable,
                  versions={k:importlib.metadata.version(k) for k in ['torch','onnxruntime','numpy','mediapipe']},
                  hashes=provenance,weights=[0,0.3],shared_landmarks=True,
                  latency_policy='single measurement; extraction shared; model load excluded',
                  spot=pipe.spot,bank_note='production conversation bank; not signer-held-out evaluation')
    save(OUT/'manifest.json',manifest)
    results=[]
    for index,video in enumerate(videos()):
        print(f'[{index+1}/6] Extract {video.name}',flush=True)
        t=time.perf_counter(); raw=pipe.landmarks_from_video(video); extraction=time.perf_counter()-t
        if raw is None: raise RuntimeError(f'No decoded landmarks: {video}')
        modes={}
        # Alternate order to reduce consistent warm-up/order bias.
        for name,weight in ([('ai_only',0.),('hybrid030',.3)] if index%2==0 else [('hybrid030',.3),('ai_only',0.)]):
            pipe.symbolic_weight=weight
            t=time.perf_counter(); r=pipe.run(video,raw=copy.deepcopy(raw)); elapsed=time.perf_counter()-t
            if r.get('error'): raise RuntimeError(r['error'])
            for s in r['segments']: s.pop('emb',None)
            modes[name]=dict(weight=weight,inference_s=elapsed,result=r)
            print(f'  {name}: {elapsed:.2f}s, {len(r["segments"])} segments',flush=True)
        results.append(dict(video=video.name,sha256=digest(video),extraction_s=extraction,modes=modes))
        save(OUT/'results.json',results)  # preserve completed clips on interruption
    return results


def load_results():
    """Only accept current scientific-run output, not old mislabeled hybrid files."""
    p=OUT/'results.json'
    if not p.exists(): raise FileNotFoundError(f'Run inference cell first: {p}')
    data=read(p)
    if {r['video'] for r in data}!={p.name for p in videos()}: raise ValueError('Incomplete six-video run')
    for r in data:
        if r['modes']['hybrid030']['weight']!=.3: raise ValueError('Incorrect hybrid weight')
    return data


def edit_alignment(ref, hyp):
    """Levenshtein alignment preserving repeated words and insertion/deletion errors."""
    d=np.zeros((len(ref)+1,len(hyp)+1),int); d[:,0]=np.arange(len(ref)+1); d[0]=np.arange(len(hyp)+1)
    for i in range(1,len(ref)+1):
        for j in range(1,len(hyp)+1): d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
    i,j=len(ref),len(hyp); pairs=[]
    while i or j:
        if i and j and d[i,j]==d[i-1,j-1]+(ref[i-1]!=hyp[j-1]):
            pairs.append((i-1,j-1,'C' if ref[i-1]==hyp[j-1] else 'S'));i-=1;j-=1
        elif i and d[i,j]==d[i-1,j]+1: pairs.append((i-1,None,'D'));i-=1
        else: pairs.append((None,j-1,'I'));j-=1
    return list(reversed(pairs))


def iou(a,b):
    """Temporal intersection over union in seconds."""
    return max(0,min(a['t1'],b['t1'])-max(a['t0'],b['t0']))/max(1e-9,max(a['t1'],b['t1'])-min(a['t0'],b['t0']))


def temporal_pairs(pred,truth,threshold=.5):
    """Maximum-cardinality one-to-one temporal matches at IoU threshold, label agnostic."""
    edges={i:sorted([j for j,b in enumerate(truth) if iou(a,b)>=threshold],key=lambda j:-iou(a,truth[j])) for i,a in enumerate(pred)}
    matched={}
    def assign(i,visited):
        for j in edges[i]:
            if j in visited: continue
            visited.add(j)
            if j not in matched or assign(matched[j],visited): matched[j]=i;return True
        return False
    for i in range(len(pred)): assign(i,set())
    return {i:j for j,i in matched.items()}


def tables(data):
    """Compute metrics and detailed long-form candidate tables from real outputs."""
    annotations=read(annotation_template())['videos']; dictionary=read(HERE/'dictionary_attributes.json')
    metrics=[]; segments=[]; candidates=[]
    for row in data:
        ann=annotations.get(row['video'],{}); verified=ann.get('verified',False)
        ref=ann.get('glosses') if verified else None; truth=ann.get('segments') if verified else None
        if ref is not None and (not isinstance(ref,list) or not all(isinstance(w,str) for w in ref)): raise ValueError('glosses must be token list')
        if truth is not None:
            for g in truth:
                if not (0<=g['t0']<g['t1']) or not isinstance(g['word'],str): raise ValueError('Invalid timed annotation')
        for mode,m in row['modes'].items():
            r=m['result']; seg=[s for s in r['segments'] if s.get('status')!='transition']
            hyp=[s['nearest'] for s in seg]; pairs=edit_alignment(ref,hyp) if ref is not None else []
            counts=Counter(p[2] for p in pairs); gt_by_pred={j:ref[i] for i,j,op in pairs if i is not None and j is not None}
            matches=temporal_pairs(seg,truth) if truth is not None else {}
            if truth is not None: gt_by_pred={i:truth[j]['word'] for i,j in matches.items()}
            accepted=[i for i,s in enumerate(seg) if s.get('status')=='accepted']
            n=len(ref) if ref is not None else 0
            sentence_ref=tokenize(ann.get('sentence_text','')) if verified else []
            sentence_hyp=tokenize(r.get('sentence',''))
            retrieval_tokens=tokenize(''.join(hyp))
            retrieval_ops=Counter(p[2] for p in edit_alignment(sentence_ref,retrieval_tokens))
            sentence_ops=Counter(p[2] for p in edit_alignment(sentence_ref,sentence_hyp))
            metrics.append(dict(sentence_WER=(sentence_ops['S']+sentence_ops['D']+sentence_ops['I'])/len(sentence_ref) if sentence_ref else None,
                sentence_output=r.get('sentence',''),reference_type='filename sentence; proposed sign tokens',video=row['video'],mode=mode,n_pred=len(seg),n_ref=n if ref is not None else None,
                WER=(retrieval_ops['S']+retrieval_ops['D']+retrieval_ops['I'])/len(sentence_ref) if sentence_ref else None,
                proposed_gloss_WER=(counts['S']+counts['D']+counts['I'])/n if n else None,
                token_accuracy=retrieval_ops['C']/len(sentence_ref) if sentence_ref else None, exact_sentence=float(ref==hyp) if ref is not None else None,
                substitutions=retrieval_ops['S'] if sentence_ref else None,deletions=retrieval_ops['D'] if sentence_ref else None,insertions=retrieval_ops['I'] if sentence_ref else None,
                segment_F1=2*len(matches)/(len(seg)+len(truth)) if truth is not None and len(seg)+len(truth) else None,
                presence_recall1=sum(min(Counter(ref)[w],Counter(hyp)[w]) for w in set(ref))/n if n else None,
                presence_recall5=sum(min(Counter(ref)[w],sum(w in [c['word'] for c in s['candidates'][:5]] for s in seg)) for w in set(ref))/n if n else None,
                recall1=sum(seg[i]['nearest']==truth[j]['word'] for i,j in matches.items())/len(truth) if truth else None,
                recall5=sum(truth[j]['word'] in [c['word'] for c in seg[i]['candidates'][:5]] for i,j in matches.items())/len(truth) if truth else None,
                accepted_correct=sum(hyp[i]==gt_by_pred.get(i) for i in accepted) if truth is not None else None,
                accepted_total=len(accepted),accepted_precision=sum(hyp[i]==gt_by_pred.get(i) for i in accepted)/len(accepted) if truth is not None and accepted else None,
                count_error=abs(len(seg)-n) if ref is not None else None,
                inference_s=m['inference_s'],extraction_s=row['extraction_s'],latency_s=m['inference_s']+row['extraction_s'],
                real_time_factor=(m['inference_s']+row['extraction_s'])/r['duration_s'] if r['duration_s'] else None,
                prediction=' | '.join(hyp),reference=' | '.join(ref) if ref is not None else 'NOT ANNOTATED'))
            for i,s in enumerate(seg):
                target=gt_by_pred.get(i); obs=s.get('symbolic',{})
                segments.append(dict(video=row['video'],mode=mode,segment=i,t0=s['t0'],t1=s['t1'],word=s['nearest'],
                    status=s['status'],p_correct=s.get('p_correct'),ground_truth=target,
                    correctness=('✔' if target==s['nearest'] else '✗') if target is not None else ('✗' if truth is not None else '?'),
                    top1_covered=s['nearest'] in dictionary,top5_covered=any(c['word'] in dictionary for c in s['candidates'][:5]),
                    left_observed=obs.get('hand_observed',{}).get('left'),right_observed=obs.get('hand_observed',{}).get('right'),
                    observed_attributes=json.dumps(obs,ensure_ascii=False)))
                for rank,c in enumerate(s['candidates'][:5],1):
                    candidates.append(dict(video=row['video'],mode=mode,segment=i,t0=s['t0'],t1=s['t1'],rank=rank,word=c['word'],
                        cosine=c.get('score'),ai_softmax=c.get('ai_prob'),pose_score=c.get('pose_score'),final_score=c.get('fused_score'),
                        dictionary_covered=c['word'] in dictionary,ground_truth=target,
                        correctness=('✔' if target==c['word'] else '✗') if target is not None else '?'))
    return pd.DataFrame(metrics),pd.DataFrame(segments),pd.DataFrame(candidates)


def compare_segments(segments):
    """Pair exact intervals; do not claim score-only effects when boundaries changed."""
    a=segments[segments['mode']=='ai_only']; b=segments[segments['mode']=='hybrid030']
    result=a.merge(b,on=['video','t0','t1'],suffixes=('_ai','_hybrid'),how='outer',indicator=True,validate='one_to_one')
    result['changed']=result.word_ai!=result.word_hybrid
    def reason(r):
        if r['_merge']!='both': return 'boundary changed / unmatched; inspect postprocess'
        if not r.changed: return 'unchanged'
        if r.correctness_ai=='✗' and r.correctness_hybrid=='✔': return 'helped'
        if r.correctness_ai=='✔' and r.correctness_hybrid=='✗': return 'harmed'
        if not r.top1_covered_ai or not r.top1_covered_hybrid: return 'coverage asymmetry; truth not established'
        return 'ranking changed; review attributes and ground truth'
    result['assessment']=result.apply(reason,axis=1)
    return result


def show_table(df, explanation: str):
    """Always display explanatory Markdown immediately below a table."""
    from IPython.display import display,Markdown
    display(df.round(3)); display(Markdown(explanation))


def bar(df,x,y,title,explanation):
    """Render and save a labeled scientific chart, then explain its interpretation."""
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    fig,ax=plt.subplots(figsize=(11,max(3,len(df)*.26)))
    ax.barh(df[x].astype(str),df[y]);ax.invert_yaxis();ax.set(title=title,xlabel=y)
    fig.tight_layout();OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(hashlib.sha1(title.encode()).hexdigest()[:10]+'.png'))
    plt.show();display(Markdown(explanation))


def overview(metrics,comparison):
    """Paired metrics plots; missing labels explicitly shown rather than zero-filled."""
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    for field in ['WER','token_accuracy','segment_F1','recall1','recall5','count_error','latency_s']:
        if metrics[field].notna().any():
            ax=metrics.pivot(index='video',columns='mode',values=field).plot.bar(figsize=(12,4),title=field,rot=25)
            ax.figure.tight_layout();ax.figure.savefig(OUT/f'{field}.png');plt.show()
            display(Markdown(f'**อ่านกราฟ {field}:** เปรียบเทียบคลิปเดียวกันสองโหมด; latency เป็นการวัดครั้งเดียว รวม extraction ที่ใช้ร่วมกัน จึงยังไม่ใช่ latency benchmark ที่ทำซ้ำ'))
        else: display(Markdown(f'**{field}: N/A** — ยังไม่มี ground truth ที่จำเป็น ไม่แทนด้วย 0'))
    show_table(comparison.groupby('assessment',observed=True).size().rename('segments').reset_index(),
        'การเปลี่ยนคำไม่เท่ากับดีขึ้น: helped/harmed แสดงได้เมื่อมีเฉลยเท่านั้น ส่วน boundary changed ต้องตรวจแยกจาก fusion')


def visual_video(video,segments,candidates):
    """Per-segment real frame, timeline and three-score bars for every real video."""
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    ss=segments[(segments.video==video)&(segments['mode']=='hybrid030')]
    frame_rows = video_worker({'operation':'frames','videos':[{
        'path':str(ROOT/'Boom/boom datatest'/video),
        'frames':[{'seconds':float((s.t0+s.t1)/2),
                   'output':str(OUT/'frames'/f'{Path(video).stem}_{int(s.segment):02d}.png')}
                  for _,s in ss.iterrows()]}]})
    frame_paths = dict(zip(ss.segment, [r['path'] for r in frame_rows]))
    fig,ax=plt.subplots(figsize=(13,3))
    for level,mode in enumerate(['ai_only','hybrid030']):
        for _,s in segments[(segments.video==video)&(segments['mode']==mode)].iterrows():
            ax.broken_barh([(s.t0,s.t1-s.t0)],(level-.3,.6),facecolors='steelblue' if mode=='ai_only' else 'darkorange')
            ax.text((s.t0+s.t1)/2,level,s.word,ha='center',fontsize=8)
    ax.set(yticks=[0,1],yticklabels=['AI','Hybrid'],xlabel='seconds',title=video);fig.tight_layout();plt.show()
    display(Markdown('**Timeline:** ตรวจว่าจำนวนและขอบเขตช่วงเปลี่ยนหรือไม่ เพราะ pipeline มี postprocessing หลัง fusion; ความต่างจึงไม่ได้จำกัดอยู่ที่อันดับคำเสมอ'))
    for _,s in ss.iterrows():
        cc=candidates[(candidates.video==video)&(candidates['mode']=='hybrid030')&(candidates.segment==s.segment)]
        fig,axes=plt.subplots(1,2,figsize=(13,3.4))
        frame_path = frame_paths[s.segment]
        if frame_path: axes[0].imshow(plt.imread(frame_path))
        else: axes[0].text(.1,.5,'Frame decode failed')
        axes[0].axis('off');axes[0].set_title(f'{s.t0:.2f}–{s.t1:.2f}s | {s.word} {s.correctness}\np_correct={s.p_correct:.3f} (original calibrator)')
        yy=np.arange(len(cc))
        for offset,column,label in [(-.25,'ai_softmax','AI softmax'),(0,'pose_score','Rule agreement'),(.25,'final_score','Hybrid')]:
            axes[1].barh(yy+offset,cc[column].astype(float),height=.24,label=label)
        axes[1].set_yticks(yy,cc.word);axes[1].invert_yaxis();axes[1].legend(fontsize=8);fig.tight_layout()
        fig.savefig(OUT/f'{Path(video).stem}_segment_{int(s.segment):02d}.png');plt.show()
        display(Markdown(f'**อ่านภาพ:** เฟรมจริงตรงกลางช่วง ไม่ใช่หลักฐานการเคลื่อนไหวทั้งช่วง; `{s.observed_attributes}`. Rule score และคะแนนรวมไม่ใช่ probability; ? คือยังไม่มีเฉลย'))
        show_table(cc,'คะแนน cosine, AI softmax, rule agreement และ final score อยู่คนละความหมาย; dictionary_covered=False คือไม่มีข้อมูล ไม่ใช่พิสูจน์ว่าท่าผิด')


def fusion_audit(data):
    """Verify actual score fields and rounded fusion arithmetic on all candidates."""
    rows=[]
    for r in data:
        for s in r['modes']['hybrid030']['result']['segments']:
            for c in s.get('candidates',[]):
                if 'fused_score' not in c: continue
                residual=abs(c['fused_score']-(.7*c['ai_prob']+.3*c['pose_score']))
                rows.append(dict(video=r['video'],word=c['word'],residual=residual,passed=residual<.0002))
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError('No fused scores found: Hybrid execution not proven')
    if not df.passed.all(): raise ValueError('Fusion arithmetic differs from documented formula')
    return df.groupby('video').agg(candidates=('word','size'),max_residual=('residual','max'),all_pass=('passed','all')).reset_index()


def paired_summary(metrics):
    """Paired bootstrap of observed clips; resampling is uncertainty analysis, not mock data."""
    rows=[]
    for field in ['WER','sentence_WER','token_accuracy','count_error','latency_s']:
        pivot=metrics.pivot(index='video',columns='mode',values=field).dropna()
        if pivot.empty: continue
        delta=(pivot['hybrid030']-pivot['ai_only']).to_numpy(float)
        rng=np.random.default_rng(20261018)
        boots=rng.choice(delta,size=(5000,len(delta)),replace=True).mean(axis=1)
        rows.append(dict(metric=field,n_clips=len(delta),ai_mean=pivot.ai_only.mean(),hybrid_mean=pivot.hybrid030.mean(),
                         delta=delta.mean(),ci_low=np.quantile(boots,.025),ci_high=np.quantile(boots,.975),
                         lower=int((delta<0).sum()),same=int((delta==0).sum()),higher=int((delta>0).sum())))
    return pd.DataFrame(rows)


def score_diagnostics(candidates):
    """Describe evidence coverage and score variation without treating scores as probabilities."""
    h=candidates[candidates['mode']=='hybrid030'].copy()
    h['zero_rule_score']=h.pose_score==0
    return h.groupby('dictionary_covered',dropna=False).agg(candidate_rows=('word','size'),
        zero_rule_fraction=('zero_rule_score','mean'),mean_rule=('pose_score','mean'),
        mean_ai=('ai_softmax','mean'),mean_final=('final_score','mean')).reset_index()


def export_report(metrics,segments,candidates,comparison):
    """Export auditable raw tables and honest auto-summary, including missing metrics."""
    OUT.mkdir(parents=True,exist_ok=True)
    for name,df in [('metrics',metrics),('segments',segments),('top5',candidates),('comparison',comparison)]:
        df.to_csv(OUT/f'{name}.csv',index=False)
    text=['# Hybrid phase 2.1 — real six-video experiment',
          'Results are descriptive on these six videos; no population-level superiority claim.',
          'Production scoring: 0.70 * candidate softmax(T=1) + 0.30 * rule agreement. This is not a calibrated probability.',
          'Uncovered dictionary entries receive zero in current production code; inspect coverage bias.',
          'Original p_correct calibrator is not validated for hybrid rankings.',
          'Timed ground truth is required for Segment F1, aligned Recall@1/5 and accepted precision.',
          '\n## Metrics\n```\n'+metrics.round(3).to_string(index=False)+'\n```',
          '\n## Change categories\n```\n'+comparison.assessment.value_counts().to_string()+'\n```']
    (OUT/'REPORT.md').write_text('\n\n'.join(text),encoding='utf-8')
    return OUT/'REPORT.md'


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='run': run_experiment()
    else: print(environment().to_string(index=False))


def launch_hand_tracking(video: str | None = None) -> None:
    """Run real framewise finger tracking in the isolated project interpreter."""
    args=[str(PYTHON),'-u',str(HERE/'handshape_tracking.py')]
    if video is not None:
        path=ROOT/'Boom/boom datatest'/video
        if path not in videos(): raise ValueError('Select one of the six authorized videos')
        args.extend(['--video',str(path)])
    subprocess.run(args,cwd=ROOT,env=subprocess_environment(),check=True)


def view_hand_tracking(video: str) -> None:
    """Show real overlay video, per-finger state timeline, events and detection coverage."""
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown,Video
    folder=HERE/'boom_datatest/handshape'/Path(video).stem
    csv=folder/'frame_states.csv'
    if not csv.exists(): raise FileNotFoundError(f'Run hand tracking first: {csv}')
    df=pd.read_csv(csv);events=pd.read_csv(folder/'events.csv')
    if (folder/'tracking.mp4').exists(): display(Video(str(folder/'tracking.mp4'),embed=True,width=540))
    else: display(Markdown(f'Video output: {folder}/overlay_source.avi'))
    display(Markdown('**วิดีโอ:** LEFT/RIGHT คือด้านของผู้ทำท่าตาม MediaPipe ไม่ใช่ด้านจอ; เขียว=เหยียด ส้ม=งอ เทา=ไม่แน่ใจ. extended หมายถึงนิ้วเหยียด ไม่ได้แปลว่าชี้ขึ้นบนเสมอ'))
    summary=df.groupby(['side','shape','quality'],dropna=False).size().rename('frames').reset_index()
    summary['fraction']=summary.frames/df.groupby('side').size().reindex(summary.side).to_numpy()
    show_table(summary,'**สัดส่วนเฟรม:** เป็นอัตราการให้คำตอบและคุณภาพสัญญาณ ไม่ใช่ accuracy; ต้องติดป้ายรูปมือจริงเพื่อตรวจความแม่น')
    fig,axes=plt.subplots(2,1,figsize=(14,5),sharex=True)
    from matplotlib.colors import ListedColormap
    names=['thumb','index','middle','ring','pinky'];thai=['โป้ง','ชี้','กลาง','นาง','ก้อย']
    for ax,side in zip(axes,['left','right']):
        part=df[df.side==side]
        arr=part[names].apply(lambda column: column.map({'unknown':0,'folded':1,'extended':2})).to_numpy(dtype=float).T
        ax.imshow(arr,aspect='auto',interpolation='nearest',origin='lower',vmin=0,vmax=2,
            cmap=ListedColormap(['#aaaaaa','#f28e2b','#59a14f']),
            extent=[part.time_s.min(),part.time_s.max(),-.5,4.5])
        ax.set_yticks(range(5),thai);ax.set_title(side)
    axes[-1].set_xlabel('seconds');fig.tight_layout();fig.savefig(folder/'finger_timeline.png');plt.show()
    display(Markdown('**Timeline:** เทา=unknown/มือหาย/หลักฐานยังไม่นิ่ง, ส้ม=folded, เขียว=extended. ใช้ consensus ย้อนหลัง 5 เฟรม และไม่คงคำตอบเก่าเมื่อมือหาย จึงมีหน่วงเล็กน้อย'))
    show_table(events,'**เหตุการณ์:** ช่วงที่ผลรูปมือและสถานะนิ้วคงเดิม; ช่องว่าง unknown เก็บไว้ ไม่เติมเป็นกำมือ. fist หมายถึงสี่นิ้วงอ ไม่ระบุว่าโป้งอยู่ในหรือนอกกำปั้น; four_extended ยังยืนยันโป้งไม่ได้')
    show_table(df.head(30),'**หลักฐานรายเฟรม:** ค่า angle/ratio เป็นเรขาคณิตจาก landmarks; threshold เป็นค่าเริ่มต้นที่ยังไม่ผ่าน validation และไม่ใช่ probability')
