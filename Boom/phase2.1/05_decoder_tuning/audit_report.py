"""Tables, plots and source-file review links for the real v7.2 experiment."""
from pathlib import Path
import html
import numpy as np
import pandas as pd
import audit_experiments as a
OUT=a.OUT

def enrich_sources():
    """Resolve original video paths from the manifest, not the shard's dataset source code."""
    m=pd.read_parquet(a.ex.ROOT/'cloud_s3/data_prep/manifest/clips.parquet').drop_duplicates('clip_id').set_index('clip_id')
    for file,key in [('bank_inventory.csv','clip'),('visual_review_queue.csv','nn_clip')]:
        p=OUT/file
        if not p.exists():continue
        df=pd.read_csv(p);sources=[];poses=[];urls=[]
        for cid in df[key]:
            base=cid.split('@')[0]
            if base in m.index:
                r=m.loc[base];sources.append(str(a.ex.ROOT/'cloud_s3'/str(r.source_file)));poses.append(str(a.ex.ROOT/'cloud_s3/data_prep'/str(r.pose_file)));urls.append(str(r.source_url))
            else:
                sources.append('');urls.append('')
                prefix,key=base.split(':',1)
                possible=a.ex.ROOT/'cloud_s3/data_prep/pose_mp/agent'/('agent_'+key.replace(':','_')+'.npz')
                poses.append(str(possible) if prefix=='agent' and possible.exists() else '')
        df['source']=sources;df['source_exists']=[Path(x).is_file() if x else False for x in sources];df['pose_path']=poses;df['pose_exists']=[Path(x).is_file() if x else False for x in poses];df['source_url']=urls
        df.to_csv(p,index=False)
    queue=pd.read_csv(OUT/'visual_review_queue.csv');prior=queue.groupby(['bank','nn_clip','nn_label','source','source_exists','pose_path'],dropna=False).agg(hits=('truth','size'),queried_truths=('truth',lambda s:' / '.join(sorted(set(s))))).reset_index().sort_values('hits',ascending=False)
    prior.to_csv(OUT/'review_priority.csv',index=False)
    # A local review page exposes source paths even when RGB was not downloaded.
    body='<meta charset="utf-8"><h1>Bank nearest-neighbour audit</h1><p>Candidate label noise only: nearest wrong label is not proof of a bad label. Auto-aligned query boundaries require inspection.</p>'
    for _,r in prior.iterrows():
        body+=f'<h3>{html.escape(r.queried_truths)} → {html.escape(r.nn_label)} ({r.hits})</h3><p>{html.escape(r.nn_clip)} · {r.bank}</p><p>RGB: {html.escape(str(r.source))}</p><p>Pose: {html.escape(str(r.pose_path))}</p>'
        if r.source_exists:body+=f'<video width="400" controls preload="none" src="{Path(r.source).as_uri()}"></video>'
    (OUT/'review.html').write_text(body)

def query_coverage():
    """Show how much of the tune reference actually has auto-aligned queries."""
    from collections import Counter
    align=a.ex.read(a.ex.ROOT/'artifacts/runs/seq-v7/tsl51_alignment.json')
    refs=Counter(w for v in align.values() if v['role']=='tagger_train' for w in v['glosses'])
    q=pd.read_csv(OUT/'nearest_neighbours.csv');q=q[q.bank=='eval'];rows=[]
    for w in sorted({p[0] for p in a.PAIRS}):
        g=q[q.truth==w];rows.append(dict(word=w,reference_occurrences=refs[w],aligned_queries=len(g),median_auto_alignment_score=g.alignment_score.median(),query_fraction=len(g)/refs[w] if refs[w] else None))
    result=pd.DataFrame(rows);result.to_csv(OUT/'query_coverage.csv',index=False);return result

def table(file):
    """Load only actual completed output; missing results stay missing."""
    return pd.read_csv(OUT/file).round(3)

def plot_grid():
    """Compare optimistic original grammar with held-out-template predictions."""
    import matplotlib.pyplot as plt
    df=table('grammar_cv_summary.csv');fig,ax=plt.subplots(figsize=(12,5))
    for version,g in df.groupby('version'):ax.plot(range(len(g)),g.sort_values('config').macro_WER,label=version)
    ax.set(xlabel='Parameter grid index (sorted config)',ylabel='Tune macro WER',title='Original grammar vs leave-role-template-out');ax.legend();fig.tight_layout();fig.savefig(OUT/'grammar_cv.png',dpi=160);return fig

def comparison_plots():
    """Plot S/D/I and WER separately for each dataset/speed from completed runs."""
    import matplotlib.pyplot as plt
    outputs=[]
    for path in sorted(OUT.glob('*x/summary.csv')):
        df=pd.read_csv(path)
        if path.parent.name.startswith('boom'):
            df['accepted_correct']=np.nan;df.to_csv(path,index=False)
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        df.plot.barh(x='mode',y=['S','D','I'],ax=axes[0]);df.plot.barh(x='mode',y='macro_WER',ax=axes[1],legend=False)
        fig.suptitle(path.parent.name+' | baseline = grammar_original');fig.tight_layout()
        target=path.parent/'comparison.png';fig.savefig(target,dpi=150);plt.close(fig);outputs.append(target)
    return outputs

def render_summary():
    """Generate one narrative report from verified output files, with limitations."""
    enrich_sources();query_coverage();comparison_plots();lines=['# v7.2 — grammar audit / filters / confusing words','',
    'No retraining; production configuration and banks unchanged. Experiments are opt-in. Current local model is v7.2, so supplied v8 results are not treated as reproduced results.',
    '','## A. Grammar leakage','',str(a.ex.read(OUT/'provenance.json')),'','```',table('grammar_overlap.csv').to_string(index=False),'```',
    'The exact reconstruction includes test reference sentences. This is text-label leakage, not proof of video training leakage. Recording membership is reconstructed from the current script and manifest; original grammar has no clip-ID provenance.',
    'CV holds out entire role-templates and all their tune recordings. Selected CV score is still a model-selection estimate, not an unbiased final score. Test was already reported by the user and must not be described as a never-seen holdout.','',
    '## B. Full pipeline comparisons','']
    for path in sorted(OUT.glob('*x/summary.csv')):
        lines+=['### '+path.parent.name,'','```',pd.read_csv(path).round(3).to_string(index=False),'```','```',pd.read_csv(path.parent/'paired_ci.csv').round(4).to_string(index=False),'```']
    lines+=['','Timing uses shared real neural caches: not a fresh CPU latency benchmark. Boom WER uses filename text/newmm; TSL51 uses ordered gloss labels. Do not pool them. Hand-filter visibility uses standardised confidence, which can include interpolated points. Accepted correctness is ordered alignment, not human temporal verification.','',
    '## C. Confusing words','', '```',table('pair_summary.csv').to_string(index=False),'```',
    'Query segments use automatic alignment; a cosine mismatch alone cannot distinguish segmentation error, variant, label error and a real visual similarity. Missing aligned query words are not silently counted as failures. Production bank neighbours may include same-source sentence exemplars; use eval bank for primary diagnosis. Review exact video/pose paths in review_priority.csv and review.html. No exemplar removed.']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    return OUT/'REPORT.md'
def render_html():
    """Browse completed tables, comparisons and real-pose review in one local page."""
    parts=['<meta charset="utf-8"><style>body{font:17px Tahoma,sans-serif;margin:32px;background:#f6f8fc;color:#182333}table{border-collapse:collapse;background:white;font-size:14px}td,th{padding:7px;border:1px solid #ccd3df}img{max-width:100%}section{overflow:auto;margin:30px 0}</style><h1>v7.2: Grammar leakage / Filters / Bank audit</h1><p>Actual local results. No retraining or production change. Test has previously been reported by the user; this is a frozen reproduction, not a virgin holdout. Original grammar has test-reference leakage.</p>']
    for label,file in [('A: Grammar source overlap','grammar_overlap.csv'),('Tune CV grid','grammar_cv_summary.csv'),('C: Confusing pairs','pair_summary.csv'),('Query coverage','query_coverage.csv'),('Priority source clips to review','review_priority.csv')]:
        parts+=['<section><h2>'+label+'</h2>',table(file).to_html(index=False),'</section>']
    for folder in sorted(OUT.glob('*x')):
        if not (folder/'summary.csv').exists():continue
        parts+=['<section><h2>'+folder.name+'</h2>',table(str(folder.relative_to(OUT)/'summary.csv')).to_html(index=False),table(str(folder.relative_to(OUT)/'paired_ci.csv')).to_html(index=False),f'<img src="{folder.name}/comparison.png"></section>']
    parts+=['<p>Bootstrap is clip-level, not signer-level. If CI crosses zero: ไม่มีนัยสำคัญ. Cached timing is not independent CPU latency. Auto-aligned boundaries and nearest-neighbour mismatches cannot certify label noise.</p><a href="pose_review/index.html">Real pose review</a> | <a href="review.html">Original file paths</a>']
    (OUT/'index.html').write_text(''.join(parts));return OUT/'index.html'

if __name__=='__main__':
    print(render_summary());print(render_html())
