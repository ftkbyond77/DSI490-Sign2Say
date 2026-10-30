"""Create figures and segment diagnostics from real Phase 2.2 predictions."""
from pathlib import Path
import json, os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
import pandas as pd
import numpy as np

def build(dest: Path | None = None) -> Path:
    """Read only produced results; never create placeholder metrics."""
    dest=dest or Path(json.loads((HERE/'latest.json').read_text())['path'])
    if not (dest/'metrics.csv').exists():raise RuntimeError('Inference has not completed its first progress checkpoint')
    from tune import align
    rows=[];changes=[];base={}
    for line in (dest/'predictions.jsonl').read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        segs=[s for s in r['result']['segments'] if s['status']!='transition'];hyp=[s['nearest'] for s in segs];a=align(r['reference'],hyp)
        mapping={j:(i,op) for i,j,op in a if j is not None}
        key=(r['speed'],r['clip'])
        if r['case']=='baseline':base[key]=r
        elif key in base:
            b=base[key];changes.append(dict(speed=r['speed'],clip=r['clip'],case=r['case'],baseline_WER=b['WER'],WER=r['WER'],delta=r['WER']-b['WER'],reference=' | '.join(r['reference']),prediction=' | '.join(hyp)))
        for j,s in enumerate(segs):
            i,op=mapping[j];rows.append(dict(speed=r['speed'],clip=r['clip'],case=r['case'],segment=j,t0=s.get('t0'),t1=s.get('t1'),reference=r['reference'][i] if i is not None else '[insertion]',prediction=s['nearest'],correct=op=='C',status=s['status'],p_correct=s.get('p_correct'),top5=json.dumps(s['candidates'][:5],ensure_ascii=False)))
    df=pd.DataFrame(rows);df.to_csv(dest/'segments.csv',index=False);pd.DataFrame(changes).to_csv(dest/'changes.csv',index=False)
    calibration=[]
    for (speed,case),g in df.groupby(['speed','case']):
        g=g.dropna(subset=['p_correct']).copy();g['bin']=np.minimum((g.p_correct*10).astype(int),9)
        ece=0.
        for _,b in g.groupby('bin'):ece+=len(b)/len(g)*abs(b.p_correct.mean()-b.correct.mean())
        calibration.append(dict(speed=speed,case=case,ECE=ece,n_segments=len(g),definition='10 equal bins; top-1 edit-aligned correctness; includes uncertain/rejected non-transition segments'))
    pd.DataFrame(calibration).to_csv(dest/'calibration.csv',index=False)
    import matplotlib.pyplot as plt
    summary=pd.read_csv(dest/'summary.csv')
    fig,ax=plt.subplots(figsize=(11,5));summary.pivot(index='case',columns='speed',values='WER').plot.bar(ax=ax);ax.set_ylabel('Macro WER (lower is better)');ax.set_title('Phase 2.2 — '+('frozen test' if dest.name.startswith('test_') else 'tune')+'; partial until DONE.json exists');fig.tight_layout();fig.savefig(dest/'tune_wer.png',dpi=160);plt.close(fig)
    return dest
if __name__=='__main__':print(build())
