"""Run actual combined tune comparison, then derive paired exploratory confidence intervals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import test_engine as tune

def finish() -> None:
    """Write final measured comparison against baseline and the best previous single setting."""
    dest=Path(json.loads((tune.HERE/'latest_test.json').read_text())['path'])
    if not (dest/'DONE.json').exists():raise RuntimeError('Inference incomplete')
    df=pd.read_csv(dest/'metrics.csv');rows=[];rng=np.random.default_rng(20261024)
    for speed,g in df.groupby('speed'):
        wide=g.pivot(index='clip',columns='case',values='WER')
        if wide.isna().any().any() or len(wide)!=126:raise RuntimeError('Missing paired clips')
        for control in ['baseline','A']:
            delta=(wide.Combined-wide[control]).to_numpy()
            low,high=np.quantile(rng.choice(delta,(10000,len(delta))).mean(axis=1),[.025,.975])
            rows.append(dict(speed=speed,control=control,delta_WER=delta.mean(),ci_low=low,ci_high=high,improved=int((delta<-1e-9).sum()),unchanged=int((abs(delta)<=1e-9).sum()),worsened=int((delta>1e-9).sum()),conclusion='ไม่มีนัยสำคัญ' if low<=0<=high else ('WER ลดลงบน test' if high<0 else 'WER เพิ่มขึ้นบน test')))
    ci=pd.DataFrame(rows);ci.to_csv(dest/'paired_ci.csv',index=False)
    import report
    report.build(dest)
    with (dest/'REPORT.md').open('a') as f:
        f.write('\n\n## Combined versus baseline and A\n\n'+ci.round(5).to_string(index=False)+'\n\nFrozen test comparison; clip-level bootstrap, not signer-clustered. Test previously evaluated and original grammar leakage persists: not an uncontaminated holdout. No retuning or production promotion.\n')
    tune.save(dest/'ANALYSIS_DONE.json',{'paired_bootstrap':10000,'test_run':True,'production_enable':False})
    print(ci.round(4).to_string(index=False),flush=True)
if __name__=='__main__':
    tune.run(tune.HERE/'FROZEN_TEST.json')
    finish()
