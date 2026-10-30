"""Uncalibrated soft prior; never replaces probability or acceptance calibration."""
import math
import numpy as np
from features import FEATURES
FLOORS=dict(is_two_handed=.25,gross_location=.4,face_location=.4,motion_energy=1.,symmetry=.4)
def rerank(candidates,query,table,alpha=.1,min_samples=3):
    if not 0<=alpha<=1:raise ValueError('alpha must be in [0,1]')
    out=[]
    for c in candidates:
        ref=table.get(c['word'],{});parts={}
        for f in FEATURES:
            if min(query['observed_rate'])<.5 and f in ('is_two_handed','symmetry'):continue
            q=query.get(f);v=ref.get(f+'_median');n=ref.get(f+'_count',0)
            if q is None or v is None or float(n)<min_samples or not math.isfinite(float(v)):continue
            scale=max(float(ref.get(f+'_std',0)),FLOORS[f])
            parts[f]=float(np.clip(1-abs(q-float(v))/scale,-1,1))
        rule=float(np.mean(list(parts.values()))) if parts else 0.
        bonus=float(np.clip(alpha*rule,-.20,.15))
        out.append(dict(c,rule_score=rule,bonus=bonus,final_score=float(c['support'])+bonus,rule_parts=parts,p_correct=None))
    return sorted(out,key=lambda x:-x['final_score'])
