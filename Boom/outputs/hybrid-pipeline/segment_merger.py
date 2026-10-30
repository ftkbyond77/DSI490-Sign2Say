"""Conservative wrist-only merge proposal. Caller must rescore every accepted union."""
from dataclasses import dataclass,asdict
import numpy as np
@dataclass(frozen=True)
class MergeConfig:
    max_gap:float=.20
    max_duration:float=3.20
    radius:float=.16
    max_jump:float=.40
    min_speed:float=.15
    min_observed:float=.75
    min_speed_ratio:float=.25
    max_speed_ratio:float=4.
def assess(a,b,s,cfg=MergeConfig()):
    reasons=[];gap=b['t0']-a['t1']
    if gap < -1e-8 or gap>=cfg.max_gap-1e-8:reasons.append('gap')
    if b['t1']-a['t0']>cfg.max_duration+1e-8:reasons.append('duration')
    overlap=sorted(set(a['top5'][:3])&set(b['top5'][:3]))
    if not overlap:reasons.append('top3_disjoint')
    t=s['t'];x=s['xy'];o=s['observed'];dt=np.diff(t)
    local=(t>=a['t1']-cfg.radius)&(t<=b['t0']+cfg.radius)
    idx=np.flatnonzero(local); evidence=[]
    if len(idx)<4:reasons.append('insufficient_frames')
    else:
        # Every hand must stay detected around the seam; otherwise absence is ambiguous.
        oo=o[idx]
        if np.any(oo[:-1]&~oo[1:]):reasons.append('hand_disappears')
        if np.any(oo.mean(0)<cfg.min_observed):reasons.append('missing_hand')
        edges=local[:-1]&local[1:]&(dt>0)
        step=np.linalg.norm(np.diff(x,axis=0),axis=-1)
        speed=step/np.maximum(dt[:,None],1e-9)
        for h in range(2):
            good=edges&o[:-1,h]&o[1:,h]
            before=good&(t[1:]<=a['t1']);after=good&(t[:-1]>=b['t0'])
            vb=float(np.median(speed[before,h])) if before.any() else 0.
            va=float(np.median(speed[after,h])) if after.any() else 0.
            jump=float(step[good,h].max()) if good.any() else None
            ratio=va/max(vb,1e-9)
            continuous=(jump is not None and jump<=cfg.max_jump and min(va,vb)>=cfg.min_speed and cfg.min_speed_ratio<=ratio<=cfg.max_speed_ratio)
            evidence.append(dict(hand=h,before_speed=vb,after_speed=va,max_step=jump,continuous=continuous))
            if jump is not None and jump>cfg.max_jump:reasons.append('coordinate_jump')
        # One moving hand is enough for a one-handed sign, but both must remain visible.
        if not any(h['continuous'] for h in evidence):reasons.append('motion_discontinuous')
    return dict(merge=not reasons,reasons=sorted(set(reasons)),overlap=overlap,hands=evidence,config=asdict(cfg))
