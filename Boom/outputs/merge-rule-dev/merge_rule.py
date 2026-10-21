"""Experimental conservative gate. No word-specific exceptions and no production writes."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class MergeConfig:
    min_continuous_hands: int = 1
    max_gap_s: float = .2
    min_common_candidates: int = 2
    motion_radius_s: float = .16
    min_observed_fraction: float = .6
    min_speed: float = .15  # shoulder widths / second
    max_step: float = .35  # shoulder widths between adjacent observed frames
    min_boundary_speed_ratio: float = .25
    max_boundary_speed_ratio: float = 4.
    max_duration_s: float = 3.2

def assess(a,b,t,xy,observed,cfg=MergeConfig()):
    # xy [T,2,21,2] is shoulder-centred/scaled, observed [T,2] before imputation.
    gap=b['t0']-a['t1'];common=sorted(set(a['top5'])&set(b['top5']))
    boundary=(a['t1']+b['t0'])/2
    dt=np.diff(t);middle=(t[1:]+t[:-1])/2
    local=(middle>=boundary-cfg.motion_radius_s)&(middle<=boundary+cfg.motion_radius_s)
    valid=observed[1:]&observed[:-1]&(dt[:,None]>0)
    step=np.linalg.norm(np.diff(xy,axis=0),axis=-1).mean(-1)
    speed=step/np.maximum(dt[:,None],1e-8)
    # Require reliable observation on at least one same anatomical hand across the boundary.
    hands=[]
    for h in range(2):
        v=local&valid[:,h];den=int(local.sum());frac=float(v.sum()/den) if den else 0.
        across=(middle>=a['t1']-.04)&(middle<=b['t0']+.04)&valid[:,h]
        context=(middle>=boundary-.4)&(middle<=boundary+.4)&valid[:,h]
        med=float(np.median(speed[v,h])) if v.any() else None
        context_speed=float(np.median(speed[context,h])) if context.any() else None
        boundary_speed=float(np.median(speed[across,h])) if across.any() else None
        jump=float(step[v,h].max()) if v.any() else None
        ratio=boundary_speed/max(context_speed,1e-8) if boundary_speed is not None and context_speed is not None else None
        continuous=bool(frac>=cfg.min_observed_fraction and med is not None and med>=cfg.min_speed and jump<=cfg.max_step and ratio is not None and cfg.min_boundary_speed_ratio<=ratio<=cfg.max_boundary_speed_ratio)
        hands.append(dict(hand='left' if h==0 else 'right',observed_fraction=frac,speed=med,max_step=jump,boundary_ratio=ratio,continuous=continuous))
    checks=dict(gap=bool(0<=gap<cfg.max_gap_s-1e-9),candidate_overlap=len(common)>=cfg.min_common_candidates,motion=sum(h['continuous'] for h in hands)>=cfg.min_continuous_hands,duration=b['t1']-a['t0']<=cfg.max_duration_s)
    return dict(merge=all(checks.values()),gap_s=gap,common_candidates=common,checks=checks,hands=hands,config=asdict(cfg))
