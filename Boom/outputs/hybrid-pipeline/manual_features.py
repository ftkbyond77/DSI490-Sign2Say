"""Observed-only 2D wrist descriptors. These are proxies, not linguistic labels."""
import numpy as np

def _longest_run(mask):
    ids=np.flatnonzero(mask)
    if not len(ids): return ids
    return max(np.split(ids,np.flatnonzero(np.diff(ids)>1)+1),key=len)

def _trajectory(t,xy,observed):
    # Split at missing samples, large jumps, and timestamp gaps; never bridge them.
    good=observed.copy()
    good[1:] &= (np.linalg.norm(np.diff(xy,axis=0),axis=1)<=.4)&(np.diff(t)>0)&(np.diff(t)<=.12)
    ids=_longest_run(good)
    unknown=dict(circling=None,linear=None,path_length=None)
    if len(ids)<10 or len(ids)<.5*len(t):return unknown
    p=xy[ids]
    p=np.array([np.median(p[max(0,i-1):min(len(p),i+2)],axis=0) for i in range(len(p))])
    v=np.diff(p,axis=0);length=np.linalg.norm(v,axis=1);total=length.sum()
    if total<.15:return unknown
    direction=v[length>.015]
    if len(direction)<5:return unknown
    angles=np.arctan2(direction[:,1],direction[:,0]);turns=np.angle(np.exp(1j*np.diff(angles)))
    signed=abs(turns.sum());coherence=signed/max(np.abs(turns).sum(),1e-9)
    spread=np.ptp(p,axis=0);roundness=min(spread)/max(max(spread),1e-9)
    circle=np.clip(signed/(2*np.pi),0,1)*coherence*np.clip(roundness/.4,0,1)
    return dict(circling=float(circle),linear=float(np.linalg.norm(p[-1]-p[0])/total),path_length=float(total))

def detailed_profile(s,start,end):
    mask=(s['t']>=start)&(s['t']<end)
    t=s['t'][mask];xy=s['xy'][mask];obs=s['observed'][mask]
    out=dict(observed_rate=obs.mean(0).tolist() if len(t) else [0.,0.],wrist_y=[None,None],median_speed=[None,None],
             distance_lr_wrist=None,speed_balance=None,alternating=None,circling=[None,None],linear=[None,None])
    if len(t)<3:return out
    for h in range(2):
        if obs[:,h].sum()>=3:out['wrist_y'][h]=float(np.median(xy[obs[:,h],h,1]))
        tr=_trajectory(t,xy[:,h],obs[:,h]);out['circling'][h]=tr['circling'];out['linear'][h]=tr['linear']
    dt=np.diff(t);step=np.linalg.norm(np.diff(xy,axis=0),axis=-1)
    valid=obs[1:]&obs[:-1]&(dt[:,None]>0)&(dt[:,None]<=.12)&(step<=.4)
    speed=step/np.maximum(dt[:,None],1e-9)
    for h in range(2):
        if valid[:,h].sum()>=3:out['median_speed'][h]=float(np.median(speed[valid[:,h],h]))
    both=obs.all(1)
    if both.sum()>=3:out['distance_lr_wrist']=float(np.median(np.linalg.norm(xy[both,0]-xy[both,1],axis=1)))
    shared=valid.all(1)
    if shared.sum()>=3:
        med=np.median(speed[shared],axis=0)
        if max(med)>.15:out['speed_balance']=float(min(med)/max(med))
    # Relative vertical order reversals on one continuous valid run: weak alternation proxy.
    ids=_longest_run(both)
    if len(ids)>=10 and len(ids)>=.5*len(t) and shared[max(0,ids[0]):ids[-1]].all():
        diff=xy[ids,0,1]-xy[ids,1,1];sign=np.sign(diff[np.abs(diff)>.08])
        if len(sign)>=3:out['alternating']=float(min(1,np.count_nonzero(np.diff(sign))/2))
    return out
