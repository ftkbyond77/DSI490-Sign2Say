"""Physical features in shoulder widths; missing detections are never interpolated here."""
import numpy as np
FEATURES=('is_two_handed','gross_location','face_location','motion_energy','symmetry')
def stream(kp,sc,hw,t):
    q=np.asarray(kp,float)*[hw[1]/hw[0],1]
    width=np.linalg.norm(q[:,5]-q[:,6],axis=1)
    good=(sc[:,5]>.5)&(sc[:,6]>.5)&(width>.02)
    xy=(q[:,[91,112]]-q[:,[5,6]].mean(1)[:,None])/np.maximum(width[:,None,None],.02)
    obs=(sc[:,[91,112]]>.5)&good[:,None]
    face=(q[:,0,1]-q[:,[5,6],1].mean(1))/np.maximum(width,.02)
    face=np.where((sc[:,0]>.5)&good,face,np.nan)
    return dict(t=np.asarray(t),xy=xy,observed=obs,face=face)
def profile(s,start=None,end=None):
    take=np.ones(len(s['t']),bool)
    if start is not None: take &= s['t']>=start
    if end is not None: take &= s['t']<end
    t=s['t'][take]; x=s['xy'][take]; o=s['observed'][take]; face=s['face'][take]
    out={f:None for f in FEATURES};out['observed_rate']=o.mean(0).tolist() if len(o) else [0.,0.]
    if len(t)<3:return out
    dt=np.diff(t); delta=np.diff(x,axis=0); step=np.linalg.norm(delta,axis=-1)
    valid=o[1:]&o[:-1]&(dt[:,None]>0)&(step<=.4)
    velocity=delta/np.maximum(dt[:,None,None],1e-9)
    speed=np.linalg.norm(velocity,axis=-1)
    med=[np.median(speed[valid[:,h],h]) if valid[:,h].sum()>=2 else np.nan for h in range(2)]
    if np.isfinite(med).all() and max(med)>.15:out['is_two_handed']=float(min(med)/max(med))
    if o.any():out['gross_location']=float(np.median(x[:,:,1][o]))
    vf=o&np.isfinite(face[:,None])
    if vf.any():out['face_location']=float(np.median((x[:,:,1]-face[:,None])[vf]))
    if valid.sum()>=4:out['motion_energy']=float(np.var(speed[valid]))
    both=valid.all(1)
    if both.sum()>=3:
        a=velocity[both,0];b=velocity[both,1]*[-1,1]
        den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);active=den>.0225
        if active.any():out['symmetry']=float(np.mean(np.sum(a[active]*b[active],axis=1)/den[active]))
    return out
