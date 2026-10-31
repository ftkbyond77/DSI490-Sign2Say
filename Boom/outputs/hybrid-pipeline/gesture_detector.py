"""Explainable observed-only 2D hand-shape diagnostics, not word recognition.
All thresholds are experimental. Palm facing/contact cannot be verified in 2D.
"""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class GestureConfig:
    extended_angle:float=155.
    curled_angle:float=120.
    wrist_distance_max:float=.5
    palm_distance_max:float=.5
    wrist_y_min:float=-.5
    wrist_y_max:float=.65
    upward_cos_min:float=.5
    min_observed:float=.5
    min_duration:float=.20
    min_fraction:float=.25

FINGERS={'index':(5,6,7,8),'middle':(9,10,11,12),'ring':(13,14,15,16),'little':(17,18,19,20)}
def angle(a,b,c):
    u=a-b;v=c-b;den=np.linalg.norm(u)*np.linalg.norm(v)
    if den<1e-9:return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/den,-1,1))))
def hand_features(p,cfg=GestureConfig()):
    p=np.asarray(p,float)
    if p.shape!=(21,2) or not np.isfinite(p).all() or np.linalg.norm(p[9]-p[0])<.01:return None
    states={};angles={}
    for name,(mcp,pip,dip,tip) in FINGERS.items():
        a=angle(p[mcp],p[pip],p[dip]);b=angle(p[pip],p[dip],p[tip]);angles[name]=[a,b]
        states[name]='unknown' if a is None or b is None else ('extended' if min(a,b)>=cfg.extended_angle else 'curled' if min(a,b)<=cfg.curled_angle else 'intermediate')
    axis=p[9]-p[0];cos=float(-axis[1]/np.linalg.norm(axis))
    thumb=angle(p[2],p[3],p[4])
    return dict(fingers=states,joint_angles_deg=angles,thumb_ip_angle_deg=thumb,palm_center=p[[0,5,9,13,17]].mean(0).tolist(),upward_cos=cos,
                four_fingers_extended=all(v=='extended' for v in states.values()),four_fingers_curled=all(v=='curled' for v in states.values()),
                index_pointing=states['index']=='extended' and all(states[k]=='curled' for k in ('middle','ring','little')),
                palm_facing='unknown_2d',contact='unknown_2d')
def _evidence(t,flags,valid,cfg):
    longest=0.;begin=None;intervals=[]
    for i,yes in enumerate(flags):
        if begin is not None and (not yes or t[i]-t[i-1]>.12):
            intervals.append([float(t[begin]),float(t[i-1])]);begin=None
        if yes and begin is None:begin=i
    if begin is not None:intervals.append([float(t[begin]),float(t[-1])])
    if intervals:longest=max(b-a for a,b in intervals)
    rate=float(np.mean(valid)) if len(valid) else 0.;fraction=float(np.mean(flags)) if len(flags) else 0.
    status='insufficient_data' if rate<cfg.min_observed else 'supported' if longest>=cfg.min_duration-1e-9 and fraction>=cfg.min_fraction else 'not_supported'
    return dict(status=status,observed_rate=rate,support_fraction=fraction,longest_support_s=longest,intervals=intervals)
def detect_gestures(raw,start,end,cfg=GestureConfig()):
    t=np.asarray(raw['t']);take=(t>=start)&(t<end);t=t[take]
    pose=np.asarray(raw['pose'])[take];hands=np.stack([raw['lh'],raw['rh']],1)[take,:,:,:2]
    scale=np.array([raw['hw'][1]/raw['hw'][0],1.]);shoulders=pose[:,[11,12],:2]*scale
    width=np.linalg.norm(shoulders[:,0]-shoulders[:,1],axis=1);origin=shoulders.mean(1)
    good=(pose[:,[11,12],3]>.5).all(1)&np.isfinite(shoulders).all((1,2))&(width>.02)
    obs=np.isfinite(hands).all((2,3))&(np.abs(hands).sum((2,3))>0)&good[:,None]
    xy=(hands*scale-origin[:,None,None,:])/np.maximum(width[:,None,None,None],.02)
    frames=[];valid=np.zeros((len(t),2),bool);prayer=np.zeros(len(t),bool)
    patterns={name:np.zeros((len(t),2),bool) for name in ('four_fingers_extended','four_fingers_curled','index_pointing')}
    for i in range(len(t)):
        feats=[hand_features(xy[i,h],cfg) if obs[i,h] else None for h in range(2)]
        pair=None
        for h,f in enumerate(feats):
            if f is not None:
                valid[i,h]=True
                for name in patterns:patterns[name][i,h]=f[name]
        if all(f is not None for f in feats):
            dist=float(np.linalg.norm(xy[i,0,0]-xy[i,1,0]));pdist=float(np.linalg.norm(np.array(feats[0]['palm_center'])-feats[1]['palm_center']))
            ys=xy[i,:,0,1];checks=dict(wrists_close=dist<=cfg.wrist_distance_max,palms_close=pdist<=cfg.palm_distance_max,fingers_extended=all(f['four_fingers_extended'] for f in feats),hands_upward=all(f['upward_cos']>=cfg.upward_cos_min for f in feats),upper_torso=bool(((ys>=cfg.wrist_y_min)&(ys<=cfg.wrist_y_max)).all()))
            prayer[i]=all(checks.values());pair=dict(wrist_distance=dist,palm_distance=pdist,wrist_y=ys.tolist(),checks=checks)
        frames.append(dict(t=float(t[i]),hands=feats,pair=pair))
    labels={name:[_evidence(t,flags[:,h],valid[:,h],cfg) for h in range(2)] for name,flags in patterns.items()}
    labels['prayer_like_2d']=_evidence(t,prayer,valid.all(1),cfg)
    return dict(start=start,end=end,config=asdict(cfg),labels=labels,frames=frames,units='shoulder_width; y positive downward from mid-shoulder',limitations=['2D projection may hide finger flexion','palm-facing and contact unknown','curled four fingers is not a confirmed fist; thumb not classified','prayer-like geometry is not a word label'])
