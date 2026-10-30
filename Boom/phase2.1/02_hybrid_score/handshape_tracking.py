"""Framewise geometric finger-state recognition on actual MediaPipe hand landmarks.

Experimental thresholds, not a trained classifier or calibrated probability.
No dictionary label or sentence prediction is used to decide finger states.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE/'runtime_packages'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.mplconfig'))
os.environ['MP_HOLISTIC_MODEL'] = str(ROOT/'models/mediapipe/holistic_landmarker.task')
import numpy as np

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
THAI = dict(zip(FINGERS, ('โป้ง','ชี้','กลาง','นาง','ก้อย')))
JOINTS = ((1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20))
EDGES = [(0,1),(0,5),(5,9),(9,13),(13,17),(17,0)] + [(a,b) for finger in JOINTS for a,b in zip(finger,finger[1:])]
OUTPUT = HERE/'boom_datatest/handshape'


def angle(a: np.ndarray,b: np.ndarray,c: np.ndarray) -> float:
    """Internal joint angle in degrees; degenerate joints remain unknown."""
    u,v=a-b,c-b
    d=np.linalg.norm(u)*np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/d,-1,1)))) if d>1e-8 else float('nan')


def geometry(hand: np.ndarray, hw: tuple[int,int]) -> dict:
    """Measure extension in aspect-corrected local 3D; missing is never folded."""
    result={'observed':False,'quality':'missing','palm_px':None,'fingers':{},'points':None}
    if hand.shape!=(21,3) or not np.isfinite(hand).all() or not np.any(hand): return result
    h,w=hw; points=hand*np.array([w,h,w])
    palm=float(np.linalg.norm(points[5]-points[17]))
    result.update(observed=True,palm_px=palm,points=points)
    if palm<14: result['quality']='too_small'; return result
    if np.any((hand[:,:2]<0)|(hand[:,:2]>1)): result['quality']='clipped'; return result
    result['quality']='usable'
    for name,ids in zip(FINGERS,JOINTS):
        a,b,c,d=points[list(ids)]
        proximal,distal=angle(a,b,c),angle(b,c,d)
        lengths=[np.linalg.norm(b-a),np.linalg.norm(c-b),np.linalg.norm(d-c)]
        ratio=float(np.linalg.norm(d-a)/max(sum(lengths),1e-9))
        state='unknown'
        if min(lengths)>palm*.025 and np.isfinite([proximal,distal,ratio]).all():
            # Thumb opposition differs from flexion. Conservative extension only;
            # no claim about palm direction or whether the finger points upwards.
            if proximal>=155 and distal>=150 and ratio>=.86: state='extended'
            elif proximal<=110 or distal<=105 or ratio<=.60: state='folded'
        result['fingers'][name]=dict(state=state,proximal_angle=proximal,distal_angle=distal,reach_ratio=ratio)
    return result


def posture(states: dict[str,str]) -> str:
    """Compose interpretable shape from finger states; do not infer a sign word."""
    four=[states[f] for f in FINGERS[1:]]
    if all(s=='folded' for s in four): return 'fist'
    if all(s=='extended' for s in four):
        return 'open' if states['thumb']=='extended' else 'four_extended'
    if states['index']=='extended' and all(states[f]=='folded' for f in ('middle','ring','pinky')): return 'index_point'
    if states['index']==states['middle']=='extended' and states['ring']==states['pinky']=='folded': return 'two_fingers'
    return 'unknown' if 'unknown' in states.values() else 'mixed'


class HandStateTracker:
    """Causal five-frame consensus, with immediate abstention on missing/invalid hands."""
    def __init__(self) -> None:
        self.history={f:deque(maxlen=5) for f in FINGERS}
        self.previous=None

    def update(self,hand: np.ndarray,hw: tuple[int,int]) -> dict:
        """Return both raw geometric evidence and stabilized per-finger decisions."""
        obs=geometry(hand,hw)
        if obs['quality']=='usable':
            center=obs['points'][[0,5,9,13,17]].mean(axis=0)
            if self.previous is not None and np.linalg.norm(center-self.previous)>2.0*obs['palm_px']:
                obs['quality']='coordinate_jump'
            self.previous=center
        else: self.previous=None
        states={f:'unknown' for f in FINGERS}
        if obs['quality']!='usable':
            for history in self.history.values(): history.clear()
        else:
            for f in FINGERS:
                raw=obs['fingers'][f]['state']; self.history[f].append(raw)
                winner,count=Counter(self.history[f]).most_common(1)[0]
                # Never keep an old known answer when current geometry is ambiguous.
                if raw!='unknown' and winner==raw and count>=3 and count/len(self.history[f])>=.6:
                    states[f]=winner
        obs.update(states=states,shape=posture(states),extended=[f for f,s in states.items() if s=='extended'])
        return obs


def event_intervals(rows: list[dict],end_s: float) -> list[dict]:
    """Compress identical stabilized states into time intervals, retaining unknown gaps."""
    events=[]
    for side in ('left','right'):
        items=[r for r in rows if r['side']==side]; current=None
        for index,row in enumerate(items):
            signature=(row['shape'],*(row[f] for f in FINGERS))
            if current is None or signature!=current['_signature']:
                if current is not None: current['t1']=row['time_s'];events.append(current)
                current=dict(side=side,t0=row['time_s'],t1=end_s,shape=row['shape'],
                    extended=row['extended'],_signature=signature)
        if current: events.append(current)
    for event in events:
        event.pop('_signature');event['duration_s']=event['t1']-event['t0']
    return events


def run_video(video: Path) -> Path:
    """Process every decoded frame with real Holistic and save annotated video + measurements."""
    import cv2
    import mediapipe
    import pandas as pd
    from modules.mediapipe_pose import Holistic
    from experiment_analysis import digest,save
    output=OUTPUT/video.stem;output.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video));fps=float(cap.get(cv2.CAP_PROP_FPS));count=0
    if not cap.isOpened() or fps<=0: raise RuntimeError(f'Invalid video: {video}')
    h,w=int(cap.get(4)),int(cap.get(3))
    writer=cv2.VideoWriter(str(output/'overlay_source.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h))
    if not writer.isOpened(): raise RuntimeError('Cannot open video writer')
    trackers={side:HandStateTracker() for side in ('left','right')}
    rows=[];landmarks=[];started=time.perf_counter();hol=None
    try:
        hol=Holistic()
        while True:
            ok,frame=cap.read()
            if not ok: break
            timestamp=count/fps
            _,left,right,_,_=hol.frame(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),timestamp*1000)
            landmarks.append([left,right])
            banner=np.zeros_like(frame)
            for panel,(side,hand) in enumerate((('left',left),('right',right))):
                obs=trackers[side].update(hand,(h,w))
                row=dict(frame=count,time_s=timestamp,side=side,observed=obs['observed'],quality=obs['quality'],
                    palm_px=obs['palm_px'],shape=obs['shape'],extended=' '.join(obs['extended']),**obs['states'])
                for name,m in obs['fingers'].items():
                    row.update({f'{name}_{key}':val for key,val in m.items()})
                rows.append(row)
                if obs['observed']:
                    xy=(hand[:,:2]*[w,h]).astype(int)
                    for a,b in EDGES: cv2.line(frame,tuple(xy[a]),tuple(xy[b]),(180,180,180),2)
                    for name,ids in zip(FINGERS,JOINTS):
                        color={'extended':(0,230,0),'folded':(0,110,255),'unknown':(160,160,160)}[obs['states'][name]]
                        for j in ids: cv2.circle(frame,tuple(xy[j]),4,color,-1)
                        cv2.putText(frame,name[0:2],tuple(xy[ids[-1]]),cv2.FONT_HERSHEY_SIMPLEX,.40,color,1)
                y=28+panel*64
                label=f'{side.upper()}: {obs["shape"]} [{obs["quality"]}]'
                cv2.putText(banner,label,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                cv2.putText(banner,'extended: '+(','.join(obs['extended']) or '?'),(8,y+23),cv2.FONT_HERSHEY_SIMPLEX,.43,(90,255,90),1)
            frame[:145]=cv2.addWeighted(frame[:145],.25,banner[:145],.9,0)
            cv2.putText(frame,f'{timestamp:.2f}s | geometry estimate',(8,h-14),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            writer.write(frame)
            if count%90==0:
                cv2.imwrite(str(output/f'preview_{count:04d}.jpg'),frame)
                print(video.name,count,'frames',flush=True)
            count+=1
    finally:
        cap.release();writer.release()
        if hol is not None:hol.close()
    if not rows: raise RuntimeError('No frames decoded')
    pd.DataFrame(rows).to_csv(output/'frame_states.csv',index=False)
    pd.DataFrame(event_intervals(rows,count/fps)).to_csv(output/'events.csv',index=False)
    np.savez_compressed(output/'raw_hands.npz',hands=np.asarray(landmarks),fps=fps,hw=[h,w])
    save(output/'manifest.json',dict(video=str(video),sha256=digest(video),mediapipe=mediapipe.__version__,
        code_sha256=digest(Path(__file__)),frames=count,fps=fps,elapsed_s=time.perf_counter()-started,
        method='unvalidated joint-angle rules + causal consensus; not trained handshape classifier',
        side='anatomical MediaPipe L/R; no canonical_hands or auto-flip',
        min_palm_px=14,extended_angles=[155,150],folded_angles=[110,105],history_frames=5))
    ffmpeg='/opt/homebrew/bin/ffmpeg'
    if Path(ffmpeg).exists():
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(output/'overlay_source.avi'),
            '-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'tracking.mp4')],check=True)
        (output/'overlay_source.avi').unlink()
    print('SAVED',output,flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--video',type=Path);args=parser.parse_args()
    paths=[args.video] if args.video else sorted((ROOT/'Boom/boom datatest').glob('*.MOV'))
    for path in paths:run_video(path)
