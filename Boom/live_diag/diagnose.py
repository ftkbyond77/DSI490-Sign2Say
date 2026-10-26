"""Read recorded live JSON only. Never call the server or modify pipeline/config."""
from __future__ import annotations
import argparse,csv,gzip,json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
BODY={'nose':0,'left_shoulder':11,'right_shoulder':12,'left_elbow':13,'right_elbow':14,'left_wrist':15,'right_wrist':16}

def complete(frame:dict,key:str)->bool:
    """21 finite coordinate triples; presence only, not detector confidence."""
    a=frame.get(key) or []
    return len(a)==21 and all(len(p)>=3 and all(math.isfinite(float(v)) for v in p[:3]) for p in a)
def gaps(t:list,ok:list)->list:
    """Missing runs: duration uses half-open sample intervals; final interval estimated."""
    dt=statistics.median([b-a for a,b in zip(t,t[1:]) if b>a]) if len(t)>1 else None
    rows=[];i=0
    while i<len(ok):
        if ok[i]:i+=1;continue
        a=i
        while i<len(ok) and not ok[i]:i+=1
        end=t[i] if i<len(t) else (t[-1]+dt if dt is not None else None)
        ms=end-t[a] if end is not None else None
        rows.append(dict(start_frame=a,end_frame_exclusive=i,frames=i-a,start_ms=t[a],duration_ms=ms,bounded=a>0 and i<len(ok),final_interval_estimated=i==len(ok),longer_than_500ms=ms>500 if ms is not None else None))
    return rows

def run(src:Path)->dict:
    """Inventory actual fields and calculate raw timestamp/body/hand availability."""
    files=sorted(src.glob('*.json.gz')) if src.is_dir() else []
    rows=[];runs=[];fields=set()
    for path in files:
        with gzip.open(path,'rt',encoding='utf-8') as f:r=json.load(f)
        frames=r.get('frames',[])
        if not frames:continue
        fields.update(k for frame in frames for k in frame)
        t=[float(f['t']) for f in frames]
        if not all(math.isfinite(v) for v in t) or any(b<=a for a,b in zip(t,t[1:])):
            rows.append(dict(request=path.name,status='invalid timestamps; inspect without silently resampling'));continue
        diffs=[b-a for a,b in zip(t,t[1:])];row=dict(request=path.name,status='raw audit only',n_frames=len(frames),fps_elapsed=(len(t)-1)*1000/(t[-1]-t[0]) if len(t)>1 else None,fps_median=1000/statistics.median(diffs) if diffs else None)
        for key in ['lh','rh']:
            ok=[complete(f,key) for f in frames];row[key+'_observed_rate']=sum(ok)/len(ok)
            for gap in gaps(t,ok):runs.append(dict(request=path.name,hand=key,**gap))
        for name,j in BODY.items():
            scores=[float(f['pose'][j][3]) if len(f.get('pose') or [])==33 and len(f['pose'][j])>=4 else None for f in frames]
            row[name+'_visibility_recorded_rate']=sum(v is not None for v in scores)/len(scores)
            row[name+'_visible_gt_05']=sum(v is not None and v>.5 for v in scores)/len(scores)
        row.update(handedness_switch_count=None,rotation_at_dropout=None,standard_present_zeroed_rate=None,correctness=None,reason='Raw body visibility is not post-harmonisation Standard Schema; no handedness IDs or ground truth assumed')
        rows.append(row)
    def output(name,values):
        if values:
            keys=list(dict.fromkeys(k for r in values for k in r))
            with (HERE/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(values)
    output('requests.csv',rows);output('gaps.csv',runs)
    summary=dict(input=str(src),directory_exists=src.is_dir(),recordings=len(files),analysed=len(rows),actual_frame_fields=sorted(fields),status='missing_input' if not files else 'raw_audit_only',replay_run=False)
    (HERE/'inventory.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=HERE.parents[1]/'cache/live_requests');a=p.parse_args();print(json.dumps(run(a.input),ensure_ascii=False,indent=2))
