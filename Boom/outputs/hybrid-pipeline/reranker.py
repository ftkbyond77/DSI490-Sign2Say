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


def load_manual(path):
    """Vocabulary lives in external data, never in scoring branches."""
    import json
    data=json.loads(__import__('pathlib').Path(path).read_text())
    if data.get('schema_version')!=1:raise ValueError('Unsupported manual-rule schema')
    entries={}
    for entry in data['concepts']:
        if entry['concept'] in entries:raise ValueError('Duplicate concept')
        if not entry['variants']:raise ValueError('Empty variants')
        for v in entry['variants']:
            if v['num_hands'] not in (1,2):raise ValueError('Invalid num_hands')
            if v['bimanual_type'] not in ('single','symmetric','asymmetric'):raise ValueError('Invalid bimanual_type')
            if v['wrist_y_min']>v['wrist_y_max']:raise ValueError('Invalid wrist bounds')
            if not v['source'].get('pdf_page'):raise ValueError('Missing source page')
            if set(v['checks'])-{'circling_min','speed_balance_max','alternating_min'}:raise ValueError('Unknown check')
        entries[entry['concept']]=entry
    return entries


def _finite(x):return x is not None and math.isfinite(float(x))
def _bound(x,low=None,high=None,tolerance=.4):
    if not _finite(x):return None
    distance=max((low-x) if low is not None else 0,(x-high) if high is not None else 0,0)
    return float(np.clip(1-2*distance/tolerance,-1,1))


def manual_match(query,entry):
    """Best supported book variant; unavailable evidence contributes zero, never a penalty."""
    both=min(query['observed_rate'])>=.5
    results=[]
    for v in entry['variants']:
        parts={};skipped=[];expected=1
        reliable=[h for h in range(2) if query['observed_rate'][h]>=.5 and _finite(query['wrist_y'][h])]
        if v['num_hands']==2:
            hands=[0,1] if both and len(reliable)==2 else []
        else:
            # Mirror invariant: choose the faster reliably observed hand, not a fixed anatomical side.
            moving=[h for h in reliable if _finite(query['median_speed'][h])]
            hands=[max(moving,key=lambda h:query['median_speed'][h])] if moving else reliable[:1]
        if hands:parts['wrist_y']=float(np.mean([_bound(query['wrist_y'][h],v['wrist_y_min'],v['wrist_y_max']) for h in hands]))
        else:skipped.append('wrist_y: insufficient reliable hands')
        limit=v.get('target_distance_lr_wrist_max')
        # Symmetric does NOT imply proximity: only an explicitly sourced variant sets this limit.
        if limit is not None:
            expected+=1
            value=_bound(query.get('distance_lr_wrist'),high=limit)
            if both and value is not None:parts['wrist_distance']=value
            else:skipped.append('wrist_distance: missing hand mask')
        for key,threshold in v['checks'].items():
            expected+=1;value=None
            if both:
                if key=='circling_min':
                    values=[x for x in query['circling'] if _finite(x)]
                    if len(values)==2:
                        x=min(values) if v['active_hand']=='both' else max(values)
                        value=_bound(x,low=threshold,tolerance=.5)
                elif key=='speed_balance_max':value=_bound(query.get('speed_balance'),high=threshold,tolerance=.5)
                elif key=='alternating_min':value=_bound(query.get('alternating'),low=threshold,tolerance=.5)
            if value is None:skipped.append(key+': unavailable or masked')
            else:parts[key]=value
        # Fixed denominator prevents a single surviving check receiving a full bonus.
        score=sum(parts.values())/expected
        results.append(dict(variant=v['id'],score=score,coverage=len(parts)/expected,parts=parts,skipped=skipped,source=v['source']))
    best=max(results,key=lambda x:x['score'])
    return dict(best=best,variants=results,rule_bonus=(.15 if best['score']>=0 else .20)*best['score'])


def rerank_detailed(candidates,query,table,manual,detailed,alpha=.1,manual_weight=.5):
    if not 0<=manual_weight<=1:raise ValueError('manual_weight must be in [0,1]')
    rows=rerank(candidates,query,table,alpha=alpha)
    for row in rows:
        row['bank_bonus']=row['bonus'];row['manual_bonus']=0.;row['manual_evidence']=None
        entry=manual.get(row['word'])
        if entry:
            evidence=manual_match(detailed,entry)
            row['manual_evidence']=evidence;row['manual_bonus']=manual_weight*evidence['rule_bonus']
        row['bonus']=float(np.clip(row['bank_bonus']+row['manual_bonus'],-.20,.15))
        row['final_score']=float(row['support'])+row['bonus']
    return sorted(rows,key=lambda x:-x['final_score'])
