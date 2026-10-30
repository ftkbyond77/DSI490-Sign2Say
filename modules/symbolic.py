"""Evidence-limited symbolic pose scorer for the Hybrid retrieval experiment.

Only four attributes with support in the reviewed Thai Sign Language book are used:
number of hands, coarse location, two-hand relation, and main motion verbs.
Unknown dictionary fields never count as a mismatch.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
from .schema import NOSE, SH_L, SH_R, WR_L, WR_R

_MOTION = {"วน":"circle", "หมุน":"circle", "ยก":"up", "ขึ้น":"up", "ลด":"down", "ลง":"down", "แตะ":"touch", "แยก":"apart", "เข้าหา":"together", "ประกบ":"together", "สลับ":"alternate", "เลื่อน":"side"}
def _norm(s: str) -> str: return re.sub(r"[\s—–-].*", "", str(s)).strip()
def _verbs(s: str) -> set[str]: return {v for k,v in _MOTION.items() if k in str(s)}

def build_dictionary(reviewed: Path, concepts: list[str]) -> dict[str, list[dict]]:
    """Map reviewed book rows to production concepts, retaining all valid variants."""
    rows = json.loads(reviewed.read_text(encoding="utf-8")); out = {c: [] for c in concepts}
    by_norm = {_norm(c): c for c in concepts}
    for r in rows:
        raw = str(r.get("concept", "")); c = by_norm.get(_norm(raw))
        if not c or len(raw.split()) > 8: continue
        loc = str(r.get("location", "")); mov = str(r.get("movement", ""))
        hands = r.get("num_hands") if r.get("num_hands") in (1,2) else None
        relation = next((x for x in ("together","apart","alternate") if x in _verbs(mov)), None)
        out[c].append(dict(hands=hands, location_zone=_zone(loc), motion=sorted(_verbs(mov)), relation=relation, source=f"book{r.get('book')}:{r.get('printed_page')}"))
    return {k:v for k,v in out.items() if v}

def load_dictionary(path: Path, concepts: list[str]) -> dict[str, list[dict]]:
    """Load normalized attributes or build them from raw reviewed book rows."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, list)}
    return build_dictionary(path, concepts)

def _zone(text: str) -> str | None:
    t=str(text)
    if any(x in t for x in ("ปาก","คาง")): return "mouth"
    if any(x in t for x in ("หน้าอก","ลำตัว")): return "chest"
    if any(x in t for x in ("หน้าผาก","ศีรษะ","หน้า")): return "face"
    if any(x in t for x in ("เอว","ต่ำ")): return "low"
    if "แขน" in t or "ข้อศอก" in t: return "arm"
    return None

def observe(kp: np.ndarray, sc: np.ndarray) -> dict:
    """Extract robust coarse attributes from canonical, isotropic landmarks for one segment."""
    L,R=kp[:,WR_L],kp[:,WR_R]; okL=sc[:,WR_L]>.5; okR=sc[:,WR_R]>.5
    shoulder=np.linalg.norm(kp[:,SH_L]-kp[:,SH_R],axis=1); scale=float(np.nanmedian(shoulder[shoulder>1e-4]) or 1)
    active=[float(okL.mean())>.5,float(okR.mean())>.5]; hands=sum(active)
    w=np.nanmean(np.where((okL|okR)[:,None], (L+R)/2, np.nan),0); nose=np.nanmedian(kp[:,NOSE],0); sh=np.nanmedian((kp[:,SH_L]+kp[:,SH_R])/2,0)
    y=(w[1]-sh[1])/scale
    zone="face" if w[1] < nose[1]+.18*scale else ("chest" if y < .65 else "low")
    d=np.linalg.norm(L-R,axis=1); relation="together" if hands==2 and np.nanmedian(d)/scale<.45 else None
    v=np.diff((L+R)/2,axis=0); motion=set()
    if len(v)>3:
        if abs(np.nanmean(v[:,1]))>.006: motion.add("up" if np.nanmean(v[:,1])<0 else "down")
        if np.nanstd(np.arctan2(v[:,1],v[:,0]))>1.4: motion.add("circle")
    return dict(hands=hands, location_zone=zone, relation=relation, motion=sorted(motion), hand_observed=dict(left=round(float(okL.mean()),3),right=round(float(okR.mean()),3)))

def score(obs: dict, specs: list[dict]) -> float:
    """Best supported attribute agreement across a word's variants, in [0,1]."""
    best=0.0
    for x in specs:
        terms=[]
        if x.get("hands") is not None: terms.append((.35, obs["hands"]==x["hands"]))
        if x.get("location_zone"): terms.append((.30, obs["location_zone"]==x["location_zone"]))
        if x.get("relation"): terms.append((.20, obs.get("relation")==x["relation"]))
        if x.get("motion"): terms.append((.15, bool(set(obs["motion"]) & set(x["motion"]))))
        if terms: best=max(best,sum(w*float(ok) for w,ok in terms)/sum(w for w,_ in terms))
    return float(best)
