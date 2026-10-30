"""Read-only input audit before the v8.1 integration experiment. No inference/test access."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT.parent/'data/Eval-Test'
OUT=HERE/'outputs'
VIDEO={'.mov','.mp4','.mkv','.avi','.webm'}
def digest(path:Path)->str:
    """Stream SHA256 for exact duplicate checks, never facial identity matching."""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path:Path,value:object)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
def readiness()->dict:
    """Require actual v8.1 code and artifacts; do not infer the version from a report."""
    config=json.loads((ROOT/'models/config.json').read_text())
    required=['scripts/decision.py','modules/handshape.py','models/hand_bank.npz','artifacts/eval/hand_bank_eval.npz']
    missing=[p for p in required if not (ROOT/p).is_file()]
    return dict(version=config.get('version'),ready=config.get('version')=='v8.1' and not missing,missing=missing,python=sys.executable,root=str(ROOT),no_inference_run=True,test_predictions_opened=False)
def require_v81()->None:
    """Fail before any inference, backup or test access when the specified baseline is missing."""
    r=readiness()
    if not r['ready']:raise RuntimeError(f"ต้องมี v8.1 จริงก่อน: local={r['version']}; missing={r['missing']}. ไม่รัน v7.2 แทนและไม่สร้าง backup ชื่อ v8.1 จาก v7.2")
def run()->None:
    """Inventory real media, prior development reuse and exact training-file duplicates."""
    import cv2
    OUT.mkdir(exist_ok=True);ready=readiness();save(OUT/'readiness.json',ready)
    olddir=ROOT/'Boom/boom datatest'
    old={digest(p):str(p) for p in olddir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO} if olddir.is_dir() else {}
    rows=[]
    for person in ['Boom','Jack','Pitty']:
        for p in sorted((DATA/person).iterdir()):
            if not p.is_file() or p.suffix.lower() not in VIDEO:continue
            h=digest(p);cap=cv2.VideoCapture(str(p));opened=cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
            rows.append(dict(person_folder=person,file=p.name,path=str(p),sha256=h,bytes=p.stat().st_size,readable=opened,fps=fps,frames=frames,width=width,height=height,duration_s=frames/fps if fps else None,reference_filename=p.stem,reference_needs_review=p.stem.endswith('_'),prior_development_duplicate=h in old,prior_path=old.get(h,''),reference_glosses=None,oov_status='pending v8.1 tokenizer + 328-word variant mapping'))
    df=pd.DataFrame(rows);df.to_csv(OUT/'eval_inventory.csv',index=False)
    df.groupby('person_folder').agg(clips=('file','size'),duration_s=('duration_s','sum'),prior_development_duplicates=('prior_development_duplicate','sum')).to_csv(OUT/'eval_summary.csv')
    # Use only declared metadata and exact file hashes. No face identification.
    manifest=pd.read_parquet(ROOT/'cloud_s3/data_prep/manifest/clips.parquet')
    train=manifest[manifest.role.astype(str).isin(['train','tagger_train'])]
    candidates=[];sizes=set(df.bytes);hashes=set(df.sha256);checked=0
    for _,r in train.iterrows():
        if not isinstance(r.get('source_file'),str):continue
        p=ROOT/'cloud_s3'/r.source_file
        if not p.is_file() or p.suffix.lower() not in VIDEO:continue
        checked+=1
        if p.stat().st_size not in sizes:continue
        h=digest(p)
        if h in hashes:candidates.append(dict(clip_id=str(r.clip_id),signer=str(r.signer),path=str(p),sha256=h,role=str(r.role)))
    save(OUT/'training_file_duplicates.json',dict(matches=candidates,available_training_videos_checked=checked,training_manifest_rows=len(train),conclusion='Exact-file check only; unavailable files and person identity remain unresolved'))
    pd.DataFrame([dict(person=p,declared_training_signer_id=None,person_overlap='unknown',reason='No user-verified mapping from folder person to training signer IDs; no facial identity inference') for p in ['Boom','Jack','Pitty']]).to_csv(OUT/'signer_audit.csv',index=False)
    save(OUT/'protocol.json',dict(baseline_required='v8.1',gate=dict(max_set_wer_regression=.003,minimum_mean_wer_improvement_exclusive=.002,minimum_accepted_precision_each_set=.90),bootstrap=10000,selection='tune only; Eval-Test observation only; freeze before final test batch',test_history='TSL51 test previously evaluated in earlier work; do not describe as never seen',baseline_inference='test baseline delayed until final batch despite numbered step 2',face_identity_matching=False,backup_policy='Create artifacts/baselines/v8.1 only after verifying actual v8.1; current v7.2 not copied under that name'))
    (OUT/'REPORT.md').write_text('# v8.2 integration — preflight only\n\n'+json.dumps(ready,ensure_ascii=False,indent=2)+'\n\n'+df.groupby('person_folder').size().to_string()+'\n\nPrior development duplicates: '+str(int(df.prior_development_duplicate.sum()))+'\n\nNo baseline scores, OOV counts or model changes were fabricated. Tokenizer/variant mapping must come from actual v8.1. Folder names are supplied labels, not verified unseen signer IDs. Exact-file duplicate tests do not prove signer separation. Filename ending in underscore requires review; no silent ground-truth rewriting.\n')
    print(df[['person_folder','file','prior_development_duplicate']].to_string(index=False));print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':run()
