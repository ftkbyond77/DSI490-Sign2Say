"""Extract independent source clips; no synthetic train/test split."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from extract_keypoints import extract_video_keypoints, PIPELINE_VERSION

DEFAULT_IDS='919,1577,36827,1866,1289,1246,1565,26341,6877,7553,8245,55791,28860,7219,56139,32614,32636,51405,36879,36952'

def features(raw,frames=64):
    pose=raw['pose']; t=len(pose)
    if t<2: raise ValueError('Video has fewer than two decoded frames')
    shoulders=pose[:,[5,6],:2]
    good=np.isfinite(shoulders).all((1,2))
    width=np.linalg.norm(shoulders[:,0]-shoulders[:,1],axis=-1)
    good &= width>0.02
    if good.mean()<0.5: raise ValueError('Shoulders detected in fewer than half the frames')
    origin=shoulders.mean(1)
    for j in range(2): origin[:,j]=np.interp(np.arange(t),np.flatnonzero(good),origin[good,j])
    width=np.interp(np.arange(t),np.flatnonzero(good),width[good])
    hands=np.stack([raw['left_hand'][:,:,:2],raw['right_hand'][:,:,:2]],axis=1)
    valid=np.isfinite(hands).all(-1)
    if valid.any(-1).sum()<4: raise ValueError('Too few observed hand tokens')
    hands=(hands-origin[:,None,None,:])/width[:,None,None,None]
    hands[~valid]=0
    # Fixed-length nearest temporal sampling preserves the actual observation mask.
    idx=np.linspace(0,t-1,frames).round().astype(int)
    if valid[idx].any(-1).sum()<2: raise ValueError('Too few observed tokens after sampling')
    return hands[idx].astype('float32'),valid[idx],valid.mean((0,2)).tolist()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--videos',type=Path,default=Path('/Users/thawaphorn/Pee4/Term1/DSI490/data/videos'))
    parser.add_argument('--ids',default=DEFAULT_IDS)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'data')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    records=[]; failures=[]; seen=set(); sources=[]
    for ident in args.ids.split(','):
        found=sorted(args.videos.glob(ident.strip()+'_*.mp4'))
        if not found:
            failures.append({'id':ident,'error':'No source video'}); continue
        path=found[0]; digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen: continue
        seen.add(digest); dest=args.output/(ident+'.npz')
        sources.append(dict(id=ident,label=path.stem.split('_',1)[1].replace('_',' '),video=str(path),sha256=digest,signer_id=None,split='train'))
        try:
            if dest.exists():
                with np.load(dest,allow_pickle=False) as d:
                    if str(d['sha256'])!=digest or str(d.get('pipeline_version',''))!=PIPELINE_VERSION: raise ValueError('Source or extraction pipeline changed; remove stale feature cache')
                    coverage=d['coverage'].tolist()
            else:
                raw=extract_video_keypoints(str(path)); x,mask,coverage=features(raw)
                np.savez_compressed(dest,xy=x,visible=mask,coverage=coverage,sha256=digest,pipeline_version=PIPELINE_VERSION)
            record=dict(id=ident,label=path.stem.split('_',1)[1].replace('_',' '),video=str(path),
                features=str(dest.resolve()),sha256=digest,signer_id=None,split='train',hand_coverage=coverage)
            records.append(record); print('OK',ident,record['label'],coverage,flush=True)
        except Exception as exc:
            failures.append(dict(id=ident,error=str(exc))); print('FAILED',ident,str(exc),flush=True)
    (args.output/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2))
    manifest=dict(schema=1,pipeline_version=PIPELINE_VERSION,evaluation='training-only; no independent validation/test recordings',records=records,failures=failures)
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    if len(records)<2: raise SystemExit('Need at least two usable classes')

if __name__=='__main__': main()
