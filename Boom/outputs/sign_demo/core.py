from pathlib import Path
import hashlib
import numpy as np
from extract_keypoints import extract_video_keypoints

BASE = Path(__file__).parent
VIDEO_DIR = Path('/Users/thawaphorn/Pee4/Term1/DSI490/data/videos')
IDS = ['919', '1577', '36827', '1866', '1289']

def references():
    result = []
    for ident in IDS:
        paths = sorted(VIDEO_DIR.glob(f'{ident}_*.mp4'))
        if not paths:
            raise ValueError(f'ไม่พบคลิปอ้างอิง ID {ident} ใน {VIDEO_DIR}')
        result.append(paths[0])
    return result

def label(path):
    return Path(path).stem.split('_', 1)[1].replace('_', ' ')

def normalize(raw):
    pose = raw['pose'].astype(float)
    if len(pose) < 2:
        raise ValueError('คลิปต้องมีอย่างน้อย 2 เฟรมที่อ่านได้')
    valid = np.isfinite(pose[:, [5,6], :2]).all(axis=(1,2))
    scale = np.linalg.norm(pose[:,5,:2] - pose[:,6,:2], axis=1)
    valid &= scale > 0.02
    coverage = float(valid.mean())
    if coverage < 0.5 or valid.sum() < 2:
        raise ValueError('ตรวจพบไหล่ไม่เพียงพอ กรุณาใช้คลิปที่เห็นครึ่งตัวบนชัดเจน')
    origin = (pose[:,5,:2] + pose[:,6,:2]) / 2
    # Use 2D coordinates only: hand and pose z have different origins.
    parts = [raw['left_hand'], raw['right_hand'], pose]
    points = np.concatenate([a[:,:,:2] for a in parts], axis=1)
    mask = np.isfinite(points).all(axis=2)
    normalized = (points-origin[:,None,:])/scale[:,None,None]
    normalized[~mask] = 0
    normalized, mask = normalized[valid], mask[valid]
    idx = np.unique(np.linspace(0,len(normalized)-1,min(80,len(normalized))).astype(int))
    return dict(points=normalized[idx], mask=mask[idx],
                pose_coverage=coverage, left_coverage=float(np.isfinite(raw['left_hand'][:,0,0]).mean()),
                right_coverage=float(np.isfinite(raw['right_hand'][:,0,0]).mean()), frames=len(pose))

def extract(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = BASE/'cache'
    cache.mkdir(exist_ok=True)
    dest = cache/f'v1_{digest}.npz'
    if dest.exists():
        with np.load(dest, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    result = normalize(extract_video_keypoints(str(path)))
    np.savez_compressed(dest, **result)
    return result

def distance(a,b):
    ap,bp = a['points'],b['points']
    # Equal group influence. Missing hand observations are excluded, never zero-filled as evidence.
    costs = np.zeros((len(ap),len(bp)))
    total = np.zeros_like(costs)
    for region,weight in [(slice(0,21),0.25),(slice(21,42),0.25),(slice(42,None),0.5)]:
        common = a['mask'][:,None,region] & b['mask'][None,:,region]
        sq = ((ap[:,None,region,:]-bp[None,:,region,:])**2).sum(axis=-1)
        count = common.sum(axis=-1)
        available = count > 0
        costs += weight * np.sqrt((sq*common).sum(axis=-1)/np.maximum(count,1)) * available
        total += weight*available
    costs /= np.maximum(total,1e-12)
    n,m=costs.shape
    dp=np.full((n+1,m+1),np.inf); dp[0,0]=0
    steps=np.zeros((n+1,m+1),dtype=int)
    for i in range(1,n+1):
        for j in range(1,m+1):
            options=[(i-1,j-1),(i-1,j),(i,j-1)]
            pi,pj=min(options,key=lambda pos: dp[pos])
            dp[i,j]=costs[i-1,j-1]+dp[pi,pj]
            steps[i,j]=steps[pi,pj]+1
    return float(dp[n,m]/steps[n,m])
