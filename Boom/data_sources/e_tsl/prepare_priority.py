"""Download a reviewed URL sample and audit duplicates without modifying any bank."""
from __future__ import annotations
import csv,json,hashlib,time,urllib.request,html
from pathlib import Path
import pandas as pd
import cv2
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def digest(p:Path)->str:
    """Compute byte-level SHA256; perceptual duplicates are not inferred."""
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def run()->None:
    """Retrieve checked priority media and output provenance plus visual review."""
    videos=HERE/'priority_videos';videos.mkdir(exist_ok=True)
    thumbs=HERE/'thumbnails';thumbs.mkdir(exist_ok=True)
    checks=list(csv.DictReader((HERE/'priority_url_checks.csv').open(encoding='utf-8-sig')))
    manifest=pd.read_parquet(ROOT/'cloud_s3/data_prep/manifest/clips.parquet')
    sources=manifest[manifest.dataset.astype(str).isin(['thsl','th_sl'])]
    sources[['clip_id','source_file','gloss','concept','source_url','role']].to_csv(HERE/'thsl_source_inventory.csv',index=False)
    metadata=pd.read_csv(ROOT/'metadata/metadata_word.csv')
    hashes={str(h):list(g.uid.astype(str)) for h,g in metadata.dropna(subset=['sha256']).groupby('sha256')}
    existing=[]
    # Only inspect source videos referenced by the manifest, never unrelated user media.
    for value in manifest.source_file.dropna().astype(str).unique():
        path=ROOT/'cloud_s3'/value
        if path.is_file() and path.suffix.lower() in {'.mp4','.mov','.avi','.webm'}:existing.append(path)
    bysize={}
    for path in existing:bysize.setdefault(path.stat().st_size,[]).append(path)
    local_hashes={};rows=[];cards=[]
    for r in checks:
        if r['status']!='200':continue
        filename=Path(r['video_url']).name;path=videos/filename
        if not path.exists():
            temp=path.with_suffix('.part')
            with urllib.request.urlopen(r['video_url'],timeout=45) as response,temp.open('wb') as out:
                if 'video' not in response.headers.get('Content-Type',''):raise ValueError('Not a video response')
                while block:=response.read(1048576):out.write(block)
            if r['bytes'] and temp.stat().st_size!=int(r['bytes']):raise ValueError('Incomplete download')
            temp.replace(path);time.sleep(1)
        sha=digest(path);matches=[]
        for candidate in bysize.get(path.stat().st_size,[]):
            if candidate not in local_hashes:local_hashes[candidate]=digest(candidate)
            if local_hashes[candidate]==sha:matches.append(str(candidate))
        source_match=sources[sources.source_url.fillna('').astype(str)==r['video_url']]
        label_match=sources[(sources.concept.fillna('').astype(str)==r['word']) | (sources.gloss.fillna('').astype(str)==r['word'])]
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));decoded=0
        for frac in [.25,.5,.75]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames*frac));ok,frame=cap.read()
            if ok:cv2.imwrite(str(thumbs/f'{path.stem}_{int(frac*100)}.jpg'),frame);decoded+=1
        cap.release()
        row=dict(word=r['word'],sign_id=r['sign_id'],video_url=r['video_url'],path=str(path),sha256=sha,bytes=path.stat().st_size,frames=frames,fps=fps,duration_s=frames/fps if fps else None,decoded_sample_frames=decoded,exact_metadata_hash_matches=' | '.join(hashes.get(sha,[])),exact_local_file_matches=' | '.join(matches),exact_source_url_matches=' | '.join(source_match.clip_id.astype(str)),same_label_thsl_clips=len(label_match),same_label_clip_ids=' | '.join(label_match.clip_id.astype(str)),dedup_status='exact duplicate' if hashes.get(sha) or matches else ('same source URL; inspect' if len(source_match) else 'no exact match in available evidence; unseen not established'),human_review='pending',bank_added=False)
        rows.append(row)
        e=html.escape
        cards.append(f'<article><h2>{e(r["word"])} — sign {e(r["sign_id"])}</h2><video controls preload="metadata" width="400" src="priority_videos/{e(filename)}"></video><p>{e(row["dedup_status"])}</p><p>THSL ชื่อตรงกัน: {len(label_match)} คลิป — ไม่ได้ยืนยันว่าเป็นวิดีโอซ้ำ</p><p><a href="{e(r["video_url"],quote=True)}">ต้นทาง</a></p></article>')
        pd.DataFrame(rows).to_csv(HERE/'priority_download_audit.csv',index=False,encoding='utf-8-sig');print(r['word'],row['dedup_status'],flush=True)
    summary=dict(downloaded=len(rows),all_sample_frames_decoded=all(r['decoded_sample_frames']==3 for r in rows),exact_duplicates=sum(r['dedup_status']=='exact duplicate' for r in rows),manifest_thsl_rows=len(sources),thsl_source_url_nonmissing=int(sources.source_url.notna().sum()),metadata_hash_rows=int(metadata.sha256.notna().sum()),available_manifest_rgb_files=len(existing),bank_modified=False,limitations=['No visual identity matching','Hash mismatch does not rule out re-encoding or edited duplicates','Absent source files limit duplicate audit','No human validation of meaning or sign variant yet'])
    (HERE/'download_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (HERE/'local_review.html').write_text('<!doctype html><meta charset="utf-8"><title>e-TSL local review</title><style>body{font:17px sans-serif;margin:30px;background:#f5f6fa}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}article{background:white;padding:18px;border-radius:12px}video{max-width:100%}</style><h1>e-TSL: วิดีโอจริง 12 ตัวอย่าง</h1><p>วางวิดีโอข้างกันเพื่อเปิดดูท่า ยังไม่เพิ่ม bank และยังไม่ยืนยันว่าคลิปใหม่ไม่ซ้ำข้อมูลเดิม</p><div class="grid">'+''.join(cards)+'</div>')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':
    run()
    import source_labels
    source_labels.build()
