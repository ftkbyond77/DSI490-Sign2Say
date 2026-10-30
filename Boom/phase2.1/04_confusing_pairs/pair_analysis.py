"""Auditable real THSL pair inventory, raw-hand evidence and human review preparation."""
from __future__ import annotations
from pathlib import Path
import json
import subprocess
import sys
import hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
HYBRID=HERE.parent/'02_hybrid_score'
OUT=HERE/'outputs'
PYTHON=ROOT/'Boom/venv-sign2say/bin/python'
PAIRS=[('ชอบ','ข้าว'),('ฉัน','คุณ'),('ฉัน','แฟน'),('ลืม','ญาติ'),('ช่วย','ขนมปัง')]
sys.path.insert(0,str(HYBRID))


def write_json(path: Path,value) -> None:
    """Persist UTF-8 metadata, leaving annotation ownership explicit."""
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path: Path) -> str:
    """Fingerprint input files for reproducibility."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    """Audit real manifests; distinguish physical source clips from extractor copies."""
    from handshape_tracking import HandStateTracker,FINGERS
    OUT.mkdir(parents=True,exist_ok=True)
    shards=ROOT/'cloud_s3/data_prep/shards/clips.parquet'
    manifest=ROOT/'cloud_s3/data_prep/manifest/clips.parquet'
    d=pd.read_parquet(shards)
    d=d[(d.dataset=='th_sl')&(d.extractor=='mp')].drop_duplicates('clip_id')
    m=pd.read_parquet(manifest).drop_duplicates('clip_id').set_index('clip_id')
    words=sorted({w for pair in PAIRS for w in pair})
    clips=[];finger_rows=[];dictionary={w:[] for w in words}
    for _,r in d[d.concept.isin(words)].iterrows():
        key=r.clip_id.split(':',1)[1];base=ROOT/'cloud_s3/data_prep'
        raw=base/'pose_mp_raw/th_sl'/f'{key}.npz'
        standard=base/'pose_mp/th_sl'/f'{key}.npz'
        source=str(m.loc[r.clip_id,'source_file']) if r.clip_id in m.index else ''
        paths=[ROOT/source,ROOT/'cloud_s3'/source,base/source] if source else []
        local=next((p for p in paths if p.is_file()),None)
        item=dict(clip_id=r.clip_id,word=r.concept,signer=r.signer,role=r.role,
            raw_path=str(raw),raw_exists=raw.is_file(),standard_path=str(standard),standard_exists=standard.is_file(),
            source_file=source,local_video=str(local) if local else None)
        if raw.is_file():
            with np.load(raw,allow_pickle=False) as z:
                n=len(z['t']);hw=tuple(int(x) for x in z['hw']); ts=z['t'].astype(float)
                item.update(n_frames=n,duration_s=float(ts[-1]) if n else 0,
                    left_observed=float(np.any(z['lh']!=0,axis=(1,2)).mean()),
                    right_observed=float(np.any(z['rh']!=0,axis=(1,2)).mean()))
                for side,k in [('left','lh'),('right','rh')]:
                    tracker=HandStateTracker()
                    for i,hand in enumerate(z[k].astype(float)):
                        obs=tracker.update(hand,hw)
                        finger_rows.append(dict(clip_id=r.clip_id,word=r.concept,side=side,time_s=float(ts[i]),
                            shape=obs['shape'],quality=obs['quality'],**obs['states']))
        clips.append(item)
        dictionary[r.concept].append(item)
    df=pd.DataFrame(clips);df.to_csv(OUT/'thsl_clips.csv',index=False)
    pd.DataFrame(finger_rows).to_csv(OUT/'estimated_finger_states.csv',index=False)
    errors=pd.read_csv(HYBRID/'boom_datatest/scientific/segments.csv')
    errors=errors[(errors['mode']=='ai_only')&errors.ground_truth.notna()]
    pair_rows=[];evidence=[]
    for index,(a,b) in enumerate(PAIRS,1):
        row=dict(pair_id=f'pair{index:02d}',reference=a,prediction=b)
        hits=errors[(errors.ground_truth==a)&(errors.word==b)]
        row['observed_errors']=len(hits)
        for prefix,word in [('a',a),('b',b)]:
            sub=df[df.word==word];train=sub[sub.role=='train']
            row.update({f'{prefix}_clips':len(sub),f'{prefix}_signers':sub.signer.nunique(),
                f'{prefix}_train_clips':len(train),f'{prefix}_train_signers':train.signer.nunique(),
                f'{prefix}_raw_available':int(sub.raw_exists.sum()),
                f'{prefix}_videos_available':int(sub.local_video.notna().sum())})
        row['ready_for_review']=row['a_raw_available']>0 and row['b_raw_available']>0
        row['multi_signer_training_ready']=min(row['a_train_signers'],row['b_train_signers'])>=3
        pair_rows.append(row)
        for _,hit in hits.iterrows():evidence.append(dict(pair_id=row['pair_id'],**hit.to_dict()))
    pd.DataFrame(pair_rows).to_csv(OUT/'pairs.csv',index=False)
    pd.DataFrame(evidence).to_csv(OUT/'boom_evidence.csv',index=False)
    write_json(OUT/'thsl_example_dictionary.json',dictionary)
    review=HERE/'manual_review.json'
    if not review.exists():
        write_json(review,{'instructions':'Geometry estimates are not human labels. Review true sign and boundaries before training. Do not fill labels from filename alone.',
            'segments':[dict(pair_id=e['pair_id'],video=e['video'],t0=e['t0'],t1=e['t1'],
                             proposed_reference=e['ground_truth'],boundary_verified=False,true_word=None,notes='') for e in evidence],
            'hand_attributes':[dict(clip_id=c['clip_id'],word=c['word'],intervals=[],verified=False) for c in clips]})
    write_json(OUT/'provenance.json',{'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [shards,manifest,HYBRID/'boom_datatest/scientific/segments.csv']},
        'pairs':PAIRS,'deduplication':'dataset=th_sl, extractor=mp, unique clip_id',
        'training_readiness':'at least 3 training signers per class (screening only, not sufficient for reliable accuracy)',
        'evidence':'Errors come from proposed text edit alignment; no verified temporal ground truth',
        'geometry_code_sha256':sha(HYBRID/'handshape_tracking.py')})
    print('Prepared',len(df),'real THSL clips across',len(words),'concepts',flush=True)


def run_prepare() -> None:
    """Use compatible project Python for PyArrow and raw-data processing."""
    from experiment_analysis import subprocess_environment
    subprocess.run([str(PYTHON),str(Path(__file__)),'prepare'],cwd=ROOT,env=subprocess_environment(),check=True)


def table(name: str) -> pd.DataFrame:
    """Load a generated real-data table."""
    return pd.read_csv(OUT/f'{name}.csv')


def show(df: pd.DataFrame,note: str) -> None:
    """Round tables and explain their evidential limits below each output."""
    from IPython.display import display,Markdown
    display(df.round(3));display(Markdown(note))


def pair_examples(pair_id: str) -> None:
    """Render actual raw hand trajectories for both concepts at observed frames."""
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    from handshape_tracking import EDGES
    pair=table('pairs').set_index('pair_id').loc[pair_id]
    clips=table('thsl_clips')
    for word in [pair.reference,pair.prediction]:
        for _,r in clips[(clips.word==word)&clips.raw_exists].iterrows():
            with np.load(r.raw_path,allow_pickle=False) as z:
                valid=np.flatnonzero(np.any(z['lh']!=0,axis=(1,2))|np.any(z['rh']!=0,axis=(1,2)))
                if not len(valid):display(Markdown(f'{r.clip_id}: ไม่พบมือ'));continue
                indices=np.unique(valid[np.linspace(0,len(valid)-1,3).astype(int)])
                fig,axes=plt.subplots(1,len(indices),figsize=(4*len(indices),4),squeeze=False)
                h,w=z['hw']
                for ax,i in zip(axes[0],indices):
                    for key,color,label in [('lh','tab:blue','left'),('rh','tab:orange','right')]:
                        hand=z[key][i].astype(float)
                        if not np.any(hand):continue
                        xy=hand[:,:2]*[w,h]
                        for a,b in EDGES:ax.plot(xy[[a,b],0],xy[[a,b],1],color=color,lw=1)
                        ax.scatter(xy[:,0],xy[:,1],c=color,s=8,label=label)
                    ax.set(xlim=(0,w),ylim=(h,0),title=f'{float(z["t"][i]):.2f}s');ax.set_aspect('equal')
                fig.suptitle(f'{word} | {r.clip_id} | {r.signer} | {r.role}');fig.tight_layout()
                fig.savefig(OUT/f'{pair_id}_{r.clip_id.replace(":","_")}.png');plt.show()
                display(Markdown('**ภาพโครงมือจริง:** เลือกต้น/กลาง/ท้ายจากเฟรมที่เห็นมือ ไม่ใช่ภาพ RGB และไม่ใช่การยืนยันช่วงท่าหลัก; แกนถูกปรับตามสัดส่วนภาพจริง มือที่บังกันยังทำให้ landmark ผิดได้'))


def boom_examples(pair_id: str) -> None:
    """Show video, actual segment frames and top-five for the suspected error."""
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown,Video
    from experiment_analysis import video_worker
    ev=table('boom_evidence');ev=ev[ev.pair_id==pair_id]
    top=pd.read_csv(HYBRID/'boom_datatest/scientific/top5.csv')
    for j,r in ev.iterrows():
        reviewed=next((x for x in json.loads((HERE/'manual_review.json').read_text())['segments'] if x['pair_id']==pair_id and x['video']==r.video),{})
        source=ROOT/'Boom/boom datatest'/r.video
        display(Video(str(source),embed=True,width=360))
        times=np.linspace(r.t0,r.t1,5,endpoint=False)+(r.t1-r.t0)/10
        frames=video_worker({'operation':'frames','videos':[{'path':str(source),'frames':[
            {'seconds':float(t),'output':str(OUT/f'{pair_id}_{j}_{i}.png')} for i,t in enumerate(times)]}]})
        fig,axes=plt.subplots(1,5,figsize=(14,5))
        for ax,f in zip(axes,frames):
            if f['path']:ax.imshow(plt.imread(f['path']))
            label=next((a['label'] if a['kind']=='sign' else 'พัก / ไม่ได้ทำท่า' for a in reviewed.get('frame_annotations',[]) if abs(a['time_s']-f['seconds'])<.015),'ยังไม่ตรวจ')
            ax.set_title(f'{f["seconds"]:.2f}s\n{label}');ax.axis('off')
        fig.tight_layout();plt.show()
        display(Markdown(f'**ช่วงต้องตรวจ:** {r.t0:.2f}–{r.t1:.2f}s; proposed reference={r.ground_truth}, AI={r.word}. ภาพจริงห้าเวลาใช้ตรวจว่าตัดท่าครบไหม ไม่ถือ edit alignment เป็นเฉลยเวลา'))
        if reviewed.get('review_status')=='mixed_signs_and_rest':
            display(Markdown('**ผู้ใช้ตรวจแล้ว:** ช่วงนี้ปน ฉัน → ชอบ → พัก จึงไม่ใช่ pure recognition error ของ ชอบ–ข้าว. Labels ยืนยันเฉพาะภาพที่แสดง ยังไม่ใช่ขอบเขตเวลาทั้งคำ'))
        show(top[(top.video==r.video)&(top.segment==r.segment)&(top['mode']=='ai_only')],
             '**Top-5 จริง:** ตรวจว่าคำจริงเข้า candidates หรือไม่ก่อนเลือกฝึกแยกคู่; score เป็น cosine ไม่ใช่ probability')


def review_status() -> pd.DataFrame:
    """Read manual decisions without overwriting human annotations on reruns."""
    return pd.DataFrame(json.loads((HERE/'manual_review.json').read_text())['segments'])


if __name__=='__main__':
    if sys.argv[1:] == ['prepare']:prepare()


def attribute_summary() -> None:
    """Compare estimated states without converting pseudo-labels into ground truth."""
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    df=table('estimated_finger_states')
    counts=df.groupby(['word','side','shape']).size().rename('frames').reset_index()
    counts['fraction']=counts.frames/counts.groupby(['word','side']).frames.transform('sum')
    show(counts,'**ผลประมาณ:** นับทุกเฟรมรวมช่วงพัก ไม่ใช่ handshape label ของคำทั้งคำ; unknown ไม่ใช่ folded และห้ามใช้จำนวนเฟรมแทนจำนวนตัวอย่างอิสระ')
    pivot=counts.pivot_table(index=['word','side'],columns='shape',values='fraction',fill_value=0)
    ax=pivot.plot.barh(stacked=True,figsize=(12,8));ax.set_xlabel('fraction of frames');ax.figure.tight_layout()
    ax.figure.savefig(OUT/'estimated_shape_distribution.png');plt.show()
    display(Markdown('**อ่านกราฟ:** ความต่างที่เห็นเป็นสมมติฐานให้ตรวจต่อ อาจเกิดจาก detector/mุมกล้อง/ช่วงพัก ไม่ได้พิสูจน์ว่าคู่นี้ต่างกันที่รูปมือ'))


def readiness() -> pd.DataFrame:
    """Training gate: verified pair boundaries and signer diversity are both required."""
    pairs=table('pairs');review=review_status()
    for i,r in pairs.iterrows():
        matched=review[review.pair_id==r.pair_id]
        verified=bool(len(matched) and matched.boundary_verified.all() and matched.true_word.notna().all())
        blocked=bool(len(matched) and 'review_status' in matched and matched.review_status.eq('mixed_signs_and_rest').any())
        verified=verified and not blocked
        pairs.loc[i,'review_status']='mixed_signs_and_rest' if blocked else 'awaiting_verification'
        pairs.loc[i,'manual_evidence_verified']=verified
        pairs.loc[i,'train_gate_pass']=verified and bool(r.multi_signer_training_ready)
    return pairs
