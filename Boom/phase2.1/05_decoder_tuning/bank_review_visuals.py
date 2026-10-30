"""Render real pose sequences for neighbour review when RGB is unavailable."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import audit_report as report
ROOT=report.a.ex.ROOT;OUT=report.OUT

def pose_path(cid):
    base=cid.split('@')[0];prefix,key=base.split(':',1);dirs={'ttrs':('ttrs','ttrs_'+key),'thsl':('th_sl',key),'agent':('agent','agent_'+key.replace(':','_')),'t51s':('tsl51_user_sentence',key)}
    if prefix in dirs:
        folder,stem=dirs[prefix];p=ROOT/'cloud_s3/data_prep/pose_mp'/folder/(stem+'.npz')
        if p.exists():return p
    inv=pd.read_csv(OUT/'bank_inventory.csv');g=inv[inv['clip']==cid]
    if len(g) and isinstance(g.iloc[0].get('pose_path'),str):
        p=Path(g.iloc[0].pose_path)
        if p.is_file():return p
    return None

def render():
    """Five real sampled poses for each side; no fabricated RGB or inferred label truth."""
    report.enrich_sources();queue=pd.read_csv(OUT/'visual_review_queue.csv');priority=pd.read_csv(OUT/'review_priority.csv');dest=OUT/'pose_review';dest.mkdir(exist_ok=True)
    font='/System/Library/Fonts/Supplemental/Tahoma.ttf';exports=[]
    for _,r in priority[priority.bank=='eval'].head(12).iterrows():
        match=queue[(queue.bank=='eval')&(queue.nn_clip==r.nn_clip)].iloc[0];paths=[pose_path(match['clip']),pose_path(r.nn_clip)]
        if any(p is None for p in paths):continue
        im=Image.new('RGB',(1200,690),'#111b2c');draw=ImageDraw.Draw(im)
        def txt(x,y,s,n=19):draw.text((x,y),s,font=ImageFont.truetype(font,n),fill='white')
        txt(12,8,f"Query: {match.truth} -> nearest bank: {r.nn_label} | cos={match.nn_cos:.3f}",24)
        txt(12,42,'POSE ONLY · ภาพจากพิกัดจริง ไม่ใช่ภาพ RGB · query boundary เป็น auto-alignment')
        for row,p in enumerate(paths):
            with np.load(p) as z:kp=z['kp'];sc=z['sc'];hw=z['hw']
            lo,hi=(int(match.f0),int(match.f1)) if row==0 else (0,len(kp))
            txt(12,80+row*300,('TSL51 tune' if row==0 else r.nn_clip)+f' [{lo}:{hi}]')
            for col,f in enumerate(np.linspace(lo,min(hi-1,len(kp)-1),5).astype(int)):
                scale=min(220/float(hw[1]),240/float(hw[0]));points=kp[f]*np.array([hw[1],hw[0]])*scale+np.array([col*240+10,115+row*300])
                edges=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12)]
                for base in (91,112):
                    for finger in range(5):
                        chain=[base]+list(range(base+1+finger*4,base+5+finger*4));edges+=list(zip(chain,chain[1:]))
                for a,b in edges:
                    if b<len(points) and min(sc[f,a],sc[f,b])>.5:draw.line([tuple(points[a]),tuple(points[b])],fill='#65caff',width=2)
                txt(col*240+10,365+row*300,f'frame {f}',15)
        file=dest/f'pair_{len(exports)+1:02d}.jpg';im.save(file,quality=90);exports.append(dict(image=str(file),query_clip=match['clip'],bank_clip=r.nn_clip,query_pose=str(paths[0]),bank_pose=str(paths[1])))
    pd.DataFrame(exports).to_csv(dest/'index.csv',index=False)
    body='<meta charset="utf-8"><h1>Real pose review — not RGB</h1>'+''.join(f'<p>{x["query_clip"]} → {x["bank_clip"]}</p><img width="1200" src="{Path(x["image"]).name}">' for x in exports)
    (dest/'index.html').write_text(body);print('Rendered',len(exports))
if __name__=='__main__':render()
