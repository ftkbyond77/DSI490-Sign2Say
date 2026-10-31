"""Render actual saved predictions alongside source video or explicitly labelled pose."""
from pathlib import Path
import json, csv, subprocess, html
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'outputs/visuals'; OUT.mkdir(exist_ok=True)
FONT='/System/Library/Fonts/Supplemental/Tahoma.ttf'
def read(p):
    return json.loads(p.read_text())
def text(im,xy,s,size=19,color='#e5edf7'):
    ImageDraw.Draw(im).text(xy,str(s),font=ImageFont.truetype(FONT,size),fill=color)
def segs(r):
    return [s for s in r['segments'] if s.get('status')!='transition']
def interval(s):
    return s.get('t0',s['f0']/25),s.get('t1',s['f1']/25)
def panel(im,r,t,x,y,label):
    text(im,(x,y),label,23,'#65caff')
    active=[s for s in segs(r) if interval(s)[0]<=t<interval(s)[1]]
    if not active:
        text(im,(x,y+38),'ไม่มีช่วงคำที่ decoder เลือก',18); return
    s=active[0]; a,b=interval(s)
    text(im,(x,y+35),f"{s['nearest']}  [{a:.2f}–{b:.2f}s]",23)
    text(im,(x,y+70),f"{s.get('status')} | p_correct={s.get('p_correct',0):.3f}",16)
    for k,c in enumerate(s.get('candidates',[])[:5]):
        text(im,(x,y+102+29*k),f"{k+1}. {c['word']}   cos={c['score']:.3f}",18)
def compose(frame,r0,r1,t,title):
    im=Image.new('RGB',(1100,640),'#111b2c')
    frame=frame.copy();frame.thumbnail((345,485)); im.paste(frame,(15,80))
    text(im,(18,15),title,25);text(im,(18,48),f'{t:.2f}s | ซ้าย: ภาพต้นทาง  ·  ขวา: ผลจริงสองรูปแบบ',17)
    panel(im,r0,t,380,95,'เดิม · baseline')
    panel(im,r1,t,740,95,'ปรับ · combined')
    text(im,(380,390),'cos = cosine similarity ไม่ใช่ probability',17,'#ffcb75')
    text(im,(380,420),'p_correct = calibrator เดิม ยังไม่ได้ยืนยัน calibration หลังปรับ',16)
    text(im,(380,450),'ช่วงว่าง = decoder ไม่เลือกช่วง ไม่ยืนยันว่าผู้ทำท่าหยุด',16)
    text(im,(20,590),'เปรียบเทียบ decoder; ไม่ใช่ Hybrid 70:30 | เปลี่ยนคำ ≠ ทายถูกขึ้น',19,'#ffcb75')
    return im

def encode(name,frames):
    target=OUT/f'{name}.mp4'
    proc=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1100x640','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    try:
        for im in frames:proc.stdin.write(im.tobytes())
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError('ffmpeg failed')
    return target

def contact(name,r0,r1,get_frame,title):
    samples=[]
    for label,r in [('เดิม',r0),('ปรับ',r1)]:
        ss=segs(r); sheet=Image.new('RGB',(max(1,len(ss))*245,610),'#111b2c')
        for j,s in enumerate(ss):
            a,b=interval(s); t=(a+b)/2
            frame=get_frame(t);frame.thumbnail((230,290));sheet.paste(frame,(j*245,65))
            text(sheet,(j*245+6,8),f'{label} #{j+1} {a:.2f}–{b:.2f}s',15)
            text(sheet,(j*245+6,35),s['nearest'],20,'#65caff')
            text(sheet,(j*245+6,365),f"p_correct={s.get('p_correct',0):.3f}",16)
            for k,c in enumerate(s.get('candidates',[])[:5]):text(sheet,(j*245+6,400+k*32),f"{k+1} {c['word']} {c['score']:.3f}",16)
        samples.append(sheet)
    im=Image.new('RGB',(max(x.width for x in samples),1300),'#111b2c')
    text(im,(10,8),title,24);text(im,(10,43),'ภาพกลางช่วงคำ / Top-5 cosine ตามลำดับจริง / ไม่มี timed ground truth',19)
    im.paste(samples[0],(0,80));im.paste(samples[1],(0,690));im.save(OUT/f'{name}.jpg',quality=90)


def main():
    run=Path(read(HERE/'outputs/latest.json')['path']); items=[]
    for i,row in enumerate(read(run/'results.json')):
        src=ROOT/'Boom/boom datatest'/row['video'];cap=cv2.VideoCapture(str(src))
        if not cap.isOpened():raise RuntimeError(src)
        duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)
        def frame(t):
            cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'Cannot decode {src} at {t}')
            return Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        r0=row['modes']['baseline']['result'];r1=row['modes']['combined_without_csls']['result'];name=f'boom_{i+1:02d}'
        contact(name,r0,r1,frame,src.stem)
        encode(name,(compose(frame(float(t)),r0,r1,float(t),src.stem) for t in np.arange(0,duration-.1,.1)))
        cap.release();items.append((name,src.stem,'RGB video'));print('Rendered',name,flush=True)
    run=Path(read(HERE/'outputs/latest_tsl51.json')['path']); rows=list(csv.DictReader((run/'paired_wer.csv').open()))
    predictions=read(run/'predictions.json')
    # Deterministic first improved and first worsened examples; no selection by visual appeal.
    for kind,test in [('improved',lambda d:d<0),('worsened',lambda d:d>0)]:
        row=next(r for r in rows if test(float(r['delta'])));cid=row['clip'];rs={r['mode']:r for r in predictions if r['clip']==cid}
        path=ROOT/'cloud_s3/data_prep/pose_mp/tsl51_user_sentence'/f'{cid.split(":",1)[1]}.npz'
        z=np.load(path);kp=z['kp'];sc=z['sc']
        def frame(t):
            n=min(int(t*25),len(kp)-1);im=Image.new('RGB',(345,480),'#21334c');draw=ImageDraw.Draw(im)
            edges=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12)]
            for base in (91,112):
                for finger in range(5):
                    chain=[base]+list(range(base+1+4*finger,base+5+4*finger));edges.extend(zip(chain,chain[1:]))
            for a,b in edges:
                if min(sc[n,a],sc[n,b])>.5:
                    pa=tuple((kp[n,a]*[345,480]).astype(int));pb=tuple((kp[n,b]*[345,480]).astype(int));draw.line([pa,pb],fill='#65caff',width=3)
            text(im,(8,8),'POSE ONLY · ไม่ใช่ภาพ RGB',16)
            return im
        r0=rs['baseline']['result'];r1=rs['combined_without_csls']['result'];name=f'tsl51_{kind}'
        title='TSL51 '+kind+' | GT: '+' '.join(rs['baseline']['ref'])
        contact(name,r0,r1,frame,title)
        encode(name,(compose(frame(float(t)),r0,r1,float(t),title) for t in np.arange(0,len(kp)/25,.1)))
        items.append((name,title,f"pose; WER {row['baseline']} → {row['combined_without_csls']}"));print('Rendered',name,flush=True)
    body='<meta charset="utf-8"><style>body{background:#111b2c;color:#eee;font:18px sans-serif;margin:30px}video{max-width:100%}img{max-width:100%}section{margin:50px 0}a{color:skyblue}</style><h1>Actual decoder comparison</h1><p>baseline vs combined_without_csls; symbolic weight=0. TSL51 examples are tune data, not held-out test. Original audio omitted. Images show predicted segment midpoints, not verified sign boundaries.</p>'
    for name,title,kind in items:body+=f'<section><h2>{html.escape(title)}</h2><p>{html.escape(kind)}</p><video controls preload="metadata" src="{name}.mp4"></video><p><a href="{name}.jpg">Full resolution contact sheet</a></p><img src="{name}.jpg"></section>'
    (OUT/'index.html').write_text(body)
    (OUT/'manifest.json').write_text(json.dumps({'source_boom':str(Path(read(HERE/'outputs/latest.json')['path'])),'source_tsl51':str(run),'items':items},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
