"""Blind listening page for one generated folder -> <out>/listen.html (open it in a browser, files stay local).
Each listener: name, then per clip (random order, ids only) pick the emotion heard + naturalness 1-5; 'Download CSV' at the end.
  python make_listening.py --out out          then  python score.py --out out answers_*.csv
"""
import argparse, csv, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--k", type=int, default=2, help="sentences per listener (random, balanced over all clips)"); ap.add_argument("--fixed", default="", help="comma list of sentence indices every listener gets, e.g. 0,4,7")
a = ap.parse_args()
sent = {r["id"]: int(r["sent"]) for r in csv.DictReader(open(Path(a.out) / "key.csv", encoding="utf-8"))}   # sentence index only: no emotion/method leaks
clips = [[r["id"], sent[r["id"]]] for r in csv.DictReader(open(Path(a.out) / "clips.csv", encoding="utf-8"))]
html = """<!doctype html><meta charset=utf-8><title>ฟังเสียง</title>
<style>body{font-family:sans-serif;max-width:640px;margin:2em auto;padding:0 1em}button{margin:.2em;padding:.5em .9em;font-size:1em}
.sel{background:#2563eb;color:#fff}#p{color:#666}</style>
<h2>ทดสอบการฟัง</h2><p>ฟังแต่ละคลิป แล้วเลือก <b>อารมณ์ที่ได้ยินจากน้ำเสียง</b> (ไม่ใช่จากความหมายของประโยค) และความเป็นธรรมชาติ 1–5</p>
<div id=start>ชื่อผู้ฟัง: <input id=who> <button onclick=go()>เริ่ม</button></div>
<div id=q style=display:none><p id=p></p><audio id=au controls autoplay></audio>
<p>อารมณ์: <span id=emo></span></p><p>ความเป็นธรรมชาติ (1 = เหมือนหุ่นยนต์, 5 = เหมือนคนจริง): <span id=mos></span></p>
<button onclick=next()>ถัดไป ▶</button></div><div id=done style=display:none><h3>เสร็จแล้ว ขอบคุณ</h3><button onclick=dl()>ดาวน์โหลด CSV</button></div>
<script>
const CLIPS=__CLIPS__, EMO={neutral:'ปกติ',happy:'ดีใจ',sad:'เศร้า',angry:'โกรธ'};
const FIX=__FIX__; const NS=FIX.length?FIX:[...new Set(CLIPS.map(c=>c[1]))].sort(()=>Math.random()-.5).slice(0,__K__);
let order=CLIPS.filter(c=>NS.includes(c[1])).map(c=>c[0]).sort(()=>Math.random()-.5), i=0, ans=[], cur={}, who='';
function btns(id,vals,labels,key){const e=document.getElementById(id);e.innerHTML='';vals.forEach((v,k)=>{const b=document.createElement('button');
 b.textContent=labels[k];b.onclick=()=>{cur[key]=v;[...e.children].forEach(c=>c.classList.remove('sel'));b.classList.add('sel')};e.appendChild(b)})}
function show(){cur={};document.getElementById('p').textContent=`คลิป ${i+1}/${order.length}`;document.getElementById('au').src=order[i]+'.wav';
 btns('emo',Object.keys(EMO),Object.values(EMO),'emo');btns('mos',[1,2,3,4,5],['1','2','3','4','5'],'mos')}
function go(){who=document.getElementById('who').value.trim()||'anon';start.style.display='none';q.style.display='';show()}
function next(){if(!cur.emo||!cur.mos){alert('เลือกอารมณ์และคะแนนก่อน');return}ans.push([who,order[i],cur.emo,cur.mos]);
 i++;if(i<order.length)show();else{q.style.display='none';done.style.display=''}}
function dl(){const t='listener,id,heard,mos\\n'+ans.map(r=>r.join(',')).join('\\n');const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([t],{type:'text/csv'}));a.download=`answers_${who}.csv`;a.click()}
</script>""".replace("__CLIPS__", json.dumps(clips)).replace("__K__", str(a.k)).replace("__FIX__", json.dumps([int(x) for x in a.fixed.split(",") if x.strip()]))
(Path(a.out) / "listen.html").write_text(html, encoding="utf-8"); print(f"-> {a.out}/listen.html ({len(clips)} clips; {a.k} random sentences per listener = {len(clips)*a.k//len(set(sent.values()))} clips each)")
