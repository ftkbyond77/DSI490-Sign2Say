"""One shared listening session: one person plays, every listener fills the same table.
-> <out>/play.html (numbered clips in one fixed random order, no answers) + <out>/session_order.csv (no -> clip id; keep private)
  python make_session.py --out out --fixed 0,4,7 --seed 12
"""
import argparse, csv, json, random
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--fixed", default="0,4,7"); ap.add_argument("--seed", type=int, default=12)
a = ap.parse_args(); out = Path(a.out); keep = {int(x) for x in a.fixed.split(",")}
ids = [r["id"] for r in csv.DictReader(open(out / "key.csv", encoding="utf-8")) if int(r["sent"]) in keep]
random.Random(a.seed).shuffle(ids)
with open(out / "session_order.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["no", "id"]); [w.writerow([i + 1, c]) for i, c in enumerate(ids)]
html = """<!doctype html><meta charset=utf-8><title>ฟังเสียง</title><style>body{font-family:sans-serif;text-align:center;margin-top:8vh}
#n{font-size:120px;font-weight:bold}button{font-size:28px;padding:.4em 1.2em;margin:.3em}</style>
<div>ข้อ</div><div id=n>1</div><div id=t></div><audio id=au></audio><br>
<button onclick=prev()>◀ ก่อนหน้า</button><button onclick=play()>▶ เล่น</button><button onclick=next()>ถัดไป ▶</button>
<p style=color:#888>คีย์ลัด: Space = เล่นซ้ำ, → = ถัดไป, ← = ก่อนหน้า</p>
<script>const C=__IDS__;let i=0;const au=document.getElementById('au');
function show(){n.textContent=i+1;t.textContent=`(${i+1}/${C.length})`;au.src=C[i]+'.wav'}
function play(){au.currentTime=0;au.play()}function next(){if(i<C.length-1){i++;show();play()}}function prev(){if(i>0){i--;show();play()}}
document.onkeydown=e=>{if(e.key===' '){e.preventDefault();play()}if(e.key==='ArrowRight')next();if(e.key==='ArrowLeft')prev()};show()</script>"""
(out / "play.html").write_text(html.replace("__IDS__", json.dumps(ids)), encoding="utf-8")
print(f"{len(ids)} clips -> {out}/play.html, {out}/session_order.csv")
