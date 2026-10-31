import json
from pathlib import Path
O=Path('outputs/hybrid-pipeline');old=json.loads((O/'dev_results.json').read_text());new=json.loads((O/'dev_results_manual.json').read_text())
lines=['# ผลทดลองกฎจากหนังสือบน Dev', '', 'เปรียบเทียบบน segment และ candidate ชุดเดียวกัน: ก่อน = bank heuristic เดิม; หลัง = bank + manual rule จากหนังสือ ไม่ใช้คำอ้างอิงมาแก้อันดับ', '', '## สูตรและข้อจำกัด', '', '`final_score = support + clip(bank_bonus + 0.5 × manual_rule_bonus, -0.20, +0.15)` โดย manual_rule_bonus ใช้ขอบเขต +0.15/-0.20 และเลือก variant ที่สอดคล้องกับหลักฐานมากที่สุด ส่วนที่ไม่ทราบได้ 0; หารด้วยจำนวนกฎทั้งหมดเพื่อไม่ให้หลักฐานข้อเดียวได้โบนัสเต็ม', '', 'ตัวเลขเป็น heuristic ที่ตั้งก่อนรันครั้งนี้ ไม่ใช่ค่าจากหนังสือหรือค่าที่ calibrate แล้ว กฎมีเพียง 6 คำ ทำให้ coverage ไม่เท่ากันทั้ง vocabulary ต้องประเมินผลเสียต่อคำอื่นก่อนนำใช้จริง ไม่ให้ p ของอันดับหลังปรับ', '', 'การทดสอบ 17 ข้อผ่าน; fingerprint models ตรงกันก่อน/หลัง; เปิด cache เฉพาะ clip1/clip2 ไม่เปิดหรือรัน clip3', '', '## ผลรวมหลัง merge', '', '| Clip | เวลา (s) | ก่อน: top-1 / final_score | หลัง: top-1 / final_score |','|---|---|---|---|']
changes=[]
for prev,r in zip(old,new):
 assert prev['clip']==r['clip']
 for mode in ['without_merge','combined']:
  assert len(prev[mode])==len(r[mode])
  for a,b in zip(prev[mode],r[mode]):
   assert a['segment']==b['segment'] and a['temporal_top5']==b['temporal_top5'] and a['reranked_top5']==b['reranked_top5'],'Baseline drift'
 for x in r['combined']:
  a=x['reranked_top5'][0];b=x['detailed_top5'][0];seg=x['segment']
  lines.append(f"| {r['clip']} | {seg['t0']:.2f}–{seg['t1']:.2f} | {a['word']} / {a['final_score']:.4f} | {b['word']} / {b['final_score']:.4f} |")
for r in new:
 for mode in ['without_merge','combined']:
  for x in r[mode]:
   seg=x['segment'];before=x['reranked_top5'];after=x['detailed_top5'];bm={c['word']:(i+1,c) for i,c in enumerate(before)}
   lines+=['',f"## {r['clip']} / {mode} / {seg['t0']:.2f}–{seg['t1']:.2f} s",'',f"Raw observed L/R: {x['detailed_profile']['observed_rate'][0]:.1%} / {x['detailed_profile']['observed_rate'][1]:.1%}",'','| คำ | อันดับก่อน→หลัง | support | final ก่อน | manual contribution | final หลัง | variant |','|---|---|---|---|---|---|---|']
   for j,c in enumerate(after):
    rank,p=bm[c['word']];e=c['manual_evidence'];v=e['best']['variant'] if e else 'no rule'
    lines.append(f"| {c['word']} | {rank}→{j+1} | {c['support']:.4f} | {p['final_score']:.4f} | {c['manual_bonus']:+.4f} | {c['final_score']:.4f} | {v} |")
   if before[0]['word']!=after[0]['word']:changes.append((r['clip'],mode,seg['t0'],before[0]['word'],after[0]['word']))
lines+=['','## การตีความ','','- หลัง merge top-1 ทั้ง 4 ช่วงไม่เปลี่ยน กฎละเอียดจึงยังไม่มีหลักฐานว่าช่วยแก้คำอันดับ 1 เพิ่มขึ้น','- คลิป 1 ช่วงแรกยังเป็น น้อง ไม่ใช่ สวัสดี; กฎไม่สามารถกู้ landmarks มือที่หายได้','- ภาษามือ ในคลิป 2 ช่วงรวมขึ้นอันดับ 1 ตั้งแต่ temporal merging แล้ว ไม่ควรนับเป็นความสำเร็จของ manual reranker','- ก่อน merge ดูการเปลี่ยนอันดับในตารางเป็น diagnostic เท่านั้น ยังไม่มี verified time-aligned gloss จึงไม่คำนวณ accuracy/false merge','- กฎ 2 มือถูกปิดเมื่อ observed <50%; num_hands เป็นคำอธิบายท่า ไม่ใช้จำนวนมือที่ detector เห็นแทนจำนวนมือที่ใช้จริง','- wrist trajectory เป็นภาพฉาย 2D; รูปมือ การสัมผัส ทิศฝ่ามือ การเคลื่อนลึก และการทำซ้ำนิ้ว ยังไม่ได้ตรวจ จึงไม่ใช่ symbolic verification ครบทุกองค์ประกอบ','',f'Top-1 changes ก่อน merge: {changes}', '']
(O/'REPORT_MANUAL.md').write_text('\n'.join(lines));print('\n'.join(lines[:22]));print('changes',changes)
