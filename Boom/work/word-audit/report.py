from pathlib import Path
import json,collections,hashlib
O=Path('outputs/word-audit');groups=json.loads((O/'groups.json').read_text());lex=json.loads((O/'lexicon_19_review.json').read_text());wrong=json.loads((O/'top1_errors.json').read_text())
lines=['# ตรวจคำและ lexicon — 11 ต.ค. 2026','','## ขอบเขต','ผลด้านล่างเป็นการคำนวณ retrieval ใหม่จาก production query embeddings ที่ cache ไว้ กับ artifacts/eval/bank_eval.npz; ไม่ใช่ผล webcam ใหม่ ไม่ใช่ end-to-end translation และไม่ได้รัน visual encoder ใหม่ ใช้เฉพาะ role=test, คลังบทสนทนา, ตัด signer เดียวกับ query และ clip ซ้ำออก และตัด signer ไม่ทราบค่าออก ข้อมูล query เดิมอาจเคยใช้รายงานผลแล้ว ไม่ใช่ชุด blind test ใหม่','','## 10 คำ','|คำ|ตัวอย่างใน production bank|signer IDs|Top-1 บน test|','|---|---:|---:|---|']
for r in groups:lines.append(f"|{r['word']}|{r['bank_examples']}|{r['signers']}|"+(f"{r['correct']}/{r['test_queries']}" if r['test_queries'] else 'ไม่มี test query')+'|')
lines+=['','5 คำกลุ่มแรก: 64/66 (96.97%); กลุ่มพจนานุกรมไม่มี test query จึงไม่มีคะแนนเปรียบเทียบ ห้ามแทนข้อมูลที่ขาดด้วย 0% หรือสรุปยืนยันเหตุจากจำนวนข้อมูล จำนวน signer เป็น IDs ตาม metadata ไม่ใช่จำนวนคนที่ตรวจยืนยันด้วยตาใหม่','','## Lexicon 19 คำ','19 คำนี้มาจาก daily_missing.csv ของ audit เดิม ทุกคำไม่มี exact concept และไม่อยู่ในคลังบทสนทนาปัจจุบัน การไม่มีชื่อ exact ไม่เท่ากับไม่มีความหมายใกล้เคียง คำเสนอด้านล่างต้องให้ผู้รู้ภาษาและภาษามือตรวจ ไม่แก้ mapping อัตโนมัติ','','|คำ|ผลตรวจ / สิ่งที่ควรตรวจต่อ|','|---|---|']
for r in lex:
 note='ไม่พบ exact concept; ต้องตรวจแหล่งคำ/ความหมายเพิ่มเติม'
 if r['candidate_concepts_review_only']:note='พบ concept ที่ควรตรวจเทียบ: '+', '.join(r['candidate_concepts_review_only'])+' (ยังไม่ยืนยันว่าใช้แทนกันได้)'
 if r['word']=='ทำ':note='มี 1 รายการใน manifest แต่ถูกตัดด้วย pose_missing; ต้องแก้ข้อมูลสกัด ไม่ใช่เพิ่มชื่ออย่างเดียว'
 if r['metadata_annotated']:note+='; พบ label พร้อมวงเล็บใน metadata ต้นทาง ต้องตรวจการ ingest/cleaning'
 lines.append(f"|{r['word']}|{note}|")
lines+=['','## รายการ Top-1 ผิด','พบ 211/762 test queries ผิด จาก '+str(len({r['expected'] for r in wrong}))+' คำจริง; นี่เป็น retrieval ก่อน threshold/reject และก่อนสร้างประโยค จึงไม่เท่ากับคำที่เว็บยอมแสดงหรือพูดทั้งหมด ดู top1-errors.md สำหรับคำจริง → คำที่ตอบแทนทุกคลิป','','## ทดสอบเว็บ','เว็บที่ตรวจ: http://localhost:3010 (model v7.2, conversation 326 concepts). แยก origin จากหน้าผู้ใช้เดิมเพื่อไม่ใช้ personal memory 2 คำ; ไม่ลบ memory ของผู้ใช้ ตั้ง LLM off, speech off, spelling off; light correction on. ยังขาดชุดคลิปทดสอบใหม่ครบ 10 คำ คลิปพจนานุกรมเดิมไม่ใช่ held-out evidence จึงไม่ใช้ยืนยันความแม่นกับคนใหม่']
(O/'REPORT-th.md').write_text('\n'.join(lines))
e=['# คำที่ Top-1 ผิด','','Offline cross-signer retrieval, conversation bank. ไม่ใช่ผล webcam ใหม่','','|คลิป|คำจริง|คำอันดับ 1|อันดับคำจริง|','|---|---|---|---:|']
for r in wrong:e.append(f"|{r['clip_id']}|{r['expected']}|{r['predicted']}|{r['rank'] if r['rank'] is not None else 'ไม่มีคำจริงใน gallery'}|")
(O/'top1-errors.md').write_text('\n'.join(e))
R=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say');W=Path('/Users/thawaphorn/Documents/Codex/2026-10-07/new-chat');files=[R/'models/bank.npz',R/'artifacts/eval/bank_eval.npz',R/'models/concepts.json',R/'models/vocab_conversation.json',W/'outputs/full-audit/production_query_embeddings.npy',W/'outputs/full-audit/evaluation_queries.csv']
(O/'provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('reports saved; wrong concepts',len({r['expected'] for r in wrong}))
