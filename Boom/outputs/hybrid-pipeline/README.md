# เพิ่มการตรวจลักษณะมือรายเฟรม

`gesture_detector.py` ตรวจนิ้วเหยียด/งอ ชี้นิ้ว และลักษณะคล้ายประนมมือใน 2D พร้อมหลักฐานและสถานะ insufficient_data ผลล่าสุดจาก `evaluate_dev.py` อยู่ใน `dev_results_gestures.json`, `verification_gestures.json` และ `REPORT_GESTURES.md` ไม่ merge และยังไม่ใช้ผล gesture เปลี่ยนคะแนนคำ; ยังไม่เชื่อมหน้าเว็บ

# สถานะล่าสุด: ปิด experimental merge

`evaluate_dev.py` ประเมินเฉพาะช่วงจาก pipeline เดิม แล้ว rerank ภายในแต่ละช่วง ผลล่าสุดอยู่ใน `dev_results_no_merge.json`, `verification_no_merge.json` และ `REPORT_NO_MERGE.md` โค้ด segment_merger และรายงานที่มี combined/merge ด้านล่างเป็นประวัติการทดลอง ไม่ถูกเรียกโดยตัวประเมินปัจจุบัน

# Experimental two-stage retrieval pipeline

สถานะ: สร้างและรัน Dev แล้ว ยังไม่เชื่อมเข้าหน้าเว็บ production

## ผลที่ตรวจได้

| คลิป / ช่วงหลัง quality gate | Temporal top-1 ก่อนรวม | หลังรวม: support | หลัง rerank: final score |
|---|---|---|---|
| clip1 0.80–1.60 s | น้อง 0.4836 | ไม่รวม | น้อง 0.4080 |
| clip1 1.60–3.68 s | เที่ยว / ภาษามือ (สองช่วง) | ภาษามือ 0.4000 | ภาษามือ 0.4481 |
| clip2 0.16–1.28 s | สวัสดี 0.4907 | ไม่รวม | สวัสดี 0.4898 |
| clip2 1.44–4.48 s | เพื่อน / กรุงเทพ (สองช่วง) | ภาษามือ 0.3383 | ภาษามือ 0.3758 |

- quality gate ปิด auto-flip ของ clip1 เพราะมือขวาปรากฏ 48.94% ของทั้งคลิป; ทำให้ segmentation เปลี่ยนจาก 2 เป็น 3 ช่วงก่อน merge จึงห้ามเทียบคะแนนช่วงเดิมกับใหม่เสมือนเป็นช่วงเดียวกัน
- ช่วงแรก clip1 ยังผิด: baseline เดิมเป็น ข้าว ปัจจุบันเป็น น้อง ไม่ใช่หลักฐานว่าแก้ recognition สำเร็จ
- clip2 merge ช่วยให้ ภาษามือ ขึ้นอันดับ 1; reranker เพียงอย่างเดียวยังไม่เปลี่ยน top-1 ของสองช่วงเดิม
- ยังไม่มี annotation ขอบเขต gloss ที่ยืนยันโดยคน จึงยังสรุป accuracy, false-merge rate หรือการไม่กลืนคำไม่ได้ โดยเฉพาะช่วง คุยกัน/ภาษามือ
- synthetic negative tests ผ่าน 9 ข้อ รวม gap, duration, Top-3 overlap, rest, coordinate jump, hand disappearance และ missing-mask rules แต่ไม่ทดแทน real negative clips
- ไม่เปิดหรือรัน clip3; SHA256 ของไฟล์ models .onnx/.npz/.json ก่อนและหลังตรงกัน

## โค้ด

- `features.py`: พิกัดข้อมือเทียบไหล่ในหน่วย shoulder width, raw visibility mask และสถิติเชิงกายภาพ ไม่ interpolate mask
- `build_bank_heuristics.py`: join bank clip IDs → pose store → เฉพาะ role=train → mean/median/std/count ต่อ concept
- `segment_merger.py`: gap < .20, duration ≤3.20, Top-3 มีคำร่วม, wrist velocity ต่อเนื่อง, jump ≤.40; abstain เมื่อมือหลุดรอบรอยต่อ
- `reranker.py`: support + clipped alpha×rule_score, alpha=.10; ไม่มีกฎชื่อคำ; ไม่ใช้สองมือ/symmetry เมื่อ observed rate ข้างใดต่ำกว่า .50; reference แต่ละ feature ต้องมี ≥3 ตัวอย่าง
- `evaluate_dev.py`: subclass quality gate ไม่ monkeypatch production; rerun tagger, propose merge, recompute temporal support จากหน้าต่างภายในช่วงรวม, rerank; allowlist เฉพาะ clip1/clip2
- `test_hybrid.py`: unittest

เกณฑ์เพิ่มเติมที่เลือกสำหรับการทดลอง: รอยต่อ ±.16 s, observed อย่างน้อย .75 ต่อข้าง, ความเร็ว ≥.15 shoulder-width/s, อัตราส่วนความเร็วก่อน/หลัง .25–4.0 ข้างที่ขยับอย่างต่อเนื่องอย่างน้อยหนึ่งข้าง แต่ทั้งสองข้างต้องตรวจพบ กฎนี้อาจ abstain มากกับท่ามือเดียวที่อีกมืออยู่นอกเฟรม ต้องวัดบน Dev เพิ่มก่อนปรับ

ตาราง 326 คำมี reference ใช้ได้ 310 คำ / 2,332 คลิป ตัด val 178, test 762 และรายการไม่มี pose/metadata 638 ออก ไม่เดาค่าให้คำที่ข้อมูลไม่พอ Prepared pose scores อาจผ่าน interpolation มาแล้ว จึงไม่ใช่ raw observed rates; เป็นข้อจำกัดของ reference heuristics ความเร็ว pose store ใช้ฐานเวลา 25fps ของระบบเดิม

คะแนน `cosine` ในรายงานเป็นค่าสูงสุดต่อ concept จากหน้าต่างที่เกี่ยวข้อง; `support` คือผลรวมคะแนน concept ที่อยู่ Top-5 ของแต่ละหน้าต่าง หารจำนวนหน้าต่างทั้งหมด `final_score` ไม่ใช่ probability จึงให้ `p_correct=null` หลัง rerank/merge และยังไม่ใช้ acceptance threshold .8 กับคะแนนใหม่ คะแนน baseline p เก็บแยกไว้เพื่อ audit เท่านั้น

## รันซ้ำ

ใช้ Python env ที่มี numpy, pandas, pyarrow, torch และ dependencies ของ Sign2Say:

```sh
PY=/Users/thawaphorn/Documents/Codex/2026-10-07/new-chat/work/training-venv/bin/python
REPO=/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say
DATA=/Users/thawaphorn/Documents/Codex/2026-10-07/new-chat/work/training-data
CACHE=/Users/thawaphorn/Documents/Codex/2026-09-17/good-id-word-category-video-url/outputs/three-clip-baseline
cd "$REPO/experiments/hybrid_pipeline"
"$PY" build_bank_heuristics.py --repo "$REPO" --data "$DATA" --out .
"$PY" -m unittest discover -s . -p 'test_*.py'
"$PY" evaluate_dev.py --repo "$REPO" --cache "$CACHE" --out .
```

ขั้นต่อไป: ยืนยันขอบเขตคำใน Dev โดยคน, เพิ่มคู่คำต่างกันที่อยู่ติดกันเป็น negative cases, ประเมิน false merge และคำที่ reranker ทำให้ถูก→ผิด จากนั้น freeze config และสอบเทียบ confidence ใหม่ ก่อนพิจารณาเปิด held-out ตามเงื่อนไขผู้ใช้

## Update: manual rules from book illustrations

เพิ่ม `manual_heuristics_tsl_detailed.json` (6 concepts / 9 variants), `manual_features.py` และ `rerank_detailed` ใน `reranker.py` รายละเอียดแหล่งอ้างอิงอยู่ใน `BOOK_RULES.md`; ผลทุกอันดับอยู่ใน `REPORT_MANUAL.md` และ `dev_results_manual.json`

คำสั่ง `evaluate_dev.py` เดิมจะประเมินทั้ง bank-only และ bank+manual ในการรันเดียว โดยคง segment/candidates เดียวกัน ผลใหม่เขียน `dev_results_manual.json` และ `verification_manual.json` เพื่อเก็บ `dev_results.json` รุ่นก่อนหน้าไว้ ทุก manual variant อ้างเลขหน้า PDF และเลขหน้าพิมพ์แยกกัน ไม่มีชื่อคำในเงื่อนไข Python

Run tests ใช้คำสั่งเดิม ปัจจุบัน 17 tests รวม missing-hand abstention, symmetric ไม่บังคับมือชิด, alternative variants, bounded penalties และ circle/line/gap trajectories

Book classifications are interpretations of the supplied illustrations, not expert-validated annotations. Numerical bounds are engineering hypotheses, not standards specified by the books. One-handed signs are mirror invariant. The number of visible hands is never treated as proof of the number of hands used. Repetition and finger articulation are recorded as metadata but not scored by wrist-only rules.
