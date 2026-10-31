# Boom Phase 2.1 Combined — Current baseline → Boom integration

## เป้าหมาย
นำแนวคิดจากสามสายมาทดลองบนโค้ดที่มีจริง รักษาคำที่ทำเร็ว ลดคำแทรก แยกท่าคล้าย และยืนยันคำได้เหมาะสม ไม่อ้างว่าเป็น v8.1 และไม่คัดลอกคะแนนจากรายงานมาเป็นผลทดลอง

## สถาปัตยกรรม
วิดีโอ → MediaPipe / Standard Schema → Tagger → window encoder + bank scoring → Decoder → post filters / grammar → calibrated status → ข้อความ

ส่วนทดลองเพิ่มเติมคือ hand descriptor จากรูปนิ้วและการเคลื่อนไหวของ landmark เทียบ descriptor ของคลิปคำจริงใน bank เสริม ก่อนรวมกับคะแนน encoder ต้องตรวจคุณภาพมือและปรับสเกลคะแนนบน tune ช่องที่ตรวจไม่ได้เป็น unknown ไม่ตีความว่าเป็นมือกำ และ cosine ไม่ใช่ probability

## ลำดับทดลอง
1. ล็อก revision, config, encoder, tagger, bank และ vocabulary เป็น baseline เดียวกัน บันทึก hash ก่อนรัน
2. รัน baseline บน tune ที่มีจริงที่ 1×/1.5×/2× แยกประโยคจริงกับประโยคสังเคราะห์
3. ทดลอง A/B/B2 เกณฑ์ p_sign_min, D/E/E2 เกณฑ์ตัดช่วง, F exact_rescore ทีละตัวแปร
4. C ตรวจ effective tau_accept ก่อน การเปลี่ยน accept_min อย่างเดียวอาจไม่มีผลเพราะ constructor ใช้ max กับ calibration; เก็บ threshold ที่ใช้จริงทุกครั้ง
5. G พัฒนาตัวกรองมือหายแยก flag ไม่ใช้ confidence ที่ interpolate แล้วแทน raw observed rate โดยไม่ระบุข้อจำกัด
6. Hand-shape และ grammar เป็นรอบแยกหลัง baseline พร้อม ไม่ hardcode คำ และไม่บังคับน้ำหนัก 70:30; ตรวจ grammar leak ก่อนเลือกค่า
7. H รวมเฉพาะค่าที่ผ่านแล้วทดสอบ tune ซ้ำ การผ่านแยกตัวไม่รับประกันว่ารวมแล้วผ่าน
8. ล็อก config ก่อนเปิด test รอบสุดท้าย และประเมิน Eval-Test เพื่อรายงานเท่านั้น

## เกณฑ์ผ่าน
ไม่มี tune set ใด WER แย่ลงเกิน 0.003; WER เฉลี่ยดีขึ้นมากกว่า 0.002; accepted precision ≥0.90 ทุกชุด หากไม่มีคำ accepted ให้เป็น N/A และไม่ถือว่าผ่าน

## รายงานที่ต้องได้เมื่อรันจริง
- WER และ S/D/I ทั้ง macro ต่อคลิปและ corpus WER แยกนิยาม
- Ordered Recall@1/5 จาก alignment เดียวกัน; ระบุ denominator และไม่ใช่นับคำที่โผล่ตรงไหนก็ได้
- Segment F1 แบบ 1:1 ที่ IoU≥0.3 เฉพาะชุดที่มีขอบเขตอ้างอิง ระบุถ้าเป็น automatic annotation; ชื่อวิดีโออย่างเดียววัด F1 ไม่ได้
- Count error = abs(predicted count-reference count), accepted correct/count/precision และ acceptance coverage
- ECE ของ p_correct เทียบ correctness พร้อม bin definition; ห้ามใช้ cosine เป็น probability
- Latency แยก cold load, extraction และ warm end-to-end บนเครื่องเดียวกัน ไม่ใช้เวลา replay cached แทน
- Paired bootstrap 10,000 ครั้งของ delta WER; cluster ตาม signer หากมี ID และจำนวน cluster เพียงพอ; CI คร่อมศูนย์รายงานว่า ไม่มีนัยสำคัญ
- ภาพแต่ละ segment, เวลาเริ่ม–จบ, GT alignment, Top-5, cosine/support, p_correct, hand score (ถ้ามี), final score และ Accept/Uncertain/Reject
- ตารางช่วยแก้/ทำแย่ลง, confusion pairs และตัวอย่างคำสั้นหรือคำใกล้ปากที่หาย โดยแยกหลักฐานจากข้อสันนิษฐาน

## วินัยข้อมูล
ไม่ retrain ไม่แก้ bank/models ต้นฉบับ ทุกฟีเจอร์ใหม่ default ปิด ใช้ config/descriptor bank แยก ผล Eval-Test ห้ามใช้เลือกค่า Boom 6 คลิปเคยใช้พัฒนาแล้วจึงไม่ใช่ unseen test; signer overlap ยังต้องตรวจด้วย metadata และ IDs ที่ยืนยันได้

## ไฟล์และสถานะ
เปิด boom_phase2_2.ipynb เพื่ออ่านแผนและตรวจ config ปัจจุบันผ่าน phase22.py ไฟล์ experiment_plan.json เป็นแผนที่ยังไม่เปิดใช้งาน ไม่ใช่ config ผู้ชนะ
**มี tune.py รัน inference จริงสำหรับ baseline/A/B/B2/C/D/E/E2/F ที่สามความเร็ว และ report.py สรุปผล; ผลยังเป็น partial จนมี DONE.json ส่วน G/handshape/grammar/H ยังไม่รัน** ไม่เปลี่ยนเว็บหรือ production จึงไม่ต้อง rollback


## ชื่อโฟลเดอร์ใหม่
งานรวมอยู่ที่ Boom/phase2.1combined เปิด boom_phase2_1combined.ipynb พาธ phase2.2 เดิมเป็น symlink เพื่อรักษางาน test ที่กำลังรันและ provenance เดิม
