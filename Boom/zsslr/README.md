# ZSSLR / GZSSLR: two-tower research baseline

ระบบ PyTorch ตามสเปก ใช้ **3D-CNN** (ตัวเลือกที่โจทย์อนุญาต) ไม่ใช่ TSM/ResNet-34 และยังไม่ได้เทรนกับข้อมูลจริง ไม่แก้ production model ไม่ใช้ clip3 และไม่มี segment merge

## รัน

```bash
cd /Users/thawaphorn/Pee4/Term1/DSI490/การทดลอง/zsslr
../work/demo-venv/bin/python demo.py
../work/demo-venv/bin/python -m unittest -v
```

ใช้ Python 3.10+ และ PyTorch (ทดสอบ 2.10.0); สร้าง environment ใหม่ได้ด้วย `pip install -r requirements.txt` ไม่มีการดาวน์โหลดโมเดลระหว่าง dummy pipeline

## โมดูลและขนาด

- `zsslr/model.py`: VisualEncoder, SemanticEncoder, ZSSLRCompatibilityModel
- `zsslr/protocol.py`: training_loss, predict, evaluate, calibrated stacking
- `zsslr/attribution.py`: FlipDifferenceOperator
- `demo.py`: forward → CE+regularization → backward → optimizer step → ZSSLR/GZSSLR → flip
- `test_system.py`: ตรวจ label mapping, leakage ใน seen-only loss, gradients, calibration, H-score, flip เทียบ full recomputation

Body B,T,3,H,W และ hands B,T,3,h,w ต้องใช้เวลาเดียวกัน แต่ขนาดภาพต่างกันได้ Conv3d รับ B,3,T,H,W จึง transpose ก่อนเข้า backbone ทั้งสองมี weights อิสระ; concat → projection → normalized v[B,512]

ข้อความ K,768 → Linear 64; concat binary attributes K,53 → normalized c[K,117]. W[512,117]; logits=v@W@c.T. Loss=CE(raw logits,local seen target)+lambda*sum(W²). ไม่ softmax ก่อน CE และไม่ใช้ optimizer weight decay ซ้ำแทน regularizer นี้

## สัญญาข้อมูลจริง

เตรียมตารางคลาสหนึ่งลำดับเดียวกัน: class_id คือ row index 0..K-1, concept, dictionary description, text embedding 768 และ attributes 53. แยก seen_ids/unseen_ids ให้ไม่ซ้ำ; ตัวอย่างวิดีโอฝึกต้องมี label อยู่ใน seen เท่านั้น. `training_loss` เลือก prototype ของ seen ก่อนคำนวณ CE แม้ส่งตารางครบทุกคลาส

BERT embeddings เป็น input ที่เตรียมไว้ล่วงหน้า ไม่ใช่ random embedding จาก demo: เลือก encoder รองรับภาษาไทยที่ output 768, บันทึกชื่อ/revision, tokenizer และ pooling ให้เหมือนกันทุกคลาส แล้ว freeze encoder ใช้ dictionary description ที่ไม่มาจากวิดีโอ test. โค้ดนี้ฝึกเฉพาะ projection ของ embedding ที่รับเข้า

ผู้ใช้ยังไม่ได้ให้ **รายชื่อและลำดับ 53 attributes จากงานวิจัย** จึงไม่ประดิษฐ์ schema ขึ้นอ้างว่าเป็นต้นฉบับ ต้องจัดทำ attribute_names.json จำนวน 53 ชื่อในลำดับคงที่ ครอบคลุมหมวดที่ระบุ และ annotation จริงก่อนเทรน. ข้อมูลที่ไม่ทราบต้องตรวจเพิ่ม ไม่ใส่ 0 แทน unknown; encoder ปฏิเสธ NaN/nonbinary. ตารางหนังสือ 473 รายการยังไม่ใช่ binary annotation ที่ตรวจรับรองแล้ว และไม่ได้ถูกแปลงอัตโนมัติในงานนี้

สุ่ม/เลือก T เฟรมให้ครอบคลุมท่า, crop มือทั้งสองเป็นภาพรวมอย่างสม่ำเสมอ, ใช้ normalization เดียวกัน train/test. ถ้าไม่มีมืออย่าเปลี่ยน label เอง. Backbone นี้ไม่มี padding mask จึงต้อง resample ความยาวคงที่ก่อน batching ไม่เติมศูนย์โดยไม่จัดการผลกระทบ. ทุก tensor และ model ต้องอยู่ device เดียวกัน

## การประเมินที่ไม่รั่วไหล

แยกคลาส train-seen / validation pseudo-unseen / final unseen และแยกผู้ทำท่าหรือคลิปต้นฉบับตาม protocol. ปรับ hyperparameters และ gamma จาก validation เท่านั้น; ห้ามเลือก gamma ด้วย final-test labels. ค่า gamma ใน demo เป็นตัวอย่าง ไม่ใช่ค่าที่ผ่านการจูน

ZSSLR ใช้เฉพาะ unseen candidates และ unseen test videos. GZSSLR ใช้ seen+unseen; หัก gamma เฉพาะ seen logits ก่อนจัดอันดับ/softmax. คืน candidate IDs คู่ logits เสมอ; ใช้ IDs[logits.argmax(1)] แปลงเป็น global prediction

Top-1/5 รายงานทั้ง micro และ macro; Top-5 ใช้ min(5,K) และไม่ใช่ตัวชี้วัดที่มีความหมายมากเมื่อ K≤5. S/U เป็น mean per-class Top-1 ของกลุ่ม seen/unseen บน candidate space ร่วม; H=2SU/(S+U), 0 เมื่อทั้งคู่ 0. ต้องรวบรวม prediction ของชุดประเมินทั้งชุดก่อน evaluate ไม่เฉลี่ย H ราย batch. Macro เฉลี่ยเฉพาะคลาสที่มีตัวอย่าง จึงตรวจ coverage ทุกคลาสใน test split ด้วย

## Flip difference

ใช้ visual features จาก model.eval() ภายใต้ no_grad. Flip ทีละ attribute ของทีละ class; normalize semantic prototype ใหม่และคำนวณ softmax บน candidate space เดิม รวม gamma เดิม

ผล `probability_shift[B,K,53,K]` = P_after − P_before; `self_probability_shift[B,K,53]` คือผลต่อคำที่ถูกเปลี่ยนคุณลักษณะ; `mean_absolute_impact[53]` เป็นสรุปขนาดผล. หากมี ground truth ให้ดู shift ของ true-class column เทียบ confusion columns เพิ่มเติม. เป็น sensitivity ไม่ใช่เหตุเชิงสาเหตุหรือหลักฐานว่า attribute ถูกต้อง; บิตที่ flip อาจเป็นท่าที่เป็นไปไม่ได้. Softmax ยังไม่ใช่ confidence ที่ผ่านการ calibration

ผลทั้งหมดเก็บเต็มจึงใช้ memory O(B*K²*53); สำหรับคลังใหญ่แบ่ง visual batch หรือแก้เป็น streaming aggregate ก่อนใช้จริง

## อ้างอิง API

- https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

ตัวเลขใน smoke_result.json มาจากข้อมูลสุ่ม มีไว้ตรวจโค้ด ไม่ใช่ผลการรู้จำภาษามือ
