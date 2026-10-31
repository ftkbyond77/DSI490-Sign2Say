# Demo ภาษามือไทย — SignBERT-inspired แบบไม่ใช้ MANO

ใช้ hand graph encoder + temporal/chirality embeddings + Transformer แล้วฝึก 2 ขั้น:

1. **Pretraining**: ปิดบังจุดมือแล้วสร้างพิกัด x/y กลับด้วย MLP decoder
2. **Fine-tuning**: attention pooling + classifier เพื่อรู้จำคำเดี่ยว

ไม่ต้องใช้ไฟล์ MANO, smplx หรือบัญชี MANO รุ่นนี้เป็นการดัดแปลงจาก SignBERT ไม่ใช่การทำซ้ำ paper ต้นฉบับครบทุกส่วน รายละเอียดอยู่ใน METHOD.md

## สถานะจริง

- เตรียมคลิปต้นทาง 20 คำ พร้อม hash: `data/sources.json`
- ทดสอบ encoder / reconstruction / gradients แล้ว และทดสอบ pretraining → checkpoint → fine-tuning ด้วยข้อมูลสังเคราะห์ในพื้นที่ scratch เท่านั้น
- **ยังไม่มี checkpoint จากคลิปจริง**: MediaPipe ในสภาพแวดล้อมผู้ช่วยสร้าง OpenGL context ไม่ได้ จึงยังสกัด keypoints ไม่สำเร็จ
- ไม่ใช้ข้อมูลสังเคราะห์แทนผลฝึกจริง และไม่มีการนำ synthetic checkpoint มาแสดงใน demo
- ทางเลือก CPU ของ Apple Vision ก็ไม่สามารถสร้าง pixel buffer ในสภาพแวดล้อมนี้ได้ จึงไม่ได้เปลี่ยน backend ที่ส่งมอบ

## รันบนเครื่องนี้

ดับเบิลคลิก `train_demo.command` เพื่อสกัด → pretrain → fine-tune → เปิดเว็บตามลำดับ หากขั้นใดล้มเหลว สคริปต์จะหยุดทันที

ไฟล์ชุดนี้อยู่ที่ `/Users/thawaphorn/Documents/Codex/2026-09-17/good-id-word-category-video-url/outputs/signbert_demo` ไม่ใช่ `DSI490/exp1`

ตัวเรียกใช้พาธเต็มของ Python แล้ว จึงเรียกจากโฟลเดอร์ใดก็ได้:

```bash
bash "/Users/thawaphorn/Documents/Codex/2026-09-17/good-id-word-category-video-url/outputs/signbert_demo/train_demo.command"
```

เปิดเฉพาะหน้าเว็บหลังมี checkpoint:

```bash
bash "/Users/thawaphorn/Documents/Codex/2026-09-17/good-id-word-category-video-url/outputs/signbert_demo/run.command"
```

## ติดตั้งใหม่ / ย้ายเครื่อง

ใช้ Python 3.11:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare.py --videos /absolute/path/to/videos
python train.py pretrain
python train.py finetune
streamlit run app.py --server.address 127.0.0.1
```

`prepare.py` ใช้ 20 IDs ที่กำหนดใน DEFAULT_IDS หรือเลือกเองด้วย `--ids 919,1577,...` และอ่านจาก videos/ ไม่รวม videos2/ ที่ซ้ำกัน การย้ายเครื่องให้รัน prepare ใหม่เพื่อสร้างพาธ features ให้ตรงเครื่องนั้น

## ทดสอบและทำนายจาก command line

```bash
python test_pipeline.py
python predict.py /absolute/path/query.mp4 --checkpoint runs/finetune.pt
```

checkpoint จะระบุ `method=SignBERT-inspired without MANO` และ `decoder=coordinates` อย่างชัดเจน ไม่โหลด checkpoint MANO แทนโดยอัตโนมัติ

## ขอบเขตผลทดลอง

ข้อมูลที่เลือกมี 1 คลิปต้นฉบับต่อคำ ทั้งหมดจึงเป็น train และยังไม่มี validation/test คะแนนบน training clips ใช้ตรวจการจดจำตัวอย่าง ไม่ใช่ผลกับคนใหม่ การเพิ่มคลิปดัดแปลงไม่สร้างชุดทดสอบอิสระ ต้องเก็บผู้ทำท่าหรือการบันทึกใหม่เพื่อประเมิน และแบ่งก่อน augmentation

โมเดลเลือกคำจากชุดฝึกเสมอ แม้ท่าที่ทำจะไม่อยู่ในชุด และ softmax ไม่ใช่ความมั่นใจที่สอบเทียบแล้ว การตรวจจุดมือหายอาจทำให้ผลผิดได้

## โครงสร้าง

- prepare.py / extract_keypoints.py — เตรียมข้อมูลจริง พร้อม missing masks
- model.py — GCN, Transformer, attention classifier, coordinate decoder, masking
- train.py — ฝึกและบันทึก checkpoint ทั้งสองขั้น
- predict.py / app.py — inference ผ่าน command line และหน้าเว็บ
- test_pipeline.py — การทดสอบส่วนโมเดล
- STATUS.json — สถานะที่ตรวจยืนยัน

โฟลเดอร์ sign_demo ที่อยู่ข้างเคียงเป็นร่าง DTW เดิม ไม่ใช่ demo รุ่นนี้
