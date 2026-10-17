# pitty — สรุปงาน improve ThaiSLM (branch `pitty`)

> อัปเดตล่าสุด: 2026-10-09 (รอบ 2: แยก error, oracle, decoder grid, rerank, ล็อก config แล้วเปิด test ครั้งเดียว) · ทุกอย่างในโฟลเดอร์นี้**ยังไม่ commit** · **ไม่แตะ `main`**

---

## 1. ความต้องการ (โจทย์)

**งานหลัก:** Model Tuning (Forward Part) → ทำให้ Recognition ได้มากที่สุด โดยเน้น **Basic Daily Conversation**
- Prep · Recognition, Translation · Voice
- เช็คเว็บแอป: **มืออยู่ในเฟรมแต่ระบบจับไม่เจอ**
- เอา `data_prep/` ไปไว้ใน `cloud_s3/data_prep/` (cloud_s3 = data ดิบที่ดึงจาก S3)

**เงื่อนไขที่ตกลงกันไว้**
| เงื่อนไข | ทำตามยังไง |
|---|---|
| ทำใน branch ของตัวเองเท่านั้น | ทุกอย่างอยู่บน branch `pitty` |
| **ห้ามยุ่งกับ `main`** | ไม่ได้ switch / merge / push ไป `main` |
| **อย่าเพิ่ง commit** | ไฟล์ใหม่ทั้งหมดยังเป็น untracked |
| **ไม่เก็บข้อมูลเพิ่ม** (ไม่ถ่ายใหม่, ไม่ดึง YouTube เพิ่ม — กลัว noise) | ใช้แค่ข้อมูลที่มี |
| **ลองแล้วไม่ดีขึ้น → คืนค่า** | โค้ดของโปรเจกต์ตอนนี้ = เหมือนตอน clone (`git diff` ว่าง) |
| อยากเห็นผลใน notebook | ทุกการทดลองมี notebook ที่รันแล้ว (มีตาราง/กราฟ) |

---

## 2. เตรียมเครื่อง (Phase 0)

| อะไร | ที่ไหน |
|---|---|
| Clone + branch `pitty` + push branch ขึ้น GitHub | `ftkbyond77/DSI490-Sign2Say` |
| conda env **`thaislm`** (Python 3.10, numpy 1.26, onnxruntime 1.23.2, mediapipe 1.0.1, torch 2.7 CPU) + Jupyter kernel "Python (thaislm)" | `C:\Users\ppich\anaconda3\envs\thaislm` |
| ข้อมูลจาก S3 (เฉพาะที่ใช้): `manifest`, `params`, `pose_mp` (287 MB), `shards` (1.2 GB) | `cloud_s3/data_prep/` (อยู่ใน .gitignore) |
| ไฟล์จาก Drive เพื่อน: `data_test/`, embedding ของ encoder (`artifacts/runs/enc-v7-europe-west4/emb*.npy`, `encoder_best.pt`) | อยู่ใน .gitignore |
| Colab (T4 ฟรี) + Google Drive `MyDrive/thaislm/` | ใช้ train |

⚠️ ไดรฟ์ C เหลือที่ว่างน้อย (~5 GB)

---

## 3. วิธีวัดผล (ใช้กับทุกการทดลอง)

| ชุด | ข้อมูล | ตัววัด |
|---|---|---|
| **คำเดี่ยว** | คลิปของคนที่ไม่อยู่ใน train · **val = เลือกค่า / test = รายงาน** | R@1 |
| **ประโยค TSL51** ⭐ | 252 ประโยค (tune 126 / test 126) · ความเร็ว 1× / 1.5× / 2× · ผ่าน `SignPipeline` ตัวจริง · bank = `bank_eval` | **WER** (ต่ำ = ดี), recall, จำนวนคำที่ตัดได้ |
| **data_test** | วิดีโอจริง 5 คลิป (22 คำ) | ดูอย่างเดียว ห้ามใช้เลือกค่า |
| **จับมือ** | วิดีโอ 64 คลิป | % เฟรมที่ข้อมือยกแต่หามือไม่เจอ |

**กฎตัดสิน:** เปลี่ยนทีละอย่าง · เลือกจาก val/tune → ดู test · **paired bootstrap 95% CI** (ไม่คร่อม 0 = ดีขึ้นจริง) · train ใหม่ต้องมี**ตัวควบคุม**
**เช็คตัววัดก่อนใช้:** ได้ตัวเลขตรงกับรายงานเพื่อน (T51U R@1 0.683 ตรงเป๊ะ · TSL51 WER test 0.424/0.380/0.406 vs เพื่อน 0.446/0.391/0.415)

**Baseline (บนเครื่องนี้):** TSL51 test WER **0.424 / 0.380 / 0.406** (1× / 1.5× / 2×) · data_test recall 0.50

---

## 4. ทำอะไรไปแล้ว + ผล

| # | สิ่งที่ลอง | ผล | ใช้ไหม | notebook / ไฟล์ |
|---|---|---|---|---|
| 1 | **มือหลุดเฟรม:** threshold มือ 0.5→0.1, ปรับแสง, ขยายภาพ, วัดแสงจากตัวคน | **ต้นเหตุ = การปรับแสงอัตโนมัติ** ทำมือหาย 8.9% → 19.8% (2.2 เท่า) · threshold ไม่มีผลเลย | ⏳ รอทดสอบกับกล้องจริง (ปิดช่อง "ปรับภาพอัตโนมัติ") | `notebooks/pitty_phase1_hands.ipynb` |
| 2 | ทำความสะอาด bank (หาคลิป label ผิด) | สะอาดอยู่แล้ว — น่าสงสัยแค่ 5 คลิป (0.2%) | – | `notebooks/pitty_lab.ipynb`, `result_reporting/pitty/suspicious_clips.csv` |
| 3 | CSLS (ลดคำที่ชอบโผล่มั่ว) | คำเดี่ยว: ไม่ชัด · ประโยค: 1× ดีขึ้น แต่ **2× แย่ลงจริง** | ❌ | `pitty_lab`, `pitty_phase3b_sentences` |
| 4 | TTA (ทายหลายเวอร์ชันแล้วเฉลี่ย) | ไม่ต่าง (±0.007) — encoder ทนอยู่แล้ว | ❌ | `notebooks/pitty_phase3_tta.ipynb` |
| 5 | Depth prior (ลดเปรียบคำที่มีคนทำเยอะ) | ไม่ต่าง / แย่ลง | ❌ | `pitty_phase3b_sentences` |
| 6 | Train tagger ใหม่ + augmentation "ทำท่าลื่น" (Colab) | A vs ตัวควบคุม B **ไม่ต่าง** · ที่ดีขึ้นมาจากการ train ใหม่เฉยๆ | ❌ **คืนค่าโค้ดแล้ว** | `notebooks/pitty_phase4_tagger.ipynb` |
| 7 | **แยก error (tune)** Sub / Del / Ins | 1×: **ทายผิดคำ 0.33** (หาย 0.04, เกิน 0.02) · 2×: คำหาย 0.165 | – | `scripts/compare_configs.py` |
| 8 | **Oracle segmentation** (ให้ขอบเขตคำถูก แล้วทาย) | R@1 0.652 vs pipeline 0.639 → **ที่ 1× การตัดคำแทบไม่ใช่ปัญหา** · **คำที่ถูกอยู่ใน top-5 ถึง 94%** | – | `scripts/oracle_segments.py` |
| 9 | Decoder `min_len` / `post.min_frames` / `lam` / tempo (12 config, tune) | ตัวการคำหายตอนเร็ว = **ตัวกรองท้าย `post.min_frames`** · `lam 4` แย่ · tempo ช่วยน้อยมาก | – | `cache/pitty_sent/c_*`, `dec_*` |
| 9b | **ล็อกบน tune: `post.min_frames` 14 → 10** (กฎ: ห่างจากดีสุด ≤ 0.005 → เลือกตัวง่าย/เสี่ยงน้อยสุด) → **เปิด test ครั้งเดียว** | **1× / 1.5× เหมือนเดิมทุกประโยค · 2× WER 0.406 → 0.394 ✅ (CI [−0.028, −0.002])** · data_test เหมือนเดิม | ✅ **ผ่าน** — ยังไม่แก้ `models/spotting.json` (รอคุยกับเพื่อน) | |
| 9c | Rerank top-5 ด้วยบริบทคำจากคลังภาษาไทย (TNC bigram) | β เล็ก = ไม่เปลี่ยน · β ใหญ่ = **แย่ลง** (ลำดับคำภาษาไทย ≠ ภาษามือ) | ❌ | `scripts/rerank_context.py` |
| 10 | **CTC** (อ่านทั้งประโยค) | มี TSL51 ใน train: WER 0.07 ← **ข้อมูลรั่ว** (คนเดียวกัน) · train แบบยุติธรรม: **0.64 / 0.63 / 0.63 แย่กว่ามาก** | ❌ | `ctc/` |
| 11 | ไป ↔ กลับ สลับเพราะกลับภาพซ้าย-ขวา? | ไม่ใช่ — คนทดสอบทำ "ไป" ต่างจากผู้เชี่ยวชาญ | – | |
| 12 | Hard-negative fine-tune encoder | **เพื่อนทำไว้แล้ว** (hard_frac 0.5 + daily ×3) → ไม่คุ้มทำซ้ำ | ไม่ทำ | |

### ข้อค้นพบสำคัญ
1. **ค่าที่เพื่อนตั้งไว้ดีอยู่แล้ว** — การปรับวิธีให้คะแนน (CSLS / TTA / prior) ไม่ช่วย
2. **คอขวดจริง = ความหลากหลายของคนทำท่า** — คำที่มีคนทำน้อย (คุณ 4 คน, ภาษามือ 3, เที่ยว 3) ถูกคำที่มีคนเยอะแย่ง (หนึ่ง 21 คน, วิ่ง 15 คน) · 155/326 คำ daily มีคนทำแค่ 1–2 คน
3. **ทำท่าเร็ว → ท่าถูกรวม/ถูกกรองทิ้ง** (2×: ตัดได้ ~2.9 คำ จากจริง 3.57) — ต้นเหตุส่วนหนึ่งคือตัวกรอง `post.min_frames` ทิ้งคำสั้น → แก้เป็น 10 ช่วยได้
6. **คอขวดใหญ่ที่สุดคือ "ทายผิดคำ" ไม่ใช่ "ตัดคำ"** (oracle ช่วยแค่ +0.013) — แต่คำที่ถูกอยู่ใน top-5 ถึง 94% → ถ้ามีตัวจัดอันดับที่ดี (เช่น LLM ที่ระบบมีอยู่ / ไวยากรณ์ภาษามือที่ไม่ได้เรียนจาก TSL51) ยังมีช่องให้ได้เพิ่ม
4. **ต้องมีตัวควบคุม / ระวังข้อมูลรั่ว** — ถ้าไม่มี tagger B จะสรุปผิดว่า augmentation ช่วย · ถ้าไม่ train CTC ใหม่แบบยุติธรรม จะสรุปผิดว่า WER ดีขึ้น 6 เท่า
5. **ไม่มีประโยคจริงจากหลายคน** → วิธีแบบ end-to-end (CTC) ยังใช้ไม่ได้ · pipeline แบบแยกขั้นของเพื่อนเหมาะกับข้อมูลที่มี

---

## 5. ขั้นต่อไป

| อะไร | ใคร |
|---|---|
| ปิด **"ปรับภาพอัตโนมัติ"** ในเว็บแล้วลองกล้องจริง → ถ้าดีขึ้น แก้ค่าเริ่มต้น / วัดแสงจากตัวคน | เพื่อน / เรา |
| เสนอเพื่อน: `models/spotting.json` → conversation `post.min_frames` 14 → 10 (หลักฐานข้อ 9b) | เรา + เพื่อน |
| ⚠️ TSL51 test ถูกดูไปหลายรอบแล้ว (≥ 8 การทดลอง) → ระบุในรายงาน · จากนี้ล็อก config ด้วย tune ก่อนเปิด test เสมอ | เรา |
| ถามคนรู้ภาษามือ: คู่คำที่ระบบสับสน (ยาย/ย่า, ทำงาน/งาน, วันนี้/ตอนนี้, ดี/ได้) ท่าเหมือนกันไหม | เรา |
| (ถ้าเปลี่ยนใจเรื่องข้อมูล) เพิ่มคนทำท่าสำหรับคำที่มีคนน้อย — ได้ผลมากที่สุดตามรายงาน | ทีม |
| Translation / Voice ยังไม่ได้แตะ | – |
| commit โฟลเดอร์ `pitty/` เมื่อพร้อม | เรา |

---

## 6. ไฟล์ในโฟลเดอร์นี้

```
pitty/
├── README.md                         ← ไฟล์นี้
├── notebooks/                        ผลการทดลอง (รันแล้ว เปิดดูได้เลย; kernel "Python (thaislm)")
│   ├── pitty_lab.ipynb                 Phase 0/2/3: baseline, วิธีวัดคำเดี่ยว val/test, CSLS, คลิปน่าสงสัย
│   ├── pitty_phase1_hands.ipynb        มือหลุดเฟรม (MediaPipe 6 แบบ + ห้องมืดจำลอง)
│   ├── pitty_phase3_tta.ipynb          TTA
│   ├── pitty_phase3b_sentences.ipynb   วัดผลระดับประโยค + CSLS + depth prior
│   └── pitty_phase4_tagger.ipynb       tagger augmentation (A vs B vs เพื่อน)
├── scripts/
│   ├── sentence_eval.py                ⭐ ตัววัดระดับประโยค (TSL51) — env: TAGGER_ONNX, SPOT_OVERRIDE
│   ├── compare_configs.py              เทียบ config บน tune: WER + Sub/Del/Ins
│   ├── oracle_segments.py              ทายคำเมื่อให้ขอบเขตคำที่ถูก (หาว่าผิดตรงขั้นไหน)
│   ├── rerank_context.py               rerank top-K ด้วยบริบทคำจาก TNC
│   ├── tta_embed.py                    สร้าง embedding แบบ TTA
│   ├── hands_extract_r1/r2/r3.py       MediaPipe หลายการตั้งค่า (มือหลุด)
│   └── notebook_builders/              สคริปต์สร้าง notebook ด้านบน + notebook Colab
├── ctc/
│   ├── train_ctc.py                    CTC head บน encoder (GPU)
│   └── eval_ctc.py                     วัด CTC บน CPU (เช็คตรงกับ Colab + data_test)
└── colab/
    ├── train_tagger_colab.ipynb        train tagger (ต้องใช้ zip ที่มีโค้ด augmentation)
    ├── train_ctc_colab.ipynb           train CTC (มี TSL51)
    └── train_ctc_synthonly_colab.ipynb train CTC (ยุติธรรม)
```

**ไม่ได้อยู่ใน git (ใหญ่ / ข้อมูล):** `cache/` (ผลขั้นกลางทั้งหมด), `cloud_s3/`, `data_test/`, `artifacts/runs/*.npy|*.pt`, `DSI490/colab_bundle/thaislm_pitty_colab.zip`

**ตัวอย่างการรัน (จาก `Main_demo/`):**
```bash
conda activate thaislm
python pitty/scripts/sentence_eval.py cache/pitty_sent/my_run.json 0 2.0
SPOT_OVERRIDE='{"min_len":10,"post":{"min_frames":10}}' python pitty/scripts/sentence_eval.py cache/pitty_sent/L10_x2.json 0 2.0
```
