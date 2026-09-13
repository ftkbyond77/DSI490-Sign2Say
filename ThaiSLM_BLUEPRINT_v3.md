# ThaiSLM Blueprint (v3)
## Thai Sign Language → Text → Speech — ออกแบบใหม่จากสิ่งที่วัดได้จริงใน result_v1

> **สถานะเอกสาร:** ฉบับต่อจาก v2 เขียนหลังการ build → test → optimize → evaluate ทั้งระบบบนเครื่องนี้ (2026-09-13) รายละเอียดตัวเลขทั้งหมดอยู่ใน `result_reporting/result_v1.md` และ `artifacts/reports/*.json`
> **ป้ายกำกับ:** **[วัดจริง]** = ได้จาก result_v1 · **[สมมติฐาน]** = เหตุผลเชิงออกแบบที่ยังไม่ได้ทดสอบ ต้องผ่าน gate ใน §8 ก่อนเชื่อ · **[ประมาณการ]** = ยังไม่ได้วัด
> **โฟกัสของ v3:** (1) Encoder ที่แยก "ท่า" ออกจาก "ตัวคน" (2) Continuous Sign Language Recognition (CSLR) (3) Inference ที่ generalize ไปยัง signer / กล้อง / ความเร็วใหม่
> **สิ่งที่ v3 ไม่เปลี่ยน:** articulators (valid-rate 1.00 ทุกโดเมน), cache job, TTS, language layer interface, schema — ส่วนเหล่านี้ทำงานได้ตามแบบแล้ว **[วัดจริง]**

---

## 0. Executive Summary

### 0.1 v2 คาดไว้อย่างไร และความจริงคืออะไร

| สมมติฐานใน v2 | ผลจริงใน result_v1 | ข้อสรุปสำหรับ v3 |
|---|---|---|
| Frozen ViT + NAP + temporal SSL + prototype head จะได้ cross-signer R@1 ≥ 0.60 | E1 (signer ใหม่ vs พจนานุกรม 4,706 ท่า) R@1 **0.000**, median rank ท่าที่ถูก **547** (chance ≈ 2,353) **[วัดจริง]** | ระบบเรียน "ท่า" ได้บางส่วน แต่ embedding ยังถูกครอบงำด้วย **ตัวตนของ signer** |
| SSL บน 25 ชม. ทำให้ encoder ทน signer/ฉาก | DINO/iBOT loss นิ่งหลัง ~1k steps · kNN ไม่ดีกว่า frozen (median rank 1,642 → 1,759) **[วัดจริง]** | positive pairs ของ SSL มาจากคลิปเดียวกัน = **คนเดียวกัน** → objective แก้ได้ด้วย identity (shortcut) |
| Continuous v1 = sliding-window spotting บน prototypes | gloss F1 ≈ **0.00–0.01** ทุก decoder · threshold ที่ tune บน studio ไม่ output อะไรเลยบน video จริง **[วัดจริง]** | CSLR ใช้ไม่ได้จนกว่า isolated ranking จะดีพอ และต้องมี model ที่ train บน transition/ความต่อเนื่อง |
| Split: คู่ข้าม signer ทั้งหมดเป็น val/test | train มี cross-signer positive = **0** → ArcFace จำคลิป (loss 0.3) แต่ retrieval นิ่ง **[วัดจริง]** | split ต้องเก็บ cross-signer positives ไว้ train และแยก protocol E1/E2 |
| NAP (fixed) พอสำหรับ normalization | ช่วย frozen (NN-same-signer 0.89 → 0.55) แต่หลัง train ยังเหลือ identity มาก **[วัดจริง]** | ต้องมี **session adaptation ตอน inference** เพิ่ม — session centering คือ knob ที่ดีที่สุด (547 → 491 และดันท่าจริงขึ้นอันดับ 1 ใน video จริง) |
| Signing speed ใกล้เคียง TTRS | คนจริง ≈ 0.6 s/ท่า vs TTRS 1.5–3 s · speed aug 0.8–2.5× ให้ E1 ดีที่สุด **[วัดจริง]** | ความเร็วเป็น domain gap หลัก ต้องเป็นส่วนหนึ่งของ encoder design |
| Face stream สำคัญต่อการรู้จำ | ตัด face ออก ผลเท่าเดิม (E1 787 vs 720, E2 R@5 0.658 เท่ากัน) **[วัดจริง]** | face crop เป็นตัวพา identity ที่แรงที่สุด → ใช้เฉพาะ NMM/mouthing ผ่าน representation ที่ไม่มี identity |

### 0.2 Root cause (เรียงตามน้ำหนักหลักฐาน)

1. **ข้อมูลไม่มีความหลากหลายของ signer ต่อท่า** — TTRS มี signer หลัก 3 คน, ท่าที่มี ≥ 2 signer = **177** จาก 4,742 · 4,201 lexeme มีคลิปเดียว **[วัดจริง]**
2. **Representation พา identity** — DINOv2 CLS/patch-mean ของ crop ที่มีแขนเสื้อ ผิว ผม หน้า และฉาก → nearest neighbour เป็นคนเดียวกัน 55–100 % **[วัดจริง]**
3. **Objective ไม่บังคับ invariance** — SSL views มาจากคลิปเดียวกัน, ArcFace บน singleton = จำ (คน + ท่า) ร่วมกัน **[วัดจริง + วิเคราะห์]**
4. **Temporal domain gap** — ความเร็ว/coarticulation ของประโยคจริงไม่มีใน TTRS **[วัดจริง]**
5. **CSLR ถูกสร้างบน ranking ที่ยังไม่พร้อม** และเลือก decoder ด้วย metric ที่ degenerate (WER เลือก "ไม่พูดอะไร") **[วัดจริง]**

### 0.3 การตัดสินใจหลักของ v3

| # | ตัดสินใจ | เหตุผล |
|---|---|---|
| V1 | **Data first:** ThaiSLM-Bench (≥ 12 signer) + cross-signer mining จาก YouTube lessons เป็นงานเฟสแรก ไม่ใช่ทางเลือก | ไม่มี loss ไหนใน v1 แก้ปัญหา 3 signer ได้; Bench เป็นทั้ง training signal และ test ที่เชื่อถือได้ |
| V2 | **Encoder สอง stream ที่ออกแบบให้ identity-free โดยโครงสร้าง:** (a) Geometry stream = body + **21-kpt hands ×2** + face landmarks, bone-length normalized (b) Hand-appearance stream = tight hand crops + hand mask pooling + motion | pose body-only ขาด handshape (E1 1,232) · RGB crop พา identity · ต้องได้ handshape โดยไม่ได้ผิว/เสื้อ |
| V3 | **SSL ต้องมี identity-breaking positives:** positive = ท่าเดียวกันคนละคน (Bench, mined, TTRS pairs) + objectives ที่แก้ด้วยรูปลักษณ์ไม่ได้ (motion/keypoint prediction) · negative ยาก = คนเดียวกันคนละท่า | SSL within-clip plateau ที่ 1k steps **[วัดจริง]** |
| V4 | **Supervised head:** SupCon ที่ weight cross-signer positives + same-signer hard negatives · ArcFace เฉพาะคลาสที่มี ≥ 2 signer · model selection ด้วย E1 MRR/median rank | ArcFace singleton = memorize **[วัดจริง]** · R@k บน ~56 queries = noise |
| V5 | **CSLR = frame-level encoder + CTC ด้วย prototype logits** (เพิ่มคำใหม่ได้โดยไม่ retrain) + boundary/blank + LM/LLM rescoring · เปิดใช้เมื่อผ่าน Encoder gate เท่านั้น | spotting ด้วย isolated prototypes = chance **[วัดจริง]** |
| V6 | **Inference = Session Adaptation Stack:** session centering → running calibration → user enrollment → vocabulary scoping → relative scoring | centering + vocab scoping ดันท่าจริง ฉัน/กิน ขึ้นอันดับ 1 **[วัดจริง]** · absolute threshold ไม่ transfer **[วัดจริง]** |
| V7 | **Function-sign lexicon** (ไป, ไหม, ไม่, อะไร, ที่ไหน, ใคร, ฉัน/ผม, เธอ, …) ต้องอัดเพิ่ม | ไป, ไหม ไม่มีใน TTRS · 2/9 คำใน data_test เป็น OOV **[วัดจริง]** |
| V8 | **Evaluation:** E1 (dictionary, new signer) เป็นตัวชี้วัดหลัก · E3 signer-disjoint Bench · real-world set ≥ 30 videos พร้อม gloss + timestamp · ≥ 300 queries ต่อ split | CI ±0.15 ใน v1 ทำให้แยก variant ไม่ออก **[วัดจริง]** |

---

## 1. Problem Statement (สิ่งที่ต้องแก้ ในรูปที่วัดได้)

### 1.1 ตัวชี้วัดที่อธิบายปัญหา

| ปัญหา | ตัวชี้วัด | ค่าปัจจุบัน **[วัดจริง]** | เป้าหมาย v3 gate |
|---|---|---|---|
| Identity dominance | NN-same-signer (query → nearest clip) | 0.55 (frozen+NAP) · 0.89–1.00 (frozen) | ≤ 0.35 |
| Dictionary retrieval ข้าม signer | E1 median rank / R@5 (gallery 4.7k) | 547 / 0.000–0.054 | ≤ 50 / ≥ 0.30 (Gate E) |
| One-shot enrollment | E2 R@5 (gallery 18) | 0.658 test / 0.895 val | ≥ 0.85 บน gallery ≥ 100 |
| Webcam robustness | Δ median rank เมื่อ query = aug1 | 547 → 779 (+42 %) | ≤ +20 % |
| Speed gap | E1 บน query ที่เร่ง 2× | ไม่ได้วัด (train speed aug ช่วย) | Δ ≤ +20 % |
| CSLR | gloss F1 / WER (pseudo + real) | ≈ 0.00 / 1.0–2.6 | F1 ≥ 0.5, WER ≤ 0.6 (Gate C1) |
| Decoder calibration | hyp/ref บน video จริง ด้วย threshold จาก studio | 0 (ไม่ output) | 0.7–1.3 |
| Stream active mask | active fraction บนช่วงพักของ video จริง | 1.00 / 0.93 (ไม่เจอช่วงพัก) | false-active ≤ 10 % |
| CPU real-time | p95 ms/frame · RSS | 108–116 ms · 2.9–3.4 GB | ≤ 80 ms · ≤ 2.5 GB |

### 1.2 ทำไมข้อมูลปัจจุบัน "เรียน invariance ไม่ได้"

```
สัญญาณที่ encoder ต้องเรียน:  f(video) ≈ f(video')  เมื่อ  sign(video)=sign(video')  แต่  signer ต่างกัน
จำนวนคู่แบบนี้ใน TTRS:          177 ท่า × (ส่วนใหญ่ 2 คลิป)  ≈ 180 คู่  →  หลัง split เหลือ train ≈ 35 ท่า
จำนวนคู่ "คนเดียวกันคนละท่า":   ~4,700 คลิป × 3 signer  →  สัญญาณ identity แรงกว่าสัญญาณ sign หลายร้อยเท่า
```
ผลที่ตามมา: loss ใดก็ตามที่ลดได้ด้วยการจำ (signer, clip) จะถูกลดด้วยทางนั้นก่อน — สอดคล้องกับ ArcFace loss → 0.3 ขณะ E1 ไม่ขยับ **[วัดจริง]**

### 1.3 ทำไม SSL ของ v2 plateau

- ทุก view (global/local, base/aug/flip) มาจาก **คลิปเดียวกัน** → คนเดียวกัน เสื้อเดียวกัน ฉากเดียวกัน (aug เปลี่ยนฉากแต่ไม่เปลี่ยนคน)
- DINO/iBOT บน features ที่ frozen แล้ว (ซึ่งเข้ารหัส identity แน่นอยู่แล้ว) สามารถ match teacher ได้ด้วย identity/texture component → prototype usage entropy คงที่ 8.31–8.32 (≈ log 4096 → ใช้ทุก prototype สม่ำเสมอแบบ Sinkhorn บังคับ แต่ไม่แยก semantic) **[วัดจริง]**
- iBOT บน frame features: frame ข้างเคียงคล้ายกันมากที่ 12.5 fps → interpolate ได้โดยไม่ต้องเข้าใจท่า

### 1.4 ทำไม CSLR แบบ spotting ล้ม

| สาเหตุ | หลักฐาน |
|---|---|
| Isolated ranking อ่อน: ท่าถูกอยู่อันดับ ~500 | E1 **[วัดจริง]** |
| Window embedding ไม่เคยถูก train บน "ท่าครึ่งท่า / transition" | head train บน active segment เต็มท่าเท่านั้น |
| Similarity scale ต่างโดเมน → threshold absolute ไม่ transfer | τ=0.7 → output ว่างบน video จริง **[วัดจริง]** |
| Pseudo-sentence จากการต่อคลิป ไม่มี coarticulation | synthetic ≠ real (ไม่มี transition จริง) |
| Metric เลือก decoder degenerate | WER เลือก decoder ที่ output 0.08 gloss/ref **[วัดจริง]** |
| OOV function signs (ไป, ไหม) | ไม่มีใน TTRS **[วัดจริง]** |

### 1.5 ทำไม inference generalize ไม่ได้ และอะไรช่วยจริง

| Knob | ผล **[วัดจริง]** | สถานะใน v3 |
|---|---|---|
| Session centering (video mean / signer mean ของ bank) | E1 median rank ดีขึ้นทั้ง 4 โมเดลบน val (770→684, 750→652, …) · video จริง: ฉัน >10 → 1, กิน >10 → 4 | **แกนหลัก** |
| Vocabulary scoping (TNC top-5000 → 1,155 ท่า) | ฉัน 1, กิน 1, เพื่อน 6 (ร่วมกับ centering) แต่ตัด ปลอบใจ/ด้วยกัน ออก | ใช้แบบ domain vocab ที่ผู้ใช้กำหนด ไม่ใช่ความถี่คำพูด |
| CSLS hubness correction | แย่ลง (720 → 804) | ตัดออก |
| Mirror TTA | ไม่ช่วย | ปิด (เปิดเมื่อ signer ถนัดซ้าย) |
| Training-time signer centering | แย่ลงมาก (1,356) | ตัดออก |
| Relative DP penalty | ทำให้ output ไม่ว่างบน video จริง แต่ F1 ยัง ≈ 0 | คงไว้แนวคิด "relative scoring" |

---

## 2. Design Principles ของ v3

1. **Invariance มาจากข้อมูลและ representation ก่อน loss** — ถ้าไม่มีคู่ข้ามคน หรือ input ยังมีผิว/เสื้อ/หน้า loss ใดก็เรียนไม่ได้
2. **ทุก positive ใน contrastive learning ต้อง "ต่างคน" ให้มากที่สุดเท่าที่ข้อมูลมี** และทุก batch ต้องมี hard negative "คนเดียวกันคนละท่า"
3. **Session คือหน่วยของ generalization** — ระบบจริงเห็น signer คนเดิมหลายวินาทีเสมอ ใช้ข้อมูลนั้น (centering, calibration, enrollment) ไม่ต้องให้ encoder invariant 100 %
4. **Relative > absolute** — คะแนนทุกชั้น (similarity, spotting, rejection) normalize ภายใน session
5. **Gate ก่อนต่อชั้น** — CSLR/LLM ไม่ถูก train/tune จนกว่า encoder ผ่าน Gate E บน signer-disjoint Bench
6. **Metric ต้องไม่ degenerate** — CSLR เลือกด้วย F1 + WER + hyp/ref ∈ [0.7, 1.3] เสมอ; retrieval เลือกด้วย MRR/median rank ไม่ใช่ R@1 บนชุดเล็ก
7. **สิ่งที่ผู้ใช้ทำได้ ต้องเป็นส่วนหนึ่งของระบบ** — enroll ท่าเฉพาะตัว, เลือก vocab ตามบริบท, calibration phrase 10 วินาที

---

## 3. Data Plan v3 (เงื่อนไขจำเป็นของ Encoder และ CSLR)

### 3.1 แหล่งข้อมูลและบทบาท

| ชุด | ขนาดเป้าหมาย | บทบาทใหม่ | เหตุผล |
|---|---|---|---|
| **Bench-Word** (อัดเอง) | 300 ท่า × **≥ 12 signer** × 2 ฉาก/แสง · phone 720p + webcam | train (8 signer) · val (2) · **test signer-disjoint (2)** · หมุน k-fold ตาม signer | สัญญาณ cross-signer หลัก + test ที่ตรงกับการใช้งาน |
| **Bench-Function** | 60 function/question/pronoun signs × 12 signer | เติม OOV ที่พบใน data_test | ไป, ไหม ไม่มีใน TTRS **[วัดจริง]** |
| **Bench-Sent** | 150 ประโยค (3–8 ท่า) × 8 signer · gloss sequence + **gloss timestamps** (label ด้วย tool คลิกช่วงเวลา) | CSLR train/val/test (signer-disjoint) | CTC ต้องมีประโยคจริงที่มี coarticulation |
| **TTRS** | 5,043 คลิป | lexicon ครอบคลุม 4.7k ท่า (prototype bank) · pretrain | ยังเป็นพจนานุกรมที่ใหญ่ที่สุด |
| **YouTube lessons + VTT** | 186+57 วิดีโอ · 658 lexeme hits **[วัดจริง v2]** | **cross-signer mining** (ครูหลายคน) | เพิ่มคู่ข้ามคนโดยไม่ต้องอัด |
| **Big Sign / continuous** | 6 ชม. | SSL geometry/motion · weak sentence pairs (v4) | caption เป็นเสียงพูด ไม่ใช่ gloss |
| **data_test + real-world set** | ≥ 30 วิดีโอ (ขยายจาก 2) พร้อม gloss + timestamp | **test เท่านั้น** (sha256 blocklist) | 2 วิดีโอไม่พอสรุป |

### 3.2 Bench recording protocol (ย่อ)
- signer: หลากหลายเพศ อายุ สีผิว ความถนัดมือ ≥ 2 คนถนัดซ้าย; ผู้ใช้ภาษามือจริง ≥ 50 %
- ต่อ signer: ท่าละ 2 ครั้ง (ความเร็วปกติ + เร็วแบบสนทนา) · เสื้อ 2 แบบ · ฉาก 2 แบบ
- ใช้ `cli.py record`: แสดงคำ + คลิปตัวอย่าง TTRS (เพื่อให้เป็นท่าเดียวกัน) · ตรวจ valid-rate ทันที · บันทึก metadata ตาม schema §3 ของ v2 (`source=bench`, `signer_id`, `is_test_only`)
- **Split ล็อกก่อนอัด** (signer ถูกกำหนด fold ล่วงหน้า) ป้องกันการเลือก test signer ย้อนหลัง

### 3.3 Cross-signer mining (อัตโนมัติ)

```
VTT word timestamps (lesson) ──► candidate window [t−0.5, t+2.0] s ──► encoder v3 (round r)
      ──► score vs TTRS prototype ของคำนั้น (session-centred, relative rank ภายในวิดีโอ)
      ──► accept ถ้า rank ≤ 3 และ margin ≥ δ และ duration 0.4–2.5 s ──► InstanceRecord(source=mined, weight=0.5, round=r)
      ──► human audit 50 ตัวอย่าง/รอบ: precision ≥ 0.7 จึงใช้รอบถัดไป
```
- ใช้ **หลังผ่าน Gate E-1** เท่านั้น (ถ้า encoder ยังอ่อน mining จะเก็บ noise)
- cap 5 instance/lexeme/signer ป้องกันครูคนเดียวครอบงำ

### 3.4 Splits และ protocol (แทนของ v2)

| Protocol | Query | Gallery | Held-out จาก training ทุกชั้น |
|---|---|---|---|
| **E1 Dictionary** (หลัก) | คลิปของ signer ที่ไม่มีท่านั้นใน train | prototypes ทุกท่าในพจนานุกรม | คลิปของ query signer สำหรับท่านั้น |
| **E2 Enrollment** | signer A | enroll ท่าที่ไม่เคย train จาก signer ≠ A | ทุกคลิปของท่านั้น |
| **E3 Signer-disjoint Bench** (gate สุดท้าย) | Bench test signers (ไม่เคยเห็นเลย) | TTRS + Bench-train prototypes | signer ทั้งคน |
| **C1 Bench-Sent** | ประโยคของ test signers | vocab ของ Bench + function signs | signer ทั้งคน |
| **R Real-world** | ≥ 30 วิดีโอนอกระบบ | full vocab และ domain vocab | ทั้งหมด (sha256) |

กฎ: ≥ 300 queries ต่อ protocol ต่อ split · รายงาน 95 % CI เสมอ · selection บน val เท่านั้น · real-world ห้ามใช้ tune (อนุญาตเฉพาะ diagnostic ที่ประกาศชัด)

---

## 4. Encoder v3

### 4.1 ภาพรวม

```
frame @12.5 fps (native res)
 │
 ├─ L1 Articulators v3 ─ YOLOX + RTMPose-s body (เดิม, valid 1.00)
 │                      + RTMPose hand-21 ×2 (crop จาก wrist box, ทุกเฟรมบน GPU / ทุก 2 เฟรมบน CPU)
 │                      + face landmarks (brows, eyes, mouth ~ 68–106 จุด, ทุก 2 เฟรม)
 │
 ├─ S_geo   Geometry stream (identity-free by construction)
 │           3D-lift (hand depth ratio + body) → bone-length canonicalization → torso-centred coords
 │           handshape angles (15 joint angles/มือ) · hand-to-face / hand-to-chest relations · velocities
 │
 ├─ S_hand  Hand-appearance stream (identity-reduced)
 │           tight crop จาก hand-21 bbox (pad 1.2) → hand/skin mask → DINOv2 patch tokens pooled **ภายใน mask**
 │           + motion: |Δ patch tokens| ระหว่างเฟรม · grayscale+CLAHE option · หัก per-session mean
 │
 └─ S_nmm   Non-manual stream (ไม่เข้า word embedding)
             face landmark deltas (brow raise, mouth open/shape, head yaw/pitch/roll) → NMM + mouthing head
                    │
                    ▼
 Temporal Encoder v3: per-stream Transformer → cross-stream fusion (S_geo ↔ S_hand attention)
                    ▼
 frame tokens Z_t (สำหรับ CSLR)  ·  pooled sign embedding z (สำหรับ ISLR / prototypes)
```

### 4.2 Input representation — ทำไมเปลี่ยน

| v2 | ปัญหาที่วัดได้ | v3 |
|---|---|---|
| Hand crop จาก wrist + forearm ขนาด ~0.45–1.2 × shoulder width | crop มีแขนเสื้อ ลำตัว หน้า (ตรวจด้วยตา) · NN-same-signer ของ hands-only 0.79 **[วัดจริง]** | Tight crop จาก hand keypoints + mask pooling → ตัดเสื้อ/ฉาก |
| CLS ‖ patch-mean ทั้ง crop | global descriptor เข้ารหัส texture/สีผิว | patch tokens pooled ใน mask + motion difference (ส่วนที่คงที่ของรูปลักษณ์หักล้างกัน) |
| Pose body 11 จุด normalized ด้วย shoulder width | ไม่มี handshape → E1 1,232 **[วัดจริง]** | + 21 จุด/มือ + joint angles + bone-length canonicalization (สัดส่วนร่างกายไม่เหลือ) |
| Face crop เข้า word embedding | ตัดออกได้โดยไม่เสีย **[วัดจริง]** แต่พา identity | Face landmark deltas → NMM/mouthing เท่านั้น |
| 112 px ทุก stream | ไม่ได้ทดสอบ 224 (T18) | hand crop 112 (CPU) / 160–224 (GPU) — ablation บังคับใน Phase B |

**Bone-length canonicalization (S_geo)** [สมมติฐาน]
- สร้าง skeleton มาตรฐาน (ความยาวกระดูกเฉลี่ยจาก Bench) → ทุกเฟรมหมุนแต่ละ segment ตามทิศทางจริง แต่ใช้ความยาวมาตรฐาน → ตำแหน่งมือสัมพัทธ์ใบหน้า/อกคงอยู่ แต่สัดส่วนร่างกายหาย
- handshape: มุม MCP/PIP/DIP + abduction ต่อมือ (invariant ต่อขนาดมือ)
- confidence-aware: joint conf < 0.3 → mask token (เหมือน [MISSING] ของ v2)

**ต้นทุน** [ประมาณการ จาก v2 วัดจริง: hand model 78–88 ms/เฟรม CPU แบบ detector+pose]
- GPU: RTMPose-hand ×2 crops batch ≈ 4–6 ms · face landmarks ≈ 2–3 ms
- CPU: ใช้ wrist box แทน hand detector (ตัด RTMDet) → RTMPose-hand-m ×2 ≈ 20–30 ms → รันทุก 2 เฟรม + interpolate

### 4.3 Frame backbone

| ตัวเลือก | ใช้เมื่อ | หมายเหตุ |
|---|---|---|
| DINOv2-S frozen, patch tokens + mask pooling | ค่าเริ่มต้น | ไม่ต้อง re-cache ทั้งหมด: cache เฉพาะ hand crops ใหม่ (≈ 2 crops/เฟรม) |
| DINOv3-S/B frozen | เมื่อได้ `HF_TOKEN` + transformers ≥ 4.56 | ablation เดียวใน Phase B |
| **Hand-specialist fine-tune** (LoRA r8 บน 4 blocks สุดท้าย) ด้วย cross-signer handshape contrastive | ถ้า Gate E-1 ไม่ผ่านด้วย frozen | positive = hand crop ของท่า/เฟรม-phase เดียวกันคนละ signer (จาก Bench alignment ด้วย DTW บน S_geo) |

### 4.4 Temporal encoder

- per-stream: S_geo (d 192, 4 layers) · S_hand (d 256, 4 layers) — ขนาดใกล้ v1 (≈ 3.3 M/stream) ซึ่ง train ได้บน 4 GB **[วัดจริง]**
- fusion: 2 layers cross-attention (geo query ↔ hand key/value) → frame tokens Z_t (d 256)
- positional: relative time (ms) ไม่ใช่ frame index → ทนต่อ fps/ความเร็ว
- **Speed-equivariant training:** input resample 0.6–2.5× (v1: 0.8–2.5× ดีที่สุด **[วัดจริง]**) + random frame drop 10 % (จำลอง webcam fps ไม่นิ่ง)
- pooled embedding z: attention pooling ด้วย active/boundary weights (ไม่ใช่ tCLS อย่างเดียว) → ทนต่อ rest frames ที่ติดมาใน window

### 4.5 Self-supervised pre-training v3

| Objective | ทำอะไร | ทำไมแก้ shortcut ได้ |
|---|---|---|
| **Cross-signer DINO** | teacher view = ท่าเดียวกันจาก signer อื่น (Bench/TTRS pairs/mined) เมื่อมี; มิฉะนั้น within-clip | บังคับให้ match ข้ามตัวตน **[สมมติฐาน]** |
| **Masked keypoint-motion prediction (MKP)** | mask 30–50 % frames ของ S_hand tokens → ทำนาย S_geo velocity/handshape angles ของเฟรมที่ถูก mask | target ไม่มี texture → identity ช่วยทำนายไม่ได้ |
| **Temporal order + speed prediction** | สลับ 2 ช่วง / เร่ง 1–2.5× → classify order และ speed bin | ต้องเข้าใจ dynamics ไม่ใช่รูปลักษณ์; ช่วย speed gap |
| **Signer-confusion regularizer** | GRL signer classifier บน pooled z (signer id จาก cluster/Bench) | ลด identity ที่ยังเหลือ (λ 0.5 ใน v1 ไม่ทำให้แย่ลง **[วัดจริง]**) |
| **Gram anchoring** (คงจาก v2) | stage 2 | กัน frame-level collapse (loss 0.022–0.027 ทำงานปกติ **[วัดจริง]**) |
| ~~within-clip iBOT บน frozen CLS~~ | ตัด | interpolate ได้จากเฟรมข้างเคียง |

**Batch composition:** 50 % ตัวอย่างที่มี cross-signer partner · 25 % YouTube/Big Sign (geometry diversity) · 25 % TTRS singleton (lexicon coverage)

**SSL monitors (หยุด/เปลี่ยนแผนเมื่อ):**
- NN-same-signer บน Bench-val ไม่ลดลง ≥ 0.1 ภายใน 2k steps → หยุด, ตรวจ input representation
- kNN E1 median rank (val) ไม่ดีขึ้นกว่า frozen baseline ≥ 30 % ภายใน 3k steps → หยุด (v1 plateau ที่ 1k **[วัดจริง]**)
- prototype entropy: ตรวจ *per-class* usage ไม่ใช่ batch-mean (batch-mean ถูก Sinkhorn บังคับให้สูงเสมอ)

### 4.6 Supervised ISLR head v3

| ส่วน | v2/v1 | v3 | เหตุผล |
|---|---|---|---|
| Loss หลัก | SupCon + ArcFace (ทุกคลาส) + adversary 0.1–0.5 | **Cross-signer SupCon** (positive ต่างคน weight 3×, positive คนเดียวกัน = aug views weight 1×) + **same-signer hard negatives** | ArcFace singleton = memorize **[วัดจริง]** |
| ArcFace | ทุก 4.6k คลาส | เฉพาะคลาสที่มี ≥ 2 signer ใน train (Bench + pairs + mined) | ป้องกัน class center = signer center |
| Sampler | P classes × 2 views | P classes × 2 **signers** (ถ้ามี) + K same-signer negatives | batch ต้องสอนสิ่งที่ต้องการ |
| Fine-tune | full FT ดีกว่า LoRA **[วัดจริง]** | full FT, lr encoder 1e-4 | — |
| Speed aug | 0.8–2.5× ดีที่สุด **[วัดจริง]** | 0.6–2.5× + frame drop | — |
| Face stream | ใช้ | **ไม่ใช้** ใน z | ตัดแล้วไม่เสีย **[วัดจริง]** |
| Training-time centering | แย่ลง **[วัดจริง]** | ไม่ใช้ | — |
| Prototype | mean ต่อ sign_variant | mean ต่อ sign_variant **ต่อ signer** แล้วเฉลี่ยข้าม signer (signer-balanced) + เก็บ signer id ไว้ทำ centering | ป้องกัน signer ที่มีคลิปมากครอบงำ |
| Selection | MRR(E1) + MRR(E1 centred) + 0.05·E2 | เหมือนเดิม + NN-same-signer penalty บน Bench-val | MRR เสถียรกว่า R@k **[วัดจริง]** |

### 4.7 Encoder Gates

| Gate | เงื่อนไข (บน val, CI รายงาน) | ถ้าไม่ผ่าน |
|---|---|---|
| **E-0 Representation** | S_geo-only frozen kNN: NN-same-signer ≤ 0.5 และ E1 median rank ดีกว่า RGB frozen | ตรวจ canonicalization / hand-pose quality (valid-rate hand ≥ 0.9) |
| **E-1 Encoder** | E1 (TTRS+Bench) median rank ≤ 150, R@5 ≥ 0.15 · NN-same-signer ≤ 0.4 | เพิ่ม Bench signer / เปิด hand-specialist fine-tune |
| **E-2 Deployment** | **E3 signer-disjoint Bench** R@5 ≥ 0.30 (full vocab) และ ≥ 0.60 (domain vocab 300) · aug/webcam Δ ≤ 20 % | ห้ามเข้า CSLR; ใช้ assisted mode (top-k) เท่านั้น |

---

## 5. Continuous Sign Language Recognition (CSLR) v3

### 5.1 สถาปัตยกรรม

```
frame tokens Z_t (Encoder v3, speed-equivariant)
   │
   ├─ Boundary head ── per-frame {sign, transition, rest} ── BiGRU/2-layer Transformer, causal สำหรับ stream
   │
   ├─ Prototype-CTC head
   │     logits_t(k) = s · cos( W_proj Z_t , P_k ) ,  k ∈ vocab ∪ {blank}
   │     P_k = prototype bank (session-centred) → เพิ่ม/ลบคำได้โดยไม่ retrain
   │     blank = transition/rest (ผูกกับ boundary head ด้วย auxiliary loss)
   │
   ├─ Decoder: prefix beam search (beam 8)
   │     + lexical constraints (domain vocab) + gloss n-gram LM (Bench-Sent + Thai text→gloss mapping)
   │     + insertion penalty แบบ relative (per-session calibrated)
   │
   └─ N-best (≤ 5) + per-gloss posterior + timestamps ──► Language layer (LLM rerank + compose) / rules
```

### 5.2 ทำไมเลือก Prototype-CTC แทน spotting และแทน CTC ปกติ

| ทางเลือก | ข้อดี | ข้อเสียในข้อมูลนี้ |
|---|---|---|
| Sliding-window spotting (v2) | ไม่ต้องมี sentence labels | window ไม่เคยเห็น transition, threshold absolute, F1 ≈ 0 **[วัดจริง]** |
| CTC softmax classifier ปกติ | มาตรฐาน CSLR | ต้องมี sentence data ต่อคำจำนวนมาก, เพิ่มคำต้อง retrain, 4.7k คลาส singleton |
| **Prototype-CTC** | ใช้ prototype bank เดิม (enrollment), train alignment ด้วย sentence ไม่มาก, คำนอก Bench-Sent ยังถูก score ได้ผ่าน prototype | ขึ้นกับคุณภาพ encoder (จึงมี Gate E-2) |
| Seq2seq / ByT5 (v2 option) | แปลเป็นประโยคตรง | ข้อมูลไม่พอ (6 ชม. weak captions) → เลื่อนไป v4 |

### 5.3 ข้อมูลสำหรับ CSLR

| ชุด | ใช้อย่างไร |
|---|---|
| **Bench-Sent** (ประโยคจริง + gloss timestamps) | train/val/test หลัก (signer-disjoint) |
| **Synthetic sentences v3** จาก Bench-Word + TTRS | ต่อ active segments **ของ signer เดียวกัน** (Bench มี signer เดียวทำหลายท่า) + speed 1.0–2.5× + **transition synthesis**: interpolate S_geo keypoints ระหว่างท่า 3–8 เฟรม และตัด rest ทิ้ง → ลด gap กับ coarticulation จริง |
| TTRS pseudo-sentences แบบ v1 (ต่างคนต่อท่า) | **ห้ามใช้ train**; ใช้เป็น regression test เท่านั้น (ต่าง signer ในประโยคเดียวไม่มีจริง และทำให้ session centering ใช้ไม่ได้) |
| Big Sign weak pairs | v4 (LLM-level sentence supervision) |

### 5.4 Training recipe
1. Freeze encoder (ผ่าน E-2) → train boundary head (TTRS active masks + Bench-Sent timestamps) 
2. Train prototype-CTC projection + blank on synthetic v3 (ratio 70 %) + Bench-Sent (30 %) · SpecAugment-style time masking บน Z_t · speed perturb
3. Unfreeze encoder top 2 layers, lr 3e-5, Bench-Sent เท่านั้น (ป้องกันลืม isolated)
4. Joint loss: `L = L_CTC + 0.3·L_boundary + 0.3·L_ISLR(SupCon on segments)` — ป้องกัน CSLR ทำลาย metric space ของ prototypes

### 5.5 Decoding & calibration (บทเรียนจาก "decoder ที่เงียบ")

- **Session score normalization:** logits ต่อเฟรม z-normalize ด้วยค่าเฉลี่ย/SD ของ top-1 cosine ใน session (running window 20 s ใน stream)
- **Insertion penalty** = quantile ของ session (เช่น median top-1 + δ) ไม่ใช่ค่าคงที่ (relative DP ทำให้ output ไม่ว่างบน video จริง **[วัดจริง]**)
- **Output-rate guard:** ถ้า glosses/วินาทีของ active time ต่ำกว่า 0.3 หรือสูงกว่า 3 → ปรับ penalty อัตโนมัติภายใน [δ_min, δ_max] และ flag `uncertain`
- **Unknown token `<UNK>`**: ช่วงที่ boundary = sign แต่ max posterior ต่ำ → ส่ง `<UNK>` + top-5 ให้ language layer แทนการเดาคำผิด
- **Function-sign prior:** LM ให้ prior กับ ไป/ไหม/ไม่/อะไร ตามโครงสร้าง topic-comment (เช่น ไหม ท้ายประโยค เมื่อ NMM question)

### 5.6 NMM สำหรับประโยค
- question (yes/no): brow raise (landmark eyebrow–eye distance z-score ต่อ session) + head forward/tilt + ท่า "ไหม" · wh-question: brow furrow + wh-sign
- negation: head-shake (คงจาก v2; ทำงานถูกบน 2 วิดีโอที่ไม่ปฏิเสธ **[วัดจริง]**) + ท่า "ไม่"
- affect: head distilled จาก FER (agreement 0.60 YouTube held-out **[วัดจริง]**) + damp เมื่อ NMM active (คงจาก v1) · ต้องมี Bench-Face labels ก่อนประกาศ accuracy

### 5.7 CSLR Metrics & Gates

| Metric | นิยาม | หมายเหตุ |
|---|---|---|
| Gloss WER | edit distance / ref | รายงานคู่กับ insertion / deletion / substitution rate เสมอ |
| Gloss F1 (bag) | precision/recall ของ gloss set | ป้องกัน decoder เงียบ |
| hyp/ref ratio | จำนวน gloss output ต่อ ref | ต้องอยู่ 0.7–1.3 จึงนับเป็น config ที่ valid |
| Timestamp IoU | ช่วงเวลา gloss vs label | ใช้ Bench-Sent timestamps |
| Sentence chrF | หลัง language layer | แยก rules vs LLM |
| Top-5 oracle WER | WER ถ้าเลือก gloss ที่ดีที่สุดใน top-5 ต่อ segment | บอกเพดานของ LLM rerank |

| Gate | เงื่อนไข (Bench-Sent val signer-disjoint) |
|---|---|
| **C-1** | F1 ≥ 0.50 · WER ≤ 0.60 · hyp/ref 0.7–1.3 · top-5 oracle WER ≤ 0.35 |
| **C-2 (deploy)** | test: WER ≤ 0.45 · chrF ≥ 40 (เป้า v1 เดิม) · real-world set R: WER ≤ 0.60 |

---

## 6. Inference v3 — How to Generalize

### 6.1 Session Adaptation Stack (ลำดับการทำงาน)

| ชั้น | ทำอะไร | ต้นทุน | หลักฐาน / สถานะ |
|---|---|---|---|
| **A0 Capture QA** | ตรวจ hand px (≥ 40 px), valid-rate มือ, แสง, fps จริง → แนะนำผู้ใช้ | ~0 | v1 valid-rate 1.00 ในทุกวิดีโอ **[วัดจริง]** แต่ต้องมีคำเตือนสำหรับกล้องไกล |
| **A1 Session centering** | z ← z − μ_session (EMA ของ window embeddings ในช่วง active, half-life 20 s); bank ← P − μ_signer(bank) | matmul | ดีขึ้นทุกโมเดล (val) + ท่าจริงขึ้นอันดับ 1 **[วัดจริง]** |
| **A2 Calibration phrase** (option, 10–15 s) | ผู้ใช้ทำท่าชุดสั้นที่รู้คำตอบ (เช่น สวัสดี ขอบคุณ ฉัน ชื่อ) → ประมาณ μ_session เร็วขึ้น + fit per-session score scale (temperature, penalty) | 1 ครั้ง/ผู้ใช้ | [สมมติฐาน] ลด cold-start ของ A1 ในช่วงแรกของ stream |
| **A3 User enrollment** | ผู้ใช้เพิ่ม/แทนที่ prototype ของท่าที่ใช้บ่อยด้วยคลิปของตัวเอง 1–3 ครั้ง | ไม่ต้อง retrain | E2 one-shot R@5 0.66–0.90 **[วัดจริง]** |
| **A4 Vocabulary scoping** | domain vocab (เช่น โรงพยาบาล, ร้านอาหาร, ห้องเรียน, ชีวิตประจำวัน 300–800 ท่า) + function signs เสมอ | index subset | TNC vocab + centering: ฉัน 1, กิน 1 **[วัดจริง]** แต่ตัดคำจำเป็นทิ้ง → ต้องเป็น domain ที่ออกแบบ ไม่ใช่ความถี่คำพูด |
| **A5 Relative decoding** | §5.5 session score normalization + output-rate guard | ~0 | absolute threshold ล้ม **[วัดจริง]** |
| **A6 OOD / rejection** | max-posterior distribution เทียบกับ calibration → `<UNK>` / แจ้ง "ไม่แน่ใจ" | ~0 | ป้องกันการพูดคำผิดด้วยความมั่นใจ (v1 พูด "ขนมปัง…" ออกเสียง) |
| **A7 Language layer** | LLM rerank N-best + compose; rules fallback เมื่อ offline | 0.4–0.9 s/ประโยค [ประมาณการ] | rules fallback ทำงานทุกครั้ง **[วัดจริง]** |
| ~~Mirror TTA~~ | ปิดค่าเริ่มต้น; เปิดเมื่อ calibration ตรวจพบถนัดซ้าย | — | ไม่ช่วย **[วัดจริง]** |
| ~~CSLS~~ | ตัด | — | แย่ลง **[วัดจริง]** |

### 6.2 Stream mode v3

```
T1 capture (OpenCV) ──queue(2)──► T2 perception (GPU/CPU)
                                   body pose (ทุกเฟรม) · hand-21 (ทุกเฟรม GPU / 2 เฟรม CPU) · face lmk (2 เฟรม)
                                   hand crops → DINOv2-S (batch 2) · S_geo features
                                   ring buffer 64 เฟรม (5 s) · temporal encoder incremental (ทุก 2 เฟรม)
                                   boundary head (causal) → segment close เมื่อ rest ≥ 0.4 s
                                   A1 EMA session mean · A5 normalization
                                   ──segment events──► T3 CTC beam on closed segment + LM → language → TTS (async)
```
- **Latency เป้า:** gloss ≤ 400 ms หลังจบท่า · GPU ≤ 40 ms/เฟรม · CPU ≤ 80 ms/เฟรม p95
- ปัจจุบัน **[วัดจริง]**: GPU 29 ms p50 / 64–73 ms p95 · CPU 70 ms p50 / 108–116 ms p95 · RSS 2.6–3.4 GB
- แผนลด (เรียงตามผล): ONNX int8 DINOv2-S (encoder 3.6–5.3 s/100 เฟรม บน CPU คือ ~60 % ของเวลา **[วัดจริง]**) · lazy-load TTS/FER หลัง perception · hand-21 ทุก 2 เฟรม + interpolate · batch 2 hand crops · pre-allocated ring buffer

### 6.3 Active / rest detection v3 (แก้ active = 1.00 บน video จริง)
- v1 stream mode ใช้ percentile ของ wrist-y ทั้งวิดีโอ → วิดีโอสั้นที่มือยกเกือบตลอดไม่มี "rest" ให้เทียบ **[วัดจริง]**
- v3: rest ต้องเข้าเงื่อนไข **ทั้ง** (a) มือทั้งสองต่ำกว่า elbow หรือประสานใกล้เอว (b) speed ≤ threshold ต่อเนื่อง ≥ 0.4 s (c) handshape เปลี่ยน < ε → จากนั้นใช้ boundary head แทน rule เมื่อผ่าน C-1
- ประเมิน T16 ด้วย label ช่วงพักใน real-world set

### 6.4 Generalization checklist ต่อ deployment ใหม่

| คำถาม | เครื่องมือ |
|---|---|
| กล้อง/ระยะใหม่? | A0 capture QA (hand px, valid-rate) |
| signer ใหม่? | A1 centering → A2 calibration → A3 enrollment |
| บริบทใหม่ (โรงพยาบาล, ร้าน)? | A4 domain vocab + LM เฉพาะบริบท |
| ความเร็วแปลก (เร็ว/ช้า)? | speed-equivariant encoder + relative time positional |
| คำที่ไม่มีในพจนานุกรม? | A6 `<UNK>` + enrollment UI |
| ถนัดซ้าย? | calibration ตรวจ dominant hand → swap streams / mirror |
| ประโยคคำถาม/ปฏิเสธ? | NMM landmarks + function signs + LM prior |

---

## 7. Evaluation v3

### 7.1 ตาราง test cases (แทน T01–T20 ของ v2 เฉพาะที่เปลี่ยน)

| ID | กรณี | ชุด | เกณฑ์ |
|---|---|---|---|
| V3-E1 | Dictionary retrieval signer ใหม่ | E1 (TTRS+Bench) | median rank ≤ 150 · R@5 ≥ 0.15 (Gate E-1) |
| V3-E3 | Signer-disjoint Bench | E3 | R@5 ≥ 0.30 full vocab · ≥ 0.60 domain 300 (Gate E-2) |
| V3-ID | Identity leakage | Bench-val | NN-same-signer ≤ 0.35 · signer probe accuracy บน z ≤ 1.5 × chance |
| V3-SP | Speed robustness | E1 query เร่ง 2× | Δ median rank ≤ 20 % |
| V3-WC | Webcam robustness | Bench phone vs studio / aug1 | Δ ≤ 20 % |
| V3-EN | One-shot enrollment | E2 gallery ≥ 100 | R@5 ≥ 0.85 |
| V3-C1 | CSLR val | Bench-Sent val | Gate C-1 |
| V3-C2 | CSLR test + real-world | Bench-Sent test · R | Gate C-2 |
| V3-SA | Session adaptation ablation | E3 · R | A1 ≥ +20 % MRR · A3 (3 clips) ≥ +30 % R@5 ของท่าที่ enroll |
| V3-OOD | `<UNK>` | non-sign motions + OOV signs 50 คลิป | rejection ≥ 0.8 @ FPR 0.1 |
| V3-NMM | question / negation | Bench-Sent subset | F1 ≥ 0.75 |
| V3-RT | Real-time | stream replay 3 นาที + webcam | GPU p95 ≤ 40 ms · CPU p95 ≤ 80 ms · RSS ≤ 2.5 GB คงที่ |

### 7.2 กฎการทดลอง (จากความผิดพลาดใน v1)
1. ≥ 300 queries ต่อ split · รายงาน CI · ถ้า CI ของสอง variant ทับกันมาก ให้ถือว่าเท่ากัน
2. Model selection บน val ด้วย MRR/median rank · CSLR ด้วย F1 + WER + hyp/ref guard
3. ทุก metric ที่เป็น 0.000 หรือ 1.000 ต้องตรวจ NaN/degenerate ก่อนรายงาน (v1 เจอ NaN → R@1 = 1.000 **[วัดจริง]**)
4. ห้ามใช้ real-world set ในการ tune; diagnostic ต้องประกาศ
5. SSL และ training ยาวต้อง save checkpoint ทุก eval (v1 SSL 2 ชม. ไม่มี checkpoint กลางทาง)
6. GPU 4 GB: ห้ามรัน job GPU สองงานที่รวม VRAM > 3.5 GB (v1 thrash 8 s/step **[วัดจริง]**)

---

## 8. Roadmap และ Go/No-go

| เฟส | งาน | ระยะ | Go/No-go |
|---|---|---|---|
| **A0** | Freeze v1 codebase เป็น baseline · เพิ่ม E3/C1 loaders · checkpoint ระหว่าง train | 2 วัน | reproduce ตัวเลข result_v1 ±CI |
| **A1 Data** | อัด Bench-Word (4 signer ก่อน) + Bench-Function · annotate tool | 1–2 สัปดาห์ | valid-rate hand-21 ≥ 0.9 บน Bench |
| **B1 Representation** | articulators v3 (hand-21, face lmk) · S_geo canonicalization · hand mask pooling · cache ใหม่เฉพาะ hand crops | 1 สัปดาห์ | **Gate E-0** |
| **B2 Ablations** (เรียงตาม expected value) | ① S_geo vs S_hand vs fusion (frozen kNN) ② cross-signer SupCon + hard negatives ③ ArcFace multi-signer only ④ SSL v3 objectives (MKP, order/speed) ⑤ 112 vs 224 hand crops ⑥ DINOv3 ⑦ hand-specialist LoRA | 2 สัปดาห์ | **Gate E-1** (ถ้าไม่ผ่านหลัง ①–④ → อัด Bench เพิ่มเป็น 12 signer ก่อนทำต่อ) |
| **A2 Data** | Bench ครบ 12 signer · mining รอบ 1 จาก YouTube lessons (audit precision ≥ 0.7) | 2 สัปดาห์ (ขนานกับ B2) | mined precision ≥ 0.7 |
| **B3** | retrain encoder บน Bench + mined · signer-balanced prototypes | 1 สัปดาห์ | **Gate E-2** |
| **C1 CSLR** | Bench-Sent · synthetic v3 · boundary + prototype-CTC · decoder calibration | 2–3 สัปดาห์ | **Gate C-1** |
| **D1 Inference** | Session Adaptation Stack A0–A6 · stream v3 · active/rest v3 · CPU int8 | 1–2 สัปดาห์ | V3-SA, V3-RT, V3-OOD |
| **C2** | LLM rerank (ต้องมี `OPENAI_API_KEY`) · NMM question · real-world set 30 วิดีโอ | 1–2 สัปดาห์ | **Gate C-2** |
| **v4** | seq2seq จาก Big Sign weak pairs · NADT licence · self-training จาก user-confirmed sentences | ต่อเนื่อง | ชนะ C-2 เท่านั้นจึงสลับ |

**ทางลัดถ้ายังอัด Bench ไม่ได้** (ยอมรับว่าเพดานต่ำ): ทำ B1 + ②③④ บน TTRS + mined + domain vocab 300 + A1–A4 ในโหมด **assisted** (แสดง top-5 ต่อ segment ให้ผู้ใช้/ผู้ช่วยเลือก) — ไม่ประกาศว่าเป็น automatic translation

---

## 9. การเปลี่ยนแปลงของโค้ด (เล็กที่สุดที่ทำให้ v3 เป็นไปได้)

```
modules/
  articulators.py   + HandPose21 (RTMPose-hand จาก wrist box) · FaceLandmarks · rest detector v3
  geometry.py       ใหม่: 3D-lift, bone-length canonicalization, handshape angles, relations (S_geo)
  encoder.py        + hand-mask patch pooling · motion tokens · cache version `hand_v3`
  ssl.py            + cross-signer view sampler · MKP head · order/speed head · checkpoint ทุก eval
  heads.py          + CrossSignerSupCon · same-signer hard negative sampler · multi-signer ArcFace · signer-balanced prototypes
  cslr.py           ใหม่: boundary head · PrototypeCTC · prefix beam search + LM · session score normalization · output-rate guard
  session.py        ใหม่: SessionState (EMA mean, score scale, dominant hand) · calibration · enrollment API · domain vocab
  mining.py         ใหม่: VTT candidates → verify → audit export
  pipeline.py       ใช้ session.py + cslr.py · stream v3
  evalkit.py        + E3/C1 loaders · signer probe · NN-same-signer · CSLR metrics (F1, ins/del/sub, IoU, oracle top-5) · NaN guard
scripts/cli.py      + record (Bench) · annotate · enroll · mine
configs/            data.yaml (+hand/face models) · train.yaml (v3 objectives, gates) · infer.yaml (session stack, domains)
configs/domains/    daily.yaml · hospital.yaml · school.yaml (vocab lists + function signs)
```

---

## 10. ความเสี่ยงและทางออก

| ความเสี่ยง | สัญญาณ | ทางออก |
|---|---|---|
| อัด Bench ไม่ได้ครบ 12 signer | Gate E-1 ไม่ผ่านด้วย 4 signer | assisted mode + domain vocab + enrollment; ขอความร่วมมือโรงเรียนโสตฯ/สมาคมคนหูหนวก; NADT licence |
| Hand-21 pose ไม่แม่นเมื่อมือซ้อน/เบลอ/ความละเอียดต่ำ (640×360) | valid-rate hand < 0.9 | fusion กับ S_hand appearance · temporal smoothing · fallback เป็น crop แบบ v2 |
| Canonicalization ลบข้อมูลเชิงไวยากรณ์ (ขนาดท่า/ระยะ) | ท่าที่ต่างกันด้วยขนาดสับสน | เก็บ relative-scale feature แยกช่อง (ไม่ใช่ absolute) |
| Mining สะสม noise | audit precision < 0.7 | ลด τ, จำกัดรอบ, weight 0.3 |
| Prototype-CTC ดึง metric space เสีย | E1 ลดลง > 10 % หลัง CSLR fine-tune | joint ISLR loss (§5.4) · freeze encoder |
| Session centering ผิดเมื่อ session สั้น/มีท่าเดียว | ช่วง 3–5 วินาทีแรกผิดมาก | A2 calibration phrase · prior mean จาก Bench |
| Domain vocab ตัดคำจำเป็น | OOV สูงใน real-world | function signs บังคับรวม · `<UNK>` + enrollment |
| LLM เติมเนื้อหาเกิน gloss | chrF สูงแต่ความหมายผิด | prompt ห้ามเพิ่ม, ตรวจด้วย gloss coverage, แสดง uncertain |
| CPU ไม่ถึง 80 ms | p95 > 80 | int8, hand-21 ทุก 2 เฟรม, GPU/NPU profile |
| ลิขสิทธิ์และความยินยอมของ Bench signer | — | consent form, ใช้เพื่อวิจัย, `license_status` ใน schema |

---

## 11. Appendix — Evidence Log จาก result_v1 (ตัวเลขที่ v3 อ้างอิง)

| หัวข้อ | ค่า |
|---|---|
| Cache | TTRS 279,180 frames × 4 versions (31.8 min) · YouTube 582,774 frames × 2 (99.8 min) · 0 errors |
| Signer inventory | named CV acc 0.999 · cluster purity 0.999 · signer หลัก 3 คน · cross-signer signs 177 |
| Frozen baselines (test) | E1 median rank 2,046 · +NAP 1,018 · E2 R@5 0.289 / 0.395 · NN-same-signer 1.00 / 0.66 |
| SSL v1 (6k steps, 2 h 15 min) | DINO 33.6→27.7 · iBOT 33.3→26.2 · kNN E1 median rank 1,642→1,759 · E2 R@5 0.55–0.58 |
| ISLR scratch (test) | E1 median 720 (centred 590) · E2 R@5 0.658 |
| ISLR no-face + adv 0.5 | E1 787 (706) · E2 R@5 0.658 |
| ISLR pose-only | E1 1,232 · E2 R@5 0.500 |
| ISLR SSL + LoRA r4 | E1 706 (740) · E2 R@5 0.553 |
| ISLR SSL + full FT | E1 712 (594) · E2 R@5 0.526 (val 0.842) |
| ISLR SSL + train centering | E1 1,356 (628) |
| **ISLR SSL + speed 0.8–2.5× (selected)** | **E1 547 (491)** · E2 R@5 0.658 (val 0.895) · aug1 queries E1 779 |
| CSLS k=10 | scratch 720 → 804 |
| Pseudo-continuous | gloss F1 0.000–0.010 ทุก decoder · WER 1.0–2.7 |
| Real-world (final config) | video 1 WER 1.25 chrF 5.7 · video 2 WER 1.00 chrF 8.1 · best window rank ฉัน 1, กิน 1–4 (centred) · ไป/ไหม OOV |
| Face affect | teacher agreement held-out TTRS 0.94 (baseline 0.92) · YouTube 0.60 (baseline 0.49) |
| Latency | GPU 29 ms p50 / 64–73 p95 · CPU 70 p50 / 108–116 p95 · RSS 2.6–3.4 GB · CPU encoder ≈ 60 % ของเวลา |
| Speed of real signing | ≈ 0.6 s/ท่า (on-screen text timing, video 2) vs TTRS active 1.5–3 s |

## 12. References
- SignDino (arXiv:2609.06296) · SHuBERT (arXiv:2411.16765) · SignMusketeers (arXiv:2406.06907)
- DINOv2 / DINOv3 · RTMPose / rtmlib (body, hand-21) · YOLOX
- CTC (Graves 2006) และงาน CSLR ที่ใช้ CTC บน visual features (เช่น VAC, CorrNet, TwoStream-SLR) — แนวคิด auxiliary alignment และ keypoint+RGB two-stream
- Nuisance Attribute Projection (speaker verification) · Session/speaker mean normalization (CMN) ในงานเสียง — ต้นแบบของ Session Adaptation Stack
- โปรเจกต์: `ThaiSLM_BLUEPRINT_v1.md`, `ThaiSLM_BLUEPRINT_v2.md`, `result_reporting/result_v1.md`, `artifacts/reports/*.json`
