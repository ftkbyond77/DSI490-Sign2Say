# ThaiSLM Blueprint
## Thai Sign Language → Text → Speech, built on the SignDINO idea

> เอกสารนี้คือ **แบบพิมพ์เขียว (Blueprint) ฉบับเดียวจบ** สำหรับสร้างระบบรู้จำภาษามือไทย (ระดับคำ + ต่อเนื่อง) พร้อม Face Impression แล้วแปลงเป็นข้อความและเสียง
> ออกแบบจาก **ข้อมูลจริงที่อยู่ในโฟลเดอร์นี้** (สำรวจแล้ว ณ 2026-09-13) และจากสเปกของเครื่องที่ใช้อยู่ (RTX 2050 4 GB, RAM 16 GB, 12 cores)
> หลักคิด: *ไม่ซับซ้อนเกินจำเป็น แต่ทุกชิ้นต้องมีเหตุผลจากข้อมูลและใช้งานจริงได้*

---

## 0. สรุปการตัดสินใจหลัก (Design Decisions at a Glance)

| # | คำถาม | ตัดสินใจ | เหตุผลจากข้อมูล/ข้อจำกัด |
|---|---|---|---|
| D1 | Feature หลักคืออะไร | **RGB crops ของ มือซ้าย / มือขวา / ใบหน้า** ผ่าน frozen **DINOv3 ViT** (ตาม SignDINO) + pose keypoints เป็น stream เสริมราคาถูก | landmark-only (MediaPipe) ทิ้ง handshape/texture/mouthing และ generalize ข้าม signer/กล้องได้แย่ ซึ่งตรงกับอาการที่คุณเจอ |
| D2 | จะ train อะไร / freeze อะไร | freeze image backbone, **train เฉพาะ temporal encoders (~3.5M params/stream) ด้วย SSL** บน embedding ที่ cache ไว้ | ทำให้ SSL ทั้งหมด **รันได้บน GPU 4 GB** และ iterate ได้เร็ว (paper รายงานเร็วขึ้น ~50×) |
| D3 | Word-level ทำเป็น classifier ไหม | **ไม่** → ใช้ **Prototype / Retrieval head** (metric learning) | TTRS มี 4,599 lexeme แต่ **4,201 lexeme มีคลิปเดียว** classifier แบบ N-class เรียนไม่ได้ และ prototype รองรับ "เพิ่มคำใหม่ด้วยคลิปเดียว" (one-shot enrollment) ซึ่งจำเป็นในงานจริง |
| D4 | Continuous ทำอย่างไรกับข้อมูล ~6 ชม. ที่ caption มาจาก *เสียงพูด* | v1 = **Sign Spotting + LLM Language Layer** (ตรวจจับคำจาก prototype แล้วให้ LLM เรียบเรียงเป็นประโยคไทย) / v2 = seq2seq (ByT5-small) เมื่อมี weak-aligned pairs พอ | caption ของ Big Sign เป็นคำพูดของผู้ประกาศ ไม่ใช่ gloss และเหลื่อมเวลากับล่าม 1–3 s → ใช้เป็น *weak supervision* เท่านั้น |
| D5 | Face Impression | face stream เดียวกันกับที่ใช้รู้จำ → **affect head** (pseudo-label จาก FER teacher) + **NMM detector** จาก pose (ส่ายหัว = ปฏิเสธ, คิ้ว/ก้มเงย) → fuse เข้า LLM prompt + TTS style | corpus ไม่มี label อารมณ์ → ต้อง distill จาก teacher; แยก *อารมณ์* ออกจาก *ไวยากรณ์บนใบหน้า* (non-manual markers) เพื่อไม่ตีความผิด |
| D6 | Augmentation ที่ "เข้ากับข้อมูล" | **Chroma-key background replacement** สำหรับ TTRS (ฉากฟ้าเรียบ), mirror-swap L/R, speed 0.8–1.25×, crop jitter | ฉากฟ้า+เสื้อดำ+ไฟสตูดิโอของ TTRS คือ domain gap ใหญ่สุดเทียบกับ webcam; chroma key แก้ตรงจุด ต้นทุนต่ำ |
| D7 | Leakage ที่ต้องกัน | **ตัดแถบล่าง 15 % ของ TTRS ออกเสมอ** (มี gloss เป็นตัวอักษรฝังในเฟรม) และ mask โลโก้ | ถ้าไม่ตัด โมเดล "อ่านคำบรรยาย" แทนอ่านมือ metric สวยแต่ใช้จริงไม่ได้ |
| D8 | Real-time budget | sampling **12.5 fps**, ring buffer 48 เฟรม, ONNX Runtime fp16 (GPU) / int8 (CPU), ทั้ง pipeline **< 120 ms/เฟรม บน CPU, < 30 ms บน RTX 2050** | ดูตารางงบ latency §5.7 |
| D9 | LLM / TTS | OpenAI (gpt-4.1-mini) เป็น Language Layer แบบถอดเปลี่ยนได้; TTS ผ่าน interface เดียว รองรับ edge-tts (ฟรี, ไทย), OpenAI TTS (สั่ง style ได้), local TTS ภายหลัง | ออกแบบ style/accent hook ไว้ก่อนตามที่ต้องการ |
| D10 | Compute | pre-compute embedding cache ครั้งเดียว (ViT-S/16 ≈ 3–4 ชม., ViT-B/16 ≈ 10–12 ชม. บน RTX 2050) → ทุกอย่างหลังจากนั้นเบา | ไม่จำเป็นต้องเช่า GPU ยกเว้นอยาก scale ไป ViT-B/L หรือ seq2seq ใหญ่ |

---

## 1. เป้าหมายและนิยาม "ใช้งานจริงได้"

**Input ที่ระบบต้องรับ**
1. ไฟล์ `.mp4` ประโยคภาษามือไทย จาก signer หลายคน กล้อง/ฉาก/แสงต่างกัน (ไม่ใช่สตูดิโอ)
2. Webcam แบบ real-time (720p, 25–30 fps, notebook ทั่วไป)

**Output**
- ข้อความภาษาไทย (gloss sequence + ประโยคที่เรียบเรียงแล้ว) พร้อม timestamp และความมั่นใจ
- Face Impression: อารมณ์ (เช่น ดีใจ/เศร้า/โกรธ/ประหลาดใจ/กลาง) + ความเข้ม + grammatical markers (คำถาม/ปฏิเสธ)
- เสียงพูดภาษาไทย (สไตล์/น้ำเสียงปรับตามอารมณ์ได้ในภายหลัง)

**เกณฑ์ผ่าน (Acceptance criteria) — ตั้งไว้ก่อนเริ่ม เพื่อไม่หลอกตัวเองด้วย test metric**

| เกณฑ์ | เป้าหมาย v1 | วัดจาก |
|---|---|---|
| Word-level, signer ที่ไม่เคยเห็น, webcam | Recall@1 ≥ 0.60, Recall@5 ≥ 0.85 บนชุดคำ 200 คำ | ThaiSLM-Bench (ชุดที่เราอัดเอง §5.3.4) |
| Continuous, .mp4 ประโยคสั้น (3–8 คำ) | Gloss WER ≤ 0.45, ประโยคจาก LLM ได้ chrF ≥ 40 เทียบ reference | ThaiSLM-Bench-Sent |
| Face impression | agreement กับผู้ประเมิน ≥ 0.70 (5 คลาส), NMM negation F1 ≥ 0.75 | Bench-Face |
| Latency (webcam) | end-to-end gloss ≤ 400 ms หลังจบท่า; เสียงเริ่มเล่น ≤ 1.5 s หลังจบประโยค | `notebooks/08` |
| RAM | ≤ 2.5 GB resident (CPU mode) | psutil ใน pipeline |

---

## 2. ทำความเข้าใจข้อมูลก่อนออกแบบ (Data Understanding)

ตัวเลขทั้งหมดมาจาก `metadata/metadata_master.csv` และการสุ่มดูเฟรมจริง

### 2.1 ภาพรวม

| ชุด | คลิป | ชั่วโมง | ความยาว/คลิป (median) | ความละเอียดหลัก | fps | label |
|---|---|---|---|---|---|---|
| `word_level/ttrs_dictionary` | 5,043 | 6.2 | 4.1 s (1.8–20 s) | 854×480 (96 %), 720p (3 %) | 25 | **gloss ต่อคลิป** (สะอาด) |
| `word_level/youtube_word` | 186 | 10.6 | 157 s | 720p ส่วนใหญ่ | 25/30 | title (หัวข้อ) + auto-caption จากเสียงครู |
| `sentence_level/youtube_sentence` | 57 | 2.6 | 131 s | 720p | 25/30 | title + caption (49 ไฟล์ .vtt) |
| `continuous/youtube_bigsign` | 44 | 6.2 | 217 s (30 s – 54 นาที) | 720p (39) / **360×640 แนวตั้ง** (5, รวม 2.2 ชม.) | 25 | title + auto-caption **จากเสียงผู้ประกาศ** |
| `continuous/youtube_continuous` | 1 | — | — | 720p | 30 | title |
| `continuous/parliament`, `_quarantine` | 0 | — | — | — | — | ว่าง |

รวม 5,331 คลิป / 25.6 ชม. / ≈ 2.7 ล้านเฟรม (≈ 1.15 ล้านเฟรมที่ 12.5 fps)

### 2.2 ข้อสังเกตที่ **กำหนดการออกแบบ** (สำคัญมาก)

**TTRS (แกนหลักของ word-level)**
- 5,043 คลิป → 4,747 label → หลัง normalize (ตัดวงเล็บแหล่งที่มา เช่น "(ภาษามือโรงเรียนเศรษฐเสถียรฯ)") เหลือ **4,599 lexeme**
- การกระจาย: **4,201 lexeme มี 1 คลิป**, 354 มี 2, 42 มี 3, 2 มี 4 → นี่คือ *one-shot dictionary* ไม่ใช่ classification dataset
- lexeme ที่มี ≥ 2 คลิป = 398 และ **263 lexeme ในนั้นมาจากคนละ signer** → เป็น *cross-signer retrieval test set* ตามธรรมชาติ (ใช้ประเมินได้ทันทีโดยไม่ต้องอัดเพิ่ม)
- signer: ระบุชื่อได้ ~2,300 คลิป (ชาย 1,928 / หญิง 1,018) มี signer หลักเพียง **~5–6 คน** → ความหลากหลายของ signer ต่ำมาก ต้องชดเชยด้วย augmentation และ benchmark ที่อัดเอง
- ภาพ: ฉาก **ฟ้าเรียบ (chroma-blue)**, เสื้อดำ, ยืนกลางเฟรม, โลโก้ TTRS มุมขวาบน, **gloss ภาษาไทยฝังอยู่แถบล่างของเฟรม** (leakage!)
- 67 label เป็นตัวอักษรเดี่ยว (ก–ฮ, A–Z) = ชุด fingerspelling; 136 หมวด (มากสุด: คำในบทเรียน 402, COVID-19 273, กิริยาวลี 188)
- 3,944 คลิปไม่มีเสียง → ไม่มี audio ให้ใช้

**YouTube word / sentence**
- คลิปยาว (บทเรียน) มีครูพูดคำพร้อมทำท่า + ตัวอักษร/flashcard ในเฟรม → auto-caption (`*.th-orig.vtt` มี word-level timestamp) ให้ **weak timestamp ของคำที่ครูพูด ≈ เวลาที่ทำท่า (±1.5 s)** → ใช้ mine ตัวอย่างเพิ่มให้ lexeme ของ TTRS ได้ (dictionary spotting)
- บางคลิปเป็น fingerspelling ระยะใกล้ (มือใหญ่เต็มเฟรม) → ทดสอบ scale-invariance ของ crop pipeline
- หลาย channel, หลายฉาก, หลาย signer → เป็นแหล่ง *domain diversity* สำหรับ SSL ที่มีค่าที่สุดในชุดนี้

**Big Sign (continuous)**
- แบบแนวนอน: ล่ามเต็มเฟรม มีกราฟิก/ข้อความซ้อนบางช่วง
- แบบแนวตั้ง 360×640: **ล่ามอยู่กรอบล่าง แขกรับเชิญอยู่กรอบบน** → มี "คน 2 คน" ในเฟรม ต้องมีตรรกะเลือก signer (ไม่ใช่เลือกหน้าใหญ่สุด)
- caption = **คำพูดภาษาไทยของผู้ประกาศ/เพลง** ที่ล่ามแปลตาม ไม่ใช่ gloss และเหลื่อมเวลา → ใช้เป็นคู่ (video-segment, Thai text) แบบ *weakly aligned* สำหรับ v2 เท่านั้น
- เพลงประกอบละคร ≈ 40 % ของชุด → ท่าจะ "ตามจังหวะเพลง" ไม่ใช่ภาษามือสนทนาปกติ ต้อง tag แยกและถ่วงน้ำหนักต่ำ

### 2.3 ข้อสรุปเชิงการออกแบบจากข้อมูล

1. **SSL ใช้ทุกคลิป (25.6 ชม.)** — ไม่ต้องใช้ label; ความหลากหลายฉาก/คนจาก YouTube ช่วยให้ temporal encoder ไม่ overfit สตูดิโอ
2. **Word-level = prototype retrieval บน lexeme 4,599 คำ** — train metric space ด้วย augmentation-positives + คู่ 398 lexeme; ประเมินด้วย 263 คู่ข้าม signer + benchmark ที่อัดเอง
3. **Continuous v1 ไม่ train seq2seq** — ข้อมูล 6 ชม. ที่ label เป็นเสียงพูดไม่พอสำหรับ SLT ตรง ๆ (งานวิจัยใช้หลักร้อย–พันชั่วโมง); ใช้ spotting + LLM ก่อน แล้วค่อยเก็บคู่ที่ align ดีขึ้นจากผลลัพธ์ระบบเอง (self-training loop)
4. **ต้องมี benchmark ของเราเอง** เพราะไม่มีชุดทดสอบไหนในโฟลเดอร์นี้ที่สะท้อน "webcam ที่บ้าน + signer ใหม่" — ห้ามใช้ TTRS held-out เป็นตัวชี้วัดความสำเร็จหลัก
5. **Leakage guard**: crop-based features แก้ปัญหา gloss ฝังในเฟรมโดยธรรมชาติ (มือ/หน้าไม่รวมแถบล่าง) แต่ทุก full-frame/pose path ต้อง mask แถบล่างและโลโก้เสมอ

### 2.4 ทำไม MediaPipe-landmark pipeline "test ดีแต่ใช้จริงไม่ดี" และ SignDINO แก้ตรงไหน

| ปัญหาของ landmark-only | สิ่งที่เกิดขึ้นจริง | SignDINO-style แก้อย่างไร |
|---|---|---|
| landmark เป็นตัวแทน handshape ที่บาง (21 จุด) และ noisy เมื่อมือซ้อน/หันข้าง/เบลอ | เฟรมที่ landmark ผิดจะพัง feature ทั้งเฟรม; train set สตูดิโอไม่มีอาการนี้ | ใช้ RGB crop → backbone เห็น texture/นิ้วซ้อน/ motion blur ได้; validity mask จัดการเฟรมที่หาไม่เจอ |
| ไม่มี mouthing / สีหน้า | ภาษามือไทยแยกความหมายด้วยปากและคิ้วหลายคำ | face stream เป็นพลเมืองชั้นหนึ่ง (paper: ตัด face → -3.3 BLEU) |
| normalization ของพิกัดขึ้นกับระยะกล้อง/ท่ายืน | webcam ที่บ้านต่างจากสตูดิโอ | crop ต่อ articulator ทำให้ scale-invariant โดยนิยาม |
| classifier เรียน "signer + ฉาก" แทน "ท่า" เพราะ signer น้อย | metric สูงบน split ที่ signer ซ้ำ | SSL บน 25 ชม. หลายฉาก + prototype head + chroma-key augmentation |

---

## 3. สถาปัตยกรรมระบบ (System Architecture)

```
                       ┌────────────────────────────────────────────────────────────────┐
  INPUT                │  L0  Capture        .mp4 (ffmpeg/PyAV)  |  webcam (OpenCV)     │
                       │      → frames @12.5 fps, resize long side 640, ring buffer     │
                       └───────────────────────────────┬────────────────────────────────┘
                                                       ▼
                       ┌────────────────────────────────────────────────────────────────┐
                       │  L1  Articulator Extraction  (modules/articulators.py)         │
                       │      RTMPose-body17 (ONNX, ~5 ms GPU / ~25 ms CPU)              │
                       │      → signer selection → boxes: L-hand, R-hand, face           │
                       │      → 3 crops 112×112 (+ validity mask) + pose vector (34-d)   │
                       └───────────────────────────────┬────────────────────────────────┘
                                                       ▼
                       ┌────────────────────────────────────────────────────────────────┐
                       │  L2  Frame Encoder  (frozen)  modules/encoder.py               │
                       │      DINOv3 ViT-S/16 (edge) | ViT-B/16 (quality), CLS 384/768-d│
                       │      offline: cached to disk (fp16 memmap)  online: per frame  │
                       └───────────────────────────────┬────────────────────────────────┘
                                                       ▼
                       ┌────────────────────────────────────────────────────────────────┐
                       │  L3  Temporal SSL Encoders  (trainable, 3 × 3.5 M) modules/ssl │
                       │      per-stream 6-layer Transformer, D=384, temporal CLS        │
                       │      trained with temporal-DINO + iBOT + KoLeo + Gram anchoring │
                       │      + pose stream (tiny 2-layer) → fused token sequence        │
                       └───────┬───────────────────────┬───────────────────────┬────────┘
                               ▼                       ▼                       ▼
                 ┌──────────────────────┐ ┌──────────────────────┐ ┌───────────────────────┐
                 │ L4a ISLR Prototype   │ │ L4b Continuous       │ │ L4c Face Impression   │
                 │ head (metric space,  │ │ sign spotting        │ │ affect head + NMM     │
                 │ 4.6k lexeme protos,  │ │ (sliding window →    │ │ (emotion, intensity,  │
                 │ one-shot enrollment) │ │ prototypes → peaks)  │ │ question/negation)    │
                 └──────────┬───────────┘ └──────────┬───────────┘ └───────────┬───────────┘
                            └──────────────┬─────────┘                         │
                                           ▼                                   │
                       ┌────────────────────────────────────────────────────────┴───────┐
                       │  L5  Language Layer  modules/language.py                       │
                       │      gloss seq + conf + timing + affect → LLM (OpenAI) →        │
                       │      Thai sentence (+ style tag)  | offline fallback: rules     │
                       └───────────────────────────────┬────────────────────────────────┘
                                                       ▼
                       ┌────────────────────────────────────────────────────────────────┐
                       │  L6  TTS  modules/tts.py   TTSProvider(edge-tts | openai | local)│
                       │      text + SpeechStyle(emotion, rate, pitch, voice) → audio    │
                       └────────────────────────────────────────────────────────────────┘
```

**หลักการที่ทำให้เบาและเร็ว**
- โมเดลหนัก (ViT) รัน **ครั้งเดียวต่อเฟรมต่อ crop** และ frozen → export ONNX/fp16 ได้ตรง ๆ
- ทุกสิ่งที่ train ได้อยู่หลัง cache → train บน 4 GB ได้; เปลี่ยน head ไม่ต้องรัน video ใหม่
- pose stream แถมมาจาก detector ที่ต้องรันอยู่แล้ว (ไม่มีต้นทุนเพิ่ม) และให้ L/R hand identity โดยไม่ต้อง ByteTrack

---

## 4. โมดูลและไฟล์ (File Structure)

```
SLM-Labs-SignDINO/
├─ raw_data/                 (เดิม — read-only)
├─ metadata/                 (เดิม)
├─ ThaiSLM_BLUEPRINT.md
├─ .env                      ← secrets & paths (ไม่ commit)
├─ .env.example
├─ requirements.txt
├─ configs/
│   ├─ data.yaml             ← paths, fps, crop, split, augmentation
│   ├─ train.yaml            ← ssl / islr / face / continuous hyper-params
│   └─ infer.yaml            ← runtime profile (edge|quality), thresholds, llm, tts
├─ modules/
│   ├─ __init__.py
│   ├─ data.py               manifest, lexeme normalization, video decode (PyAV), splits
│   ├─ articulators.py       pose detector (rtmlib ONNX) → signer selection → crops + masks
│   ├─ encoder.py            frozen DINOv3 wrapper, ONNX export, embedding cache (memmap)
│   ├─ augment.py            chroma-key bg replace, mirror-swap, speed, crop jitter (works on crops & cache)
│   ├─ ssl.py                temporal DINO/iBOT/KoLeo/Gram: student/teacher, multi-temporal-crop, train loop
│   ├─ heads.py              ISLR prototype head, face affect head, NMM detector
│   ├─ spotting.py           continuous: sliding-window spotting, peak picking, gloss timeline
│   ├─ language.py           LLM language layer (OpenAI) + rule-based fallback + prompt templates
│   ├─ tts.py                TTSProvider interface: EdgeTTS / OpenAITTS / LocalTTS, SpeechStyle
│   ├─ pipeline.py           end-to-end: VideoPipeline (mp4) & StreamPipeline (webcam), ring buffer
│   ├─ evalkit.py            metrics (R@k, WER, chrF, IoU, latency/RAM probes), bench loaders
│   └─ utils.py              logging, timers, seed, io
├─ notebooks/
│   ├─ 00_explore_raw_data.ipynb
│   ├─ 01_articulators_and_cache.ipynb
│   ├─ 02_prepare_datasets.ipynb
│   ├─ 03_ssl_pretrain.ipynb
│   ├─ 04_train_heads_islr_face.ipynb
│   ├─ 05_continuous_spotting_llm.ipynb
│   ├─ 06_evaluation.ipynb
│   ├─ 07_tts.ipynb
│   └─ 08_inference_realworld.ipynb
├─ scripts/                  (เดิม harvester) + cli.py  (python scripts/cli.py infer --video x.mp4)
├─ artifacts/                ← ผลลัพธ์ทั้งหมด (gitignored)
│   ├─ cache/                embeddings fp16 memmap + index (ต่อ backbone profile)
│   ├─ manifests/            parquet: clips, segments, lexemes, splits, mined_instances
│   ├─ checkpoints/          ssl/, islr/, face/, spotting/
│   ├─ prototypes/           lexeme_protos.npz (+ enrollment เพิ่มได้)
│   ├─ onnx/                 backbone/pose/temporal exported
│   └─ bench/                ThaiSLM-Bench (คลิปที่อัดเอง + labels.csv)
└─ tests/                    pytest: smoke tests ต่อ module (ไม่บังคับ แต่มีโครง)
```

หนึ่ง module = หนึ่งความรับผิดชอบ, notebook แต่ละตัวเรียก module เพื่อ "โชว์" หนึ่งขั้นตอน ไม่มี logic ซ้ำใน notebook

---

## 5. รายละเอียดแต่ละขั้น

### 5.1 Exploration & Inspect Raw Data — `00_explore_raw_data.ipynb`

**เป้าหมาย:** ยืนยันสมมติฐานใน §2 ด้วยตาและตัวเลข ก่อนเสียเวลา cache

| ขั้น | ทำอะไร | output |
|---|---|---|
| E1 | โหลด `metadata_master.csv` → distribution ต่อชุด (duration, res, fps, codec) | ตาราง+histogram |
| E2 | Lexeme normalization preview: ตัดวงเล็บ, trim, unify Thai numerals; นับ singleton/multi | `lexeme_stats.csv` |
| E3 | สุ่ม 12 คลิปต่อชุด → contact sheet 8 เฟรม/คลิป | ดูฉาก, ตำแหน่ง signer, caption ฝัง |
| E4 | ตรวจ **แถบ caption ของ TTRS**: วัดตำแหน่งแถบ (สแกน row ที่มี text edge หนาแน่น) → กำหนด `caption_band_frac` ให้ config | ค่าประมาณ 0.85–1.0 ของความสูง |
| E5 | ตรวจ chroma background: histogram สี HSV ของ TTRS 200 เฟรม → ค่าฟ้าอ้างอิงและ tolerance | `chroma_ref.json` |
| E6 | Big Sign แนวตั้ง: วัด layout กรอบล่ามล่าง (สัดส่วน y) | `layout_rules.json` |
| E7 | VTT: parse `th-orig.vtt` (word timestamps) → นับคำที่ตรงกับ lexeme TTRS ต่อคลิป (candidate mining) | `vtt_lexeme_hits.parquet` |
| E8 | Signer inventory: `signer_creator` + face-embedding clustering เบื้องต้น (ใช้ DINOv3 CLS ของ face crop 1 เฟรม/คลิป) | จำนวน signer จริงของ TTRS/YouTube |
| E9 | รายการความเสี่ยงข้อมูล (สั้น/เบลอ/มีคน 2 คน/ไม่มีมือในเฟรม) → `quality_flags` | column ใน manifest |

**สิ่งที่ต้องได้ก่อนไปต่อ:** ค่าคงที่ 3 ตัวสำหรับ config: `caption_band_frac`, `chroma_ref`, `vertical_layout` และ manifest v0

### 5.2 Data Transformation & Feature Engineering — `01_articulators_and_cache.ipynb`

**หลัก:** แปลงวิดีโอทุกคลิปให้เป็น *ลำดับ embedding ต่อ articulator* หนึ่งครั้ง แล้วไม่แตะวิดีโอดิบอีก

**5.2.1 Decode policy**
- PyAV/ffmpeg decode → **12.5 fps** (stride 2 ของ 25 fps; resample สำหรับ 30/50/60) → long side 640 px
- เหตุผลเลือก 12.5 fps: ครอบคลุม dynamics ของท่าปกติ, ลด compute ครึ่งหนึ่ง, ตรงกับ budget webcam; **ข้อยกเว้น** ทดลอง 25 fps เฉพาะชุด fingerspelling (flag ใน config)
- TTRS: mask แถบล่างและโลโก้ *ก่อน* ทุกอย่าง (ป้องกัน pose detector ล็อกโลโก้/ตัวอักษร)

**5.2.2 Articulator extraction (`modules/articulators.py`)**
- Detector: **RTMPose body-17 via `rtmlib`** (Apache-2.0, ONNX, CPU-friendly) — ให้ nose/eyes/ears/shoulders/elbows/wrists
  - ทางเลือกตอน prototype: `ultralytics yolov8n-pose` (AGPL ระวัง license); ทางเลือกความแม่นสูง: hand detector แยก (YOLOv8n-hand) ตาม paper
- **Signer selection** (สำคัญกับ Big Sign แนวตั้ง / คลิปครูกับนักเรียน):
  1. candidates = คนที่ตรวจพบ ≥ 60 % ของเฟรมใน window 2 s
  2. score = 0.5·(hand-motion energy ของข้อมือ) + 0.3·(ขนาดไหล่) + 0.2·(อยู่ในโซน layout ที่คาดไว้ เช่น กรอบล่างของแนวตั้ง)
  3. lock track id, re-evaluate ทุก 2 s (hysteresis) — กันกระโดดไปมา
- **Crop geometry** (เริ่มจากกฎนี้ แล้ว validate ใน notebook กับ 200 เฟรมที่ label กล่องมือด้วยมือ)
  - hand: center = wrist + 0.30·(wrist − elbow), side = 1.1·‖wrist − elbow‖, padding 1.4× (ดูดซับ blur ตาม paper)
  - face: center = mean(eyes, nose), side = 2.2·‖ear_L − ear_R‖ (fallback 1.6·‖eye_L − eye_R‖·3), padding 1.2×
  - ทุก crop → 112×112, เก็บ `valid` flag (keypoint conf < 0.3 → invalid → learnable [MISSING] token ตอน train, mask ตอน inference)
  - L/R identity มาจาก skeleton (ไม่ต้อง track); **mirror-swap augmentation** ทำได้ง่าย: flip ภาพ + สลับ stream
- **Pose stream**: 17 จุด × (x, y) normalize ด้วยกึ่งกลางไหล่และความกว้างไหล่ → 34-d ต่อเฟรม + velocity 34-d = 68-d (ราคาถูก, ให้ตำแหน่งมือเทียบลำตัวซึ่ง crop เพียว ๆ ไม่มี)

**5.2.3 Frozen frame encoder (`modules/encoder.py`)**
- **DINOv3 ViT-S/16 (21 M)** = profile `edge` (ค่าเริ่มต้น), **ViT-B/16 (86 M)** = profile `quality` (ใช้กับ .mp4 offline ถ้าต้องการ)
  - DINOv3 weights ต้องกด accept license บน Hugging Face; fallback = DINOv2 (Apache-2.0) โดยเปลี่ยน 1 บรรทัดใน config
- input 224×224 (upsample จาก 112 ตาม paper), output = CLS (384-d S / 768-d B) **+ mean of patch tokens** (concat → 768/1536) — เก็บทั้งคู่ใน cache, ให้ temporal encoder เลือกใช้ (ablate ใน 03)
- **Embedding cache**: `artifacts/cache/{profile}/{stream}.f16.memmap` + `index.parquet` (uid, start_row, n_frames, valid mask) — ขนาดโดยประมาณ 1.15 M เฟรม × 3 stream × 768-d × 2 B ≈ **5.3 GB** (S) / 10.6 GB (B)
- ต้นทุน cache ครั้งเดียวบน RTX 2050 (fp16, batch 128): ViT-S ≈ 3–4 ชม., ViT-B ≈ 10–12 ชม. → รันข้ามคืน; ต่อได้ (resumable per uid)

**5.2.4 Augmentation ที่ทำงานกับ "ข้อมูลนี้" (`modules/augment.py`)**

| Aug | ใช้กับ | ทำงานระดับ | ทำไม |
|---|---|---|---|
| **Chroma-key background replacement** | TTRS (ฉากฟ้าเรียบ) | pixel (ก่อน encode) → cache เวอร์ชัน `aug_bg` 2 ชุดต่อคลิป (ฉากห้อง/ออฟฟิศ/กลางแจ้งจาก stock free) | ปิด domain gap สตูดิโอ↔บ้าน ตรงจุดที่สุด ต้นทุน 2× encode เฉพาะ TTRS (~1.5 ชม. บน ViT-S) |
| Color/brightness/JPEG/blur/downscale-upscale | ทุกชุด | pixel (ก่อน encode, ใช้ 1 เวอร์ชัน `aug_photo`) | จำลอง webcam คุณภาพต่ำ |
| Mirror-swap | ทุกชุด | cache-level (สลับ stream L↔R + flip pose x) — **ต้อง encode crop ที่ flip ไว้** เป็น `aug_flip` เพราะ CLS ไม่ flip-invariant | signer ถนัดซ้าย |
| Speed 0.8–1.25×, temporal jitter/drop | ทุกชุด | cache-level (resample index) | ความเร็วท่าต่างคน |
| Crop jitter ±10 %, scale ±15 % | ทุกชุด | pixel ตอน encode aug เวอร์ชัน | ความคลาดเคลื่อนของ detector ตอนใช้จริง |
| Random stream dropout (ปิดมือซ้าย/หน้า 10 %) | ทุกชุด | cache-level | ทน occlusion/ออกนอกเฟรม |

เวอร์ชัน cache ที่ต้องมี: `base`, `aug_flip`, `aug_photo` (ทุกชุด) และ `aug_bg×2` (TTRS) → TTRS 6.2 ชม. ×5 + อื่น 19.4 ชม. ×3 ≈ **89 ชม.-เทียบเท่า** ≈ 12–14 ชม. encode บน ViT-S (ทำเป็น background job)

**5.2.5 Weak-label mining จาก VTT (feature engineering เชิงข้อมูล)**
- จาก `th-orig.vtt` ของ youtube_word/sentence: ทุกคำที่ตรง lexeme TTRS (หลัง pythainlp tokenize + normalize) → candidate window [t−0.5, t+2.0] s
- ใน 02/04: หลังมี prototype รอบแรก ใช้ prototype ยืนยัน (cosine > τ_mine) → เพิ่มเป็น instance ของ lexeme นั้น (self-labeling รอบละ 1 ครั้ง, cap 5 instance/lexeme, เก็บ source ไว้ audit)
- Big Sign: ตัด segment ตาม caption cue **เลื่อนเวลา +1.5 s** (ล่ามตามเสียง) แล้วเก็บเป็นคู่ (segment, Thai text) ลง `pairs_weak.parquet` สำหรับ v2

### 5.3 Data Preparation Before Modeling — `02_prepare_datasets.ipynb`

**5.3.1 Manifests (parquet ทั้งหมดใน `artifacts/manifests/`)**
- `clips.parquet`: uid, level, source, path, n_frames@12.5, cache offsets, quality_flags, signer_id (จาก E8), aug versions
- `lexicon.parquet`: lexeme_id, lexeme_th, variants (label ดิบ), category, pos, definition, n_clips, is_fingerspelling
- `segments.parquet`: สำหรับ continuous/sentence — (uid, t0, t1, text, align_type ∈ {caption_shifted, mined, manual})
- `mined_instances.parquet`: (lexeme_id, uid, t0, t1, score, round)

**5.3.2 Splits (กฎเหล็ก: แยกด้วย signer และ channel ไม่ใช่สุ่มคลิป)**

| งาน | train | val | test |
|---|---|---|---|
| SSL | ทุกคลิป **ยกเว้น** คลิปที่ใช้ใน test ทุกชุด (กัน leakage แม้ SSL ไม่ใช้ label) | — | — |
| ISLR | TTRS ทั้งหมดที่เหลือ + mined instances | 135 lexeme จาก 263 คู่ข้าม signer (ถือคลิปหนึ่งเป็น query) | 128 lexeme ที่เหลือของคู่ข้าม signer **+ ThaiSLM-Bench** (signer นอกชุด) |
| Face | pseudo-label ทุกชุด (train) | manual check 300 เฟรม | Bench-Face |
| Continuous | Big Sign + sentence (weak pairs) ยกเว้น 6 คลิป | 3 คลิป (channel ต่างกัน) | 3 คลิป + Bench-Sent |

**5.3.3 SSL sample construction (ตาม SignDINO ปรับให้เข้ากับความยาวคลิปเรา)**
- คลิป TTRS ยาว 25–100 เฟรม @12.5 fps; YouTube ยาวหลักพันเฟรม → ใช้ **หน่วยตัวอย่าง = chunk 96 เฟรม** (7.7 s) สำหรับคลิปยาว, ทั้งคลิปสำหรับ TTRS
- ต่อ iteration: 2 global temporal crops ยาว U{32..64} เฟรม, 8 local crops ยาว U{6..16} เฟรม (ครึ่งหนึ่งของ paper เพราะ fps ครึ่งหนึ่ง); คลิปสั้นกว่า 32 → global = ทั้งคลิป
- sampling weight: TTRS 0.35, youtube_word 0.30, sentence 0.10, bigsign 0.25 (บทเพลง ×0.5) — กันชุดยาวกลืนชุดสั้น

**5.3.4 ThaiSLM-Bench (ชุดทดสอบที่ต้องอัดเอง — ส่วนที่ทำให้ "ใช้จริงได้" วัดได้จริง)**
- **Bench-Word**: 200 lexeme (เลือกจาก 136 หมวดให้ครอบคลุม + 30 คำใช้บ่อย + 20 fingerspelling) × **≥ 3 signer ที่ไม่อยู่ใน TTRS** × 2 สภาพแสง/ฉาก × webcam 720p → ~1,200 คลิป (อัดผ่าน `scripts/cli.py record` ที่ขึ้น prompt คำและบันทึกอัตโนมัติ)
- **Bench-Sent**: 60 ประโยค (3–8 คำ) × 3 signer, มี reference ทั้ง gloss sequence และประโยคไทย
- **Bench-Face**: จาก Bench-Sent เดิม กำหนดให้ signer แสดง 5 อารมณ์ (กลาง/ดีใจ/เศร้า/โกรธ/ประหลาดใจ) + คำถาม yes/no (คิ้วยก) + ปฏิเสธ (ส่ายหัว) → label ระดับประโยค
- ทุกคลิป Bench ถูก **ห้าม** เข้า SSL/train โดยตรวจ sha256 ใน `evalkit.py`

### 5.4 Modeling Process

#### 5.4.1 Stage A — Temporal SSL pre-training (`03_ssl_pretrain.ipynb`, `modules/ssl.py`)

สถาปัตยกรรม (ต่อ stream: L-hand, R-hand, face; + pose-stream แบบเล็ก)

```
input  E ∈ R^{T×768}  (frozen features from cache)
→ Linear 768→384 → + [tCLS] → + sinusoidal pos(frame idx) → 6 × PreLN Transformer (d=384, 6 heads, mlp 4×)
→ tCLS (clip-level) ,  Z ∈ R^{T×384} (frame-level)
DINO head: 384→2048→256 → prototypes K=4096 (เล็กกว่า paper 8192 เพราะข้อมูล 25 ชม.)
iBOT head (frame-level) ใช้ K เดียวกัน
pose stream: 68 → 2 × Transformer d=128 (train ร่วมใน Stage A ด้วย loss เดียวกัน)
```

Loss (ตาม SignDINO)
- `L_S1 = L_DINO(tCLS student local/global vs teacher global) + L_iBOT(masked frames, frame-only masking 30–50 %) + 0.1·L_KoLeo`
- Stage 2: `+ 2.0·L_Gram` (anchor frame-to-frame cosine Gram matrix กับ teacher ที่ freeze หลัง Stage 1 converge) — paper แสดงว่าตัดแล้วตก 1.4 BLEU / 3 pt R@1
- teacher = EMA (0.994→1.0 cosine), τ_teacher 0.04→0.07 ใน 30 % แรก, τ_student 0.1, Sinkhorn-Knopp centering

งบ: batch 64 chunk × 10 crops, AdamW lr 5e-4 (cosine, warmup 5 %), bf16 → **Stage 1: 60 epochs ≈ 4–6 ชม. ต่อ stream บน RTX 2050**, Stage 2: 20 epochs ≈ 2 ชม.; train 3 stream ตามลำดับ (มือขวาก่อน — สำคัญสุดตาม ablation)

Sanity checks ระหว่างทาง (อยู่ใน notebook):
1. kNN บนคู่ 263 lexeme ข้าม signer (frozen tCLS, ไม่ train head) — ต้องดีขึ้นทุก 10 epoch เทียบ baseline "mean-pool ของ DINOv3 CLS"
2. Prototype usage entropy (กัน collapse)
3. Visualize attention ของ face stream บนช่วง mouthing

#### 5.4.2 Stage B — ISLR Prototype head (`04_train_heads_islr_face.ipynb`, `modules/heads.py`)

```
z = concat(tCLS_L, tCLS_R, tCLS_face, tCLS_pose) → LayerNorm → MLP 1280→512 → L2-normalize
```
- Loss: **Supervised-Contrastive + ArcFace-style margin** บน lexeme id; positives = augmented views ของคลิปเดียวกัน + คลิปอื่นของ lexeme เดียวกัน (398 lexeme) + mined instances
- Fine-tune temporal encoders ด้วย **LoRA rank 4** (paper: rank-1 LoRA 0.17 M params ให้ R@1 0.67 บน ASL Citizen ใกล้ full FT) — เบา, กัน overfit signer 6 คน
- Prototypes: `P[lexeme] = mean(z ของ instance ทั้งหมด)` → `artifacts/prototypes/lexeme_protos.npz`
- Inference: cosine → top-k; **unknown/none** ถ้า top1 < τ_unknown (calibrate บน val) หรือ margin top1−top2 < δ
- **One-shot enrollment**: `enroll(clip, lexeme)` เพิ่ม/อัปเดต prototype ทันที ไม่ต้อง retrain → วิธีขยายคำศัพท์ในงานจริง
- Hard-negative mining: กลุ่มคำที่ท่าใกล้กัน (minimal pairs) จากผล confusion — สร้าง `confusion_groups.json` ให้ LLM ใช้เป็นบริบท

#### 5.4.3 Stage C — Continuous (`05_continuous_spotting_llm.ipynb`, `modules/spotting.py`)

v1 (ทำก่อน, ใช้ได้ทันทีที่ Stage B เสร็จ):
1. sliding window ยาว 12–20 เฟรม (≈1–1.6 s) stride 2 เฟรม → z_t → cosine กับ prototypes ทั้งหมด (4.6 k × 512 = matmul เดียว, ~µs)
2. score timeline ต่อ lexeme → NMS ตามเวลา (peak > τ_spot, ระยะห่าง ≥ 0.4 s) → gloss timeline `[(lexeme, t0, t1, score, top5)]`
3. segmentation ช่วย: pause detection จาก hand-motion energy (มือลง/นิ่ง > 0.6 s = จบประโยค) → ส่งชุด gloss ให้ Language Layer เป็นประโยค ๆ
4. เทรน **boundary head** เล็ก (2-layer Transformer บน fused frame features → sign/transition/idle ต่อเฟรม) ด้วย pseudo-label จาก TTRS (ต้น/ท้ายคลิป = transition) + mined instances → ลด false positive ตอนมือเปลี่ยนท่า (ตามสูตร fingerspelling-detection head ใน paper: threshold 0.5, median 3, L_min 3)

v2 (เมื่อมี weak pairs ≥ 3 ชม. ที่ align ดี หรือได้ข้อมูลเพิ่ม):
- seq2seq: fused frame features (LayerNorm+Linear ต่อ stream → concat → project) → **ByT5-small** decoder (byte-level เหมาะกับไทยที่ไม่มีช่องว่าง), label smoothing 0.2, beam 5
- train บน `pairs_weak` + ประโยคที่ระบบ v1 + LLM สร้างแล้วผู้ใช้ยืนยัน (human-in-the-loop) → ประเมิน chrF/BLEU (pythainlp tokenize) กับ Bench-Sent; ใช้แทน v1 เฉพาะเมื่อชนะบน Bench

#### 5.4.4 Stage D — Face Impression (`04`, `modules/heads.py`)

แยก 2 อย่างชัดเจน เพราะสีหน้าในภาษามือส่วนใหญ่เป็น **ไวยากรณ์** ไม่ใช่อารมณ์:

| ชั้น | สัญญาณ | วิธี | output |
|---|---|---|---|
| Affect (อารมณ์) | face-stream frame features Z_face | pseudo-label ต่อเฟรมจาก FER teacher open-source (ViT/EfficientNet ที่ train บน AffectNet, 8 class + valence/arousal) รันบน face crop ตอนสร้าง cache → train head 2-layer บน Z_face (temporal, window 1 s) ให้ **distill** teacher (KL) → เบากว่า teacher มาก และได้ประโยชน์จาก SSL | `emotion ∈ {neutral, happy, sad, angry, surprised, fear, disgust}`, `valence`, `arousal`, `intensity` (EMA smoothing 0.8 s) |
| NMM (ไวยากรณ์) | pose keypoints (nose x/y, eyes, ears) + Z_face | rules + tiny classifier: head-shake (nose x oscillation ≥ 2 รอบใน 1 s) = **negation**; head-nod = affirmation; brow-raise/chin-forward proxy (eye–nose ratio) + hold = **yes/no question**; head-tilt = wh-question | flags ต่อ segment |
| Fusion | ทั้งสอง + gloss timeline | ส่งเข้า LLM prompt เป็น structured tags; affect → `SpeechStyle` ของ TTS | ประโยคที่มี "?" / "ไม่" ถูกต้อง, เสียงมีอารมณ์ |

การประเมิน: Bench-Face (agreement กับ label), ablation ว่า NMM flag ช่วย WER ของ negation/question ใน Bench-Sent เท่าใด

#### 5.4.5 Language Layer (`modules/language.py`)

- Provider: OpenAI `gpt-4.1-mini` (เร็ว/ถูก) ค่าเริ่มต้น; `gpt-4.1` สำหรับ offline mp4 คุณภาพสูง; **rule-based fallback** (เรียง gloss + ใส่คำเชื่อมพื้นฐาน) เมื่อ offline
- Input JSON: `{glosses:[{th, conf, alt:[...], t0,t1}], nmm:{negation, question}, affect:{emotion, intensity}, context: 2 ประโยคก่อนหน้า}`
- Prompt หลักการ: (1) ภาษามือไทยเป็น topic-comment, ไม่มีคำผันกาล/ลักษณนามครบ (2) **ห้ามเพิ่มเนื้อหาที่ไม่มีใน gloss** (3) ถ้า conf ต่ำให้เลือกจาก `alt` ที่ทำให้ประโยคสมเหตุสมผล และรายงาน `uncertain:true` (4) เติม "ไหม/หรือเปล่า" เมื่อ question flag, "ไม่" เมื่อ negation (5) ตอบ JSON `{sentence_th, style_hint, uncertain}` เท่านั้น
- Few-shot 8 ตัวอย่างจาก Bench-Sent (train part) ใน `configs/infer.yaml`
- Latency: streaming ไม่จำเป็น (ประโยคสั้น) แต่ใช้ `max_tokens ≤ 80`, timeout 3 s → fallback rules
- ต่อยอด: ใช้ LLM เป็น **re-ranker ของ top-k gloss** (constrained selection) ก่อนเรียบเรียง — ทดลองใน 05 ว่าลด WER หรือไม่

### 5.5 Test Cases & Evaluation Process — `06_evaluation.ipynb`, `modules/evalkit.py`

**Metric ต่อชั้น**

| ชั้น | metric | ชุด |
|---|---|---|
| Articulators | crop-IoU กับกล่องมือที่ label เอง 200 เฟรม, valid-rate ต่อชุด, signer-selection accuracy บน Big Sign แนวตั้ง 30 คลิป | manual set |
| SSL quality | kNN R@1/R@5 (frozen) บนคู่ข้าม signer; linear probe บน 136 หมวด (sanity) | TTRS pairs |
| ISLR | R@1/R@5/R@10, macro per-class, unknown-rejection AUROC, **ต่อ signer / ต่อฉาก** | TTRS-test, Bench-Word |
| Spotting | interval mIoU, precision/recall @τ_spot, gloss WER (pythainlp tokenizer สำหรับข้อความ) | Bench-Sent |
| Sentence | chrF (หลัก), BLEU (pythainlp), human adequacy 1–5 บน 60 ประโยค | Bench-Sent |
| Face | accuracy/F1 5 คลาส, NMM F1, latency ของ flag เทียบ onset | Bench-Face |
| Runtime | p50/p95 ms ต่อเฟรม ต่อ stage, RAM peak, GPU mem, end-to-end delay | `08` บน CPU และ GPU |

**Test cases (ต้องผ่านก่อนประกาศ v1)**

| ID | กรณี | Input | คาดหวัง | เกณฑ์ |
|---|---|---|---|---|
| T01 | คำเดี่ยว signer ใหม่ webcam แสงปกติ | Bench-Word | top-1 ถูก | R@1 ≥ 0.60 |
| T02 | คำเดี่ยว แสงน้อย/ย้อนแสง | Bench-Word (low-light) | ไม่ตกเกิน 10 pt จาก T01 | ΔR@1 ≤ 0.10 |
| T03 | signer ถนัดซ้าย | Bench-Word (mirror + 1 signer จริง) | เท่า T01 | ΔR@1 ≤ 0.05 |
| T04 | ฉากหลังรก/คนเดินผ่านหลัง | Bench-Word (busy bg) | signer lock ไม่หลุด | selection acc ≥ 0.95 |
| T05 | ระยะกล้องไกล (ครึ่งตัวเล็ก) / ใกล้ (fingerspelling) | 2 ชุด | valid-rate ≥ 0.9 | crop pipeline |
| T06 | คำที่ไม่อยู่ในคลัง | 50 คลิปท่าที่ไม่ใช่คำ | ตอบ unknown | rejection ≥ 0.8 @ FPR 0.1 |
| T07 | เพิ่มคำใหม่ด้วยคลิปเดียว (enrollment) | 20 คำใหม่ × 1 คลิป | จำได้จาก signer อื่น | R@5 ≥ 0.7 |
| T08 | ประโยค 3–8 คำ .mp4 | Bench-Sent | gloss timeline + ประโยค | WER ≤ 0.45, chrF ≥ 40 |
| T09 | ประโยคปฏิเสธ/คำถาม | Bench-Sent subset | "ไม่"/"ไหม" ถูกที่ | NMM F1 ≥ 0.75 |
| T10 | อารมณ์ 5 แบบ | Bench-Face | emotion ตรง | acc ≥ 0.70 |
| T11 | Real-time 3 นาทีต่อเนื่อง | webcam | ไม่ drop เฟรม, RAM คงที่ | p95 ≤ 120 ms (CPU), RAM ≤ 2.5 GB, ไม่มี leak |
| T12 | LLM/เน็ตล่ม | ปิดเน็ต | ยังได้ประโยคจาก rules + TTS local/cached | ไม่ crash |
| T13 | Leakage guard | TTRS clip พร้อม caption vs. mask | ผลต่างเล็ก | ΔR@1 ≤ 0.02 (ถ้าต่างมาก = โมเดลอ่านตัวหนังสือ) |
| T14 | TTRS-held-out เทียบ Bench | ทั้งสอง | ช่องว่างไม่เกินที่รับได้ | Bench ≥ 0.6 × TTRS (ถ้าต่ำกว่านั้น = overfit สตูดิโอ) |

Error analysis มาตรฐาน (ใน notebook): confusion ต่อ handshape group, ต่อ signer, ต่อ stream dropout (ปิด face/มือซ้าย แล้วดูอะไรพัง) — ใช้เลือกว่าจะเก็บข้อมูลอะไรเพิ่ม

### 5.6 TTS — `07_tts.ipynb`, `modules/tts.py`

```python
@dataclass
class SpeechStyle:
    emotion: str = "neutral"      # neutral|happy|sad|angry|surprised
    intensity: float = 0.0        # 0..1  → rate/pitch/volume mapping
    rate: float = 1.0             # 0.7..1.4
    pitch_semitones: float = 0.0
    voice: str = "th-TH-PremwadeeNeural"
    persona: str | None = None    # ต่อยอด: สำเนียง/บุคลิก (เช่น "อีสาน", "ทางการ")

class TTSProvider(Protocol):
    def synthesize(self, text: str, style: SpeechStyle) -> AudioChunkIterator: ...
```

| Provider | ข้อดี | ใช้เมื่อ |
|---|---|---|
| `EdgeTTSProvider` (ฟรี, เสียงไทย Premwadee/Niwat, ปรับ rate/pitch ได้) | latency ~300–600 ms, ไม่ต้องมี GPU | ค่าเริ่มต้น real-time |
| `OpenAITTSProvider` (`gpt-4o-mini-tts`, ฟิลด์ `instructions` สั่งอารมณ์/สไตล์เป็นข้อความได้) | คุณภาพ/การแสดงอารมณ์ดี | โหมด quality, ทดลอง style |
| `LocalTTSProvider` (VITS/F5-TTS ภาษาไทย, ONNX) | ออฟไลน์, privacy | เมื่อต้อง offline; ใส่ทีหลังผ่าน interface เดิม |

- affect → style mapping ใน `configs/infer.yaml` (เช่น happy: rate 1.1, pitch +2; sad: rate 0.9, pitch −2; angry: rate 1.15, volume +)
- Streaming: แบ่งข้อความตามประโยค → synth เป็น chunk → เล่นผ่าน `sounddevice` ทันที; cache เสียงของประโยคซ้ำ (LRU) → ลด latency
- Hook สำหรับสำเนียง/บุคลิก: `persona` ถูกส่งเป็น instruction ของ provider ที่รองรับ; provider ที่ไม่รองรับจะ ignore อย่างเงียบ ๆ

### 5.7 Inference with Real-World Data — `08_inference_realworld.ipynb`, `modules/pipeline.py`

**โหมด .mp4 (`VideoPipeline.run(path)`)**
1. decode 12.5 fps → articulators → encoder (batch 64 เฟรม) → temporal encoders บน chunk ยาว (ไม่มี ring-buffer constraint) → spotting → LLM ต่อประโยค → TTS → เขียน `result.json` (gloss timeline, sentences, affect timeline), `subtitle.srt`, `audio.wav`, และ (option) วิดีโอ overlay
2. profile `quality` (ViT-B) ได้เมื่อไม่ต้อง real-time

**โหมด webcam (`StreamPipeline.start()`)** — 3 thread + queue ขนาดจำกัด (backpressure, ทิ้งเฟรมเก่าไม่ทิ้งเฟรมใหม่)
- T1 capture: OpenCV 720p → stride ให้ได้ 12.5 fps → queue(2)
- T2 perception: pose → crops → ViT (batch 3 crops/เฟรม) → temporal encoders ทำงานบน **ring buffer 48 เฟรม** แบบ incremental (คำนวณ tCLS ทุก 2 เฟรม) → spotting online (peak เมื่อ score ผ่านและเริ่มลดลง) → gloss event
- T3 language+audio: รวม gloss จน pause → LLM → TTS → play; UI overlay (gloss ล่าสุด, ประโยค, emoji อารมณ์)

**งบ latency ต่อเฟรม (ประมาณการ ต้องวัดจริงใน 08)**

| stage | RTX 2050 (fp16 ORT/TensorRT) | CPU 12-core (int8 ORT) |
|---|---|---|
| pose RTMPose-s @256 | 4–6 ms | 18–25 ms |
| 3 × crop + resize | 1 ms | 1 ms |
| DINOv3 ViT-S/16 ×3 (batch) | 6–9 ms | 45–70 ms |
| temporal encoders ×4 + heads | 2–3 ms | 6–10 ms |
| spotting matmul 4.6k×512 | <1 ms | <1 ms |
| **รวม/เฟรม** | **~15–20 ms** (รองรับ 12.5 fps สบาย) | **~75–105 ms** (12.5 fps ได้แบบชิด; ถ้าไม่ไหว stride → 8 fps) |
| LLM (ต่อประโยค) | 400–900 ms (เครือข่าย) | เท่ากัน |
| TTS เริ่มเล่น | 300–600 ms | เท่ากัน |

**การ optimize ที่ใช้ (เรียงตามผลต่อความคุ้ม)**
1. **Frozen backbone → ONNX fp16 / int8 (dynamic quant)**; TensorRT ถ้ามี — ทำครั้งเดียว
2. ViT-S/16 แทน ViT-B/16 ในโหมด edge (4× ถูกกว่า) — ยอมรับได้เพราะ SSL อยู่ชั้นบน
3. 12.5 fps + ring buffer คงที่ → RAM คงที่ (ไม่มี list โตเรื่อย ๆ), pre-allocated tensors
4. Batch 3 crops ต่อเฟรม เป็น tensor เดียว; หลีกเลี่ยง CPU↔GPU copy ซ้ำ (crop บน GPU ด้วย `roi_align` ถ้าใช้ CUDA)
5. Prototype matrix เป็น fp16 contiguous; top-k ด้วย `argpartition`
6. LLM/TTS async ไม่บล็อก perception; cache ประโยคซ้ำ
7. RAM เป้าหมาย: pose 10 MB + ViT-S fp16 42 MB + temporal 20 MB + protos 5 MB + runtime/buffers ≈ **< 1.5 GB** (CPU mode รวม Python) — วัดด้วย `psutil` ใน T11

**Robustness ที่ต้องมีในโค้ด (ไม่ใช่แค่โมเดล)**
- auto-exposure/gain hint: ถ้า face crop มืด (mean < 40) → แจ้ง UI
- ถ้า valid-rate มือขวา < 0.5 ใน 2 s → แจ้ง "ขยับเข้ากลางเฟรม"
- graceful degradation: ไม่มี GPU → profile edge-cpu อัตโนมัติ; ไม่มีเน็ต → rules + local/cached TTS

---

## 6. `.env` และ Configs

### 6.1 `.env.example`
```dotenv
# ---- paths ----
THAISLM_ROOT=D:/SLM-labs/SLM-Labs-SignDINO
THAISLM_RAW=${THAISLM_ROOT}/raw_data
THAISLM_ARTIFACTS=${THAISLM_ROOT}/artifacts
THAISLM_CACHE_PROFILE=edge            # edge (ViT-S/16) | quality (ViT-B/16)

# ---- model hubs ----
HF_TOKEN=hf_xxx                        # ต้องมี เพื่อโหลด DINOv3 (gated) ; ถ้าไม่มี ใช้ DINOv2
HF_HOME=D:/hf_cache

# ---- language layer ----
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_S=3

# ---- tts ----
TTS_PROVIDER=edge                      # edge | openai | local
TTS_VOICE=th-TH-PremwadeeNeural
OPENAI_TTS_MODEL=gpt-4o-mini-tts

# ---- runtime ----
DEVICE=auto                            # auto | cuda | cpu
ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
NUM_THREADS=8
LOG_LEVEL=INFO
```

### 6.2 `configs/data.yaml`
```yaml
fps: 12.5
frame_long_side: 640
crop_size: 112
encoder_input: 224
ttrs:
  caption_band_frac: 0.85        # จาก 00 (E4)
  logo_box: [0.86, 0.02, 0.99, 0.12]
  chroma_ref_hsv: [105, 180, 200] # จาก 00 (E5)
  chroma_tol: [12, 80, 90]
signer_selection:
  min_presence: 0.6
  weights: {motion: 0.5, size: 0.3, layout: 0.2}
  lock_seconds: 2.0
  vertical_layout_zone: [0.0, 0.45, 1.0, 1.0]   # x0,y0,x1,y1 ของกรอบล่าม (Big Sign แนวตั้ง)
crops:
  hand: {center_shift: 0.30, side_ratio: 1.1, pad: 1.4}
  face: {side_ratio_ears: 2.2, pad: 1.2}
  min_kpt_conf: 0.3
cache:
  versions: [base, aug_flip, aug_photo]
  ttrs_extra_versions: [aug_bg1, aug_bg2]
  dtype: float16
  store: [cls, patch_mean]
lexicon:
  strip_parenthetical: true
  merge_synonyms: false          # เปิดหลังตรวจด้วยตา
splits:
  seed: 42
  by: [signer_id, channel]
  bench_sha256_blocklist: artifacts/bench/sha256.txt
vtt_mining:
  window: [-0.5, 2.0]
  bigsign_caption_shift_s: 1.5
sampling_weights: {ttrs: 0.35, youtube_word: 0.30, youtube_sentence: 0.10, bigsign: 0.25, bigsign_song_factor: 0.5}
```

### 6.3 `configs/train.yaml`
```yaml
backbone:
  edge: facebook/dinov3-vits16-pretrain-lvd1689m
  quality: facebook/dinov3-vitb16-pretrain-lvd1689m
  fallback: facebook/dinov2-small
ssl:
  streams: [hand_r, hand_l, face, pose]
  d_model: 384
  layers: 6
  heads: 6
  prototypes: 4096
  crops: {n_global: 2, n_local: 8, global_len: [32, 64], local_len: [6, 16], chunk_len: 96}
  mask: {type: frame_only, ratio: [0.3, 0.5]}
  loss: {ibot: 1.0, koleo: 0.1, gram: 2.0}
  temp: {teacher: [0.04, 0.07], student: 0.1, warmup_frac: 0.3}
  ema: [0.994, 1.0]
  optim: {lr: 5e-4, wd: 0.04, batch: 64, epochs_s1: 60, epochs_s2: 20, precision: bf16}
islr:
  embed_dim: 512
  loss: {supcon_temp: 0.07, arcface_margin: 0.2, arcface_scale: 30}
  lora: {rank: 4, alpha: 8, targets: [q, v]}
  optim: {lr: 3e-4, epochs: 40, batch: 128}
  unknown: {tau: 0.55, margin: 0.05}     # calibrate ใน 06
  mining: {tau: 0.72, max_per_lexeme: 5, rounds: 2}
face:
  teacher: hsemotion/enet_b0_8_best_afew   # หรือ ViT-FER; ต้องยืนยัน license ก่อนใช้
  classes: [neutral, happy, sad, angry, surprised, fear, disgust]
  window_frames: 12
  smoothing_ema_s: 0.8
  nmm: {shake_min_cycles: 2, shake_window_s: 1.0, nod_min_cycles: 2}
spotting:
  window_frames: [12, 20]
  stride: 2
  tau_spot: 0.6
  nms_gap_s: 0.4
  pause_s: 0.6
  boundary_head: {layers: 2, thresh: 0.5, median: 3, min_len: 3}
seq2seq_v2:
  decoder: google/byt5-small
  label_smoothing: 0.2
  beam: 5
  max_len: 256
```

### 6.4 `configs/infer.yaml`
```yaml
profile: edge                 # edge | quality | edge-cpu
runtime: {engine: onnxruntime, precision: fp16, cpu_int8: true, threads: ${NUM_THREADS}}
stream: {capture_res: [1280, 720], fps: 12.5, ring_buffer: 48, tcls_every: 2, queue_size: 2}
video: {batch_frames: 64}
thresholds: {unknown_tau: 0.55, spot_tau: 0.6, pause_s: 0.6}
language:
  provider: openai
  model: ${OPENAI_MODEL}
  max_tokens: 80
  timeout_s: 3
  fallback: rules
  few_shot_file: configs/prompts/fewshot_th.json
  system_prompt_file: configs/prompts/system_th.txt
tts:
  provider: ${TTS_PROVIDER}
  voice: ${TTS_VOICE}
  style_map:
    neutral:   {rate: 1.0,  pitch: 0}
    happy:     {rate: 1.1,  pitch: 2}
    sad:       {rate: 0.9,  pitch: -2}
    angry:     {rate: 1.15, pitch: 1}
    surprised: {rate: 1.05, pitch: 3}
  persona: null
  cache_size: 256
ui: {overlay: true, show_topk: 5, warn_dark_face: 40, warn_hand_valid_rate: 0.5}
```

### 6.5 `requirements.txt` (แกน)
```
torch>=2.6 (CUDA build)  torchvision  timm  transformers  onnxruntime-gpu (หรือ onnxruntime)  rtmlib
av  opencv-python  numpy  pandas  pyarrow  pythainlp  jiwer  sacrebleu  scikit-learn
openai  edge-tts  sounddevice  python-dotenv  pyyaml  tqdm  psutil  rich
```
> หมายเหตุสภาพแวดล้อมปัจจุบัน: เครื่องมี torch 2.6 **CPU build** ต้องติดตั้ง CUDA build ใหม่ก่อนทำ cache/train (driver 610 รองรับ CUDA 12.x/13)

---

## 7. Notebooks — แต่ละตัวโชว์อะไร

| notebook | เรียก module | สิ่งที่เห็นชัด ๆ | ผลลัพธ์ที่ต้องได้ |
|---|---|---|---|
| `00_explore_raw_data` | data, utils | ตาราง/กราฟ distribution, contact sheets, caption band, chroma ref, VTT hits | ค่าคงที่ใน `data.yaml`, manifest v0 |
| `01_articulators_and_cache` | articulators, encoder, augment | ภาพ crop 3 stream + pose overlay 20 คลิป, IoU กับกล่องมือ, ความเร็ว/เฟรม, cache ที่เขียน (resume ได้) | cache ทุกเวอร์ชัน, `valid_rate` ต่อชุด |
| `02_prepare_datasets` | data | lexicon, splits, sampling weights, weak pairs, Bench loader + sha256 guard | manifests ทั้งหมด |
| `03_ssl_pretrain` | ssl | loss curves, prototype entropy, kNN R@1 ทุก 10 epoch, attention viz | `checkpoints/ssl/*` |
| `04_train_heads_islr_face` | heads | ISLR training, calibration curve ของ unknown, enrollment demo, face head vs teacher, NMM rules demo | prototypes, `checkpoints/islr, face` |
| `05_continuous_spotting_llm` | spotting, language | score timeline plot บนคลิป Big Sign, gloss timeline, LLM prompt/response, ablation re-ranker | boundary head, prompt files |
| `06_evaluation` | evalkit | ทุกตารางใน §5.5, T01–T14 pass/fail, error analysis | `reports/eval_v1.md` |
| `07_tts` | tts | provider เทียบกัน, style map, latency, ตัวอย่างเสียงต่ออารมณ์ | การตั้งค่า tts ใน infer.yaml |
| `08_inference_realworld` | pipeline | รัน .mp4 ของคุณ → result.json/srt/wav/overlay; webcam live cell; ตาราง latency/RAM จริง | demo และตัวเลข T11 |

---

## 8. แผนงานเป็นเฟส (Roadmap) + จุดตัดสินใจ

| เฟส | ระยะ | ส่งมอบ | Go/No-go |
|---|---|---|---|
| P0 Setup | 2–3 วัน | env CUDA, DINOv3 access, rtmlib ทำงาน, `00` เสร็จ | ค่าคงที่ 3 ตัวได้; contact sheet ยืนยันข้อสังเกต §2 |
| P1 Articulators + Cache | 1 สัปดาห์ (รวมรันข้ามคืน) | `01` ครบ, cache base+aug | valid-rate ≥ 0.9 (TTRS), ≥ 0.8 (YouTube), signer selection ≥ 0.95 บนแนวตั้ง |
| P2 SSL | 1–2 สัปดาห์ | `03` ครบ 3+1 stream, Stage 1+2 | kNN R@1 บนคู่ข้าม signer > baseline mean-pool DINOv3 อย่างน้อย +10 pt |
| P3 ISLR + Bench | 1–2 สัปดาห์ (อัด Bench ควบคู่) | `02`,`04`, Bench-Word | T01, T03, T06, T07, T13, T14 ผ่าน |
| P4 Continuous v1 + Face + LLM | 1–2 สัปดาห์ | `05`, `04`(face) | T08–T10 ผ่าน (ถ้า WER > 0.45 → เก็บ Bench-Sent เพิ่ม/เปิด re-ranker) |
| P5 TTS + Real-time | 1 สัปดาห์ | `07`,`08`, `scripts/cli.py` | T11, T12 ผ่าน |
| P6 (option) v2 seq2seq / NADT licence / ข้อมูลเพิ่ม | ต่อเนื่อง | ByT5-small, self-training loop | ชนะ v1 บน Bench-Sent เท่านั้นจึงสลับ |

---

## 9. ความเสี่ยงและทางออก

| ความเสี่ยง | สัญญาณ | ทางออกที่เตรียมไว้ |
|---|---|---|
| singleton lexeme ทำให้ metric space ไม่ทั่วถึง | R@1 ดีเฉพาะ 398 lexeme ที่มีหลายคลิป | mining จาก VTT, enrollment จากผู้ใช้, ให้ LLM ใช้ top-5 + บริบท |
| signer น้อย (≈6 คน) → overfit | T14 fail (Bench ≪ TTRS) | chroma-key bg + mirror + Bench-driven early stopping; LoRA แทน full FT |
| crop จาก wrist ไม่แม่นเท่าที่ hand detector | IoU < 0.6 ใน `01` | สลับเป็น YOLOv8n-hand + ByteTrack ตาม paper (interface เดิม) |
| DINOv3 license/gated | โหลดไม่ได้ | DINOv2 fallback (config 1 บรรทัด) — คาดคุณภาพต่ำลงเล็กน้อย |
| caption Big Sign เหลื่อมเวลามาก | weak pairs ใช้ไม่ได้ | v1 ไม่พึ่ง pairs; เก็บ pairs จากผลระบบ + ยืนยันโดยคน |
| CPU-only ช้ากว่าเป้า | p95 > 120 ms | stride 8 fps, int8, ViT-S ที่ 160 px input (ทดสอบผลต่อ R@1) |
| ลิขสิทธิ์ข้อมูล | เผยแพร่โมเดล | ใช้ภายใน/วิจัย; ขอ licence NADT (แหล่ง word-level ที่ดีที่สุด) |
| Face impression ตีความไวยากรณ์เป็นอารมณ์ | "โกรธ" ทุกครั้งที่ขมวดคิ้วถามคำถาม | แยก NMM ก่อน affect เสมอ; affect รายงานเมื่อ NMM ไม่ active |

---

## 10. อ้างอิงหลัก
- SignDino: Self-Supervised Sign Language Representation Learning via Temporal-Axis Self-Distillation — arXiv:2609.06296 (สถาปัตยกรรม 3 stream, frozen DINOv3, temporal DINO/iBOT/KoLeo/Gram, LoRA rank-1 สำหรับ ISLR, ByT5 สำหรับ SLT)
- SHuBERT (arXiv:2411.16765) และ SignMusketeers (arXiv:2406.06907) — multi-stream SSL / efficient SLT ที่ SignDino ต่อยอด
- DINOv3 / DINOv2 (Meta) — frozen image backbones
- rtmlib / RTMPose — pose ONNX สำหรับ articulator crops
- ข้อมูล: `README.md`, `metadata/*.csv`, `reports/collection_report.pdf` ในโฟลเดอร์นี้
