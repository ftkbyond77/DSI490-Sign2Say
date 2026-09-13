# ThaiSLM Blueprint (v2.1)
## Thai Sign Language → Text → Speech, built on SignDINO-style self-supervised learning

> **สถานะเอกสาร:** ฉบับสมบูรณ์ในตัวเอง (self-contained) เขียนหลังการทดลองจริงบนเครื่องนี้และข้อมูลชุดนี้ วันที่ 2026-09-13
> **ป้ายกำกับตัวเลข:** **[วัดจริง]** = รันแล้วบนเครื่องนี้ · **[คำนวณ]** = คำนวณจากค่าที่วัดจริง · **[ประมาณการ]** = ยังไม่ได้วัด ต้องยืนยันใน notebook
> **เครื่องที่ใช้ทดลอง:** RTX 2050 4 GB · Ryzen 12 cores · RAM 16 GB · conda env `hugging` (torch 2.7.0+cu128)

---

## 0. Executive Summary

### 0.1 ระบบที่แนะนำ (หนึ่งภาพ)

```
 .mp4 / webcam
     │  decode 12.5 fps (native resolution)
     ▼
 [L1] Articulators ── YOLOX-tiny (ทุก 8 เฟรม) + RTMPose-s ──► signer lock ──► crops: L-hand · R-hand · face
     │                                                                        + pose (34-d) + ACTIVE mask
     ▼
 [L2] Frozen ViT (DINOv3/DINOv2-S) ต่อ crop ──► [CLS ‖ patch-mean]
     ▼
 [L2.5] Signer-Nuisance Projection (fixed, fitted on train signers) ◄── ไม่ต้อง calibrate ตอนใช้งาน
     ▼
 [L3] Temporal encoders × 4 stream (SignDINO SSL: DINO + iBOT + KoLeo + Gram)
     ▼
 [L4] ├─ ISLR prototype head (4.6k lexeme, one-shot enrollment, unknown rejection)
      ├─ Continuous spotting + boundary head ──► gloss timeline
      └─ Face: affect head + grammatical NMM (negation / question)
     ▼
 [L5] Language layer (OpenAI LLM: rerank top-k + compose Thai sentence; rules fallback)
     ▼
 [L6] TTS (edge-tts | OpenAI TTS | local) + SpeechStyle จาก Face Impression
```

### 0.2 การตัดสินใจหลัก และหลักฐาน

| # | ตัดสินใจ | หลักฐาน |
|---|---|---|
| D1 | **Frozen ViT + kNN ใช้เป็นระบบไม่ได้** ต้องมี temporal encoder ที่ train ด้วย SSL และ metric learning | frozen features ทุกแบบให้ cross-signer R@1 = 0.000–0.017 และ nearest neighbour เป็น signer เดียวกัน 78–84 % **[วัดจริง]** |
| D2 | **Active-segment detector** (ตัดช่วงมือพัก) เป็น layer มาตรฐาน | TTRS active ≈ 70 % ของเฟรม; trimming ยก R@10 จาก 0.034 เป็น 0.144 **[วัดจริง]** |
| D3 | **Signer-Nuisance Projection (NAP)** แบบ fixed ที่ fit จาก signer ใน train set แทนการลบค่าเฉลี่ยต่อ session | ลบค่าเฉลี่ยต่อคลิปหรือจากเฟรมพัก ได้ R@1 0.000; NAP d=2 ที่ fit จาก 51 คลิปที่ไม่อยู่ในชุดทดสอบ ได้ R@1 0.068 / R@5 0.186 / R@10 0.314 เท่ากับหรือดีกว่า oracle ที่รู้ signer **[วัดจริง]** |
| D4 | ใช้ **RGB crops + pose** ร่วมกัน ไม่ใช่อย่างใดอย่างหนึ่ง | pose-only ได้ R@1 0.059 / R@5 0.144 ใกล้เคียง RGB frozen แต่ต่างข้อมูลกัน **[วัดจริง]** |
| D5 | Word-level = **prototype retrieval** ไม่ใช่ N-class classifier | 4,201 จาก 4,599 lexeme มีคลิปเดียว |
| D6 | Continuous v1 = **spotting + LLM**; seq2seq เป็น v2 | caption เป็นเสียงพูดของผู้ประกาศ ไม่ใช่ gloss และเหลื่อมเวลา 1–3 s |
| D7 | Real-time CPU ใช้ **112 px**; GPU ใช้ 224 px ได้ | DINOv2-S CPU 10 ms/crop @112 เทียบ 44 ms @224 **[วัดจริง]** |
| D8 | crop จาก **ความละเอียดต้นฉบับ** | มือใน TTRS มีขนาดเพียง 100–170 px ที่ 854×480 **[วัดจริง]** |
| D9 | Signer lock ด้วย presence + wrist confidence | แยกล่ามออกจากภาพซ้อนและแขกรับเชิญในคลิป Big Sign ได้ถูก **[วัดจริง]** |
| D10 | **Standard Data Schema** + adapter ต่อแหล่ง | ข้อมูลมาจาก 4+ รูปแบบ (TTRS API JSON, YouTube JSON, VTT หลายภาษา, ไฟล์อัดเอง) |

> **ข้อควรระวังเชิงสถิติ:** การทดลอง retrieval ใช้ 118 queries บน gallery 218 คลิป ความต่าง R@1 ระดับ 0.03–0.05 คือต่างกัน 3–6 queries จึงอาจเป็น noise ข้อสรุปที่เชื่อถือได้คือทิศทาง (frozen ใช้ไม่ได้, trimming และ NAP ช่วย) ไม่ใช่ค่าตัวเลขแม่นยำ ต้องยืนยันซ้ำบน TTRS-pairs เต็ม 263 lexeme ใน `06`

---

## 1. เป้าหมายและเกณฑ์ผ่าน

**Input:** `.mp4` ประโยคภาษามือไทยจาก signer หลากหลาย และ webcam real-time
**Output:** gloss timeline + ประโยคไทย + Face Impression (อารมณ์ ความเข้ม คำถาม/ปฏิเสธ) + เสียงพูด

| เกณฑ์ | Baseline วันนี้ | เป้าหมาย v1 | ชุดวัด |
|---|---|---|---|
| Cross-signer word retrieval | R@1 0.068 / R@5 0.186 (frozen + trim + NAP) | **R@1 ≥ 0.60 / R@5 ≥ 0.85** | TTRS-pairs test |
| Word-level signer ใหม่ บน webcam (200 คำ) | — | R@1 ≥ 0.55 / R@5 ≥ 0.80 | Bench-Word |
| Continuous .mp4 (3–8 คำ) | — | Gloss WER ≤ 0.45 · chrF ≥ 40 | Bench-Sent |
| Face impression | — | acc ≥ 0.70 (5 คลาส) · negation F1 ≥ 0.75 | Bench-Face |
| Latency webcam | ดู §7 | gloss ≤ 400 ms หลังจบท่า · เสียง ≤ 1.5 s หลังจบประโยค | `08` |
| RAM (CPU mode) | — | ≤ 2.5 GB resident | `08` |

**กฎเหล็ก:** ตัวชี้วัดความสำเร็จหลักคือ **Bench** (signer ใหม่, กล้องใหม่) ไม่ใช่ TTRS held-out เพราะนั่นคืออาการ "test ดีแต่ใช้จริงไม่ได้" ที่เจอกับ MediaPipe

---

## 2. ข้อมูลที่มี

### 2.1 ภาพรวม

| ชุด | คลิป | ชม. | ลักษณะ | label |
|---|---|---|---|---|
| `word_level/ttrs_dictionary` | 5,043 | 6.2 | สตูดิโอฉากฟ้า เสื้อดำ ~6 signer · 854×480 (96 %) · 25 fps · 2–8 s · โครง พัก→ท่า→พัก · gloss ฝังแถบล่าง · โลโก้มุมขวาบน | gloss ต่อคลิป |
| `word_level/youtube_word` | 186 | 10.6 | บทเรียนยาว ครูพูดพร้อมทำท่า flashcard · fingerspelling ระยะใกล้ · หลายฉาก | title + speech VTT มี word timestamps |
| `sentence_level/youtube_sentence` | 57 | 2.6 | ประโยคสั้นถึงบทสนทนา (2.6 s – 10 นาที) | title + VTT 49 ไฟล์ |
| `continuous/youtube_bigsign` | 44 | 6.2 | ล่ามเต็มเฟรม 720p (39) · แนวตั้ง 360×640 มี 2 คน (5) · เพลง ≈ 40 % | speech VTT เหลื่อมจากล่าม |
| `continuous/youtube_continuous` | 1 | — | 720p 30 fps | title |
| `parliament`, `_quarantine` | 0 | — | ว่าง | — |

รวม 5,331 คลิป · 25.6 ชม. · ≈ 1.15 ล้านเฟรมที่ 12.5 fps

### 2.2 ข้อเท็จจริงที่กำหนดการออกแบบ

**TTRS**
- 4,747 label → **4,599 lexeme** หลังตัดวงเล็บแหล่งที่มา (เช่น "(ภาษามือโรงเรียนเศรษฐเสถียรฯ)" 661 คลิป, "(พจนานุกรมสารสนเทศภาษามือไทย)" 69, "(ภาษามือวิทยาลัยราชสุดา)" 66)
- clip ต่อ lexeme: 1 คลิป = 4,201 · 2 = 354 · 3 = 42 · 4 = 2
- **263 lexeme มีคลิปจาก signer ต่างคน** ใช้เป็นชุดทดสอบข้าม signer ได้ทันที
- signer ระบุชื่อ ~2,300 คลิป ชื่อเดียวกันเขียนหลายแบบ ("นายธนูเกียรติ ลอยประโคน" / "ธนูเกียรติ ลอยประโคน") และมีค่าทดสอบ "ทดสอบ" 2 คลิป
- 67 label เป็นตัวอักษรเดี่ยว (ก–ฮ, A–Z) = fingerspelling
- 136 หมวด · definition มี 95.8 % · synonyms 12.7 % · 3,944 คลิปไม่มีเสียง
- sidecar มี URL ของ variant 720p ทุกคลิป (ดาวน์โหลดไว้ที่ 480p)

**ค่าที่วัดจากภาพ [วัดจริง]**
- พื้นหลัง TTRS HSV: H = 100, S ≈ 191, V ≈ 200 (แคบมาก) · พื้นที่พื้นหลัง 78–84 %
- แถบ gloss อยู่ที่ 0.88–0.93 ของความสูง
- ขนาด crop จริง: TTRS มือ 100–170 px, หน้า 180–210 px · Big Sign มือ ≈ 250 px · YouTube lesson มือ ≈ 110 px
- สัดส่วนเฟรม active ของ TTRS: เฉลี่ย 0.70

**YouTube / VTT [วัดจริง]**
- VTT ที่ใช้ได้: `th-orig` 152 · `th` 163 · `en` 64 · อื่น ๆ 28 · manual subtitle เพียง 5 คลิป ที่เหลือเป็น auto-caption
- auto-caption ของ YouTube มี **cue ซ้ำแบบ rolling** (บรรทัดก่อนหน้าถูกพิมพ์ซ้ำใน cue ถัดไป) ต้อง dedupe
- คำใน caption ที่ตรง lexeme TTRS: word-lessons 9,490 hits · sentence 3,302 · Big Sign 11,130 · **รวม 658 lexeme ไม่ซ้ำ**
- ffprobe อ่านได้ทุกไฟล์ แต่ **บางคลิปสั้นมาก** (เช่น 2.6 s) การ seek ไปเกินความยาวจะได้เฟรมว่าง ต้องตรวจความยาวก่อน

---

## 3. Standard Data Schema

ทุกแหล่งข้อมูลถูกแปลงเป็น **canonical records ชุดเดียว** ผ่าน **adapter ต่อแหล่ง** ทุก module หลังจากขั้น ingest รู้จักเพียง schema นี้

### 3.1 หลักการ

1. **Source-agnostic หลัง ingest:** ไม่มี module ใดหลัง `data.py` อ่าน JSON หรือ VTT ดิบ
2. **เวลาเก็บสองแบบเสมอ:** วินาที (float) และ frame index ที่ 12.5 fps เพื่อตัดปัญหา fps 24/25/30/50/60
3. **Text normalization ตายตัว:** Unicode NFC → ลบ zero-width และ whitespace ซ้ำ → เลขไทยเป็นอารบิก → `pythainlp.util.normalize` → วงเล็บท้ายคำย้ายไปเป็น `variant_tag`
4. **Provenance และ license ติดทุก record** ใช้กรองตอน train และตอนเผยแพร่
5. **ทุกไฟล์มี `schema_version`** และถูก validate ด้วย `pandera` ก่อนเขียน record ที่ไม่ผ่านไปอยู่ใน `_rejects.parquet` พร้อมเหตุผล
6. **Idempotent:** `clip_id` คงที่ รันซ้ำได้ผลเดิม
7. **Dedup ด้วย sha256:** ถ้าไฟล์เดียวกันมาจากสองแหล่ง เก็บ record เดียว และรวม provenance เป็น list

### 3.2 Records

ทั้งหมดเก็บเป็น parquet ใน `artifacts/manifests/` ยกเว้น TrackFile ที่เป็น npz ต่อคลิป

**ClipRecord** · หนึ่งแถวต่อไฟล์วิดีโอ

| field | type | ตัวอย่าง / หมายเหตุ |
|---|---|---|
| `clip_id` | str | `ttrs:008Ryz` · `yt:0VVs3FTqieM` · `bench:3f9a2c1d` |
| `source` | enum | ttrs · youtube · bench · nadt · other |
| `level` | enum | word · sentence · continuous |
| `media_path` | str | relative to project root |
| `media_sha256` | str | |
| `duration_s`, `fps_native` | float | |
| `width`, `height` | int | native |
| `vcodec` | str | h264 · vp9 · av1 |
| `has_audio` | bool | |
| `layout` | enum | full · inset · vertical_split · closeup · multi_person |
| `signer_zone` | list[4] float | normalized เช่น Big Sign แนวตั้ง `[0, 0.45, 1, 1]` |
| `mask_regions` | list[list[4]] | แถบ caption, โลโก้ |
| `chroma` | struct \| null | `{hsv_lo, hsv_hi}` |
| `title`, `description` | str | normalized |
| `provenance` | list[struct] | `{url, channel, fetched_at, adapter}` |
| `license_status` | enum | research_only · licensed · unknown |
| `domain_tags` | list | studio · lesson · broadcast · song · vlog · webcam |
| `quality_flags` | list | short · blurry · two_people · no_hands · low_light · decode_warning |
| `split` | enum | ssl_only · train · val · test · bench · excluded |
| `schema_version` | str | `2.1` |

**SegmentRecord** · ช่วงเวลาที่มีข้อความหรือ gloss กำกับ

| field | type | หมายเหตุ |
|---|---|---|
| `seg_id`, `clip_id` | str | |
| `t0_s`, `t1_s`, `f0`, `f1` | float / int | f = frame index ที่ 12.5 fps |
| `text_th` | str \| null | ประโยคไทย |
| `glosses` | list[str] | ลำดับ lemma ถ้ามี |
| `align_type` | enum | curated_gloss · caption_shifted · mined · manual · model |
| `align_conf` | float | 0..1 |
| `lag_applied_s` | float | เช่น +1.5 สำหรับ Big Sign |
| `text_lang` | enum | th · en |
| `text_origin` | enum | speech_caption · manual_caption · gloss · human |

**LexemeRecord** · พจนานุกรมของระบบ

| field | type | หมายเหตุ |
|---|---|---|
| `lexeme_id` | str | `L000123` |
| `lemma_th` | str | normalized |
| `surface_forms` | list[str] | label ดิบทุกแบบที่ map มาที่นี่ |
| `variant_tag` | str \| null | เศรษฐเสถียร · ราชสุดา · พจนานุกรมสารสนเทศ |
| `sign_variant_id` | str | `L000123#setsatian` ท่าต่างโรงเรียนคือคนละ prototype แต่ข้อความเดียวกัน |
| `category_th`, `pos_th`, `definition_th`, `synonyms_th` | str | |
| `is_fingerspelling` | bool | |
| `n_instances`, `n_signers` | int | |

**InstanceRecord** · หน่วยฝึกของ ISLR หนึ่งท่าต่อหนึ่งแถว

| field | type | หมายเหตุ |
|---|---|---|
| `instance_id`, `lexeme_id`, `sign_variant_id`, `clip_id` | str | |
| `seg_id` | str \| null | |
| `signer_id` | str | |
| `active_t0_s`, `active_t1_s`, `active_f0`, `active_f1` | float / int | หลัง rest-pose trimming |
| `source_type` | enum | ttrs · mined · bench · enrolled · manual |
| `weight` | float | ttrs 1.0 · mined 0.5 · enrolled 1.0 |
| `mining_round`, `mining_score` | int / float | |

**SignerRecord**

| field | type | หมายเหตุ |
|---|---|---|
| `signer_id` | str | `S_ttrs_thanukiat` · `S_yt_<channel>_<cluster>` · `S_bench_01` |
| `display_name`, `gender`, `handedness` | str \| null | |
| `sources` | list | |
| `face_centroid` | vector | จาก face-crop embedding ใช้รวมชื่อซ้ำและตรวจ split |
| `is_test_only` | bool | signer ที่ห้ามเข้า train |

**TrackFile** · `artifacts/tracks/<clip_id>.npz`

| key | shape | หมายเหตุ |
|---|---|---|
| `frame_idx`, `t_s` | [T] | ที่ 12.5 fps |
| `pose` | [T,17,3] | x, y (pixel native), conf |
| `pose_norm` | [T,34] | กึ่งกลางไหล่ หารด้วยความกว้างไหล่ |
| `box_hand_l`, `box_hand_r`, `box_face`, `box_body` | [T,4] | pixel native |
| `valid` | [T,3] | hand_l, hand_r, face |
| `active` | [T] | rest-pose trimmer |
| `signer_track_id` | [T] | |
| `extractor` | json | `{pose, det, det_every, hand_fallback, version}` |

**CacheIndex** · `artifacts/cache/<backbone>_<res>/index.parquet`

| field | หมายเหตุ |
|---|---|
| `clip_id`, `version`, `stream`, `res` | version = base · aug_flip · aug_photo · aug_bg1 · aug_bg2 |
| `start_row`, `n_rows`, `dim`, `dtype` | อ้างอิง memmap `<stream>.f16.bin` |

### 3.3 Adapters (`modules/data.py`)

| Adapter | Input | กฎเฉพาะ |
|---|---|---|
| `TTRSAdapter` | `ttrs_*.json` + mp4 + `metadata_master.csv` | label → lemma + variant_tag · ชื่อ signer ตัดคำนำหน้าแล้ว map เป็น signer_id · ตัดค่า "ทดสอบ" · layout = full · chroma on · mask = caption band + โลโก้ · หนึ่ง Segment ต่อคลิป align_type = curated_gloss |
| `YouTubeAdapter` | `yt_*.json` + `*.vtt` | ลำดับ track: `th-orig` > `th` > `th-th` · parse word-level timestamps `<00:00:04.200><c>คำ</c>` · **dedupe rolling cues** · Big Sign ใส่ lag +1.5 s align_type = caption_shifted · word lessons เป็น candidate ของ mining · layout และ signer_zone จาก `00` · title มีคำว่า "เพลง" หรือ "ภาษาเพลง" → domain_tag song |
| `BenchAdapter` | โฟลเดอร์อัดเอง + `labels.csv` | split = bench · signer_id จากผู้ถูกอัด · เพิ่ม sha256 ลง blocklist |
| `GenericFolderAdapter` | mp4 + csv `(file, text, glosses, t0, t1, signer)` | สำหรับแหล่งใหม่ เช่น NADT ถ้าได้ licence ไม่ต้องเขียนโค้ดเพิ่ม |

**ลำดับ transform ที่ทุกแหล่งผ่านเหมือนกัน**

```
raw files ─► Adapter ─► ClipRecord/SegmentRecord (validate) ─► decode 12.5 fps native
          ─► TrackFile (pose, boxes, active) ─► InstanceRecord ─► embedding cache (CacheIndex)
```

---

## 4. File Structure

```
SLM-Labs-SignDINO/
├─ raw_data/  metadata/  reports/  logs/  state/     (เดิม · read-only)
├─ ThaiSLM_BLUEPRINT.md
├─ .env  .env.example  environment.yml  requirements.txt
├─ configs/
│   ├─ data.yaml  train.yaml  infer.yaml  compute.yaml
│   └─ prompts/  system_th.txt  fewshot_th.json
├─ modules/
│   ├─ schema.py        pandera schemas (§3) + normalizers ของ text / time / id
│   ├─ data.py          adapters → manifests · splits · samplers · sha256 guard
│   ├─ articulators.py  detector cadence · pose · signer lock · crops (native res) · active mask · hand fallback
│   ├─ encoder.py       frozen ViT (torch / ONNX) · cache writer/reader (memmap) · 112/224 profiles
│   ├─ normalize.py     Signer-Nuisance Projection (fit / apply / save)
│   ├─ augment.py       chroma-key bg · mirror-swap · photometric · speed · stream dropout
│   ├─ ssl.py           temporal DINO + iBOT + KoLeo + Gram · multi-temporal-crop sampler
│   ├─ heads.py         ISLR prototype head (+LoRA, signer adversary) · boundary head · face affect · NMM
│   ├─ spotting.py      continuous timeline · NMS · pause segmentation
│   ├─ language.py      LLM rerank/compose · rules fallback
│   ├─ tts.py           TTSProvider · providers · SpeechStyle
│   ├─ pipeline.py      VideoPipeline (.mp4) · StreamPipeline (webcam)
│   ├─ evalkit.py       metrics · bench loaders · latency/RAM probes · leakage checks
│   └─ utils.py         logging · timers · seed · device selection
├─ notebooks/
│   ├─ 00_explore_raw_data.ipynb
│   ├─ 01_schema_articulators_cache.ipynb
│   ├─ 02_prepare_datasets.ipynb
│   ├─ 03_ssl_pretrain.ipynb
│   ├─ 04_train_heads_islr_face.ipynb
│   ├─ 05_continuous_spotting_llm.ipynb
│   ├─ 06_evaluation.ipynb
│   ├─ 07_tts.ipynb
│   └─ 08_inference_realworld.ipynb
├─ scripts/   harvest.py (เดิม) · cli.py  (ingest | tracks | cache | fit-nap | train | eval | infer | record)
├─ artifacts/ manifests/ tracks/ cache/ nap/ checkpoints/ prototypes/ onnx/ bench/ reports/   (gitignored)
└─ tests/     test_schema.py · test_articulators.py · test_normalize.py · test_pipeline_smoke.py
```

หนึ่ง module ต่อหนึ่งหน้าที่ · notebook เรียก module เพื่อแสดงผลแต่ละขั้น ไม่มี logic ซ้ำใน notebook

---

## 5. Architecture Detail

| Layer | Component | ค่าเริ่มต้น | เหตุผล |
|---|---|---|---|
| L0 | Capture | 12.5 fps · native res · ring buffer 48 เฟรม | ครอบคลุม dynamics ของท่า · ลด compute ครึ่งหนึ่ง |
| L1 | Articulators | YOLOX-tiny @416 ทุก 8 เฟรม + RTMPose-s @256×192 ทุกเฟรม + box EMA 0.6 | ≈ 13 ms/เฟรม CPU **[คำนวณ]** · valid-rate 99.9–100 % **[วัดจริง]** |
| L1 | Active mask | wrist ยกสูงกว่าตำแหน่งพัก 0.15 × ความกว้างไหล่ หรือความเร็ว > 0.03 × ความกว้างไหล่ต่อเฟรม · dilate 1 | D2 |
| L2 | Frame encoder | DINOv3 ViT-S/16 (fallback DINOv2-S) · edge 112 px · quality 224 px · `[CLS ‖ patch-mean]` 768-d | D7 |
| L2.5 | Signer-Nuisance Projection | P = I − NᵀN · N = top-d singular vectors ของค่าเฉลี่ยต่อ signer · fit บน train split ต่อ stream · d เลือกบน val | D3 |
| L3 | Temporal encoders | 3 × (Linear 768→384 · 6 PreLN blocks · 6 heads · tCLS) + pose (68-d → 128 · 2 blocks) | SignDINO |
| L4a | ISLR head | concat tCLS → MLP → 512-d L2 · prototypes ต่อ sign_variant | D5 |
| L4b | Spotting | window 12–20 เฟรม stride 2 · boundary head 2 blocks | D6 |
| L4c | Face | affect head (distilled) + NMM rules/classifier | §6.4 D |
| L5 | Language | OpenAI LLM + rules fallback | §6.4 E |
| L6 | TTS | TTSProvider + SpeechStyle | §6.6 |

---

## 6. Pipeline Stages

### 6.1 Exploration & Inspect Raw Data · `00_explore_raw_data.ipynb`

| ขั้น | ทำอะไร | Output |
|---|---|---|
| E1 | distribution ต่อชุด: duration · resolution · fps · codec · has_audio | ตาราง + กราฟ |
| E2 | lexeme normalization preview · นับ singleton / multi / variant | `lexeme_stats.csv` |
| E3 | contact sheet 8 เฟรม × 12 คลิปต่อชุด (seek ไม่เกินความยาวคลิป) | ดู layout, ฉาก, จำนวนคน |
| E4 | ตำแหน่งแถบ gloss และโลโก้ของ TTRS | `caption_band_frac`, `logo_box` |
| E5 | HSV histogram พื้นหลัง TTRS 200 เฟรม | `chroma_hsv_lo/hi` |
| E6 | layout ของ Big Sign แนวตั้งและคลิปมีภาพซ้อน | `signer_zone` ต่อ layout |
| E7 | parse VTT + dedupe rolling cues + นับ lexeme hits | `vtt_lexeme_hits.parquet` |
| E8 | signer inventory: face-crop embedding clustering + รวมชื่อซ้ำ | SignerRecord ร่าง |
| E9 | **rest-pose profile:** สัดส่วน active ต่อชุด · ตำแหน่งมือพัก | threshold ของ active mask |
| E10 | **hand-size histogram (px native)** ต่อชุด | ตัดสิน 112 vs 224 และการดึง TTRS 720p |
| E11 | schema validation report ต่อ adapter | ผ่าน / reject พร้อมเหตุผล |

**เงื่อนไขก่อนไปต่อ:** ค่าคงที่ใน `data.yaml` ครบ · schema validate ผ่าน 100 % หรือ reject มีเหตุผลทุกแถว

### 6.2 Data Transformation & Feature Engineering · `01_schema_articulators_cache.ipynb`

**6.2.1 Ingest ตาม schema** (§3) → `clips.parquet`, `segments.parquet`, `lexicon.parquet`, `signers.parquet`

**6.2.2 Decode**
- ffmpeg/PyAV → 12.5 fps ที่ความละเอียดต้นฉบับ (resample 24/30/50/60 fps) · decode เร็ว 16–18× realtime **[วัดจริง]**
- TTRS: mask แถบ gloss (y ≥ 0.85) และโลโก้ **ก่อน** ส่งเข้า detector

**6.2.3 Articulators (`articulators.py`)**
- Detector cadence: YOLOX-tiny ทุก 8 เฟรม (30 ms) · RTMPose-s ทุกเฟรม (8–10 ms) **[วัดจริง]**
- **Signer lock:** candidate ต้อง presence ≥ 0.6 ใน window 2 s → score = 0.5·wrist_conf + 0.3·(wrist motion / shoulder width) + 0.2·in_signer_zone → lock 2 s มี hysteresis
- **Crops จาก native resolution:**
  - hand: center = wrist + 0.30·(wrist − elbow) · side = 1.1 · forearm · 1.4
  - face: center = mean(nose, eyes) · side = 2.2 · ear distance · 1.2
  - body: center = shoulder-mid + (0, 0.9 · shoulder) · side = 3.2 · shoulder
  - resize เป็น 112 (และ 224 สำหรับ quality) · pad สีเทาเมื่อเกินขอบ
- **Valid flag:** keypoint conf < 0.3 → invalid → learnable [MISSING] token ตอน train · mask ตอน inference
- **Hand fallback:** ถ้า elbow conf < 0.3 (close-up fingerspelling) ใช้ rtmlib `Hand` (RTMDet-nano) **เฉพาะตอน cache offline** เพราะใช้ 78–88 ms/เฟรม **[วัดจริง]**
- **Active mask:** rest level = median ของ 3 เฟรมแรกและท้าย (คลิป) หรือ running 10th percentile (stream)
- **Pose features:** 17 จุด normalize ด้วยไหล่ → 34-d + velocity 34-d = 68-d
- Output: TrackFile ต่อคลิป

**6.2.4 Frozen encoder + cache (`encoder.py`)**
- Backbone: `facebook/dinov3-vits16-pretrain-lvd1689m` (gated ต้อง accept license และ `transformers ≥ 4.56`) · fallback `facebook/dinov2-small` (ใช้ในการทดลองทั้งหมด)
- เก็บ `[CLS ‖ patch-mean]` fp16 ทั้ง `res112` และ `res224`
- ต้นทุนบน RTX 2050 **[คำนวณจากวัดจริง]**: 3.45 M crops ต่อ version → res112 ≈ 31 นาที · res224 ≈ 2.0 ชม. · ViT-B res224 ≈ 5.5 ชม.
- เขียนแบบ resumable ต่อคลิป

**6.2.5 Augmentation (`augment.py`)**

| Aug | ใช้กับ | ระดับ | เหตุผล |
|---|---|---|---|
| Chroma-key background replacement | TTRS | pixel ก่อน encode → cache `aug_bg1`, `aug_bg2` | ปิดช่องว่างระหว่างสตูดิโอกับบ้าน · HSV range H 95–105, S ≥ 117, V ≥ 157 **[วัดจริง]** · ภาพพื้นหลังจาก YouTube frames ที่ไม่มีคน + stock ฟรี |
| Photometric: brightness · contrast · JPEG · blur · down-up scale | ทุกชุด | pixel → `aug_photo` | จำลอง webcam คุณภาพต่ำ |
| Mirror-swap | ทุกชุด | flip crop แล้ว encode → `aug_flip` · สลับ stream L↔R · flip pose x | signer ถนัดซ้าย · CLS ไม่ flip-invariant จึงต้อง encode ใหม่ |
| Speed 0.8–1.25× · frame drop | ทุกชุด | cache index | ความเร็วต่างคน |
| Crop jitter ±10 % · scale ±15 % | ทุกชุด | ตอน encode aug | ความคลาดเคลื่อนของ detector |
| Stream dropout 10 % | ทุกชุด | cache | ทน occlusion และมือออกนอกเฟรม |

**6.2.6 Signer-Nuisance Projection (`normalize.py`)**
- Fit: สำหรับแต่ละ stream ใช้เฟรม active ของ **train split เท่านั้น** → ค่าเฉลี่ยต่อ signer ลบค่าเฉลี่ยรวม → SVD → เก็บ N (d แถว) → P = I − NᵀN
- d เลือกจาก val TTRS-pairs ในช่วง 1 ถึง (จำนวน signer − 1) · ตอนทดลองกับ 3 signer ได้ d = 2 **[วัดจริง]**
- Apply: ทั้งตอน SSL, ตอน train head และตอน inference · เป็น matmul คงที่ 768×768 ไม่เพิ่ม latency ที่วัดได้
- เพิ่ม signer ด้วย YouTube face clusters และ Bench-train เพื่อให้ subspace ครอบคลุมขึ้น
- **ห้ามใช้** การลบค่าเฉลี่ยต่อคลิป/session หรือค่าเฉลี่ยจากเฟรมพัก เพราะให้ R@1 0.000 **[วัดจริง]**

**6.2.7 Weak-label mining**
- candidate จาก VTT: คำที่ตรง lemma → window [t − 0.5, t + 2.0] s (lesson) หรือ lag +1.5 s (Big Sign)
- ยืนยันหลังมี ISLR รอบแรก: cosine ≥ τ_mine และช่วง active ยาว 0.4–2.5 s · สูงสุด 5 instance ต่อ lexeme · 2 รอบ · เก็บ source สำหรับ audit

### 6.3 Data Preparation Before Modeling · `02_prepare_datasets.ipynb`

**Splits** · แยกด้วย `signer_id` และ `channel` ไม่สุ่มคลิป

| งาน | train | val | test |
|---|---|---|---|
| SSL | ทุกคลิป ยกเว้นคลิปใน test และ bench | — | — |
| NAP | train instances เท่านั้น | เลือก d | — |
| ISLR | TTRS ที่เหลือ + mined | 50 % ของ TTRS-pairs (131 lexeme) | 50 % ของ TTRS-pairs (132 lexeme) + Bench-Word |
| Face | pseudo-label ทุกชุด | ตรวจด้วยคน 300 เฟรม | Bench-Face |
| Continuous | Big Sign + sentence (weak pairs) ยกเว้น 6 คลิป | 3 คลิป ต่าง channel | 3 คลิป + Bench-Sent |

**SSL sampler**
- chunk 96 เฟรมสำหรับคลิปยาว · ทั้งคลิปสำหรับ TTRS · **TTRS ใช้เฉพาะเฟรม active**
- 2 global crops U{32..64} เฟรม · 8 local crops U{6..16} เฟรม
- น้ำหนัก: TTRS 0.35 · youtube_word 0.30 · sentence 0.10 · bigsign 0.25 (เพลง × 0.5)

**Lexeme vs sign variant:** prototype แยกตาม `sign_variant_id` · output ข้อความใช้ `lemma_th`

**ThaiSLM-Bench** (ต้องอัดเอง · ห้ามเข้า train ตรวจด้วย sha256)
- **Bench-Word:** 200 lexeme (ครอบคลุมหมวด + 30 คำใช้บ่อย + 20 fingerspelling) × ≥ 3 signer นอก TTRS × 2 สภาพแสง/ฉาก · webcam 720p → ~1,200 คลิป
- **Bench-Sent:** 60 ประโยค (3–8 คำ) × 3 signer · reference ทั้ง gloss และประโยคไทย
- **Bench-Face:** ประโยคจาก Bench-Sent ใน 5 อารมณ์ (กลาง ดีใจ เศร้า โกรธ ประหลาดใจ) + yes/no question + negation
- **Bench-Enroll:** 20 คำใหม่ × 1 คลิป enroll + 3 คลิปทดสอบจาก signer อื่น
- อัดผ่าน `cli.py record` ซึ่งแสดงคำและบันทึกอัตโนมัติ

### 6.4 Modeling Process

#### A. Temporal SSL pre-training · `03_ssl_pretrain.ipynb` · `ssl.py`

- Input ต่อเฟรม: `[CLS ‖ patch-mean]` → **NAP** → Linear 768→384
- Student/Teacher ต่อ stream: 6 PreLN blocks · d = 384 · 6 heads · tCLS · sinusoidal position ตาม frame index · validity mask ใน attention
- Heads: DINO 384→2048→256 → K = 4096 prototypes (Sinkhorn-Knopp) · iBOT head ต่อเฟรมใช้ K เดียวกัน
- Loss Stage 1: `L = L_DINO + L_iBOT + 0.1·L_KoLeo`
  - iBOT: mask เฉพาะเฟรม 30–50 %
  - τ_teacher 0.04 → 0.07 ในช่วง 30 % แรก · τ_student 0.1 · EMA 0.994 → 1.0
- Loss Stage 2: `+ 2.0·L_Gram` (anchor กับ teacher ที่ freeze หลัง Stage 1)
- Pose stream: d = 128 · 2 blocks · loss เดียวกัน
- ลำดับการ train: มือขวา → มือซ้าย → หน้า → pose
- งบ: train step 640 sequences × 48 เฟรม = 0.5 s ที่ 3.6 GB **[วัดจริง]** → batch 40 chunks × 10 crops · ~300 steps/epoch · Stage 1 60 ep + Stage 2 20 ep ≈ 2.5 ชม./stream บนเครื่องนี้ **[ประมาณการ]**
- Sanity ทุก 10 epoch:
  1. kNN บน TTRS-pairs val ต้องสูงกว่า baseline R@1 0.068 / R@5 0.186 ชัดเจน
  2. prototype usage entropy ไม่ collapse
  3. ถ้าไม่ดีขึ้นหลัง 20 epoch ให้ตรวจ NAP และ active-only sampling ก่อนปรับอย่างอื่น

#### B. ISLR prototype head · `04_train_heads_islr_face.ipynb` · `heads.py`

- z = LayerNorm(concat tCLS × 4) → MLP 1280→512 → L2
- Loss: SupCon (τ 0.07) + ArcFace margin 0.2 scale 30 + **signer adversary** (gradient reversal, λ 0.1, ใช้ signer_id)
- LoRA rank 4 บน q, v ของ temporal encoders · 40 epochs
- Positives: aug views (bg, flip, photo, speed) + คลิปอื่นของ sign_variant เดียวกัน + mined (weight 0.5)
- Hard negatives จาก confusion groups → `confusion_groups.json` ส่งให้ language layer
- Prototype = ค่าเฉลี่ย z ของทุก instance ต่อ `sign_variant_id` → `artifacts/prototypes/lexeme_protos.npz`
- Inference: cosine top-k → รวมเป็น lemma · **unknown** ถ้า top1 < τ_unknown หรือ top1 − top2 < δ (calibrate บน val)
- **Enrollment:** `enroll(clip, lemma)` เพิ่มหรืออัปเดต prototype ทันทีโดยไม่ retrain

#### C. Continuous recognition · `05_continuous_spotting_llm.ipynb` · `spotting.py`

**v1 · spotting + LLM**
1. Sliding window 12–20 เฟรม stride 2 → z_t → cosine กับ prototypes ทั้งหมด (matmul เดียว)
2. Score timeline ต่อ lexeme → temporal NMS (peak > τ_spot, ห่าง ≥ 0.4 s) → `[(lemma, t0, t1, score, top5)]`
3. **Boundary head** (2 blocks บน fused frame features → sign / transition / idle) · pseudo-label จาก active mask ของ TTRS · post-process threshold 0.5, median 3, ความยาวขั้นต่ำ 3 เฟรม
4. Pause segmentation: idle > 0.6 s = จบประโยค → ส่ง gloss ของประโยคให้ language layer

**v2 · seq2seq (เปิดใช้เมื่อชนะ v1 บน Bench-Sent เท่านั้น)**
- fused frame features → ByT5-small decoder · label smoothing 0.2 · beam 5
- train บน weak pairs + ประโยคที่ v1 สร้างและคนยืนยัน (human-in-the-loop)

#### D. Face Impression · `04` · `heads.py`

แยก **ไวยากรณ์บนใบหน้า** ออกจาก **อารมณ์** เพราะสีหน้าในภาษามือส่วนใหญ่เป็นไวยากรณ์

| ชั้น | สัญญาณ | วิธี | Output |
|---|---|---|---|
| NMM (ไวยากรณ์) | pose (nose, eyes, ears) + face features | head-shake: nose-x oscillation ≥ 2 รอบใน 1 s → negation · head-nod ≥ 2 รอบ → affirmation · brow-raise proxy (eye–nose ratio) + hold → yes/no question · head-tilt → wh-question · เริ่มจาก rules แล้ว train classifier เล็กบน Bench-Face-train | flags ต่อ segment |
| Affect (อารมณ์) | face-stream temporal features | distill จาก FER teacher (เช่น HSEmotion `enet_b0_8_best_afew` ตรวจ license ก่อนใช้) รันบน face crop ตอนสร้าง cache → head 2 blocks window 12 เฟรม · loss KL · EMA 0.8 s | emotion 7 คลาส · valence · arousal · intensity |
| Fusion | NMM + affect + gloss timeline | **ถ้า NMM active ให้ลดน้ำหนัก affect** · ส่ง tags เข้า LLM · affect → SpeechStyle | ประโยคที่มี "ไม่" / "ไหม" ถูกตำแหน่ง · เสียงมีอารมณ์ |

#### E. Language layer · `language.py`

- Provider: OpenAI `gpt-4.1-mini` (ค่าเริ่มต้น เปลี่ยนได้ใน config) · `max_tokens` 80 · timeout 3 s · fallback rules
- Input JSON:
  ```json
  {"glosses":[{"lemma":"กิน","conf":0.82,"alt":["อาหาร","หิว"],"t0":1.2,"t1":1.9}],
   "nmm":{"negation":false,"question":true},
   "affect":{"emotion":"happy","intensity":0.6},
   "context":["ประโยคก่อนหน้า 1","ประโยคก่อนหน้า 2"]}
  ```
- Output JSON: `{"sentence_th": "...", "style_hint": "happy", "uncertain": false}`
- หลักการ prompt:
  1. ภาษามือไทยเรียงแบบ topic-comment และไม่มีคำบอกกาลครบ ให้เรียบเรียงเป็นภาษาไทยพูดที่เป็นธรรมชาติ
  2. **ห้ามเพิ่มเนื้อหาที่ไม่มีใน gloss**
  3. ถ้า conf ต่ำ เลือกจาก `alt` ที่ทำให้ประโยคสมเหตุสมผล และตั้ง `uncertain: true`
  4. เติม "ไหม/หรือเปล่า" เมื่อ question · "ไม่" เมื่อ negation
  5. ตอบ JSON เท่านั้น
- Few-shot 8 ตัวอย่างจาก Bench-Sent train ใน `configs/prompts/fewshot_th.json`
- Ablation ใน `05`: LLM เป็น **reranker ของ top-k** ก่อนเรียบเรียง ลด WER ได้หรือไม่

### 6.5 Test Cases & Evaluation · `06_evaluation.ipynb` · `evalkit.py`

**Metrics ต่อชั้น**

| ชั้น | Metric | ชุด |
|---|---|---|
| Schema | round-trip · reject rate ต่อ adapter | ทุกแหล่ง |
| Articulators | crop IoU กับกล่องมือที่ label เอง 200 เฟรม · valid-rate ต่อชุด · signer lock accuracy | manual set · Big Sign 2 คน 30 คลิป |
| Active mask | frame F1 เทียบ label มือ 50 คลิป · onset delay | manual set |
| SSL | kNN R@1/5 บน TTRS-pairs val · linear probe 136 หมวด | TTRS |
| ISLR | R@1/5/10 · macro per-class · unknown AUROC · **แยกต่อ signer และต่อฉาก** | TTRS-pairs test · Bench-Word |
| Spotting | interval mIoU · precision/recall ที่ τ_spot · gloss WER (pythainlp tokenizer) | Bench-Sent |
| Sentence | chrF (หลัก) · BLEU · human adequacy 1–5 บน 60 ประโยค | Bench-Sent |
| Face | accuracy/F1 5 คลาส · NMM F1 · onset latency | Bench-Face |
| Runtime | p50/p95 ms ต่อ stage · RAM peak · VRAM · end-to-end delay | `08` CPU และ GPU |

**Test cases**

| ID | กรณี | Input | เกณฑ์ผ่าน |
|---|---|---|---|
| T01 | คำเดี่ยว signer ใหม่ แสงปกติ | Bench-Word | R@1 ≥ 0.55 |
| T02 | แสงน้อย / ย้อนแสง | Bench-Word low-light | R@1 ลดไม่เกิน 0.10 จาก T01 |
| T03 | signer ถนัดซ้าย | Bench-Word | R@1 ลดไม่เกิน 0.05 |
| T04 | ฉากหลังรก มีคนเดินผ่าน | Bench-Word busy bg | signer lock ≥ 0.95 |
| T05 | กล้องไกล (ครึ่งตัวเล็ก) และใกล้ (fingerspelling) | 2 ชุด | valid-rate ≥ 0.9 |
| T06 | ท่าที่ไม่อยู่ในคลัง | 50 คลิป | rejection ≥ 0.8 ที่ FPR 0.1 |
| T07 | Enrollment คลิปเดียว | Bench-Enroll | R@5 ≥ 0.70 |
| T08 | ประโยค 3–8 คำ .mp4 | Bench-Sent | WER ≤ 0.45 · chrF ≥ 40 |
| T09 | ประโยคปฏิเสธ / คำถาม | Bench-Sent subset | NMM F1 ≥ 0.75 |
| T10 | อารมณ์ 5 แบบ | Bench-Face | acc ≥ 0.70 |
| T11 | Webcam ต่อเนื่อง 3 นาที | live | p95 ≤ 80 ms/เฟรม (CPU @112) · RAM ≤ 2.5 GB คงที่ · ไม่ drop เฟรมเกิน 2 % |
| T12 | LLM หรือเน็ตล่ม | ปิดเน็ต | ได้ประโยคจาก rules + TTS local/cache · ไม่ crash |
| T13 | Leakage guard | TTRS ที่ mask vs ไม่ mask แถบ gloss | R@1 ต่างกันไม่เกิน 0.02 |
| T14 | ช่องว่าง studio vs จริง | TTRS-pairs test vs Bench-Word | Bench ≥ 0.6 × TTRS |
| T15 | NAP ablation | Bench-Word | เปิด NAP ดีกว่า ≥ +3 pt R@1 · ถ้าไม่ต่างแปลว่า encoder เรียน invariance ได้เอง ให้คงไว้เพราะต้นทุนเกือบศูนย์ |
| T16 | Active mask บน webcam ที่ไม่เริ่มจากท่าพัก | live 20 คลิป | false active ≤ 10 % ในช่วงนิ่ง · onset delay ≤ 160 ms |
| T17 | Schema round-trip ทุก adapter | ทุกแหล่ง | 100 % · reject ทุกแถวมีเหตุผล |
| T18 | 112 px vs 224 px | Bench-Word (GPU) | ถ้าต่างไม่เกิน 3 pt ใช้ 112 ทุกโหมด |
| T19 | Multi-person .mp4 | Big Sign แนวตั้ง + คลิปครูกับนักเรียน | signer lock ≥ 0.95 · ไม่สลับคนระหว่างประโยค |
| T20 | คลิปสั้นมาก / seek เกินความยาว | คลิป < 3 s | pipeline ไม่ error และคืนผลว่างอย่างมีเหตุผล |

**Error analysis มาตรฐาน:** confusion ต่อ handshape group · ต่อ signer · ต่อ stream dropout (ปิด face หรือมือซ้ายแล้วดูว่าอะไรพัง) → ใช้เลือกข้อมูลที่จะเก็บเพิ่ม

### 6.6 TTS · `07_tts.ipynb` · `tts.py`

```python
@dataclass
class SpeechStyle:
    emotion: str = "neutral"        # neutral | happy | sad | angry | surprised
    intensity: float = 0.0          # 0..1 → map เป็น rate / pitch / volume
    rate: float = 1.0               # 0.7..1.4
    pitch_semitones: float = 0.0
    voice: str = "th-TH-PremwadeeNeural"
    persona: str | None = None      # ต่อยอดภายหลัง: สำเนียง / บุคลิก เช่น "ทางการ", "อีสาน"

class TTSProvider(Protocol):
    def synthesize(self, text: str, style: SpeechStyle) -> Iterator[bytes]: ...
```

| Provider | จุดเด่น | ใช้เมื่อ |
|---|---|---|
| `EdgeTTSProvider` | ฟรี · เสียงไทย Premwadee / Niwat · ปรับ rate/pitch ได้ · latency 300–600 ms **[ประมาณการ]** | ค่าเริ่มต้น real-time |
| `OpenAITTSProvider` (`gpt-4o-mini-tts`) | สั่งอารมณ์และสไตล์ด้วยข้อความ `instructions` | โหมดคุณภาพ · ทดลอง style/persona |
| `LocalTTSProvider` (VITS / F5-TTS ไทย ผ่าน ONNX) | ออฟไลน์ · privacy | เพิ่มภายหลังผ่าน interface เดิม |

- affect → style ผ่าน `style_map` ใน `infer.yaml`
- Streaming: แบ่งตามประโยค → synth เป็น chunk → เล่นผ่าน `sounddevice` ทันที
- LRU cache เสียงของประโยคที่ซ้ำ
- `persona` ส่งเป็น instruction ให้ provider ที่รองรับ · provider ที่ไม่รองรับจะข้าม

### 6.7 Inference with Real-World Data · `08_inference_realworld.ipynb` · `pipeline.py`

**โหมด .mp4 · `VideoPipeline.run(path, profile="quality")`**
1. decode 12.5 fps → TrackFile → crops → ViT แบบ batch 128 → NAP → temporal encoders บน chunk ยาว → spotting → LLM ต่อประโยค → TTS
2. Output: `result.json` (gloss timeline, sentences, affect timeline, NMM flags) · `subtitle.srt` · `audio.wav` · วิดีโอ overlay (เลือกได้)
3. Hand fallback เปิดได้ในโหมดนี้

**โหมด webcam · `StreamPipeline.start(profile="edge")`**

```
T1 capture (OpenCV 720p) ──queue(2), ทิ้งเฟรมเก่า──► T2 perception ──gloss events──► T3 language + TTS (async)
                                                     │ det ทุก 8 เฟรม · pose · crops · ViT (batch 3)
                                                     │ NAP · ring buffer 48 · tCLS ทุก 2 เฟรม
                                                     │ online spotting (peak เมื่อ score เริ่มลดลง)
                                                     └ active mask + pause detection
```

**Robustness ในโค้ด**
- face crop มืด (mean < 40) → แจ้งเตือน UI
- valid-rate มือขวา < 0.5 ใน 2 s → แจ้ง "ขยับเข้ากลางเฟรม"
- ไม่มี GPU → profile `edge_cpu` อัตโนมัติ
- ไม่มีเน็ต → rules + local หรือ cached TTS
- คลิปสั้นกว่า window → คืนผลว่างพร้อมเหตุผล

---

## 7. Latency Budget

| Stage (ต่อเฟรม @12.5 fps) | CPU 12-core · edge @112 | RTX 2050 fp16 |
|---|---|---|
| decode + resize | ~2 ms | ~2 ms |
| YOLOX-tiny ทุก 8 เฟรม (เฉลี่ย) | 3.8 ms **[คำนวณ]** | ~1 ms **[ประมาณการ]** |
| RTMPose-s | 8–10 ms **[วัดจริง]** | ~3 ms ORT-CUDA **[ประมาณการ]** |
| crops × 3 + active mask + NAP | ~1 ms | ~1 ms |
| ViT-S × 3 crops | ≈ 31 ms torch fp32 **[คำนวณ]** (@224 ≈ 133 ms) · int8 ORT ≈ 18 ms **[ประมาณการ]** | 19–25 ms batch 3 **[วัดจริง]** · CUDA graph / TensorRT ≈ 8 ms **[ประมาณการ]** |
| temporal encoders × 4 + heads (incremental) | 6–10 ms **[ประมาณการ]** | 2–3 ms **[ประมาณการ]** |
| prototype matmul 4.6k × 512 | < 1 ms | < 1 ms |
| **รวม** | **≈ 55–60 ms** → 12.5 fps ได้ | **≈ 30 ms** (≈ 15 ms หลัง optimize) |
| LLM ต่อประโยค | 400–900 ms (เครือข่าย) | เท่ากัน |
| TTS เริ่มเล่น | 300–600 ms | เท่ากัน |

**ข้อสรุป:** CPU real-time ใช้ได้ที่ 112 px เท่านั้น · เครื่องที่มี GPU ใช้ 224 px ได้ · .mp4 offline ใช้ ViT-B @224 ที่ 175 crops/s **[วัดจริง]** ≈ 13 s ต่อวิดีโอ 1 นาที **[คำนวณ]**

**การ optimize เรียงตามความคุ้ม**
1. ONNX Runtime fp16 (GPU) / int8 dynamic quant (CPU) สำหรับ ViT และ pose
2. 112 px ในโหมด edge
3. Detector cadence 8 เฟรม + box EMA
4. Batch 3 crops เป็น tensor เดียว · crop บน GPU ด้วย `roi_align` เมื่อใช้ CUDA
5. Ring buffer และ tensor แบบ pre-allocated → RAM คงที่
6. Prototype matrix fp16 contiguous · `argpartition` สำหรับ top-k
7. LLM และ TTS แบบ async · cache ประโยคซ้ำ
8. RAM เป้าหมาย < 1.5 GB: pose ~40 MB + ViT-S fp16 ~45 MB + temporal ~25 MB + prototypes ~5 MB + runtime

---

## 8. Compute Plan · Local vs Cloud

### 8.1 งานไหนรันที่ไหน

| งาน | ปริมาณ | Local RTX 2050 | GCP L4 · g2-standard-8 | GCP A100 40GB · a2-highgpu-1g | แนะนำ |
|---|---|---|---|---|---|
| Decode + tracks ทั้งคลัง | 1.15 M เฟรม | ≈ 1.5 ชม. ด้วย 4 process **[คำนวณ]** | ≈ 40 นาที **[ประมาณการ]** | — | Local |
| Cache res112 ViT-S (5 versions TTRS, 3 versions อื่น) | ≈ 12 M crops | ≈ 1.8 ชม. **[คำนวณ]** | ≈ 25 นาที | ≈ 10 นาที | Local |
| Cache res224 ViT-S (เท่ากัน) | ≈ 12 M crops | ≈ 7 ชม. **[คำนวณ]** | ≈ 1.5 ชม. | ≈ 35 นาที | Local ข้ามคืน หรือ L4 |
| Cache res224 ViT-B (quality) | ≈ 12 M crops | ≈ 19 ชม. **[คำนวณ]** | ≈ 4 ชม. | ≈ 1.5 ชม. | L4 / A100 |
| SSL 4 streams (S1 + S2) | 80 ep × 4 | ≈ 10 ชม. **[ประมาณการ]** | ≈ 2–2.5 ชม. | ≈ 1 ชม. | Local · cloud เมื่อ sweep |
| ISLR + LoRA | 40 ep | ≈ 1 ชม. | ≈ 15 นาที | — | Local |
| FER teacher pseudo-label | 1.15 M face crops | ≈ 2 ชม. | ≈ 30 นาที | — | Local |
| ByT5-small seq2seq (v2) | weak pairs 6 ชม. | ไม่แนะนำ (VRAM) | ≈ 3–4 ชม. | ≈ 1.5 ชม. | Cloud |
| Evaluation / Bench | เล็ก | นาที | — | — | Local |

ความเร็ว cloud ประมาณจาก throughput สัมพัทธ์ L4 ≈ 4–6× และ A100 ≈ 10–15× ของ RTX 2050 **[ประมาณการ]**
ราคา on-demand โดยประมาณ L4 ≈ $0.7–1.0/ชม. · A100 40GB ≈ $3.5–4/ชม. · **ตรวจสอบราคาปัจจุบันก่อนใช้** · งาน v1 ทั้งหมดบน L4 รวมราว $20 หรือน้อยกว่า · Spot VM ลดได้มาก เพราะ cache และ checkpoint เป็น resumable อยู่แล้ว

### 8.2 `configs/compute.yaml`

```yaml
profiles:
  local_gpu:  {device: cuda, precision: fp16, vram_gb: 4,  batch_encode: 128,  batch_ssl_chunks: 40,  grad_accum: 2, num_workers: 6,  cache_res: [112, 224]}
  local_cpu:  {device: cpu,  precision: fp32, threads: 10, batch_encode: 32,   batch_ssl_chunks: 8,   grad_accum: 8, num_workers: 4,  cache_res: [112]}
  cloud_l4:   {device: cuda, precision: bf16, vram_gb: 24, batch_encode: 512,  batch_ssl_chunks: 128, grad_accum: 1, num_workers: 8,  cache_res: [112, 224]}
  cloud_a100: {device: cuda, precision: bf16, vram_gb: 40, batch_encode: 1024, batch_ssl_chunks: 256, grad_accum: 1, num_workers: 12, cache_res: [112, 224]}
storage:
  local: {root: D:/SLM-labs/SLM-Labs-SignDINO/artifacts}
  cloud: {root: gs://<bucket>/thaislm/artifacts}
sync:
  upload_before_cloud_job: [raw_data, manifests, tracks]      # raw_data 11.8 GB อัปครั้งเดียว
  download_after_cloud_job: [cache, nap, checkpoints, prototypes]
resume: {cache_per_clip: true, checkpoint_every_epoch: true, spot_safe: true}
```

โค้ดทุกจุดอ่าน `THAISLM_COMPUTE_PROFILE` · ไม่มี path หรือ batch ฝังตายตัว · ย้าย local ↔ cloud ด้วยตัวแปรเดียว

### 8.3 Environment

- ใช้ conda env `hugging` (torch 2.7.0+cu128 · transformers 4.51 · mediapipe · ultralytics · openai · pythainlp มีอยู่แล้ว)
- ต้องเพิ่ม: `rtmlib` · `onnxruntime-gpu` · `pandera` · `edge-tts` · `sounddevice` · `av` · อัปเกรด `transformers ≥ 4.56` สำหรับ DINOv3
- **ติดตั้ง `rtmlib` ด้วย `--no-deps`** เพราะมันดึง `opencv-contrib-python 5.x` และ numpy 2.x มาทับ ซึ่งเกิดขึ้นจริงระหว่างการทดลอง
- ล็อกเวอร์ชันทั้งหมดใน `environment.yml` · ตรวจ `torch.cuda.is_available()` และตั้ง `cudnn.benchmark = True` ตอน startup · log device ทุก run

```bash
conda activate hugging
pip install --no-deps rtmlib
pip install onnxruntime-gpu pandera edge-tts sounddevice av "transformers>=4.56"
```

---

## 9. Notebooks

| Notebook | เรียก module | สิ่งที่เห็น | ผลลัพธ์ |
|---|---|---|---|
| `00_explore_raw_data` | data · schema · utils | distribution · contact sheets · rest-pose profile · hand-size histogram · chroma/caption constants · signer clusters · VTT hits | ค่าคงที่ใน `data.yaml` · manifest v0 |
| `01_schema_articulators_cache` | schema · data · articulators · encoder · augment · normalize | validation report · pose overlay + crops 20 คลิป · active mask timeline · ms/เฟรมต่อ stage · cache · NAP fit | manifests · tracks · cache · `nap/*.npz` |
| `02_prepare_datasets` | data | lexicon/variant review · splits · TTRS-pairs · SSL sampler preview · Bench loader + sha256 guard | split tables |
| `03_ssl_pretrain` | ssl | loss/entropy curves · kNN บน pairs ทุก 10 ep เทียบ baseline · attention viz | `checkpoints/ssl/*` |
| `04_train_heads_islr_face` | heads | ISLR + LoRA + signer adversary · unknown calibration · enrollment demo · face head vs teacher · NMM demo | prototypes · `checkpoints/islr`, `face` |
| `05_continuous_spotting_llm` | spotting · language | score timeline บน Big Sign · boundary head · LLM prompt/response · rerank ablation | boundary head · prompt files |
| `06_evaluation` | evalkit | T01–T20 pass/fail · error analysis ต่อ signer / handshape / stream | `reports/eval_v1.md` |
| `07_tts` | tts | เปรียบเทียบ provider · style_map · latency · ตัวอย่างเสียงต่ออารมณ์ | ค่า TTS ใน `infer.yaml` |
| `08_inference_realworld` | pipeline | .mp4 ของคุณ → result.json / srt / wav / overlay · webcam live · latency และ RAM จริง | demo + ตัวเลข T11 |

---

## 10. `.env` และ Configs

### 10.1 `.env.example`

```dotenv
# ---- paths ----
THAISLM_ROOT=D:/SLM-labs/SLM-Labs-SignDINO
THAISLM_RAW=${THAISLM_ROOT}/raw_data
THAISLM_ARTIFACTS=${THAISLM_ROOT}/artifacts
THAISLM_COMPUTE_PROFILE=local_gpu        # local_gpu | local_cpu | cloud_l4 | cloud_a100
THAISLM_INFER_PROFILE=edge               # edge (112) | quality (224) | edge_cpu
CONDA_ENV=hugging

# ---- model hub ----
HF_TOKEN=hf_xxx                          # จำเป็นสำหรับ DINOv3 (gated) · ถ้าไม่มีจะใช้ DINOv2
HF_HOME=D:/hf_cache

# ---- language layer ----
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_S=3

# ---- tts ----
TTS_PROVIDER=edge                        # edge | openai | local
TTS_VOICE=th-TH-PremwadeeNeural
OPENAI_TTS_MODEL=gpt-4o-mini-tts

# ---- cloud (optional) ----
GCP_PROJECT=
GCS_BUCKET=

# ---- runtime ----
DEVICE=auto                              # auto | cuda | cpu
ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
NUM_THREADS=10
LOG_LEVEL=INFO
```

### 10.2 `configs/data.yaml`

```yaml
schema_version: "2.1"
fps: 12.5
crop_from_native_res: true
crop_size: 112
encoder_res: {edge: 112, quality: 224}

articulators:
  det:  {model: yolox_tiny, input: 416, every_n_frames: 8, box_ema: 0.6}
  pose: {model: rtmpose_s_body7, input: [256, 192]}
  hand_fallback: {offline_only: true, when_elbow_conf_below: 0.3}
  crops:
    hand: {center_shift: 0.30, side_ratio: 1.1, pad: 1.4}
    face: {side_ratio_ears: 2.2, pad: 1.2}
    body: {down_shift: 0.9, side_ratio_shoulder: 3.2}
  min_kpt_conf: 0.3

signer_lock: {min_presence: 0.6, window_s: 2.0, weights: {wrist_conf: 0.5, motion: 0.3, zone: 0.2}, lock_s: 2.0}
active_segment: {raise_thr: 0.15, move_thr: 0.03, dilate: 1, rest_level: {clip: edges3_median, stream: running_p10}}

ttrs:
  caption_band_frac: 0.85
  logo_box: [0.86, 0.02, 0.99, 0.12]
  chroma_hsv_lo: [95, 117, 157]
  chroma_hsv_hi: [105, 255, 255]
layouts:
  bigsign_vertical: {signer_zone: [0.0, 0.45, 1.0, 1.0]}

text_norm: {unicode: NFC, strip_zero_width: true, thai_digits_to_arabic: true, pythainlp_normalize: true, parenthetical_to_variant: true}
vtt: {track_priority: [th-orig, th, th-th], dedupe_rolling_cues: true, mining_window_s: [-0.5, 2.0], bigsign_lag_s: 1.5}

cache:
  backbone: {default: facebook/dinov3-vits16-pretrain-lvd1689m, fallback: facebook/dinov2-small, quality: facebook/dinov3-vitb16-pretrain-lvd1689m}
  store: [cls, patch_mean]
  dtype: float16
  versions: [base, aug_flip, aug_photo]
  ttrs_extra_versions: [aug_bg1, aug_bg2]

splits: {by: [signer_id, channel], seed: 42, ttrs_pairs_val_test: [0.5, 0.5], bench_blocklist: artifacts/bench/sha256.txt}
sampling_weights: {ttrs: 0.35, youtube_word: 0.30, youtube_sentence: 0.10, bigsign: 0.25, song_factor: 0.5}
```

### 10.3 `configs/train.yaml`

```yaml
normalization:
  method: nap                       # Signer-Nuisance Projection
  fit_on: train_split_active_frames
  dims: auto                        # เลือกบน val TTRS-pairs ในช่วง 1..(n_signers-1)
  per_stream: true
  forbidden: [per_clip_mean, rest_frame_mean]   # วัดแล้วได้ R@1 0.000

ssl:
  streams: [hand_r, hand_l, face, pose]
  input_dim: 768
  d_model: 384
  layers: 6
  heads: 6
  pose: {d_model: 128, layers: 2, input_dim: 68}
  prototypes: 4096
  crops: {n_global: 2, n_local: 8, global_len: [32, 64], local_len: [6, 16], chunk_len: 96}
  active_frames_only_for: [ttrs]
  mask: {type: frame_only, ratio: [0.3, 0.5]}
  loss: {dino: 1.0, ibot: 1.0, koleo: 0.1, gram: 2.0}
  temp: {teacher: [0.04, 0.07], student: 0.1, warmup_frac: 0.3}
  ema: [0.994, 1.0]
  optim: {lr: 5.0e-4, wd: 0.04, warmup_frac: 0.05, epochs_stage1: 60, epochs_stage2: 20, precision: bf16}

islr:
  embed_dim: 512
  loss: {supcon_temp: 0.07, arcface_margin: 0.2, arcface_scale: 30, signer_adversary_lambda: 0.1}
  lora: {rank: 4, alpha: 8, targets: [q, v]}
  optim: {lr: 3.0e-4, epochs: 40, batch: 128}
  prototype_unit: sign_variant_id
  output_unit: lemma_th
  unknown: {tau: 0.55, margin: 0.05}    # calibrate ใน 06
  mining: {tau: 0.72, max_per_lexeme: 5, rounds: 2, active_len_s: [0.4, 2.5], weight: 0.5}

face:
  teacher: hsemotion/enet_b0_8_best_afew   # ตรวจ license ก่อนใช้
  classes: [neutral, happy, sad, angry, surprised, fear, disgust]
  window_frames: 12
  smoothing_ema_s: 0.8
  nmm: {shake_min_cycles: 2, shake_window_s: 1.0, nod_min_cycles: 2, suppress_affect_when_active: true}

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
  enable_only_if_beats_v1_on: bench_sent
```

### 10.4 `configs/infer.yaml`

```yaml
profile: ${THAISLM_INFER_PROFILE}     # edge | quality | edge_cpu
runtime: {engine: onnxruntime, gpu_precision: fp16, cpu_int8: true, threads: ${NUM_THREADS}}
stream: {capture_res: [1280, 720], fps: 12.5, ring_buffer: 48, tcls_every: 2, queue_size: 2, det_every: 8}
video:  {batch_frames: 128, hand_fallback: true, write: [result_json, srt, wav], overlay: false}
normalization: {nap: true}
gates: {active_mask: true}
thresholds: {unknown_tau: 0.55, spot_tau: 0.6, pause_s: 0.6}

language:
  provider: openai
  model: ${OPENAI_MODEL}
  max_tokens: 80
  timeout_s: ${OPENAI_TIMEOUT_S}
  rerank_topk: 5
  fallback: rules
  system_prompt_file: configs/prompts/system_th.txt
  few_shot_file: configs/prompts/fewshot_th.json

tts:
  provider: ${TTS_PROVIDER}
  voice: ${TTS_VOICE}
  openai_model: ${OPENAI_TTS_MODEL}
  style_map:
    neutral:   {rate: 1.00, pitch: 0}
    happy:     {rate: 1.10, pitch: 2}
    sad:       {rate: 0.90, pitch: -2}
    angry:     {rate: 1.15, pitch: 1}
    surprised: {rate: 1.05, pitch: 3}
  persona: null
  cache_size: 256

ui: {show_topk: 5, warn_dark_face_mean: 40, warn_hand_valid_rate: 0.5}
```

### 10.5 `requirements.txt` (นอกเหนือจากที่มีใน `hugging`)

```
rtmlib            # install with --no-deps
onnxruntime-gpu
transformers>=4.56
peft
pandera
pyarrow
av
edge-tts
sounddevice
python-dotenv
pyyaml
jiwer
sacrebleu
psutil
rich
```

---

## 11. Roadmap และ Go/No-go

| เฟส | ระยะ | ส่งมอบ | Go/No-go |
|---|---|---|---|
| P0 | 2–3 วัน | env ล็อกเวอร์ชัน · schema + adapters · `00` | schema validate 100 % · ค่าคงที่ครบ |
| P1 | 1 สัปดาห์ | tracks ทั้งคลัง · active masks · cache res112 (+224 ข้ามคืน) · NAP fit | valid-rate ≥ 0.9 ทุกชุด · active frac TTRS 0.6–0.8 · signer lock ≥ 0.95 บนคลิป 2 คน · **ทำซ้ำ baseline frozen + trim + NAP บน TTRS-pairs เต็ม** |
| P2 | 1–2 สัปดาห์ | SSL 4 streams | kNN TTRS-pairs val **R@1 ≥ 0.30** · ถ้าไม่ถึงหลัง Stage 1 ตรวจ NAP และ active-only ก่อน |
| P3 | 1–2 สัปดาห์ (อัด Bench ควบคู่) | ISLR + enrollment + Bench-Word | TTRS-pairs test R@1 ≥ 0.60 · Bench-Word R@1 ≥ 0.55 · T13, T14, T15 ผ่าน |
| P4 | 1–2 สัปดาห์ | spotting + face + LLM | T08–T10, T19 ผ่าน |
| P5 | 1 สัปดาห์ | TTS + real-time + `cli.py` | T11, T12, T16, T20 ผ่าน |
| P6 | ต่อเนื่อง | v2 seq2seq (cloud) · ขอ licence NADT · ดึง TTRS 720p | สลับไป v2 เมื่อชนะ v1 บน Bench เท่านั้น |

---

## 12. ความเสี่ยงและทางออก

| ความเสี่ยง | หลักฐาน | ทางออก |
|---|---|---|
| โมเดลเรียนตัวตน signer แทนท่า | NN-same-signer 0.78–0.84 **[วัดจริง]** | NAP (fixed) · signer adversary · chroma-key bg · mirror · LoRA · gate T14/T15 |
| การ normalize แบบ session ใช้ไม่ได้ | per-clip และ rest-frame mean ได้ R@1 0.000 **[วัดจริง]** | ใช้ NAP ที่ fit จาก train set แทน |
| rest-pose ครอบงำ | active ≈ 70 % · trimming ยก R@10 ≈ 4× **[วัดจริง]** | active mask เป็น layer มาตรฐาน · SSL บนเฟรม active สำหรับ TTRS |
| signer น้อยสำหรับ fit NAP | TTRS มี signer หลัก ~6 คน · ทดลองได้ 3 | เพิ่ม signer จาก YouTube face clusters และ Bench-train · เลือก d บน val |
| CPU real-time ไม่ทัน | @224 = 133 ms/เฟรม **[คำนวณ]** | edge profile @112 · int8 · detector cadence |
| มือเล็กและรายละเอียดนิ้วหาย | TTRS มือ 100–170 px **[วัดจริง]** | crop จาก native · ดึง TTRS 720p ถ้า fingerspelling ต่ำ |
| หลายคนในเฟรม | Big Sign แนวตั้งและคลิปเพลง | presence + wrist-conf lock **[วัดจริง]** · T19 |
| Close-up มือไม่มีข้อศอก | wrist conf 0.39–0.55 ในคลิป close-up **[วัดจริง]** | hand fallback offline · T05 |
| Dependency ทับกัน | rtmlib ดึง opencv 5 และ numpy 2 **[เกิดจริง]** | ติดตั้งด้วย `--no-deps` ใน `hugging` เท่านั้น · environment.yml |
| VRAM 4 GB | SSL step 3.6 GB **[วัดจริง]** | batch 40 + grad accumulation · cloud profile |
| ผลทดลองมี noise | 118 queries | ทำซ้ำบน TTRS-pairs เต็มใน P1 ก่อนเชื่อค่าตัวเลข |
| ลิขสิทธิ์ข้อมูล | README ระบุ research use | ใช้ภายใน · `license_status` ใน schema · ขอ licence NADT |
| ตีความไวยากรณ์เป็นอารมณ์ | คิ้วยกเวลาถามคำถามถูกอ่านเป็นประหลาดใจ | NMM ก่อน affect เสมอ · ลดน้ำหนัก affect เมื่อ NMM active |

---

## 13. Appendix · Evidence Log (2026-09-13)

**Setup:** 59 lexeme จาก TTRS ที่มีคลิปจาก 2 signer ต่างคน (118 queries) + 100 คลิปสุ่ม → gallery 218 คลิป · 12,305 เฟรมที่ 12.5 fps · crops จาก RTMPose-s · DINOv2-S/B · query หาคลิปคู่ที่เป็น signer อื่น

| การทดลอง | R@1 | R@5 | R@10 | NN same signer |
|---|---|---|---|---|
| Frozen ViT-S @224 · hands · mean pool | 0.000 | 0.017 | 0.034 | 0.78 |
| Frozen ViT-B · ทุกชุด stream / res / pooling (72 configs) | ≤ 0.017 | ≤ 0.025 | ≤ 0.042 | 0.74–0.93 |
| + active trimming | 0.025 | 0.085 | 0.144 | 0.74 |
| + per-signer centering (oracle ใช้ label) | 0.068 | 0.178 | 0.280 | 0.81 |
| + per-clip mean subtraction (label-free) | 0.000 | 0.017 | 0.025 | 0.77 |
| + rest-frame mean subtraction (label-free) | 0.000 | 0.017 | 0.025 | 0.77 |
| + PCA drop top-1 fit บน train clips | 0.008 | 0.042 | 0.093 | 0.76 |
| **+ NAP d=2 fit บน 51 train clips (3 signers) · fixed ตอนทดสอบ** | **0.068** | **0.186** | **0.314** | **0.68** |
| Pose-only (landmark-like) + trimming | 0.059 | 0.144 | 0.178 | 0.48 |

| Runtime | ค่า |
|---|---|
| RTMPose Body lightweight (det ทุกเฟรม) CPU | 48 ms/เฟรม |
| YOLOX-tiny / RTMPose-s / RTMPose-t แยกกัน CPU | 30.4 / 9.6 / 8.1 ms |
| RTMPose Body balanced CPU | 312 ms/เฟรม |
| rtmlib Hand (RTMDet-nano + RTMPose-m) CPU | 78–88 ms/เฟรม |
| DINOv2-S CPU fp32 (batch 32) @112 / @224 | 10.2 / 44.4 ms/crop |
| DINOv2-B CPU fp32 (batch 32) @112 / @224 | 35.9 / 148.8 ms/crop |
| DINOv2-S GPU fp16 @112 / @224 | 1,868 / 488 crops/s (batch 64) |
| DINOv2-B GPU fp16 @112 / @224 | 699 / 175 crops/s (batch 64) |
| Temporal encoder train step 640 × 48 เฟรม bf16 | 497 ms · 3.62 GB |
| ffmpeg decode + scale @12.5 fps | 16–18× realtime |
| Pose + crops ทั้ง subset (CPU, python loop) | 60.5 ms/เฟรม · valid 99.9–100 % |

Scripts อยู่ใน scratchpad ของ session (`exp/exp1_pose.py`, `exp3b_gpu.py`, `exp3c_control.py`, `exp3d_trim.py`, `exp4_labelfree_norm.py`, `exp5_nap.py`, `gpu_bench.py`) · ควรย้าย logic เข้า `modules/` และทำซ้ำใน `01` และ `06`

---

## 14. References

- SignDino: Self-Supervised Sign Language Representation Learning via Temporal-Axis Self-Distillation · arXiv:2609.06296
- SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction · arXiv:2411.16765
- SignMusketeers: An Efficient Multi-Stream Approach for Sign Language Translation at Scale · arXiv:2406.06907
- DINOv3 / DINOv2 (Meta) · rtmlib / RTMPose / YOLOX · ByT5 · HSEmotion
- Nuisance Attribute Projection (มาจากงาน speaker verification) ปรับใช้กับ signer identity
- ข้อมูลโปรเจกต์: `README.md`, `metadata/*.csv`, `reports/collection_report.pdf`
