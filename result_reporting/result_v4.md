# ThaiSLM — Result Report v4

**Scope:** v4 optimisation. The work covers data cleaning, a standard schema, and labels without manual annotation. It adds a large vocabulary bank, a pretrained encoder, continuous spotting/segmentation, open-set "unknown ≈ nearest word" output, a separate Face model, and LLM fusion.
**Hardware:** `conda activate hugging`, single CUDA GPU.
**Executed notebooks:** `notebooks/v4/00…08` (+ `04b`).
**Main scripts:** `scripts/v4.py`, `scripts/v4_build.py`, `scripts/make_notebooks_v4.py`.

---

## 0. TL;DR

| Question | Answer |
|---|---|
| What changed from v1? | The frozen RGB DINOv2 + self-trained heads were replaced by **pose-first recognition with a pretrained sign-language encoder**: RTMW-X 133 whole-body keypoints → **Uni-Sign** (ICLR 2025) part-wise ST-GCN + mT5 encoder. Recognition is retrieval from a **vocabulary bank** of 5,281 instances / 4,600 Thai words. **Recognition-driven spotting** finds where the words are. The Face model is separate. LLM fusion is constrained to the Sign model's candidates, and a deterministic **evidence guard** stops invented sentences. |
| Is the encoder better? (Priority 1) | **Much better.** The test is a new signer against the full 4,568-word dictionary (E1). **R@1 0.196, R@5 0.321, median rank 18.** v1 on the same protocol had R@1 0.000, median rank 547. On a 932-sign pilot gallery, zero-shot Uni-Sign reached R@1 0.23 / R@5 0.50, while the v1 model reached R@1 0.00 / R@5 0.05. |
| Did fine-tuning help? | **No.** A residual metric adapter (SupCon, cross-signer positives) did not beat zero-shot on the untouched split. Test MRR went 0.255 → 0.245 and median rank 18 → 38. It was rejected. With only 3 main TTRS signers there is too little signer-invariance signal (see §4.3). |
| Does it know *where* a word is? (Priority 1) | **On held-out-signer synthetic sentences, yes.** Segment detection F1 is **0.93** and the word count is right within ±0.6 per sentence. **On the real `data_test/` videos:** video 1 gets **4 segments for 4 signs**; video 2 gets 3 segments for 5 signs. The synthetic-trained boundary tagger alone fails on real fluent signing. It marks one long "signing" run (no internal begin peaks), which is why v4 adds recognition-driven spotting (§6). |
| Does it read `data_test/`? | **Partly.** Video 1 (ฉันปลอบเพื่อนร้องไห้): **เพื่อน is found at #1** in segment 1, and **ร้องไห้ at #2** in segment 2. ฉัน and ปลอบใจ are not found. Video 2 (ไปทานข้าวด้วยกันมั้ย): no correct word in any top-5. The nearest words are semantically close (ข้าวเหนียว, ปาก, กินอาหาร…) but wrong. ไป and ไหม are not in the vocabulary at all. |
| Does it still guess? | **No sentence is invented any more.** Every segment is output as `word` (accepted), `word?` (uncertain) or `[?]≈nearest` (unknown). The evidence guard **withheld** the LLM's fluent-but-wrong sentence for video 2 ("คุณศูนย์กินอาหารเหล่านั้นไม่ได้"), which is kept only as a *tentative* field. One false accept remains: the hands returning to rest at the end of video 1 were accepted as "สะโพก" (§9.1). |
| Face model | Separate model: ViT FER plus landmark non-manual cues. Both videos: **neutral** (0.90 / 0.70). Weak sad (0.08) in video 1 and surprise (0.13) in video 2. No yes/no-question brow raise was detected in video 2, although it is a question (มั้ย). |
| Honest bottom line | Spotting and the "there is a sign here, nearest known word is ___" output work. **Word identity on a brand-new signer is still the bottleneck:** about 20% top-1 for isolated dictionary signs, and about 12% top-1 / 30% top-5 inside spotted segments. Confidence scores separate right from wrong only weakly. The missing ingredient is still **multi-signer data for the same words**: only 412 of 4,600 bank words have ≥ 2 signers. |

---

## 1. v4 solution design

```
video ─► signer lock (v1 body track) ─► RTMW-X whole-body 133 kp @12.5 fps  (upper-body crop, batched ONNX-CUDA)
      │
      ├─► SIGN MODEL
      │     Uni-Sign pose encoder (WLASL ISLR checkpoint; body 9 · hands 21+21 · face 18 → ST-GCN → mT5-base encoder, "ctx" states)
      │     ├─ boundary tagger (BiGRU on ctx ‖ 18 kinematic features)  → active signing region
      │     ├─ recognition-driven spotting: every 0.5–2.4 s window × vocabulary bank → semi-Markov DP → segments
      │     ├─ temporal ensemble of candidates inside each segment (Σ top-5 scores of all inner windows)
      │     └─ open-set status (E1-calibrated score/margin): accepted | uncertain | unknown(≈ nearest)
      │
      ├─► FACE MODEL (independent)
      │     ViT FER (trpakov/vit-face-expression) on the face crop · landmark NMM (brow raise, head shake/nod, mouth)
      │
      └─► FUSION
            OpenAI LLM: may only use each segment's candidates; unknown → [?]; question/negation from the face
            deterministic evidence guard: sentence shown only if ≥1 sign accepted AND ≥ half of the signs resolved
            outputs: result.json · subtitle.srt (word / word? / [?] ≈ nearest) · segments.jpg · audio.wav (MMS-TTS, only when a sentence is shown)
```

**Why this design (lessons from v1 → blueprint v3):**
- v1's RGB features clustered by signer and background. Pose removes appearance, and Uni-Sign was pretrained on large sign-language corpora (CSL-News, WLASL). So we fine-tune/reuse an existing SignDINO-like model instead of learning from scratch on 3 signers.
- Recognition is retrieval against a bank rather than a classifier. The vocabulary can then grow with any labelled clip (TTRS, YouTube titles, speech-mined instances) with no retraining.
- The system must say "unknown" rather than guess. Every level (segment status, LLM rules, deterministic guard) is designed to fall back to "[?] ≈ nearest".

---

## 2. Data cleaning & Standard Schema v4

**Files:** `artifacts/manifests_v4/clips_v4.parquet`, `speech_words.parquet`, `cleaning_report.json`. Code: `modules/lexicon.py::build_schema_v4`.

### 2.1 Schema (one row per clip)

| field | meaning |
|---|---|
| `clip_id`, `source` (ttrs / youtube), `subset`, `media_path`, `duration_s`, `width`, `height` | identity / media |
| `signer_id` | TTRS signer id; YouTube: one video ≈ one signer |
| `title` | YouTube title (label source when explicit) |
| `label` | the word/sentence the clip shows |
| `label_source` | `gloss` (TTRS) · `title` (parsed from the filename/title) |
| `usable_for` | `vocab` (clean single sign) · `weak_word` (lesson with timed speech words, used for mining) · empty = excluded |
| `exclude_reason` | why a clip was dropped |
| `sign_variant_id` | TTRS variant of a lemma |

`speech_words.parquet` holds one row per *spoken* dictionary word inside a lesson video: `clip_id, word, t0, t1, source`. Words come from the Thai auto-captions' inline word timestamps (`th-orig` VTT), tokenised with pythainlp against the TTRS lexicon.

### 2.2 Label extraction without manual labels
- **TTRS:** gloss (5,036 usable clips).
- **YouTube title/filename:** patterns like `…/หมวดกิริยาอาการ/ขับรถยนต์` and `(ศัพท์ภาษามือ) ยา` → `label`. 26 clips (24 embedded).
- **Speech inside the clip:** 7,468 word hits across 819 distinct dictionary words (7,248 from `th-orig` captions, 220 from `th`). A Thai Whisper ASR module (`biodatlab/whisper-th-medium-combined`, `modules/lexicon.py::run_asr`) was implemented for lessons without captions. **It was not run to completion:** it was paused because of GPU contention with pose extraction, so no ASR labels are used in v4.

### 2.3 Removed as noise (`cleaning_report.json`)

| reason | clips |
|---|---|
| no usable label (no Thai speech captions / title is only a topic) | 77 |
| broadcast / song interpretation (caption = announcer speech with lag, 2-person layouts) | 45 |
| fingerspelling / alphabet drills (letters, not lexical signs) | 40 |
| TTRS clips > 15 s (explanations / several signs) | 6 |
| song interpretation (rhythm-driven) | 2 |
| data_test entry (never used for building) | 1 |

Kept for building: TTRS 5,036 · YouTube word lessons 86 · sentence lessons 38.

---

## 3. Pose

- RTMW-X (`rtmw-dw-x-l_simcc-cocktail14_270e-256x192`) whole-body 133 keypoints, batched ONNX Runtime CUDA. `modules/wholebody.py`.
- Crop = v1 signer-locked body box expanded to the upper body, which gives the hands more pixels.
- **5,202 pose files** (`artifacts/pose_v4/`): all TTRS + labelled YouTube lessons + data_test.
- data_test hand-keypoint confidence ≈ 0.9. The video 2 signer uses one hand while the other rests (notebook 01).

---

## 4. Encoder (Priority 1)

### 4.1 Choosing the pretrained model (pilot, 932-sign gallery, new signer)

| encoder | val R@1 | val R@5 | val MRR | test R@1 | test R@5 | test MRR | test median rank |
|---|---|---|---|---|---|---|---|
| **Uni-Sign WLASL-ISLR · 2× upsample · mT5 ctx · mean** | 0.207 | 0.397 | 0.293 | **0.232** | **0.500** | **0.342** | **6** |
| Uni-Sign WLASL · ST-GCN only (no mT5) | 0.069 | 0.138 | 0.113 | 0.054 | 0.179 | 0.119 | 60 |
| Uni-Sign CSL stage-1 · ctx | 0.052 | 0.086 | 0.080 | — | — | — | — |
| Uni-Sign CSL stage-1 · ST-GCN | 0.052 | 0.172 | 0.106 | 0.071 | 0.125 | 0.112 | 72 |
| v1 RGB (DINOv2 + SSL + ISLR head) | 0.000 | 0.052 | 0.043 | not run¹ | | | |

¹The v1 test row was stopped during a VRAM contention incident; the val row is representative (v1 full-dictionary test R@1 was 0.000 in `result_v1.md`).

**Selected:** Uni-Sign WLASL-ISLR, contextual mT5 states, mean-pooled, 2× temporal upsampling (12.5 → 25 fps).

### 4.2 Full dictionary (E1: new signer vs every TTRS word; leakage-free splits from v1)

| bank | val R@1 | val R@5 | val median | test R@1 | test R@5 | test MRR | test median | vocab |
|---|---|---|---|---|---|---|---|---|
| TTRS train only | 0.138 | 0.276 | 75 | 0.161 ±0.10 | 0.286 | 0.219 | 30 | 4,558 |
| **TTRS train + YouTube (title + mined)** | 0.121 | 0.276 | 75 | **0.196 ±0.10** | **0.321** | **0.255** | **18** | 4,568 |
| *v1 (for reference)* | | | | *0.000* | *R@10 0.018* | | *547* | *4,706* |

- **E5** (YouTube title clips = different signer/camera vs full TTRS bank): **R@1 0.267, R@5 0.400, median 45** (n=15 in-vocab of 24). Typical errors are compositional near misses, e.g. ขับรถยนต์ → [รถยนต์, ขับ].
- The "lesson-domain silver" R@1 0.688 (n=221) is **biased and not a real result**: mining itself required the word to rank in the top-3.

### 4.3 Fine-tuning (metric adapter) — rejected

Residual MLP on frozen embeddings, SupCon with cross-signer positives + augmented views, 2,000 steps.

| | val MRR | val median | test MRR | test R@1 | test median |
|---|---|---|---|---|---|
| zero-shot | 0.191 | 75 | **0.255** | **0.196** | **18** |
| adapter | 0.203 | 110 | 0.245 | 0.179 | 38 |

The +0.012 val MRR is within noise and the median rank got worse, so the selection rule (val MRR +0.01 **and** median rank not worse) rejects it. `artifacts/v4/config.json: use_adapter=false`.
**Leakage caught and fixed:** an earlier run applied the adapter (trained with title/mined positives) inside `eval`. This inflated E5 to R@1 0.933, because the adapter had seen those very clips. After rejecting the adapter, E5 returned to the honest 0.267. The same zero-shot space is used by the bank, calibration and inference.

---

## 5. Vocabulary bank & weakly-supervised mining

**Bank** (`artifacts/v4/bank.npz`): **5,281 instances · 4,600 words**.

| source | instances |
|---|---|
| TTRS gloss | 5,036 |
| speech-mined lesson instances | 221 |
| YouTube title clips | 24 |

Only **412 words** have instances from ≥ 2 signers.

**Scoring:** a word's score is the mean of its top-2 instance cosine similarities (robust to one odd instance).

**Mining** (`modules/mining.py`): for a spoken word at time t, candidate windows are taken from t−1.5 s to t+3 s. An instance is accepted only if the pose encoder ranks that word **top-3 among all 4.6k words**, and the hands must be raised.
- 7,468 speech hits → **221 instances of 127 words from 98 videos**.
- Visual audit (`result_reporting/v4_mined_examples.jpg`): about 9/14 random instances clearly match the TTRS reference, 5 are unclear.
- Coverage is limited because lesson speech often names a word long before or after signing it.

data_test words in the bank:
- ฉัน, ปลอบใจ, เพื่อน, ร้องไห้, กิน (also ทาน / รับประทาน), ข้าว, ด้วยกัน are all present (TTRS only).
- **ไป and ไหม are absent** (out of vocabulary).

---

## 6. Continuous segmentation / spotting (Priority 1)

### 6.1 Boundary tagger (trained on synthetic sentences)
- **Model:** BiGRU tagger (outside / begin / inside) on [Uni-Sign ctx 768 ‖ 18 kinematic features].
- **Training data:** 2,400 synthetic sentences of 2–7 TTRS signs by the same signer. Augmentations: speed 1–2.5×, interpolated transitions, occasional pauses, signing-space compression, one-hand dropout.
- **Held-out signer S_ttrs_00:** boundary F1 **0.716**, count error 0.36/sentence (kinematic heuristic: 0.278 / 1.69).

**Failure on real video (diagnosis, `result_reporting/inference_v4/_tagger_llm/*/segment_prob.npy`):**
- Video 1: P(sign) > 0.5 over one continuous 94-frame run (1.7–9.1 s) with **a single begin peak** (frame 21, p=0.99). The next highest internal peak is 0.19, below threshold.
- Video 2: the whole clip is one run starting at frame 0.
- The old decoder therefore cut the run into equal ~2.5 s chunks, unrelated to the signs. Real fluent signing has co-articulated transitions, unlike the synthetic concatenations.

### 6.2 Recognition-driven spotting (new, `modules/spotting.py`)
1. Active region = tagger P(sign) > 0.5.
2. Candidate windows: every [a, b) on a 2-frame grid, 6–30 frames (0.5–2.4 s), ≥ 60% inside the active region. Each window is embedded exactly like a dictionary clip and scored against the bank.
3. Gain(window) = z − λ, where z = (top-1 score − vocabulary mean) / vocabulary std, i.e. how much the best word stands out.
4. A semi-Markov DP picks non-overlapping windows (length ≥ `min_len`), maximising Σ gain − γ·(uncovered active frames). Adjacent picks with the same word are merged.
5. **Candidates of a segment** = temporal ensemble: Σ of top-5 scores over all windows inside the segment (±2 frames).

**Tuning protocol (no data_test involvement):**
- Synthetic sentences of held-out signer **S_ttrs_00**; the bank **excludes every clip of that signer**, so 78% of that signer's words are OOV, as in real use.
- 120 sentences to tune λ/γ/min_len, **120 different sentences to report**.
- Grid: λ ∈ {8, 9, 10, 12} (after a first grid λ ≤ 5 over-segmented to 13.8 words/sentence), γ ∈ {0.3, 0.6, 1, 2}, min_len ∈ {8…16}.
- Objective: spot-F1 + 0.5·seg-F1 + 0.5·boundary-F1 − 0.05·count error.

**Selected:** λ=9, γ=1.0, min_len=14 (`artifacts/v4/spotting.json`, `reports/v4_spotting.json`).

| held-out test split (120 sentences, 4.8 signs each) | seg F1 (IoU≥0.3) | boundary F1 (±2 fr) | count err | words/sent | in-vocab recall@1 | recall@5 |
|---|---|---|---|---|---|---|
| tagger decoder (in its own synthetic domain) | 0.963 | 0.720 | 0.42 | 5.23 | 0.157 | 0.276 |
| **spotting, recognition only (used for real video)** | **0.928** | 0.191 | 0.58 | 5.08 | 0.118 | 0.236 → **0.299** with ensemble |
| spotting + tagger begin prior (β=2) | 0.945 | 0.422 | 0.57 | 5.24 | 0.118 | 0.276 |

**Reading the table:**
- Spotting finds almost as many correct segments as the tagger, without needing begin peaks, which is the property that transfers to real video.
- Its exact boundaries are looser (±2-frame F1 0.19).
- The begin-prior variant is better on synthetic data but depends on the peaks that vanish on real signing, so it is not used.
- The candidate ensemble was selected on the tune split (sum 0.102/0.352 vs single 0.093/0.296 R@1/R@5) and confirmed on test (`reports/v4_spot_agg_check.txt`).

---

## 7. Open-set status & calibration

Thresholds are fitted on **E1 val** in-vocab correct matches (`v4/calibration.json`):
- **accepted:** score ≥ 0.638 and margin ≥ 0.071.
- **uncertain:** score ≥ 0.564.
- **unknown:** otherwise, reported as `[?] ≈ nearest`.

For spotted segments, *accepted* is also downgraded if the ensemble top-1 differs from the best window's top-1.

| empirical top-1 precision (in-vocab) | accepted/likely | uncertain | unknown |
|---|---|---|---|
| E1 val | 0.29 (22% of queries) | 0.10 | 0.17 |
| E1 test | 0.30 (17%) | 0.16 | 0.17 |
| E5 YouTube titles | 0.50 (n=2) | 0.25 | 0.00 |
| spotted segments, held-out synthetic test | 0.14 | 0.10 | 0.11 |

**Honest limitation:** score and margin separate correct from wrong only weakly. Test-time augmentation consensus and DTW re-scoring were also tried in v4 and did not help. The status therefore mainly controls how the output is *worded* (`word` / `word?` / `[?]≈`), and the LLM layer is forbidden from turning low-evidence words into a sentence (§8).

---

## 8. Face model, LLM fusion, evidence guard

- **Face model** (`modules/face_v4.py`), independent of the sign model:
  - ViT FER emotion on face crops (7 classes, per-frame, averaged).
  - Landmark NMM relative to the signer's own baseline: brow raise near the end → yes/no question; head-shake cycles → negation; nods → affirmation.
- **Fusion** (`modules/translate.py`): an OpenAI model (`OPENAI_MODEL` from `.env.example` = `gpt-4.1-mini`) receives per-segment status + top-5 candidates + face output. It must return JSON: chosen word per segment or `[?]`, sentence, meaning, emotion description, confidence, notes.
- **Evidence guard (deterministic, v4 final):** a sentence is shown only if ≥ 1 sign is *accepted* and ≥ half of the segments are resolved. Otherwise the sentence is withheld: `tentative_sentence_th` keeps the LLM reading for debugging, and `meaning_th` is replaced by "ตรวจพบท่ามือ N ช่วง … คำที่ใกล้ที่สุดที่ระบบเคยเรียน: ช่วง 1 ≈ …".

**Does the LLM decode or guess?**
- **Setup:** 12 GPT-written sentences built only from words that ≥ 2 signers have. Each word is a real TTRS clip, and the bank excludes that signer's clips (cross-signer). Oracle segments.

| | word acc (all words) | precision of words it commits to | `[?]` rate | out-of-candidate words | sentences shown |
|---|---|---|---|---|---|
| top-1 only | 0.22 | 0.22 | 0 | — | — |
| oracle (correct word if in top-5) | 0.36 | | | | |
| LLM prompt v1 (pick from top-3 freely) | 0.22 | 0.22 | 0.00 | 0 | 12/12 |
| **LLM prompt v2 + guard (final)** | 0.14 | **0.26** | 0.47 | 0 | 0/12 (all segments uncertain → withheld by design) |

**Conclusion:** the LLM never adds vocabulary (0 violations). It does not recover the correct word from the top-5 better than top-1. The final design trades recall for not asserting sentences on weak evidence, which is what "ไม่เดามั่ว" requires.

---

## 9. Real-world inference on `data_test/`

The command is `python scripts/v4_build.py infer --tag final --tts` (also run live in notebook 07). Outputs go to `result_reporting/inference_v4/<video>/`: `result.json`, `subtitle.srt`, `segments.jpg`. Report: `artifacts/reports/v4_infer_final.json`. The test videos' audio and on-screen text are **never** used.

### 9.1 ฉันปลอบเพื่อนร้องไห้ (reference glosses: ฉัน · ปลอบใจ · เพื่อน · ร้องไห้)

| segment | status | top-5 (temporal ensemble) | what the frames show (manual look at `segments.jpg`) |
|---|---|---|---|
| 1.60–3.68 s | uncertain (0.757, margin 0.053) | **เพื่อน**, มิตรภาพ, เพื่อนสนิท, เย็บผ้า, ถักโครเชต์ | hooked index fingers = เพื่อน ✅ #1 |
| 3.68–6.08 s | uncertain (0.608) | จมูกแบน, **ร้องไห้**, สายตาดี, มอง, จมูก | hand at the eye → ร้องไห้ ✅ #2 (the rest are face-area signs) |
| 6.08–8.48 s | unknown (0.548) | [?] ≈ ไม่พอ, ภาคเหนือ, องศา, นครปฐม, ไม้ตรี | hands together in front (likely ปลอบใจ, not found) |
| 8.48–9.60 s | **accepted** (0.681, margin 0.124) | สะโพก, ผิวปาก, ท้อง, บุกรุก, กระโถนถ่าย | hands returning to rest ❌ **false accept** |

- **Evidence gloss:** `เพื่อน? | จมูกแบน? | [?]≈ไม่พอ | สะโพก`.
- **Sentence:** withheld by the LLM itself (confidence 0.3). Meaning: "จับได้เพียงบางคำ เช่น เพื่อน จมูก สะโพก แต่ไม่มีความหมายที่สมเหตุสมผลครบถ้วน".
- **Face:** neutral 0.90 (weak sad 0.08), 2 nods, no question/negation.
- **Segments:** 4 found vs 4 signs. 2 of 4 reference words are in a segment top-5 (เพื่อน #1, ร้องไห้ #2). ฉัน was not isolated as its own segment.

### 9.2 ไปทานข้าวด้วยกันมั้ย (reference glosses: ไป · กิน · ข้าว · ด้วยกัน · ไหม; ไป and ไหม are OOV)

| segment | status | top-5 | frames |
|---|---|---|---|
| 0.00–2.40 s | uncertain (0.607) | คุณ, ทวิภาค, ข้าวเหนียว, ปาก, หนึ่งหมื่น | one hand moving from the chest to near the mouth |
| 2.40–4.16 s | uncertain (0.587) | ศูนย์, งูเห่า, แปดสิบ, มิลลิลิตร, หนึ่งร้อยสอง | O / pinch hand shapes at the mouth |
| 4.16–6.56 s | uncertain (0.639) | กินอาหารเหล่านั้นไม่ได้, กลับบ้าน, ถ่มน้ำลาย, ดอกเบี้ย, ไปแล้ว | pinch at the mouth, then open hand, then rest |

- **Evidence gloss:** `คุณ? | ศูนย์? | กินอาหารเหล่านั้นไม่ได้?`.
- **Sentence:** **withheld by the evidence guard** (no accepted sign). The tentative LLM reading "คุณศูนย์กินอาหารเหล่านั้นไม่ได้" is not shown to users.
- **Face:** neutral 0.70, surprise 0.13, happy 0.12. The question brow raise was **not** detected.
- **Segments:** 3 found vs 5 signs. No reference word is in any top-5. The nearest words are in the right semantic field (mouth/eating/rice), but none is correct.

### 9.3 Progress on the same videos

| | video 1 | video 2 |
|---|---|---|
| v1 | wrong sentence (WER 1.25), random-looking words | wrong sentence (WER 1.00) |
| v4, tagger decoder + LLM prompt v1 | 3 equal chunks; เพื่อน #1; output "เพื่อนใบหน้า[?]" (invented) | 3 equal chunks; output "คุณ [?] ถ่มน้ำลาย" (invented) |
| **v4 final** | **4 segments / 4 signs; เพื่อน #1, ร้องไห้ #2; no invented sentence**; 1 false accept (rest) | 3 segments / 5 signs; nothing correct; **sentence withheld, nearest words listed** |

**Runtime per video (GPU):**
- pose 3.5 s / 1.9 s.
- tagger 0.45 s / 0.15 s.
- spotting 4.2 s / 3.1 s (≈ 500–800 windows through mT5).
- face 0.5 s / 0.2 s.
- LLM 3.8 s / 1.8 s.

---

## 10. Leakage controls

- v1 signer-level splits are reused. E1 queries are never in the bank variant used to evaluate them (`bank_train`). data_test was never used for training, threshold fitting or grid search.
- Spotting was tuned on a held-out signer whose clips were removed from the bank, and reported on separate sentences.
- The adapter was rejected; the E5 inflation it caused (0.933) was traced to training on those clips and removed.
- Mining accuracy on lessons is reported as biased and not used for decisions.
- Test videos' audio and burned-in text are not inputs. The on-screen text change times of video 2 are drawn only as reference lines in notebook 04.

---

## 11. Optimisation log (v4, chronological)

1. Schema v4 + cleaning (§2).
2. RTMW-X pose for pilot → all TTRS → labelled YouTube → data_test.
3. Encoder pilot (Uni-Sign WLASL vs CSL vs v1) → WLASL ctx + 2× upsampling.
4. TTRS bank → E1 full dictionary.
5. Title clips + speech mining → bank v2 (E1 test R@1 0.161 → 0.196).
6. Adapter v1 (TTRS only): no gain. Adapter v2 (+ YouTube positives): no gain on test; leakage found; rejected.
7. Rejection calibration v1: a precision-target search found no thresholds, because 60% of queries are OOV. Replaced by three-tier quantile calibration.
8. Test-time augmentation consensus and DTW expert re-scoring were tried and did not improve confidence separation (not adopted).
9. Boundary tagger: begin-label bug fixed (a 2-frame begin split every sign); cached training; signing-space compression augmentation for compact signers → held-out boundary F1 0.716.
10. LLM decoding eval: first 0 valid sentences (GPT inflected glosses) → numbered list + repair → 12 sentences.
11. Face model + fusion + TTS integrated. First data_test run: tagger chunks + LLM sentences that were fluent but wrong.
12. Diagnosis: the tagger finds no internal boundaries on real signing → recognition-driven spotting. First grid (λ ≤ 5) over-segmented → extended grid + min_len → seg F1 0.93 on held-out sentences.
13. Temporal candidate ensemble (tune-selected).
14. LLM prompt v2 + deterministic evidence guard → final data_test run.

---

## 12. Limitations (what still does not work)

1. **Signer generalisation of word identity.**
   - 20% R@1 on isolated dictionary signs; about 12% inside spotted segments for a hard signer.
   - The bank has 1 signer for most words (412/4,600 words have ≥ 2).
2. **Confidence is weakly informative.** "accepted" is right only 14–30% of the time on held-out data. Case in point: the hands-to-rest transition in video 1 was accepted as สะโพก.
3. **Citation vs conversational form.**
   - TTRS signs are slow, two-handed citation forms. Video 2 is fast, one-handed and mouth-centred; none of its words were retrieved.
   - ไป and ไหม are not in the vocabulary at all.
4. **Segmentation on real signing is only validated qualitatively** (2 videos). Video 2 got 3 segments for 5 signs; short function signs (ไป, ไหม) get merged.
5. **NMM** did not detect the question in video 2. The FER model is trained on posed Western faces and reads signing faces as neutral.
6. **Whisper ASR labelling** was not completed. Lessons without captions contribute nothing yet.
7. **Spotting cost** grows with window count: about 0.4 s per second of video on GPU.

## 13. Next steps (priority)

1. **Multi-signer data for the same words:** record ThaiSLM-Bench (≥ 10 signers × the 300 most frequent conversational words, phone video). It is the training signal the adapter lacked and the only reliable test set.
2. **Add function / conversational signs to the bank** (ไป, ไหม, ด้วยกัน…). Mine them from the 38 sentence lessons with finished Whisper ASR; complete `v4.py asr`.
3. **Train the tagger on real continuous signing:** use the mined lesson spans as weak boundary labels instead of synthetic concatenations. Add a "rest/transition" class so the hands-to-rest motion cannot be accepted.
4. Fine-tune Uni-Sign end-to-end (not only an adapter) once ≥ 2 signers per word are available; distil to a window-level model to cut spotting cost.
5. A Thai-signer FER / NMM model (brow raise for questions) from lesson videos with question captions.

---

## 14. Reproduce

```bash
conda activate hugging
python scripts/v4.py schema
python scripts/v4.py pose --set ttrs
python scripts/v4.py pose --set youtube_labelled
python scripts/v4.py pose_test
python scripts/v4.py eval_enc --ckpts wlasl,csl_stage1 --tag pilot
python scripts/v4_build.py embed_ttrs
python scripts/v4_build.py embed_titles
python scripts/v4_build.py mine --rank_k 3
python scripts/v4_build.py bank
python scripts/v4_build.py eval
python scripts/v4_build.py adapter --steps 2000          # rejected → config.json use_adapter=false
python scripts/v4_build.py calib
python scripts/v4_build.py seg_train --n_train 2400 --epochs 15
python scripts/v4_build.py spot_eval --n_sent 120        # writes v4/spotting.json (+ agg="sum")
python scripts/v4_spot_tiers_check.py; python scripts/v4_spot_agg_check.py
python scripts/v4_build.py lm_eval
python scripts/v4_build.py infer --tag final --tts
python scripts/make_notebooks_v4.py                     # builds + executes notebooks/v4
```

## 15. Artifact index

| path | content |
|---|---|
| `artifacts/manifests_v4/` | schema, speech words, cleaning report |
| `artifacts/pose_v4/` | 133-kp pose per clip |
| `artifacts/v4/bank.npz`, `bank_train.npz` | vocabulary banks (deployment / evaluation) |
| `artifacts/v4/segmenter.pt`, `spotting.json`, `calibration.json`, `config.json` | segmentation, spotting config, open-set thresholds, adapter switch |
| `artifacts/reports/v4_*.json` | encoder pilot, eval, adapter, mining, segmentation, spotting, calibration, LM eval (+ prompt v1), inference (final + tagger baseline) |
| `result_reporting/inference_v4/<video>/` | final outputs; `_tagger_llm/` = earlier tagger-decoder run |
| `result_reporting/v4_mined_examples.jpg` | mining audit sheet |
| `result_reporting/architecture_v4.md` | final architecture diagram, raw_data → build → inference |
| `vocab/corpus/vocab.json/.csv`, `sentence.json/.csv` | what the model knows: every bank entry (4,600) · sentence/phrase entries, lessons words were learned from, eval/test sentence coverage (`python scripts/export_vocab.py`) |
| `notebooks/v4/` | 00 cleaning/schema · 01 pose · 02 encoder · 03 bank/mining · 04 tagger · 04b spotting · 05 calibration · 06 face · 07 live data_test inference · 08 summary |
| `modules/{wholebody,unisign,recognize,kinematics,segment,spotting,lexicon,mining,face_v4,translate,pipeline_v4}.py` | v4 code |
