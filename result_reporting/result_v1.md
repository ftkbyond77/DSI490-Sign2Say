# ThaiSLM — Result Report v1

**Scope:** implement blueprint v1 and v2 → test → optimize (test cases + real-world `data_test/`) → select best → evaluate → report
**Date:** 2026-09-13 · **Machine:** RTX 2050 4 GB · 12-core CPU · 16 GB RAM · conda env `hugging` (torch 2.7.0+cu128)
**Wall-clock for this session:** ≈ 6 h 10 min (08:40–14:50), about 4 h of which was GPU/CPU compute

---

## 0. TL;DR (read this first)

| Question | Answer |
|---|---|
| Was the blueprint built? | **Yes, end to end.** Standard data schema and manifests, articulators (YOLOX + RTMPose, signer lock, native-resolution crops, active mask), frozen DINOv2-S embedding cache with blueprint augmentations (chroma-key background, mirror-swap, photometric, low-res), NAP, temporal SSL (DINO + iBOT + KoLeo + Gram), ISLR prototype head (SupCon + ArcFace + signer adversary, LoRA option), continuous spotting (NMS / DP / session-relative DP), NMM rules, a distilled face-affect head, a language layer (OpenAI with a rules fallback), TTS (offline MMS-TTS Thai, plus edge/OpenAI providers), VideoPipeline and StreamPipeline, CLI, notebooks 00–08. |
| Does it learn? | **Partly.** It learns sign identity *within* the TTRS studio domain: one-shot retrieval of unseen signs across signers on an 18-sign gallery rose from R@5 0.29 (frozen) to **0.66 test / 0.90 val** (chance 0.28). |
| Does it generalise to a new signer against the full dictionary (4,706 signs)? | **No.** Test R@1 = 0.000 and R@10 = 0.018; the median rank of the correct sign improved from 2046 (frozen) to **547** (491 with session centering). That is above chance (median ≈ 2350) but far from usable. |
| Does it work on the real-world videos in `data_test/`? | **No.** Both sentences are recognised wrongly (gloss WER 1.25 and 1.00; chrF 5.7 and 8.1). The model does rank some correct signs #1 in some window after session centering (ฉัน in video 1, กิน in video 2), but the decoder cannot pick them out of the noise. |
| Is it fast enough? | **GPU yes:** 29 ms/frame p50 (p95 64–73 ms) at 12.5 fps. **CPU borderline:** 70 ms/frame p50, 108–116 ms p95; this misses the blueprint's p95 ≤ 80 ms and RAM ≤ 2.5 GB (measured 2.9–3.4 GB). |
| Why it fails (evidence-based) | (1) TTRS has **3 main signers**, and only **177 signs** were performed by ≥ 2 of them, so there is almost no supervision for signer invariance. Embeddings still cluster by signer (nearest neighbour is the same signer 55–100% of the time). (2) Studio citation-form signs (1.5–3 s each) differ strongly from conversational 640×360 phone video (≈ 0.6 s/sign). (3) The temporal SSL losses plateaued after ~1k steps, and label-free kNN never beat frozen features. |
| Most valuable next step | Record **ThaiSLM-Bench** (blueprint §5.3.4 / §6.3), i.e. the same signs by ≥ 10 new signers on webcam/phone. It is both the missing training signal (signer invariance) and the only trustworthy test set. |

---

## 1. What was built (file map)

```
configs/        data.yaml · train.yaml · infer.yaml
modules/
  utils.py        paths, logging, seed, json
  data.py         TTRS/YouTube adapters → clips.parquet, lexicon.parquet · text normalisation · OpenCV 12.5 fps decode
                  (ffmpeg fallback) · VTT parser (rolling-cue dedupe) · open-set signer inventory · leakage-safe splits
  articulators.py YOLOX-tiny (every 8 frames) + RTMPose-s · signer lock (wrist-conf + size + layout zone, hysteresis)
                  · crops L-hand / R-hand / face at native res → 112 px · validity · pose features · active mask · mirror
  encoder.py      FrozenEncoder (DINOv2-S, [CLS‖patch-mean] 768-d) · multi-process cache job (CPU pose workers → one
                  GPU process) · packed memmap FeatureStore
  augment.py      chroma-key background replacement (roi-aligned background per crop) · mirror-swap · crop jitter ·
                  colour · low-res down-up (24–72 px) · blur · noise — parameters fixed per clip
  normalize.py    Signer-Nuisance Projection (NAP)
  ssl.py          TemporalEncoder (pre-LN, SDPA, LoRA-ready) · SignEncoder (4 streams) · DINO head · Sinkhorn-Knopp ·
                  KoLeo · iBOT · Gram anchoring · multi-temporal-crop sampler · training loop
  heads.py        ISLRModel (SupCon + ArcFace + gradient-reversal signer adversary) · ISLRSampler · PrototypeBank
                  (enroll / top-k / session-centred) · AffectTeacher (ViT FER) · AffectHead · NMM rules
  spotting.py     window scoring (optional session centering) · greedy NMS · segmental DP · session-relative DP ·
                  pause segmentation
  language.py     OpenAI composer (JSON, timeout) + RuleComposer fallback
  tts.py          SpeechStyle (emotion → rate/pitch, persona hook) · LocalMMSTTS (offline) · EdgeTTS · OpenAITTS
  pipeline.py     VideoPipeline (.mp4 → result.json / subtitle.srt / audio.wav) · StreamPipeline (ring buffer, latency)
  evalkit.py      retrieval metrics with 95% CI · CSLS · gloss WER · chrF · RSS
scripts/
  cli.py          manifests | bgpool | cache | infer
  experiments.py  pack | signers | baselines | ssl | islr | evalckpt | bank | continuous | realworld | latency | face
  collect_results.py · make_notebooks.py
notebooks/      00…08 (thin callers of modules/ + report readers)
artifacts/      manifests/ tracks/ cache/ nap/ checkpoints/{ssl,islr,face}/ prototypes/ reports/*.json backgrounds/
result_reporting/ result_v1.md · inference/<tag>/<config>/<video>/{result.json, subtitle.srt, audio.wav}
logs/thaislm/   every run log
```

### 1.1 Deviations from the blueprints (and why)

| Blueprint | Implemented | Reason |
|---|---|---|
| DINOv3 ViT-S/16 | **DINOv2-S** at 112 px | DINOv3 is gated (no `HF_TOKEN`) and needs transformers ≥ 4.56 (env has 4.51). Blueprint fallback. |
| Temporal encoder 6 layers, d = 384 | 4 layers, d = 256 (≈ 3.3 M params/stream); pose 2 × 128 | Fits the 4 GB GPU at batch 48 × 8 views; matches v1's "~3.5 M params/stream". |
| Pose 17 kpts × 2 + velocity = 68-d | 11 upper-body kpts × 2 + velocity = **44-d** | Hips/legs are out of frame in the real-world videos (waist-up). |
| Active mask `move_thr` 0.03 | **0.12** + 3-tap median | Measured RTMPose wrist jitter at rest is 0.03–0.06 shoulder-widths/frame, so 0.03 marked ~100% of frames active. |
| Signer-lock score uses motion | wrist-conf 0.5 + **size** 0.3 + zone 0.2 | Candidates are scored on detection frames, where no track history exists. Verified on a 2-person Big Sign vertical clip. |
| ffmpeg decode | **OpenCV in-process decode** (ffmpeg fallback) | Spawning ffmpeg per 2–8 s clip cost more than decoding. Throughput doubled from 55 to 125 frames/s; output is byte-identical to ffmpeg (checked). |
| YouTube: all footage | first **300 s per video** after a 10 s intro (12.9 h, 583k frames) | Diversity over a few long broadcasts; compute budget. |
| Face teacher HSEmotion | `trpakov/vit-face-expression` (HF) | Available from HF without extra packages. |
| TTS default edge-tts | **Local MMS-TTS Thai** (already cached), edge/OpenAI implemented | Offline, no text sent to an external service. |
| LLM gpt-4.1-mini | implemented; **rules fallback used** | No `OPENAI_API_KEY` present (this also exercises T12, the offline path). |
| ISLR split: all cross-signer signs → val/test | 20% train / 10+10% E2 / 30+30% E1 (see §3) | Blueprint split left **zero** cross-signer positive pairs for training (found in §5.3). |

---

## 2. Data processing

### 2.1 Manifests & lexicon
* 5,331 clips → `clips.parquet` (schema v2.1), **4,597 lexemes / 4,742 sign variants** after NFC, zero-width removal, Thai digits, pythainlp normalisation, and moving tail parentheses into `variant_tag` (blueprint said 4,599; the difference comes from normalisation).
* `data_test/*.mp4` sha256 values were checked against the corpus: **0 matches** (no real-world file is in any training pool).

### 2.2 Signer inventory (blueprint E8)
TTRS names only 2,304 clips. Implemented an **open-set** assignment: agglomerative clustering on centred face-stream CLS features, then each cluster is named by a majority vote of its named members (≥ 90% share), or becomes a new signer if it has no named members.

| Metric | Value |
|---|---|
| Face-identity CV accuracy on named clips (4 signers ≥ 10 clips) | 0.999 |
| Cluster ↔ name purity | **0.999** |
| Clips assigned by cluster name / new signer / still unknown | 2,387 / 108 / 244 |
| Main signers | S_ttrs_02 (2,437 clips), S_ttrs_01 (1,512), S_ttrs_00 (725) + 5 minor |
| Signs performed by ≥ 2 signers | **177** (only 52 with names alone) |

A visual check of 4 cross-signer pairs showed 3 clearly identical signs (ตา, ต้นหอม, เข็มขัด) and 1 visibly different variant (ใบโหระพา). **Pair labels carry some noise.**

### 2.3 Articulators & embedding cache
| Item | Value |
|---|---|
| Valid-rate (hand_l, hand_r, face) | 1.00 / 1.00 / 1.00 on sampled TTRS, Big Sign, and both real-world videos |
| Signer lock | chose the interpreter (bottom frame) over the guest in a Big Sign vertical clip |
| TTRS cache | 5,043 clips · 279,180 frames · 4 versions (base, flip, aug1, aug2) · **0 errors** · 31.8 min (+~8 min benchmarks) · 4.9 GB packed |
| YouTube cache | 288 videos · 582,774 frames · 2 versions (base, aug1) · **0 errors** · 99.8 min · 5.2 GB packed |
| TTRS active fraction | 0.62 (blueprint measured 0.70 with a looser threshold) |
| Augmentation check | contact sheet verified: blue studio keyed to YouTube/synthetic backgrounds; low-res, colour, and mirror views look correct |

---

## 3. Evaluation protocol & leakage controls

Two retrieval protocols were used. A first protocol was replaced after I found it was mis-specified (§5.3).

| Protocol | Question it answers | Query | Gallery | Held out of **all** training (SSL, NAP, ISLR) |
|---|---|---|---|---|
| **E1 — dictionary retrieval** *(primary; matches deployment)* | A new signer performs a dictionary sign: can we find it among all 4,706 signs? | clip of the held-out *query signer* | prototypes of all training signs (the target sign is represented by the *other* signer's clip) | all clips of the query signer for that sign (53 signs val / 53 test; 58 / 56 queries) |
| **E2 — one-shot enrollment** (T07) | A sign never seen in training is enrolled from one signer: can another signer's clip retrieve it? | clip of signer A | enrollments of the 18 held-out signs from signers ≠ A | *every* clip of those signs (18 val / 18 test) |
| **Pseudo-continuous** (T08 proxy) | Can the spotting decoder recover a sign sequence? | 3–6 active segments of held-out E2 clips, concatenated at 1–2× speed with 0–3 rest frames | E2 enrollments (other signers) + 4.7k train prototypes | as E2 |
| **Real-world** (`data_test/`) | End-to-end on unseen people/cameras | 2 phone-style videos, 640×360 | all 4,742 TTRS sign variants | never used for training; sha256-blocked |

Other controls: NAP fitted on train signers only; decoder/model selection on **val only**; 95% CIs reported. With ~56 queries a CI is about ±0.05–0.15, so only large differences are meaningful. Real-world references were taken from the file names: `ฉัน ปลอบใจ เพื่อน ร้องไห้` and `ไป กิน(ทาน) ข้าว ด้วยกัน ไหม(มั้ย)`. The sign order of video 2 is confirmed by its progressive on-screen text. **ไป and ไหม have no TTRS entry (out of vocabulary).**

---

## 4. Test → optimise loop (chronological)

| # | Step | Result | Decision |
|---|---|---|---|
| 1 | Frozen DINOv2 mean-pool baselines (blueprint v2 evidence replication) | E1 R@1 0.000, median rank 2046–2258; NN-same-signer 0.89–1.00 | Frozen features unusable across signers (confirms blueprint D1) |
| 2 | + active trimming, hands-only, + pose, NAP d = 1..3 | E2 R@5 0.24 → 0.55 (val); NAP d = 3 lowers NN-same-signer from 0.89 to 0.55 | Keep active trimming and NAP. A bug (d selected on an all-zero metric) was found and fixed; NAP d = 2 on the final split |
| 3 | ISLR from scratch, blueprint split (all cross-signer signs in val/test) | small-gallery R@1 0.17 at 500 steps, then flat; ArcFace loss → 0.3 (memorisation) | **Root cause:** no cross-signer positives in training → re-split 40/30/30 |
| 4 | ISLR from scratch, re-split | small R@1 0.25–0.28 but full-gallery R@1 0.000–0.009 | Full-gallery distractors were *seen* classes while queries were *unseen* (generalised zero-shot bias) → built **E1 / E2** protocols |
| 5 | CSLS hubness correction | no gain (E1 median rank 720 → 804 on test) | Not adopted |
| 6 | Session centering (embedding − signer/session mean) | frozen 2258 → 791; scratch 720 → 590 (test) | Adopted as a test-time option (bank side + video side) |
| 7 | Stream ablations: no face + adversary 0.5; pose-only | no-face similar to all streams (E1 test 787, E2 R@5 0.658); pose-only worse (1232, 0.50) | Drop the face stream for recognition (identity carrier; no loss); keep pose |
| 8 | Temporal SSL, 6000 steps (2 h 15 min) | DINO 33.6 → 27.7 and iBOT 33.3 → 26.2 (plateau after ~1k steps); kNN median rank 1642 → 1759; E2 R@5 0.55–0.58 | SSL alone ≈ frozen, but used as ISLR initialisation (step 9) |
| 9 | ISLR init from SSL: full FT vs LoRA r4 | full FT: val E2 R@5 **0.842** (scratch 0.711); LoRA: 0.737 val / 0.553 test | Full fine-tuning |
| 10 | + training-time signer centering | E1 test median rank 1356 (worse) | Rejected |
| 11 | + fast-signing speed augmentation (0.8–2.5×) | val E2 R@5 **0.895**; E1 test median rank **547** (best) | **Selected: `islr_ssl_fast`** |
| 12 | Continuous decoder grid on pseudo-sentences (DP penalty, NMS τ, window sets) | gloss F1 ≤ 0.01 for every setting; WER-based selection picked a decoder that outputs nothing (WER = 1.0) | Selection switched to F1, but F1 is at chance, so decoding is not solvable at this recognition level |
| 13 | Session-relative DP (penalty = video mean top-1 + δ) | same F1 ≈ 0.007 on pseudo-sentences, but produces non-empty output on real video (absolute thresholds emit nothing there) | Used as deployment default (δ = 0.1) so outputs and top-k are visible |
| 14 | Real-world knobs: vocabulary (full vs TNC top-5000 = 1,155 signs) × mirror-TTA × session centering | centering brings correct signs to rank 1 in *some* window; TTA does not help; TNC vocab drops ปลอบใจ/ด้วยกัน | Centering on; full vocabulary default |

---

## 5. Results — test cases (TTRS, leakage-safe)

### 5.1 Main table (test split unless noted; E1 gallery = 4,706 signs, E2 gallery = 18 signs)

| Model | E1 R@1 | E1 R@5 | E1 R@10 | E1 median rank | E1 median rank (session-centred) | E2 R@1 | E2 R@5 | E2 R@5 **val** |
|---|---|---|---|---|---|---|---|---|
| Frozen mean-pool (active) | 0.000 | 0.000 | 0.000 | 2046 | — | 0.132 | 0.289 | 0.474 |
| Frozen + NAP (d = 2) | 0.000 | 0.000 | 0.018 | 1018 | — | 0.158 | 0.395 | 0.553 |
| SSL kNN (no labels) | 0.000 | 0.000 | 0.000 | 2059 | — | 0.158 | 0.289 | 0.579 |
| ISLR scratch (4 streams) | 0.000 | 0.054 | 0.054 | 720 | 590 | 0.263 | 0.658 | 0.711 |
| ISLR no-face + adversary 0.5 | 0.000 | 0.054 | 0.089 | 787 | 706 | 0.211 | 0.658 | 0.763 |
| ISLR pose-only | 0.000 | 0.018 | 0.054 | 1232 | — | 0.132 | 0.500 | 0.763 |
| ISLR SSL-init, LoRA r4 | 0.000 | 0.018 | 0.018 | 706 | 740 | 0.184 | 0.553 | 0.737 |
| ISLR SSL-init, full FT | 0.000 | 0.036 | 0.054 | 712 | 594 | 0.211 | 0.526 | 0.842 |
| ISLR SSL-init + train centering | 0.000 | 0.000 | 0.000 | 1356 | 628 | 0.263 | 0.500 | 0.816 |
| **ISLR SSL-init + fast speed aug (selected)** | **0.000** | **0.000** | **0.018** | **547** | **491** | **0.316** | **0.658** | **0.895** |

Chance: E1 median rank ≈ 2,353, R@10 ≈ 0.002; E2 R@1 0.056, R@5 0.278. ±95% CI on E1 R@5 ≈ ±0.05; on E2 R@5 ≈ ±0.15.

**Reading:** training lifts the correct sign from the middle of the dictionary (rank ~2000) into roughly the top 12% (rank ~500). Top-1 dictionary recognition across signers is still essentially zero. On small galleries (18 signs) the model is clearly above chance.

### 5.2 Robustness to simulated webcam queries (aug1: new background + low-res + colour)
| Model | E1 median rank base → aug1 (test) | E2 R@5 base → aug1 (test) |
|---|---|---|
| ISLR scratch | 720 → 740 | 0.658 → 0.500 |
| ISLR SSL full FT | 712 → 994 | 0.526 → 0.500 |
| **ISLR SSL fast (selected)** | 547 → 779 | 0.658 → 0.632 |

Degradation is moderate. The studio → webcam gap is not the dominant failure; signer identity is.

### 5.3 Pseudo-continuous sentences (60 val / 60 test sentences, 3–6 signs)
| Model | Decoder (val-selected by F1) | Test gloss F1 | Test precision / recall | Test WER | Hypotheses per reference |
|---|---|---|---|---|---|
| ISLR scratch | NMS τ = 0.4, windows 12–32 | 0.006 | 0.004 / 0.011 | 2.56 | 2.60 |
| ISLR SSL fast | NMS τ = 0.7, windows 12–32 | 0.000 | 0.000 / 0.000 | 1.04 | 0.55 |
| ISLR SSL fast | DP penalty 0.4 (default) | 0.000 | 0.000 / 0.000 | 1.71 | 1.71 |
| ISLR SSL fast | session-relative DP δ = 0.15 (val) | val F1 0.007 | — | val 1.07 | 0.98 |

**Continuous recognition does not work (T08 fails).** This follows directly from E1: when the correct sign is ranked ~500th, no decoder can recover sequences.

---

## 6. Results — real-world inference (`data_test/`)

Final configuration (`python scripts/cli.py infer --video <file>`): `islr_ssl_fast`, 4,742-sign bank, session centering, session-relative DP (δ = 0.1, windows 12–32), rules composer (no OpenAI key), local MMS-TTS, affect head.

### 6.1 Outputs
| Video | Reference (glosses) | In TTRS vocab | System glosses | Sentence (TTS) | Gloss WER | chrF |
|---|---|---|---|---|---|---|
| `ฉันปลอบเพื่อนร้องไห้.mp4` (10.3 s, schoolgirl, classroom) | ฉัน ปลอบใจ เพื่อน ร้องไห้ | 4/4 | นั่งสมาธิ · ความคิดไม่ดี · ฤดูหนาว · สนใจ · เหงา | "นั่งสมาธิความคิดไม่ดีฤดูหนาวสนใจเหงา" | 1.25 | 5.7 |
| `ไปทานข้าวด้วยกันมั้ย.mp4` (6.7 s, man in suit, on-screen text) | ไป กิน ข้าว ด้วยกัน ไหม | 3/5 (ไป, ไหม missing) | ขนมปัง · ลูกกระเดือก · ไม้พลอง | "ขนมปังลูกกระเดือกไม้พลอง" | 1.00 | 8.1 |

Artifacts: `result_reporting/inference/final/<video>/{result.json, subtitle.srt, audio.wav, summary.json}`.
Blueprint targets (WER ≤ 0.45, chrF ≥ 40) were **not met**.

Some decoded glosses are semantically near the target: "เหงา / ความคิดไม่ดี" for comforting a crying friend, and "ขนมปัง" (bread) for eat-rice. This is anecdotal, not evidence of recognition.

### 6.2 Diagnostic: does the model ever rank the correct sign highly? (best rank over all windows, top-10 lists)
| Config | ฉัน | ปลอบใจ | เพื่อน | ร้องไห้ | กิน | ข้าว | ด้วยกัน |
|---|---|---|---|---|---|---|---|
| full vocab | >10 | >10 | >10 | >10 | >10 | >10 | >10 |
| full vocab + **session centering** | **1** | >10 | >10 | >10 | **4** | >10 | >10 |
| full vocab + centering + mirror TTA | **1** | >10 | >10 | >10 | 4 | >10 | >10 |
| TNC top-5000 vocab (1,155 signs) | 7 | (not in vocab) | >10 | 8 | >10 | >10 | (not in vocab) |
| TNC vocab + **session centering** | **1** | — | 6 | >10 | **1** | >10 | — |

For comparison, the scratch model with centering reached ฉัน 4 (full vocab), and ฉัน 1 / เพื่อน 4 / กิน 7 / ข้าว 6 (TNC vocab).
**Session centering is the single most effective real-world knob.** It was adopted because it improved E1 on val for all 4 models, not because of these 2 videos. The "which config is best on the videos" table above was computed on the test videos, so treat it as a diagnostic, not as a tuned score.

### 6.3 Face Impression and NMM (both videos)
| Component | Output | Validation |
|---|---|---|
| Affect head (distilled from ViT FER) | video 1: neutral 0.76, happy 0.14 · video 2: neutral 0.78, **surprise 0.12** (he raises his brows while asking) | top-1 agreement with the teacher on held-out clips: **TTRS 0.94** (majority-class baseline 0.92) · **YouTube 0.60** (baseline 0.49). No human affect labels exist (Bench-Face not recorded) |
| NMM rules | head-shake 0 cycles, nod 1 cycle on both → no negation (correct for both sentences) | `question` detection is **not implemented** (needs face landmarks / brow features), so "มั้ย" was not produced (T09 not met) |

### 6.4 TTS
Offline MMS-TTS Thai produces 16 kHz WAVs. Synthesis took 0.8–2.0 s per sentence on CPU. `SpeechStyle` maps affect to rate/pitch; the persona hook is implemented for the OpenAI provider (untested, no key).

---

## 7. Runtime (T11 proxy: frame-by-frame replay of the real videos)

| Device | Offline ms/frame (pose + ViT + windows) | Stream p50 | Stream p95 | Stream max | RSS (incl. TTS + FER + torch) |
|---|---|---|---|---|---|
| RTX 2050 (ORT-CUDA pose, fp16 ViT) | 28.6–30.4 | **29.0 ms** | 63.8–73.1 ms | 81 ms | 2.64–2.69 GB (stable) |
| CPU (ORT-CPU pose, fp32 ViT) | 77.6–79.4 | **70.2 ms** | 108–116 ms | 127 ms | 2.88–3.38 GB |

* GPU: comfortably real-time at 12.5 fps (80 ms budget).
* CPU: the p50 fits 12.5 fps but p95 exceeds 80 ms. **T11 (CPU p95 ≤ 80 ms, RAM ≤ 2.5 GB) is not met.** The optimisations not yet done are the ones blueprint §7 lists: int8 ONNX ViT, lazy TTS/FER loading, batching the 3 crops with pose.
* RSS stayed flat during the stream replay (no leak observed over 129 frames; a 3-minute webcam run was not possible without a camera).

---

## 8. Blueprint test-case checklist

| ID | Case | Status | Evidence |
|---|---|---|---|
| T01 | word, new signer, R@1 ≥ 0.55 | ❌ | E1 test R@1 0.000; E2 (18-sign gallery) R@1 0.316 |
| T02 | low light Δ ≤ 0.10 | ⚠️ proxy | aug1 queries: E2 R@5 0.658 → 0.632 |
| T03 | left-handed Δ ≤ 0.05 | ➖ not measured | mirror TTA implemented; no left-handed signer data |
| T04 | busy background, lock ≥ 0.95 | ⚠️ qualitative | correct lock on 2-person Big Sign clip; no labelled set |
| T05 | far/close valid-rate ≥ 0.9 | ✅ (on data seen) | valid-rate 1.00 on TTRS, Big Sign, both real videos |
| T06 | unknown rejection | ➖ | not calibrated (recognition too weak for a meaningful τ_unknown) |
| T07 | one-shot enrollment R@5 ≥ 0.70 | ⚠️ | E2 R@5 0.658 test / 0.895 val on an 18-sign gallery (blueprint's larger gallery not available) |
| T08 | sentence WER ≤ 0.45, chrF ≥ 40 | ❌ | pseudo-continuous F1 ≈ 0; real-world WER 1.25 / 1.00, chrF 5.7 / 8.1 |
| T09 | negation/question F1 ≥ 0.75 | ❌ / partial | negation rules correct on 2 non-negated videos; question not implemented |
| T10 | emotion acc ≥ 0.70 | ➖ | no human labels; teacher agreement 0.60 on held-out YouTube |
| T11 | real-time CPU p95 ≤ 80 ms, RAM ≤ 2.5 GB | ❌ CPU / ✅ GPU | §7 |
| T12 | offline LLM → rules, no crash | ✅ | no API key → RuleComposer used in every run |
| T13 | leakage guard (caption band) | ⚠️ by construction | band + logo masked before pose; crops exclude the band; ΔR@1 not measured |
| T14 | Bench ≥ 0.6 × TTRS | ➖ | no Bench recorded |
| T15 | NAP ablation ≥ +3 pt | ✅ (frozen) | frozen E2 R@5 +8 pt (val), NN-same-signer 0.89 → 0.55 |
| T16 | active mask on webcam | ⚠️ | active fraction 1.00 / 0.93 on real videos: stream-mode rest detection is too permissive |
| T17 | schema round-trip | ✅ | 5,331 clips ingested, 0 cache errors |
| T18 | 112 vs 224 px | ➖ | only 112 px cached (compute budget) |
| T19 | multi-person lock | ✅ qualitative | Big Sign vertical |
| T20 | very short clip | ✅ | 2.6 s YouTube clips and 1.8 s TTRS clips processed without error |

---

## 9. Findings

1. **The bottleneck is signer variety, not model capacity or augmentation.** The embedding is still dominated by who is signing: the nearest neighbour is the same signer even after NAP. Every model trained on 3 signers ranks a new signer's sign around #500–#1000. Adversarial training, dropping the face stream, CSLS and training-time centering did not fix it; test-time session centering helped most.
2. **Blueprint v2's split had a hidden flaw.** Putting all cross-signer signs into val/test removes the only training signal for signer invariance. It also made the "full-gallery" metric measure unseen-vs-seen classes instead of new signers. Fixed with E1/E2.
3. **Temporal SSL on 13 h of YouTube + TTRS helps as initialisation for one-shot enrollment (E2 val R@5 0.71 → 0.84–0.90), but not on its own.** Its losses plateau early. Label-free kNN never beat frozen DINOv2 + NAP.
4. **Conversational speed matters.** Real signs take ~0.6 s versus 1.5–3 s in TTRS. Speed augmentation up to 2.5× gave the best E1.
5. **Continuous decoding is premature.** With a correct-sign median rank of ~500, any decoder is at chance. Absolute similarity thresholds do not transfer from studio to phone video (they emit nothing), which is why the relative-penalty decoder was added.
6. **The engineering parts work:** robust articulators (valid-rate 1.0 across domains), a 0-error cache over 862k frames, a GPU real-time pipeline, TTS, affect, and an offline fallback.

## 10. Recommended next steps (in priority order)

1. **Record ThaiSLM-Bench-Word** (blueprint §6.3): 200 common signs × ≥ 10 signers × phone/webcam. Use ~70% of signers for training positives (cross-signer pairs) and 30% as the true test. This is the single change most likely to move E1.
2. **Mine cross-signer instances from YouTube lessons** (VTT word timestamps, 658 lexemes with hits, blueprint §6.2.7). Even noisy instances add signer variety.
3. Request the **NADT licence** (th-sl.com); more signers per sign.
4. **Restrict the deployment vocabulary** to a task domain (e.g. 200–500 daily-conversation signs) and add sentence-level enrollment from the target user. Centering + TNC vocab already brought 2 of 3 in-vocab signs to rank 1 in some window.
5. Add face landmarks (brow raise) for **question NMM**, and fix the stream-mode active mask (rest detection on real video).
6. CPU: int8 ONNX DINOv2-S, lazy-load TTS/FER, and 3-crop + pose batching to reach p95 ≤ 80 ms and RAM ≤ 2.5 GB.
7. Only after E1 R@5 ≥ ~0.3: revisit the boundary head, the LLM re-ranker (needs `OPENAI_API_KEY`), and seq2seq v2.

---

## 11. Reproduce

```bash
conda activate hugging
pip install --no-deps rtmlib && pip install onnxruntime-gpu
python scripts/cli.py manifests && python scripts/cli.py bgpool --count 400
python scripts/cli.py cache --set ttrs --workers 10          # ~32 min
python scripts/cli.py cache --set youtube --workers 9        # ~100 min
python scripts/experiments.py pack --set ttrs
python scripts/experiments.py signers --cluster_thr 0.45     # signer inventory + E1/E2 splits
python scripts/experiments.py pack --set youtube
python scripts/experiments.py baselines
python scripts/experiments.py ssl --tag ssl_v1 --steps 6000 --batch 48 --eval_every 1500
python scripts/experiments.py islr --tag islr_ssl_fast --init ssl_v1 --speed_max 2.5 --streams hand_l,hand_r,pose --w_adv 0.5 --steps 1500
python scripts/experiments.py face --device cuda
python scripts/experiments.py evalckpt --tags islr_scratch,islr_noface_adv,islr_ssl_ft,islr_ssl_fast --out final
python scripts/experiments.py bank --tag islr_ssl_fast
python scripts/experiments.py continuous --tag islr_ssl_fast --n_sent 60
python scripts/experiments.py realworld --tag islr_ssl_fast --tta --audio
python scripts/experiments.py latency --tag islr_ssl_fast --devices cuda,cpu
python scripts/cli.py infer --video "data_test/ไปทานข้าวด้วยกันมั้ย.mp4"
```

## 12. Artifact index
| Path | Content |
|---|---|
| `artifacts/reports/baselines.json` | frozen / NAP baselines (E1, E2, NN-same-signer) |
| `artifacts/reports/islr_*.json` | every ISLR run: args, training history, val/test, aug1 queries |
| `artifacts/reports/evalckpt_final.json` | E1 × CSLS × centering × base/aug1 for 4 models |
| `artifacts/reports/ssl_ssl_v1.json` | SSL loss curves and kNN evals |
| `artifacts/reports/continuous_*.json` | decoder grid (val) and test |
| `artifacts/reports/realworld_*.json` | 8 configs × 2 videos: hypotheses, WER, chrF, best window ranks, timings |
| `artifacts/reports/latency_islr_ssl_fast.json` | GPU/CPU stage timings, stream p50/p95, RSS |
| `artifacts/reports/face_affect.json` | teacher agreement, class distributions |
| `artifacts/reports/signers.json` | signer inventory statistics |
| `artifacts/checkpoints/{ssl/ssl_v1.pt, islr/*.pt, face/affect_head.pt}` | models |
| `artifacts/prototypes/islr_ssl_fast.npz` | 4,742-sign prototype bank (with signer ids for centering) |
| `result_reporting/inference/final/` | final real-world outputs (json / srt / wav) |
| `result_reporting/inference/islr_ssl_fast/`, `islr_scratch/` | outputs for every real-world config |
| `logs/thaislm/` | raw logs (including the superseded v0-split run `islr_scratch_v0split.log`) |
