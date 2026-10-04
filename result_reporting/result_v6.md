# ThaiSLM — Result Report v6

**Scope:** v6 = v5 + the new th-sl.com dictionary (7,719 clips), one extractor for all video (TTRS re-extracted at 25 fps),
a standard data-prep layer on S3 (`data_prep/`), and the improvements asked for in prompt_7 / prompt_8 — each one tried, measured on
held-out data, and **kept only if it helped**. Architecture: `architecture_v6.md`. Representation space: `latent_space_v6.md`.
Data schema: `cloud_s3/data_prep/SCHEMA.md`. Entry point: `python main.py …`; post-training check: `lab.ipynb`.

---

## 0. TL;DR

| question | answer |
|---|---|
| Is v6 better than v5 on real-world video (data_test, from the raw videos)? | **Yes.** Reference word top-1 in some segment **9/22** (v5 7/22) · in top-5 **14/22** (v5 10/22) · accepted words correct **7/7 = 100 %** (v5 6/12 = 50 %). Newly found: **ร้องไห้, ดื่ม** (top-1) and **น้ำ** (#2) — words that had 1–3 examples in v5 and got th_sl signers. Word-count error per video 1.0 (v5 0.8). |
| Better on held-out signers (isolated signs)? | **Yes, on the same queries and the same bank** (v5 encoder re-scored on v6 data): TTRS new signers R@1 0.155 → **0.190**, TSL-ONE-S 5 new signers 0.864 → **0.890**, th_sl unseen signer 0.090 → **0.105**, YouTube signers 0.176 → **0.235**. Null-vs-sign AUC (RTMW) 0.86 → **0.95**. TSL51 researcher (1 signer, 51 words) 0.368 → 0.342 (RTMW) / 0.633 → 0.620 (MediaPipe) — slightly lower (§4). |
| Better on continuous signing (held-out TSL51 templates)? | **More trustworthy and much better on fast signing; slightly worse at natural pace.** Accepted-tier precision **0.85** (RTMW, v5 0.76) / **0.91** (MediaPipe, v5 0.84). At 2× speed: exact word count **67 %** (v5 27 %), recall@1 **0.456** (v5 0.382), WER **0.558** (v5 0.597). At 1×: recall@1 0.487 (v5 0.473) but segment F1 0.793 (v5 0.835), exact count 64 % (v5 71 %), WER 0.571 (v5 0.524 before / 0.467 after the grammar re-rank). §5 explains why (the vocabulary doubled) and what was tried. |
| Did the new data help? | **Yes, measured by training without it** (same bank): without th_sl, TTRS new signers 0.131 (with: 0.190), TSL-ONE-S 0.858 (0.890). Without the MediaPipe sources, TSL-ONE-S 0.780 — neither source is noise. Vocabulary 4,570 → **9,869 concepts**; concepts with ≥ 2 signers 605 → **1,969**. |
| Is the data clean / one standard? | One Standard Schema for 19,897 clips / 26.7 h (22.7 h RTMW = the real-video extractor), every file and column documented in `data_prep/SCHEMA.md`. QC removed only 58 clips (45 duplicates of th_sl videos inside TSL51, 2 body-part close-ups with no signer, 1 corrupt video …); a first QC rule that would have dropped 175 good TSL-ONE-S clips was caught and replaced. |
| prompt_8 benchmark — which encoder / tagger? | Encoder: **Uni-Sign CSL→WLASL + Thai fine-tune** wins (val 0.5005) over CSL-only (0.442), scratch (0.342), Thai self-supervised (0.481), signer augmentation (0.4995), extractor-adversarial (0.4954). Tagger: **BiGRU** (selection 2.709) over TCN (2.642) and Transformer (2.541); self-training on 8.7 h of broadcasts did not help (2.659) and was not used. How2Sign / WLASL data were not added (§7). |
| Does the model learn TSL or lean on the ASL/CSL prior? | It learns TSL: Thai fingerspelling (handshapes that do not exist in ASL/CSL) improves as much as lexical signs (0.82 → **0.90** vs 0.78 → 0.88); the representation moved substantially (CKA 0.60); removing the ASL step costs 0.06 on validation; training from scratch costs 0.16. But **38 %** of the remaining errors are confusions the zero-shot ASL/CSL model already made — the prior's confusions are only partly unlearned. |
| Cost | GPU ≈ **$12.8** (pose extraction on 4 Spot T4 $3.1 · 8 encoder runs on Spot A100 $8.6 · sequence stage $1.1) + OpenAI ≈ $1 + storage < $1. After the round: GCS bucket emptied (kept), all 63 Vertex jobs deleted, no VM / disk / IP / endpoint exists. |

---

## 1. What changed from v5, and why (prompt_7 bottlenecks)

| v5 bottleneck (prompt_7) | v6 action | result |
|---|---|---|
| (1) bank depth decides recognition | th_sl added (7,719 clips, 8 signer clusters); ไหม checked in every source; per-depth calibration tried | multi-signer concepts 605 → 1,969; ร้องไห้ / ดื่ม / น้ำ now found on data_test. ไหม is in **no** source (only ไหม้, ไหมพรม). Depth as a ranking prior or calibration feature was **rejected** — it encodes which dataset recorded most signers (§5.3) |
| (2) RTMW ↔ MediaPipe extractor gap | every video re-extracted with the inference extractor; extractor-adversarial loss tried | adversarial loss did not help (val 0.4954 < 0.5005); the gap remains on the single TSL51 test signer (MP 0.620 vs RTMW 0.342 open-vocab), but RTMW sources now carry most of the vocabulary |
| (3) accepted precision 0.76 | calibration re-fitted on real sentences, thresholds for precision ≥ 0.85 on tune, priors kept out of segmentation | **0.85** (RTMW) / **0.91** (MP) on held-out templates; data_test 7/7 |
| (4) tagger boundaries on real data | tagger benchmark (BiGRU/TCN/Transformer), real speed-perturbed sentences, self-training on 8.7 h of unlabeled broadcasts | BiGRU kept; self-training did not improve held-out real templates → not used; boundaries vs pseudo-labels remain weak (§5.1) |
| (5) hubness | CSLS hubness correction implemented and tuned | helps RTMW dictionary queries (th_sl val +0.014 MRR, TTRS val +0.053) but hurts conversational sentences → off in the final decoder (§5.3) |
| (6) fast phone signing | tempo normalisation (time-stretch to training pace) | 2× speed exact count 27 % → **67 %**, recall@1 0.38 → 0.46 |

## 2. Data (details: `architecture_v6.md` §2–3, `data_prep/SCHEMA.md`)

![data](figures_v6/data_sources.png)

| | v5 | v6 |
|---|---|---|
| clips / videos in the standard schema | 12,131 | **19,897** (+ 102 continuous videos) |
| hours of pose | — | **26.7 h** (22.7 h RTMW) |
| isolated clips · concepts | 11,542 · 4,570 | **19,211 · 9,869** |
| concepts with ≥ 2 / ≥ 3 / ≥ 4 signers | 605 / – / – | **1,969 / 567 / 280** |
| signer ids | – | 151 (≈ 50 identifiable people + ≈ 80 YouTube signers with 1–3 clips) |
| TTRS poses | v4, 12.5 fps up-sampled | **re-extracted, 25 fps, same code as inference** |

Held-out roles: v5 roles kept for every v5 clip (identical test sets); th_sl val / test = one visually pure signer cluster each
(1,203 / 1,338 clips), checked against 340 TTRS / TSL51 faces for identity leakage (none).

## 3. Benchmarks (prompt_8)

### 3.1 Encoder (same data, losses, schedule; selection = mean new-signer MRR on TTRS / TSL-ONE-S / th_sl val signers)

| encoder | val (select) | TTRS test | TSL-ONE-S test | th_sl test | TSL51 RTMW | TSL51 MP | YouTube | GPU min |
|---|---|---|---|---|---|---|---|---|
| **Uni-Sign CSL→WLASL + Thai (all data)** | **0.5005** | **0.190** | 0.890 | 0.105 | 0.342 | 0.620 | 0.235 | 43 |
| + signer augmentation (retarget + mirror) | 0.4995 | 0.167 | **0.896** | 0.112 | 0.330 | **0.655** | 0.235 | 42 |
| + extractor-adversarial (GRL) | 0.4954 | 0.190 | 0.883 | **0.116** | 0.340 | 0.601 | **0.294** | 32 |
| Thai self-supervised (1,500 SSL steps) → fine-tune | 0.4809 | 0.179 | 0.885 | 0.112 | 0.281 | 0.622 | 0.235 | 73 |
| Uni-Sign CSL only (no ASL step) | 0.4424 | 0.107 | 0.814 | 0.090 | 0.218 | 0.608 | 0.118 | 30 |
| same architecture from scratch | 0.3422 (stopped at 4,500 steps, still far below) | – | – | – | – | – | – | 45 |
| data ablation: without th_sl | 0.4676 | 0.131 | 0.858 | 0.116 | 0.389 | 0.676 | 0.235 | 19 |
| data ablation: without MediaPipe sources | 0.3538 | 0.179 | 0.780 | 0.101 | 0.361 | 0.618 | 0.176 | 47 |

All test columns use **the same bank** (every original training clip) and vocabulary, so only the encoder differs. Choice by
validation only: the all-data encoder. Reading: pretraining is essential (scratch), the ASL step helps (CSL-only), the Thai
self-supervised stage did not add to it (its contrastive task stayed too easy — view-match accuracy 1.00 — while masked-pose loss
fell), augmentation and the adversarial loss are within noise, and th_sl helps TTRS / TSL-ONE-S but costs ~5 points on the TSL51
researcher (§4).

![training curves](figures_v6/training_curves.png)

### 3.2 Tagger (held-out real *tagger-val* templates; sentence-test templates never used)

| tagger | selection | boundary F1 | sign-frame F1 | count error |
|---|---|---|---|---|
| **BiGRU** | **2.709** | 0.202 | 0.694 | 0.52 |
| TCN (dilated) | 2.642 | 0.172 | 0.673 | 0.64 |
| Transformer encoder | 2.541 | 0.142 | 0.705 | 0.79 |
| BiGRU + self-training (1,209 pseudo-labelled 24 s chunks from 102 broadcasts) | 2.659 | – | – | – |

## 4. Isolated signs, held-out signers (`artifacts/reports/isolated.json`, `baseline_v5.json`)

Same queries, bank = training clips of v6 (open vocabulary ≈ 9.9k concepts):

| protocol | n | v5 encoder | **v6 encoder** | zero-shot Uni-Sign |
|---|---|---|---|---|
| TTRS test signers | 84 | 0.155 | **0.190** | 0.119 |
| TSL-ONE-S 5 unseen signers | 873 | 0.864 | **0.890** | 0.796 |
| th_sl unseen signer (RTMW) | 277 | 0.090 | **0.105** | 0.083 |
| YouTube title signers | 17 | 0.176 | **0.235** | 0.294 |
| TSL51 researcher, MediaPipe | 521 | **0.633** | 0.620 | 0.608 |
| TSL51 researcher, RTMW | 524 | **0.368** | 0.342 | 0.372 |
| null vs sign AUC (RTMW / MP) | 547 | – | **0.951 / 0.895** | 0.861 / 0.839 |

v5 *as shipped* (its own 4.5k-concept bank) scores TSL51-RTMW 0.525 — the drop to ~0.35 is mostly the **vocabulary doubling**
(th_sl adds ~5k concepts, many visually close), not a worse encoder. Recognition by bank depth (held-out test signers):

![depth](figures_v6/recall_by_depth.png)

R@1 ≈ 0.13–0.15 for words with 1–2 training signers, ≈ 0.45–0.48 with 3–9, 0.88 with ≥ 10. **Signers per word remain the
strongest lever** (prompt_7's diagnosis confirmed with v6 data).

A zero-shot + fine-tuned score fusion would raise TSL51-RTMW (0.488 → 0.547 MRR) but lowers the validation objective (0.5003 →
0.4933) → not used (choosing it from test numbers would be tuning on test).

## 5. Continuous signing (held-out TSL51 templates; `artifacts/reports/spotting.json`)

### 5.1 Final decoder vs v5

| test set | seg F1 | exact count | count err | recall@1 | recall@5 | prec@1 | WER |
|---|---|---|---|---|---|---|---|
| v5 RTMW | **0.835** | **0.71** | **0.35** | 0.473 | **0.718** | **0.486** | 0.524 (re-ranked 0.467) |
| **v6 RTMW** | 0.793 | 0.635 | 0.49 | **0.487** | 0.651 | 0.474 | 0.571 (re-ranked 0.588) |
| v5 RTMW 1.5× | **0.817** | 0.54 | 0.56 | 0.456 | **0.662** | **0.524** | **0.530** |
| **v6 RTMW 1.5×** | 0.800 | **0.683** | **0.43** | **0.467** | 0.647 | 0.467 | 0.570 |
| v5 RTMW 2× | 0.723 | 0.27 | 0.92 | 0.382 | 0.542 | 0.509 | 0.597 |
| **v6 RTMW 2×** | **0.786** | **0.667** | **0.42** | **0.456** | **0.629** | 0.478 | **0.558** |
| v5 MediaPipe | **0.851** | 0.71 | 0.33 | **0.653** | **0.776** | **0.689** | 0.333 (0.320) |
| **v6 MediaPipe** | 0.839 | **0.730** | 0.41 | 0.600 | 0.767 | 0.566 | 0.442 (0.425) |

**Status tiers (held-out templates)** — accepted words: RTMW **0.85 precision** (27 % of segments; v5 0.76 at 50 %), MediaPipe
**0.91** (46 %; v5 0.84 at 70 %), RTMW 2× **0.83** (v5 0.77). v6 accepts fewer words but they are right more often — the behaviour
data_test shows (7/7).

### 5.2 Components on the test templates (each switched on in turn)

| | RTMW seg F1 / exact / R@1 / WER | 2× seg F1 / exact / R@1 / WER |
|---|---|---|
| decoder only | 0.793 / 0.635 / 0.411 / 0.651 | 0.690 / 0.310 / 0.340 / 0.633 |
| + Thai word-frequency prior (recognition only) | 0.793 / 0.635 / **0.487** / **0.571** | 0.690 / 0.310 / 0.398 / 0.553 |
| + tempo normalisation (final) | 0.793 / 0.635 / 0.487 / 0.571 | **0.786 / 0.667 / 0.456 / 0.558** |

### 5.3 Decisions, including the ones that were reversed (all on held-out validation data; data_test only reported)

1. **Priors must not touch segmentation.** Adding word priors to window scores inflated window gains and over-split signs (exact
   count 0.64 → 0.38); segmentation now uses visual scores only and priors re-rank words inside each segment.
2. **Tune templates are not a valid selection set** — the tagger trained on them (segment F1 ≈ 1.0 there); decoder priors and tempo
   are chosen on the held-out tagger-val templates.
3. **Bank depth was rejected as a ranking prior and as a calibration feature.** It improved TSL51 validation strongly (0.774 → 0.894)
   but the mechanism is a dataset confound: TSL51's 51 words overlap TSL-ONE-S's family / number vocabulary, which has 29 signers. On
   real video it surfaced TSL-ONE-S words (ศูนย์, หก, แม่, เด็ก, ประเทศไทย) everywhere. Removed on that principle; reported as runs
   A/B in §6.
4. **CSLS hubness correction**: on broad dictionary validation (th_sl / TTRS / TSL-ONE-S val signers) α = 0.5 + γ = 0.1 is best
   (MRR 0.5005 → 0.5163), but on conversational sentences it penalises exactly the common words a conversation uses. The
   deployment target is conversation → selection on the conversational validation set → **α = 0, γ = 0.25**.

## 6. Real-world: data_test (from the raw videos; `artifacts/reports/infer_final.json`, `result_reporting/inference/`)

| run (decided on held-out data, data_test only reported) | top-1 in some segment | top-5 | accepted correct |
|---|---|---|---|
| v5 final (run B) | 7/22 | 10/22 | 6/12 |
| v6 run A: depth prior + depth calibration | 7/22 | 14/22 | 5/7 |
| v6 run B: frequency prior + depth calibration | 7/22 | 14/22 | 7/17 |
| v6 run C: broad-val CSLS + frequency prior | 7/22 | 10/22 | 2/2 |
| **v6 final: conversational-val frequency prior, no depth anywhere** | **9/22** | **14/22** | **7/7** |

Per video (final; `word` accepted · rank of the reference word among a segment's top-5):

| video | reference | segments | top-1 per segment | accepted | reference ranks | sentence shown |
|---|---|---|---|---|---|---|
| test1 | สวัสดี สบายดี ชอบ โกรธ แมว แฟน ปรบมือ | 7 / 7 | สวัสดี · สบายดี · ฉัน · อะโดบี อิลลัสเตรเตอร์ · แมว · เหมือนกัน · แมว | สวัสดี, สบายดี, แมว | สวัสดี 1 · สบายดี 1 · แมว 1 · โกรธ 2 · ชอบ 4 · แฟน – · ปรบมือ – | "สวัสดี ฉันสบายดี แมวเหมือนกันไหม" (partly) |
| ฉันปลอบเพื่อนร้องไห้ | ฉัน ปลอบใจ เพื่อน ร้องไห้ | 4 / 4 | เพื่อน · **ร้องไห้** · สงสาร · เด็ก | เพื่อน | เพื่อน 1 · **ร้องไห้ 1** · ปลอบใจ (สงสาร/ปลอบโยน nearby) – | "เพื่อนร้องไห้สงสารเด็ก" (partly) |
| ฉันรักเพื่อน | ฉัน รัก เพื่อน | 2 / 3 | แต่งงาน · เด็ก | – | ฉัน 2 | withheld |
| พ่อดื่มน้ำ | พ่อ ดื่ม น้ำ | 6 / 3 | ศูนย์ · พ่อ · ตา · **ดื่ม** · เด็ก · เด็ก | พ่อ | พ่อ 1 · **ดื่ม 1** · **น้ำ 2** | "พ่อ ตา ดื่ม เด็ก" (wrong extra words) |
| ไปทานข้าวด้วยกันมั้ย | ไป กิน ข้าว ด้วยกัน ไหม | 4 / 5 | ฉัน · เดียวกัน · กิน · ไป | กิน, ไป | ไป 1 · กิน 1 · ข้าว 5 · ไหม not in any data | "ฉันกินเหมือนกันไป" (partly) |

Remaining real-world errors, honestly: 3-signs-in-1.9 s phone signing (ฉันรักเพื่อน) still becomes 2 segments; extra segments on
hands moving between signs in พ่อดื่มน้ำ (6 for 3) are output as uncertain words (`เด็ก?`) that the sentence layer then uses; รัก /
ปรบมือ / แฟน / ปลอบใจ are not found (1–3 training signers); ไหม cannot be found (no source has it).

## 7. What was not done, and why

* **How2Sign / WLASL as training data (prompt_8):** How2Sign has sentence-level alignment only (no per-sign timing), so it would
  add unlabeled continuous signing — which v6 already tested with 8.7 h of Thai broadcasts (self-training: no gain) — plus the
  ASL-confusion risk you named. Not downloaded (large external download; it would need your OK).
* **Re-training the tagger on all tune templates** (v5 trained on all; v6 held ¼ out for honest selection, which likely explains
  part of the natural-pace count gap): one more A100 job (~$1.1). Not run under the cost limit of this round; it is the cheapest
  next step for §5.1.

## 8. Cost and GPU usage

| job group | GPU | minutes | ≈ USD |
|---|---|---|---|
| pose extraction (13,663 videos, 2.0 M frames; 4 shards + retries, smoke tests) | Spot T4 | 486 | 3.08 |
| encoder runs (all, adv, nothsl, nomp, csl, scratch, ssl, aug) | Spot A100 | 334 | 8.62 |
| sequence stage (alignment, synthesis, tagger benchmark, self-training) | Spot A100 | 43 | 1.10 |
| **total GPU** | | **863** | **≈ 12.8** |

Local (free): face embeddings / signer clusters, ingest, lexicon, speed + tempo tables (2 h on the laptop GPU), tuning,
evaluation, data_test, ONNX export. Lessons: one-session GPU memory caps (an ORT arena OOM made one worker per shard fail every
clip), a stall watchdog, Cloud Storage client staging instead of the bucket mount, cancel benchmark losers early.

**Clean-up done:** `gs://temp-gpu-bucket-job` emptied (0 objects, bucket kept, soft-delete retention 0 → nothing billed), all 63
Vertex AI custom jobs deleted in 6 regions, no Compute Engine instance / disk / address, no Vertex endpoint / model.

## 9. Reproduce

```bash
python main.py data pull data_prep                     # or rebuild: python main.py prep … (README)
python main.py cloud upload-code --tag v6 && python main.py cloud upload-data && python main.py cloud upload-models
python main.py cloud launch --tag v6 --stage train_encoder --args "--steps 4000 --eval_every 300 --patience 4 --P 48 --K 2 --ema 0.998"
python main.py cloud launch --tag v6 --stage sequence --args "--encoder_run <encoder_run> --epochs 20 --n_synth 3000"
python main.py eval build --run <encoder_run> --seq <sequence_run> && python main.py eval grammar
python main.py eval fast --seq <sequence_run> && python main.py eval spot --seq <sequence_run> --no_depth_calibration
python main.py eval isolated --run <encoder_run> && python main.py eval baseline --run <encoder_run>
python main.py test && python main.py eval vocab && python main.py export-onnx
```

Runs used: encoder `enc-all-0241-us-central1`, sequence `seq-all-0340-asia-southeast1` (`models/config.json`).
