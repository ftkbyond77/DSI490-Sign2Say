# v12 handoff (prompt/phase3/prompt-opt-v12.md)

## สรุปสำหรับเจ้าของงาน (TPH 1 — ข้อมูล, 2026-11-15)
- **ค่าใช้จ่าย TPH 1 = $0** (สกัด landmark บน CPU เครื่อง, embedding บน RTX 2050) — งบ Modal $6–7 เก็บไว้ทั้งหมดสำหรับ TPH 2
- ข้อมูลใหม่: e-TSL มีผู้ทำท่า **1 คน** (7 ช่วงการถ่าย), ราชบัณฑิต **3 คน** (+ 2 คนในคลิปถาม–ตอบ) · คำศัพท์ 328 คำ ได้ผู้ทำท่าเพิ่ม 211 คำ
  (คำที่มีผู้ทำท่า ≤ 1 คน ลดจาก 96 → 55 คำ)
- **BLIND-v12** = คน 2 คนของราชบัณฑิตที่ไม่เคยเห็น (ชายเสื้อแดง, หญิงเสื้อชมพู) + คลิปถาม–ตอบทั้งหมด — ไม่มีใครมีคลิปคำเดี่ยว ≥ 40 คลิปตามที่ prompt
  กำหนด (มี 30) จึงบันทึกเป็นข้อเบี่ยงเบน · NEWSIG-dev = กันคำ 15 % ของผู้ทำท่าใหม่ในชุด TRAIN (ไม่ใช่คนใหม่จริง)
- พบและแก้บั๊กข้อมูล: ผู้ทำท่า e-TSL พักมือไว้ที่เอว ทำให้ช่วงท่า = ทั้งคลิป (200 เฟรม) → แก้แล้วเหลือ 60 เฟรม;
  หลังแก้ ~50 % ของคลิป e-TSL ยังเป็น "ท่าคนละแบบ" กับคลังเดิม (ท่าตามพจนานุกรม) — เก็บเป็น form แยก
- baseline v10 / v11 ตรึงแล้ว (TUNE + dev-test) และ v10 ได้ตัวเลขตรงกับรอบ v11 ทุกชุด · เครื่องมือบันทึกโหมดสด (LIVE-L) พร้อมใช้
- **สิ่งที่ต้องทำก่อน TPH 2 (15 นาที):** บันทึก LIVE-L ตามหัวข้อ "Owner step" ด้านล่าง และดูแผ่นตรวจป้ายกำกับ
  `cloud_s3/data_prep/qc/v12/label_audit_sheet.jpg` (43 คลิปที่ระบบสงสัยว่าป้ายผิด — ต้องให้คนที่รู้ภาษามือตัดสิน)
- แผนงบใหม่สำหรับ TPH 2–3 (ด้านล่าง) — **เจ้าของงานอนุมัติแล้ว 2026-11-15**

Budget rule for this run (owner addendum at the end of the prompt, overrides C8 / C11.5): Modal ≤ $6–7 (hard ceiling ≈ $10), local RTX 2050
(490-sign) for small / medium jobs, GCP only if Modal + local cannot do it (≤ $5, delete everything afterwards). At the end: clear the workspace
and the job history on Modal / local / GCP.

## TPH 1 — Data foundation (start 2026-11-15 14:03, planned end 18:05)

GPU / cloud bill of TPH 1: **$0** (everything local: extraction on 9 CPU processes, embeddings on the RTX 2050).

### Done (as specified unless marked)

**1B ingestion** — `scripts/v12_extract.py` (MediaPipe Holistic, production extractor, `normalize=False`, hand recovery as inference) →
`data_prep/pose_mp_raw|pose_mp/{royal_institute,etsl}/`. Priority order of 1B; 202 EA-v12 extractions reused.
**All 3,440 clips extracted** (3,237 new + 202 reused; 0 errors; dialogue clips full-frame, no signer crop). **Local instead of Modal** (owner budget).
Exemplars (bank-ready): 1,945 of 3,066 usable; `shards_v12/` = 2,600 clips (TRAIN: 1,816 isolated · 261 compounds · 205 classifiers (hard negatives) · 129 letters · 152 sentences; NEWSIG-dev 37; 207 MB). BLIND-v12 manifest: 828 clips.
Extraction was paused once (to free the CPU for the baselines; one truncated file was deleted and re-extracted) and completed afterwards.

**Signer identity** (`prep v12 ingest`, DINOv2 faces, contact sheets `data_prep/qc/v12/signers_*.jpg`, checked by eye):
- e-TSL = **one** signer (`S_etsl_00`); the 7 face clusters are recording sessions (lighting / background) → kept as `session`.
- Royal Institute = **three** signers: `S_rid_00` man (red shirt, glasses) · `S_rid_01` young woman (grey) · `S_rid_02` older woman (pink);
  frozen as identity anchors (`params/v12_signer_anchors_rid.npz`, nearest-anchor assignment; margin column `id_margin`).
- Two-person Q/A clips (354): two men alternate signing → `S_rid_dialog`; signer lock NOT verified → excluded from training, caption-bag test only.

**1C labels** — `modules/lexicon.thsl_lemma` + concept map; kinds: isolated / sentence / compound (264, 25 decomposed into known
concepts) / classifier (324 → `clf:<word>`, never a positive) / letter (129). Captions → `caption_words` (newmm + lexicon + longest match):
707 / 748 sentences have ≥ 2 vocabulary words, 256 distinct vocabulary words (= EA-v12 F4). e-TSL synonym pairs: 1,309 candidates,
30 accepted (21 by lexicon, 9 by embedding cos ≥ 0.656), 1,266 without evidence (kept apart).

**1D annotation** (`modules/v12data.py`): phases f0..f3, handedness, quality, forms.
- **Bug found and fixed**: the e-TSL signer never lowers her hands (clasped at the waist between signs) → the rest rule never fired and the
  "sign span" was the whole clip (200 / 201 frames). Rest now also = both wrists within 0.35 sw of the clip's own start / end posture.
  e-TSL span 200 → 60 frames; median cos to the word's known forms 0.43 → 0.66. Audit plot of 40 clips: `qc/v12/phases_audit.png`
  (e-TSL 20 / 20 right; Royal Institute: retraction end a few frames late — body-pose wrist outlasts the hand; accepted).
- Quality is measured inside the signing span (Royal Institute hands leave the 640×480 frame at rest).
- **Forms**: Royal Institute clips sit as close to the known forms as held-out old signers do (22 % below the 0.656 threshold vs 19 %);
  e-TSL 50 % below → 313 vocabulary clips: 169 join a known form, 144 open 142 new forms over 128 words — part of e-TSL is genuinely different (dictionary) forms. Kept as separate forms (Appendix F risk 1).
- Phonology pseudo-labels for new clips: **not done** (hard-negative mining only; can be added in TPH 2 if the sampler needs them).

**1F QC** — extraction gates, 17 Royal Institute copies of TSL51/thdict excluded, 0 cross-source duplicates (embedding cos ≥ 0.98, same concept).
- Label audit (leave-signer-out, v10 primary embeddings, 328 words, old + new): first version flagged clips whose word had NO other-signer
  exemplar (rank 300+, an artifact) — fixed. Result: 43 of 4,259 vocabulary clips flagged (39 old, 4 new; the 4 new ones are also new forms → `other_form`). Contact sheet for the OWNER: `qc/v12/label_audit_sheet.jpg` (only a signer
  can judge the label; nothing in the old corpus was excluded on the audit alone).
- Attractor audit on 173 isolated val queries is too small to name attractors (top-1 0.728; max 2 captures per word) → the TUNE sentence
  attractors come from `diagnose v12` (`attractors` per set).
- YouTube hub test (bank with / without): **not done** → TPH 2A (it is a bank experiment).

**1G splits + sealing** — no new signer has ≥ 40 in-vocabulary isolated clips (Royal Institute 32 / 19 / 3; e-TSL is TRAIN by the coverage
rule). Deviation, recorded in the manifest: **BLIND-v12 = two whole unseen people** (`S_rid_00`, `S_rid_02`: 30 in-vocabulary
isolated + 206 sentences) + all dialogue clips + 20 % of `S_rid_01` sentences (caption-bag). TRAIN new signers: `S_etsl_00`, `S_rid_01`.
NEWSIG-dev = deterministic 15 % word-level hold-out (md5(word), seed 12) of the TRAIN new signers' in-vocabulary words (not unseen
people — say so when reporting E2 on dev). `artifacts/eval/blind_v12_manifest.json` (26 clips overlap EA-v12's one-time scoring, listed).
`modules/sealed.py` enforces BLIND-v11 ∪ BLIND-v12; BLIND-v12 opens only with `THAISLM_OPEN_BLIND12=1`. Tests: `tests/test_v12_sealed.py`
+ `tests/test_v11_sealed.py` 9 / 9 pass (pytest in `cache/side_pkgs`, 490-sign untouched).

**1A EDA** — `artifacts/reports/v12_eda.json`, `result_reporting/figures_v12/eda_*.png`, `result_reporting/data_coverage_matrix_v12.csv`,
`collection_list_v12.csv`. Signer depth of the 328 words (TRAIN): ≤ 1 signer 96 → 55 · 2: 92 → 70 · 3–4: 76 → 111 · 5–9: 26 → 54 · ≥ 10: 38;
211 words gain a signer, 106 of them thin. e-TSL: hands visible 100 %, rests 2.6 s before / 2.3 s after the sign, 66 % two-handed;
Royal Institute: hands out of frame at rest (any-hand 41 %), signer larger in frame (shoulder width 0.29 of the frame width).

**1E features / log replay** — `modules/v12data.Replay` (frame times, hand-loss runs, hand size from the 400 most recent saved requests;
LOG hand / shoulder ratio is 1.04–1.22 × the dataset median, consistent with F11). Training-side use is TPH 2.

**1H harness** — `python main.py diagnose v12` (`scripts/diag_v12.py`): v11 catalogue + NEWSIG sets (`nsig_dev`, `nsig_blind12`),
decoder-span R@1, matched-precision coverage (reference = frozen v10 precision), BLEU-4 / chrF++ (sacrebleu in `cache/side_pkgs`,
newmm tokens, rule composer, LLM off), `--cand` ladder flags. Baselines frozen in `artifacts/baselines/v12/{v10,v11}.json`.
v10 reproduces the v11-handoff numbers exactly on TUNE and dev-test (harness unchanged on the default path).
Sentence BLEU-4 (corpus) is 0 on most TSL51 sets: few words are accepted, composed sentences are 1–2 words, no 4-gram matches —
report chrF++ and mean sentence-BLEU (exp smoothing) as the informative sentence metrics. Experiments: `artifacts/reports/v12_experiments.csv`.

| set | v10 WER / displayed | v11 WER / displayed | other (v10 → v11) |
|---|---|---|---|
| t51_tune_x1.0 | 0.274 / 0.588 | 0.292 / 0.602 | decoder-span R@1 0.705 → 0.691; chrF++ 28.2 → 25.94; sent-BLEU 25.66 → 23.88; tax 0.053 → 0.077 |
| t51_tune_x1.5 | 0.279 / 0.599 | 0.280 / 0.600 | decoder-span R@1 0.693 → 0.698; chrF++ 27.9 → 26.22; sent-BLEU 25.12 → 24.59; tax 0.058 → 0.089 |
| t51_tune_x2.0 | 0.310 / 0.598 | 0.296 / 0.585 | decoder-span R@1 0.671 → 0.683; chrF++ 26.6 → 26.6; sent-BLEU 25.76 → 25.27; tax 0.074 → 0.074 |
| iso_tune | 0.442 / 0.488 | 0.442 / 0.471 | decoder-span R@1 0.680 → 0.680; tax 0.106 → 0.112 |
| flu_tune | 0.464 / 0.693 | 0.464 / 0.703 | decoder-span R@1 0.537 → 0.540; tax 0.124 → 0.121 |
| nsig_dev | 0.429 / 0.771 | 0.400 / 0.743 | decoder-span R@1 0.743 → 0.771; new-signer top-1 0.743 → 0.771, top-5 0.886 → 0.886 (n 35); tax 0.086 → 0.086 |
| w_t51_tune | 0.293 / 0.608 | 0.298 / 0.618 | chrF++ 25.1 → 24.32; sent-BLEU 24.99 → 23.96 |
| w_iso_tune | 0.814 / 0.898 | 0.763 / 0.898 |  |
| idle_tune | idle words / 10 s 0.000 | 0.000 | 129 s |
| w_idle_tune | idle words / 10 s 0.627 | 0.000 | 16 s |
| tail_tune | Tail accepted 0.173 | 0.177 | n 300 |
| w_tail_tune | Tail accepted 0.108 | 0.108 | n 120 |
| **tune pooled, v10** | precision 0.958 | correct accepted at v10's precision **0.576** | coverage 0.398 (target 0.958) |
| **tune pooled, v11** | precision 0.954 | correct accepted at v10's precision **0.515** | coverage 0.357 (target 0.958) |
| t51_test_x1.0 | 0.302 / 0.589 | 0.329 / 0.577 | decoder-span R@1 0.735 → 0.717; chrF++ 19.4 → 19.94; sent-BLEU 16.92 → 17.2; tax 0.093 → 0.108 |
| t51_test_x1.5 | 0.300 / 0.566 | 0.298 / 0.547 | decoder-span R@1 0.703 → 0.703; chrF++ 20.1 → 21.86; sent-BLEU 19.41 → 20.27; tax 0.118 → 0.122 |
| t51_test_x2.0 | 0.291 / 0.572 | 0.278 / 0.571 | decoder-span R@1 0.681 → 0.677; chrF++ 20.0 → 20.25; sent-BLEU 18.84 → 19.78; tax 0.136 → 0.143 |
| iso_test | 0.243 / 0.430 | 0.243 / 0.402 | decoder-span R@1 0.794 → 0.794; tax 0.103 → 0.103 |
| flu_test | 0.397 / 0.717 | 0.397 / 0.703 | decoder-span R@1 0.644 → 0.647; tax 0.236 → 0.215 |
| w_t51_test | 0.328 / 0.588 | 0.353 / 0.573 | chrF++ 18.9 → 19.89; sent-BLEU 17.6 → 18.54 |
| w_iso_test | 0.727 / 1.000 | 0.636 / 1.000 |  |
| idle_test | idle words / 10 s 0.216 | 0.000 | 46 s |
| w_idle_test | idle words / 10 s 0.000 | 0.000 | 14 s |
| tail_test | Tail accepted 0.156 | 0.160 | n 263 |
| w_tail_test | Tail accepted 0.085 | 0.061 | n 82 |
| **devtest pooled, v10** | precision 0.954 | correct accepted at v10's precision **0.593** | coverage 0.440 (target 0.954) |
| **devtest pooled, v11** | precision 0.963 | correct accepted at v10's precision **0.632** | coverage 0.462 (target 0.954) |


**1I live capture** — `/sign/live?script=livel_a` (overlay `components/sign/CaptureOverlay.tsx`, `lib/capture.ts`; engine taps are no-ops
unless a capture session is open) → `POST /v1/capture/chunk` → `cache/live_labeled/<session>/`. Scripts `livel_a` / `livel_b`
(60 words, 10 minimal-pair prompts, 20 sentences, 4 × 20 s idle, 8 OOV). `python main.py live-replay <session>` (manual vs live cut, same
pipeline). `modules/live_rule.py` = the browser rule, **frame-exact parity with the TypeScript on 20 saved requests** (`tests/test_live_rule.py`,
node runs the TS functions); LM1 / LM3 / LM4 options built in (off by default).
Self-test: visible Chrome + fake camera, local API (fastapi / uvicorn in `cache/side_api`) + Next dev → session saved and replayed
(both policies) — functional only (CPU was saturated: ~3 fps). Session parked in `cache/v12_selftest/` (not LIVE-L).

### Not done / deviations
- legacy-blind (BLIND-v11) baselines: moved to TPH 4 (scoring it now would be an extra opening) — v10 / v11 / v12 in one run there.
- Phonology pseudo-labels for the new clips; signer lock for the dialogue clips (they stay caption-bag-test only).
- Workspace note for the owner: `docker-compose.yml` has an uncommitted change (from the v11 session) that makes **model-v11** the default
  sign-api model; v10 is the shipped model — run with `THAISLM_MODELS_DIR=./models/model-v10` or revert that line.

### Owner step before TPH 2 (15 min, Appendix C)
Start the stack, open `http://localhost:3000/sign/live?script=livel_a`, press Space and follow the prompts (sit as in normal use).
Record **4 sessions** if possible (2 × `livel_a`, 2 × `livel_b`, different day / lighting): sessions 1–2 = L-tune, 3–4 = L-test (sealed).
Rebuild the images first (`docker compose build sign-api web`) — the capture endpoints and overlay are new code.

### Budget re-plan for TPH 2–3 — **APPROVED by the owner 2026-11-15** (cap $6–7 Modal vs the prompt's ~$21–25 plan)
v11 actual cost: ≈ $2 per Modal container-hour (A10G + 16 cores) → the cap buys ≈ 3 container-hours. Proposal (the owner decides):
- free rungs first: B1 bank (2A), B2 jitter (2B), caption spotting (2C) — local, $0.
- **primary reader (the main lever)**: P1 (+ new signers + distillation) on Modal A10G, one run, measure the first 200 steps and stop
  early per C8 (~1.2 h ≈ $2.5); P2 (+ pseudo spans) only if P1 beats enc-v7 on the selection score (~$2.5). P0 (control on old data)
  replaced by enc-v7 at step 0 under the same evaluation (honest limitation: no "more fine-tuning alone" control). P3 dropped.
- **seeds**: none on Modal — seed SD taken from v11 (lmenc 0.004 MRR) and enc-v7's own curve; stated as a limitation.
- **landmark encoder**: L0 only, locally on the RTX 2050 overnight (~170–210 clips/s → 6 M clips ≈ 8–10 h, $0); L1 / L2 dropped (drop order C9).
- Reserve ≈ $1.5 for an ONNX export / embedding job if local memory fails. GCP not needed.

### First command of TPH 2
Data is complete and packed (`data_prep/shards_v12/`). Start with
2A: bank-only rung B1 — embed new TRAIN exemplars with the three v10 readers and add them to the eval / prod banks.

## TPH 2-4 - readers, chain, live mode, gate, blind, overall system, report (2026-11-15 17:24 -> 21:10, owner: "do all TPH")

Full results: `result_reporting/result-v12.md` / `.pdf` section 7 (Thai). Modal bill: **$2.44** metered (P1 $1.16, P1b $1.18, OOM run $0.10), billed $0.

- **Ladder (TUNE, paired)**: B1 bank +new signers NOT kept (pooled WER +0.005, new-signer -0.029) · B2 jitter KEPT in the form "Jpc" (5 crops for
  primary + hand cue, landmark encoder 1 crop, calibration features from the single-crop read; tau 0.88 -> 0.874 = v10 pooled precision) ·
  B3 not run · B4 NOT kept (P1, P1b never beat enc-v7: saturated) · B5 two-stage calibration NOT kept (coverage -0.06) · B6 live rules KEPT ·
  B7 skipped. Rung files: `artifacts/v12/{B1,B1b,B2,B2b,J,J3,Jp,Jpc,B5}`.
- **Gate** (dev-test + BLIND-v12 opened once + legacy BLIND-v11 3rd opening): E1 -0.004 / -0.003 (non-inferior, no win) · E2 tie 0.718 · E3 0.583 vs
  0.584 · E4 live-mode displayed recall +0.018 [-0.011, +0.046], web sessions 0.566 -> 0.593 (no win) · E5 Tail +0.008 · E6 latency +2..9 % p50.
  => **all non-inferior, 0 wins -> does NOT pass the pre-registered rule. Not shipped; owner decides (report section 10: A keep v10 / B ship v12 /
  C live rules only on v10).**
- Live mode (browser rule, TS parity): live-cut WER 0.391 -> 0.345 [-0.081, -0.014] on web-rate TSL51 test = start/stop level; real Chrome sessions:
  utterances per 4 sentences 4.9 -> 4.1.
- Overall system (`python main.py e2e-eval`): v12 speaks exactly what v10 speaks on the 60 BLIND sentences (chrF++ 19.3, CER 0.70); speech is
  intelligible (ASR-CER vs TTS input 0.08, floor 0.00) -> the end-to-end error is sign reading + acceptance (oracle glosses chrF 39.9, top-1 25.1,
  accepted only 15.9). ElevenLabs 69 calls / 1.35 audio min. LLM composer +1.5 chrF.
- Tests: 88 / 88 (incl. sealed v11/v12, live-rule TS parity, robustness 8 cases). v10 reproduces exactly; models/model-v10 untouched.
- Built: `models/model-v12` (v10 + spotting.json read.jitter / jitter_lm 0 / jitter_cal single + tau 0.874 + config.json "live"; eval banks not
  needed). Rollback: THAISLM_MODELS_DIR=./models/model-v10. Web: live rules come from /v1/info "live" (null for v10 = v10 behaviour).
- NOT done as specified: LIVE-L (owner recording), web test 8 sessions instead of 30 + 10, 4-concurrent latency, calibration with quality covariates /
  stream agreement, seeds, landmark encoder, P0/P2/P3, human audits (sheets left in data_prep/qc/v12/).
- Cleanup: Modal volume thaislm-v12 deleted (uploaded copies + run outputs; histories kept in artifacts/runs/v12-P1*); TTS test container removed;
  scratch session videos / browser profiles removed. Modal app run history stays in the Modal dashboard (cannot be deleted from the CLI).

## SHIPPED (owner decision, 2026-11-15 21:45): option B
docker-compose default model -> ./models/model-v12 (sign-api + web images rebuilt). Post-ship checks all pass (health, /v1/info v12 + live rules,
12 real requests docker == offline, real Chrome live mode + capture page, no log errors; the only 503s were the 10 s restart window).
Bug found and fixed during ship: capture endpoints used repo-relative paths -> sign-api crashed in docker; now env THAISLM_CAPTURE_DIR /
THAISLM_SCRIPT_DIR + mounts (scripts read-only, sessions on this machine). tests 87/88 in the full run (test_stream timing-flaky under load;
3/3 alone). Rollback: THAISLM_MODELS_DIR=./models/model-v10 docker compose up -d sign-api. Nothing committed.
