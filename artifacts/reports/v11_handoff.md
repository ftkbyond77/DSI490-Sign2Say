# v11 handoff (prompt/phase3/prompt-opt-v11.md)

Compute policy for this run (owner, 2026-11-13): Modal first (≈ $30 credit), local RTX 2050 for small / medium work, GCP Vertex only if Modal
is exhausted. Modal L4 capacity was unavailable on 2026-11-13 → GPU fallback list A10G → L4 → T4 (A10G used), 16 CPU cores per container.

## TPH 1 — Foundation (2026-11-13 01:45 → 04:40 local, 2 h 55 min)

**Done**
- Front-pass cache rebuilt with `models/model-v10` (incl. `onnx/gate_tagger.onnx`); Harness now runs model-v10 (`THAISLM_HARNESS_MD` to override).
- v10 baseline frozen: `artifacts/baselines/v10/` + `gate_tolerance.json` (0.25 × v10 bootstrap SD). dev-test (BLIND removed):
  TSL51 1× WER 0.302 (78 utt), isolated 0.243, fluent 0.397 (`artifacts/reports/decision_v10_v11base.json`).
- `python main.py diagnose v11` (scripts/diag_v11.py): C6 catalogue through the whole chain. v10 on TUNE (`diag_v11_v10.json`):
  WER TSL51 1×/1.5×/2× 0.274/0.279/0.310 · iso 0.442 · fluent 0.464; displayed WER 0.49–0.69; segmentation tax 0.05–0.12;
  readers gold R@1 (TSL51 1×) Uni-Sign 0.662 · hand 0.379 · lmenc 0.475 · fusion 0.727 · any-of-3 0.748; correct accepted 0.55–0.73;
  open-set false accept 0.075; idle accepted words / 10 s 0.000 (34 TSL51 null clips); focus-word error 0.670; confusion-group
  within-error 0.402; in-sentence gap 0.104; lowest signer gold R@1 0.0 (a signer with ≥ 8 refs). dev-test: `diag_v11_v10_devtest.json`.
- BLIND-v11 sealed (seed 11): `artifacts/eval/blind_v11_manifest.json` — TSL-ONE-S S_one_13, S_one_23 · th_sl S_thsl_02 · TTRS S_ttrs_u00
  (TTRS roles are per word; every TTRS test person also has 671–2,359 TRAIN clips, so the unseen TTRS person with the smallest
  footprint was sealed) · YouTube yt_person_004 / 030 / 040 · 14 / 34 TSL51 sentence_test templates (48 sentence clips); 3,352 clips.
  Loaders: `modules/sealed.py` (train tables raise; dev-test sets drop sealed rows; BLIND sets need THAISLM_OPEN_BLIND=1); banks filter
  sealed rows for v10 AND v11. Unit test: tests/test_v11_sealed.py.
- Data roles: `cloud_s3/data_prep/manifest/tiers.json` (Core 38 · Extended 290, of which 106 "thin" < 2 effective signers · Tail 9,214;
  1,607 concepts with ≥ 2 TRAIN signers) · `forms.json` (327 words, 87 with ≥ 2 forms; same-source documented labels never merged,
  cross-source labels merged at the median same-label cross-signer cosine 0.656; contact sheets `artifacts/reports/v11_sheets/forms_*.png`).
  YouTube hubs yt:ICzd6znjUfM / yt:8eqTnsoH3LY: not removed (no TUNE test of their removal yet — open item for TPH 3 bank refit).
- Bank scoring v11 (`modules/runtime.FormBank`, harness `bank_v11`): on v10 embeddings it cuts confusion-group within-error 0.40 → 0.29
  at every λ, but TUNE WER is mixed (decoder bank: insertions up; reader-only λ 0.08: TSL51 better, iso / fluent +0.006) → not adopted
  on v10 embeddings; λ to be re-selected on the final encoder.
- Phonology pseudo-labels: `manifest/phonology_labels.npz` (16,815 clips; hs 100 · or 36 · loc 10 · mov 36 classes), audited on sheets
  (`v11_sheets/phon_*.png`): location boxes fixed after audit; movement re-defined on the sign core (v1 put 88 % in one class).
- Hard-case extractor set: 160 clips (8 categories × 20; "close" = zoomed copies, no owned close-to-lens footage), labelled by eye
  (`artifacts/eval/hardcase_v11/`). X0 Pose Gate: visible hands found 0.908, phantom 0.050, both 0.431, flicker 0.48 / 10 s, bone CV 0.265,
  wrist gap 0.114, 56.8 ms/frame; TUNE videos (40): found 0.480, bone CV 0.402 (`artifacts/reports/pose_gate_v11.json`).
- Wave 1 (Modal A10G, 16 cores, `modal_v11.py`; loader RAM store = 3.7× faster): measured 5.7 it/s ≈ 3,300 clips/s (cross sampler) /
  1,600 clips/s (old sampler). Local RTX 2050: ~170–210 clips/s (CPU/RAM-bound) → encoder runs stay on Modal.

| run | clips seen | best step | TUNE MRR | lowest signer | margin | in-sentence R@1 |
|---|---|---|---|---|---|---|
| v10 lmenc ensemble / best member | — | — | 0.457 / 0.447 | 0.148 / 0.111 | 0.266 / 0.250 | 0.480 / 0.398 |
| v11-R0 | 10.8 M | 6,000 / 19,532 | 0.423 | 0.125 | 0.290 | 0.386 |
| v11-R1 | 8.6 M (time-capped) | 6,000 / 15,508 | 0.410 | 0.140 | 0.275 | 0.369 |
| v11-R2 | 11.4 M | 8,000 / 19,532 | 0.421 | 0.125 | 0.376 | 0.360 |
| **v11-R3 (winner)** | 11.4 M | 8,000 / 19,532 | **0.445** | 0.135 | **0.404** | **0.425** |

Winner rule: R3 − R0 = +2.2 MRR points but only 2 of 4 signer sets agree (TTRS +11.4, ONE +0.9, th_sl −1.7, YouTube −1.9) → the MRR
difference alone is not "real" by C8; the three other selection metrics all favour R3 → R3 recipe carries into wave 2. All runs peak at
6–8 k steps and then decline (converged, not under-trained) → wave 2 uses early stopping after ≥ 6 M clips (5 evaluations without gain).

**GPU / cost:** wave 1 ≈ 6.4 container-hours (A10G + 16 cores); Modal billing at 04:00 = $7.52 metered, $0 billed (credits).
Plan estimate for TPH 1 was ~$19–21 on A100.

**Started early (TPH 2 items):** decision layer v11 code (`modules/openset.py`, `scripts/v11_decision.py`) fitted on v10 embeddings;
segmentation code (`modules/boundary.py`, `modules/protoctc.py`, `scripts/train_seg_v11.py`, smoke-tested); Path-B wiring
(`modules/pathb.py`, pipeline `_back_pathb`); verifier (`modules/verifier.py`, `scripts/v11_verifier.py`); X1 detector training
(local GPU, HaGRID 30k subject split, `scripts/v11_hand_detector.py`).

**Open issues:** conformal-singleton acceptance cannot meet G6 with the current reader (12–14 % correct accepted at ≤ 5 % error) →
acceptance uses per-group risk-controlled thresholds; open-set head AUROC 0.79 on v10 embeddings.

**First command of TPH 2:** `modal run modal_v11.py::encoder --run v11-R4 --preset R4 --clips 12e6 --hours 2 --extra "--patience 5"` (launched).

## TPH 2 — Encoder, decision layer, segmentation code, detector (04:40 → 07:40 local, ~3 h)

**Wave 2** (Modal A10G; early stop after ≥ 6 M clips): all on the R3 recipe (cumulative ladder).

| run | clips | TUNE MRR | lowest signer | margin | in-sentence R@1 | signer probe | decision |
|---|---|---|---|---|---|---|---|
| R4 sub-centres + group loss | 11.5 M | 0.438 | 0.083 | 0.400 | 0.372 | 0.281 | below R3 |
| R5 + input v11 | 9.3 M | 0.459 | 0.083 | 0.407 | 0.403 | 0.287 | +1.4 vs R3 (< 1.5) and worse lowest signer / in-sentence → not real |
| R6 + phonology sub-spaces | 8.2 M | 0.353 | 0.083 | 0.172 | 0.448 | 0.333 | stopped (flat, −11 pts vs R5) → dropped |
| R7 R6 at size M | 2.0 M | 0.201 @2k | — | — | — | — | stopped: counts only if R6 > R5 (not the case) |
| R8 R5 + adversary | 8.1 M | 0.447 | 0.083 | 0.407 | 0.331 | 0.204 | probe drops but unseen-signer R@1 does not rise → not kept |

**Final encoder:** R3 configuration × 3 seeds (R3 = seed 0) → `artifacts/runs/v11-final` (ensemble, ONNX `artifacts/v11/v11-A/onnx/lmenc.onnx`,
parity 1.04e-7). TUNE MRR 0.472 (v10 lmenc 0.457) · lowest signer 0.184 (0.148) · margin 0.393 (0.266) · in-sentence R@1 0.480 (0.480).
Seed noise: MRR 0.446 ± 0.004 SD → the R5-vs-R3 call (+1.4) is > 2 SD but below the prompt's 1.5-point rule; documented.

**Decision layer v11** (`modules/openset.py`, `scripts/v11_decision.py`) fitted on v10 embeddings: open-set false accept 0.075 → 0.025,
idle 0 words/10 s, WER unchanged, but correct-accepted down (TSL51 0.58 → 0.33) — conformal-singleton acceptance replaced by
per-group risk-controlled τ (measured: the singleton rule left 12–14 % correct accepted at ≤ 5 % error). Refit on v11-final in TPH 3.
**Segmentation code** (`modules/boundary.py`, `modules/protoctc.py`, `scripts/train_seg_v11.py`): smoke-tested; 15-min wiring head reached
segment F1 0.90 / short recall 0.91 on TUNE-signer sentences. **Path-B wiring** (`modules/pathb.py`, pipeline `_back_pathb`) runs end to end.
**Verifier** prototype on R3: 0 / 41 groups kept (TUNE check had too few clips; now includes TSL51 TUNE sentence spans).
**Adaptation simulator** prototype on R3: explicit enrolment 0.405 → 0.624 (120 s, TSL51 stream); normaliser / pseudo-enrolment unsafe → off.
**X1** detector: test mAP50 0.995 / mAP50-95 0.878 (subject-independent HaGRID split); X2 tied X0 on the Pose Gate (slower) → X0 so far.
**Tests:** tests/test_v11_components.py (10) + test_v11_sealed.py (5); full suite 77 passed / 1 skipped.

**GPU / cost:** Modal metered $25.5 of the $30 credit after wave 3 (billed $0). GCP fallback: creating the bucket/jobs was blocked by the
auto-mode safety check → not used; segmentation heads and the detector run on the local RTX 2050.
**Not done as specified:** wave-2 runs stopped early by rule/early stop (above); wave 3 = 2 new seeds (R3 is seed 0), 1 h box each.

**First command of TPH 3:** `python main.py train-seg-v11 --enc v11-final --head boundary --run v11-bnd --minutes 60` (running locally).

## TPH 3 — Segmentation, path decision, verifier, decision refit, extractor gates, adaptation (07:40 → 10:20 local, 2 h 40 min)

**Segmentation (local RTX 2050):** boundary head `artifacts/runs/v11-bnd` (4,050 steps, 60 min): TUNE-signer sentences segment F1 0.898 / 0.901
(25 / 12 fps), short-sign recall 0.944 / 0.900. Prototype-CTC `artifacts/runs/v11-ctc`: drift > 0.5 pt at step 300 → encoder frozen; val CTC best
at step 300 (3.86) then rising (train CTC → 0.2) → stopped at 5,960 steps (overfits training signers).

**S0–S3 on TUNE (WER TSL51 1×/1.5×/2× · iso · fluent):** v10 0.274/0.279/0.310 · 0.442 · 0.464 — S0 0.292/0.280/0.296 · 0.442 · 0.464 —
S1 0.542/0.537/0.560 · 0.337 · 0.569 (decoder re-tuned) — S2 0.548/0.444/0.374 · 0.872 · 0.933 — S3 0.489/0.491/0.521 · 0.355 · 0.561.
**Path decision: Path A (S0)** — Path B wins only isolated signs; it names segments with the new encoder alone (TSL51 span R@1 0.48 vs Uni-Sign 0.66+).

**S0 = shipped configuration** (`artifacts/v11/v11-S0`): v10 segmentation + Uni-Sign span + hand cue + v11 landmark encoder (3-seed ensemble) as
second opinion lm_w 0.3 (reader ablation: lm_w 0 / 0.15 / 0.3 / 0.45 → 0.45 helps fluent but hurts TSL51; primary-reader variants 0.32–0.45 TSL51)
+ bank v11 reader-only λ 0.08 + v10 calibration (the re-fitted one cost WER through transition marking) + decision layer v11 in v10-acceptance
mode with the open-set veto (bound 0.743) + conformal alternatives + verifier (9 groups; it never changed a TUNE decision in this mode).
TUNE: WER 0.292/0.280/0.296 · 0.442 · 0.464; accepted precision iso 0.915 (v10 0.904), fluent 0.898 (0.923); correct accepted 0.57/0.57/0.60 ·
0.735 · 0.526 (v10 0.58/0.57/0.60 · 0.726 · 0.548); open-set false accept 0.056 (0.075); idle 0.
Strict per-group rule (measured, not shipped): ≤ 5 % accepted error out-of-fold by signer leaves 27 % of correct words accepted.
**Extractor:** X0 kept (X1 detector / X2 re-acquisition tie on the Pose Gate, +27 / +5 ms per frame; X1 + X2 recover 5 + 7 extra hands vs 108 by X0).
**Adaptation:** off (normaliser: th_sl +3, TTRS −28 pts; pseudo ≈ 0; explicit enrolment TSL51 0.475 → 0.662 @120 s stays as the personal memory).
**Production path:** verifier exported to ONNX + NumPy (`verifier/frames.onnx` parity 2.7e-6, trees exact); SignPipeline loads bank v11, Tail
banks and the verifier from the model folder.
**Bugs found and fixed in this TPH:** Path-B front-pass cache keyed on id() (stale fronts across freed sets — earlier S1 diag invalid, re-run in
segtune); boundary-head eval set not fixed (restarted); decision fit dropped tau_accept / overwrote the v10 calibration (fixed, refit).
**GPU / cost:** no Modal use in TPH 3 (local GPU); Modal total $25.5 metered (credit).
**First command of TPH 4:** `python main.py v11-model build` → models/model-v11, then `decision report --versions v10,v11` + `decision gate`.

## TPH 4 — Integration, gate, system tests, BLIND, report (09:40 → 11:10 local)

**4A** `models/model-v11` built (`python main.py v11 model-build --cand v11-S0`): lmenc parity 1.0e-7; verifier ONNX/NumPy; `THAISLM_V11=0`
reproduces v10 on 39/39 recorded utterances. **4B** dev-test gate FAIL (21 → 20 metrics after 4G). **4C** system tests (in-process FastAPI):
contract, replay, latency, load, streaming (after 4G), robustness all PASS. **4E** BLIND opened once: ship gate FAIL (table in result_v11.md §1);
fluent WER −0.025 significant; Tail-sign / open-set false accepts down. **4G** one iteration (open-set veto off) → streaming fixed, gate still
FAIL → stop. **4F** result_reporting/result_v11.md + .pdf, collection_list_v11.csv, related-work.txt (K–Q status), model README.
**4D** (run after the owner started Docker Desktop): real Chrome + fake camera, 148 clips / 26 people, /sign/live full pool + /sign one third,
394 runs, v10/v11 alternated by blocks → **G9 FAIL**: top-1 WER 0.592 → 0.601 (tie), displayed WER 0.674 → 0.733 (worse, CI [+0.027, +0.090]),
negative clips shown as words 16.7 % → 5.6 %, 0 errors. Uses BLIND videos = 2nd BLIND opening (logged). Owner replay through the gateway:
shown 45 → 35, `[?]` 162 → 160, no cache writes. Compose rollback on model-v10 = pre-v11 HEAD code on 70/70 utterances (PASS).
Tools: D:\SLM-labs\_thaislm_v11_cleanup\browser_test\webtest\ · numbers: artifacts/reports/webtest_v11*.json · report §8.1.
**Not done:** owner webcam check (needs the owner; 10-minute script in result_v11.md §11).
**Decision for the owner:** do not ship (recommended); v10 stays default; model-v11 is an opt-in candidate.
**GPU / cost (whole v11):** Modal $25.5 metered of $30 credit (A10G, ~13 container-hours incl. CPU), GCP $0 (not used), local RTX 2050.

## v12 merge experiment (after v11; 2026-11-13/14) — result_v11.md §12
Goal: v11's strengths on top of v10 without its weaknesses. TUNE-only selection; BLIND not used again; local CPU only ($0).
Ablation: v11 full == v10 + v11 encoder + v11 bank (v11 decision layer is a no-op in v10-acceptance mode); the web "fewer words shown"
is a domain shift (browser landmarks lower v11's P(correct)), not reproduced offline. New web-domain sets (MediaPipe Holistic at
13 fps from TUNE / dev-test source videos): cache/decision/w_{t51,iso,tail,idle}_{tune,test}.pkl (+ raw in cache/v12_web/).
Candidates: v11 encoder alone (TSL51 worse) · two-stage calibration (forces τ up) · dual reader v10+v11 encoders (best: lm_w 0.3,
lm2_w 0.15, τ 0.89 by a pre-set TUNE rule) · Tail guard (no separation). Dual reader: dev-test displayed WER better, fluent top-1
0.397 → 0.362, but **dev-test gate FAIL 21 metrics** (TSL51 1× R@1/R5, false accepts, ECE, +5.6 % model time) → stopped, no web test.
v10 stays. Code kept behind flags (off by default; v10 output identical on 77 saved requests): modules/pipeline.py read.lm2_w /
cal.accept / cal.tail_guard; scripts/v12_merge.py; decision.py Harness._use_lm2 + build_items("w_*").
