# ThaiSLM — Thai Sign Language → Thai text → speech

A pose-based Thai Sign Language (TSL) translator for real video and a live camera: it finds where signing happens, spots each
sign, names it from a daily vocabulary (1,366 words with evidence; or the 327-word conversation list), says how sure it is, lets a
language layer write a Thai sentence from the evidence (never beyond it) — and speaks it.
Results: `result_reporting/result_v8.md` · architecture: `result_reporting/architecture_v8.md` · file map:
`result_reporting/file_structure.pdf` · lab: `lab.ipynb`.

## v8 — data quality, Thai-only encoder, hands that stay tracked

* **Data first:** all 561 YouTube word clips audited (cuts, people, zoom, framing, hands, label agreement + eye check) → 525 kept;
  806 "word (source)" duplicate labels merged; the **daily vocabulary** (1,366 words signed by ≥ 2 people, TNC frequency ≥ 50 +
  the conversation list) is the default — the 9.9k "full" dictionary mode is retired from the app.
* **No ASL/CSL prior:** the encoder is an ensemble of three small Thai-only landmark encoders trained from scratch (10.2 M vs 92 M
  parameters); errors copied from the ASL/CSL zero-shot model 41 % → 20 %.
* **Hands:** MediaPipe HandLandmarker fills the hands Holistic misses (browser + server); every result reports its hand tracking.
* **Faster:** 0.26 s after landmarks for 10 s of signing (v7 0.77 s); live median 0.43 s (v7 3.2 s).
* **Speech:** VachanaTTS (ONNX, `.env` `TTS_PROVIDER` / `TTS_VOICE_VACHANA`); Thai-ASR CER 0.022 vs 0.107 for MMS-TTS.
* **Honest status:** top-1 better on every unseen-person test; confident words 94–100 % correct (daily, TSL51 test); lab-sentence WER
  worse (0.446 → 0.530); natural fast signing by unseen people still ~0.1 recall (data-limited). Details: `result_v8.md` §6, §10.
* New steps: `audit-youtube`, `train-encoder`, `bench-tts`, `diagnose <recording> --intended "…"`.

## Start here

```bash
docker compose up --build                            # web demo → http://localhost:8080 (camera, live mode, speech)
```
```bash
conda activate hugging                               # Python 3.10, see requirements.txt
git lfs install && git lfs pull                      # model weights are in Git LFS
python main.py infer --video path/to/video.mp4       # translate a video with models/ (ONNX Runtime; --backend torch for the GPU)
python main.py test                                  # data_test/ end-to-end from the raw videos + scores
jupyter lab lab.ipynb                                # data, latent space, tagger, segmentation, results, self-made sentence tests
```
`python main.py` with no arguments lists every command.

## Repository map

| path | what | in git |
|---|---|---|
| `main.py` | the only CLI entry point (infer · test · data · prep · extract · harvest · audit-youtube · train-encoder · train-seq · eval · export-onnx · bench-tts · diagnose · cloud) | ✓ |
| `docker-compose.yml`, `webapp/` | web demo: `web` (Next.js) · `sign-api` (FastAPI + ONNX Runtime) · `tts` (VachanaTTS) · `gateway` (nginx) — see `webapp/README.md` | ✓ |
| `models/` | **the deployable model** — everything inference loads (+ `onnx/`, `mediapipe/`). See `models/README.md` | ✓ (LFS) |
| `modules/` | library (see below) | ✓ |
| `scripts/` | pipeline steps: `data_prep.py` · `extract_mp.py` · `harvest_words.py` · `audit_youtube.py` · `train_encoder.py` · `eval.py` · `diagnose.py` · `bench_tts.py` · `export_onnx.py` · `infer.py` · `make_lab.py` · `s3_sync.py` · `md_to_pdf.py` · `pretrained.py` · `collect/` (v4 raw-data harvester) | ✓ |
| `GCP/script/` | Vertex AI: `vertex.py` (upload · load-balanced launch · wait · **cleanup**) · `job_entry.py` (training stages) | ✓ |
| `GCP/result/` | one json per GPU job: machine, region, minutes, estimated cost | ✓ |
| `artifacts/` | `pretrained/` (Uni-Sign base — only for the v7 / zero-shot comparisons, rebuilt by `python main.py pretrained`) · `eval/` (train-only bank) · `runs/<run>/` (metrics, histories) · `reports/` (evaluation json) | ✓ (no per-clip data) |
| `vocab/` | `daily_conversation.json` (target vocabulary) · `vocab.csv` · `daily_conversation_depth.csv` | ✓ |
| `result_reporting/` | reports (`result_vN`, `architecture_vN`, `latent_space_vN`, `file_structure`, md + pdf), figures, `inference/` outputs | ✓ (no video frames) |
| `cloud_s3/` | local mirror of the S3 data lake: `raw_data/` · `data_prep/` · `agent_data/` (harvested by v7, audited by v8: `youtube_words/qc/quality_v8.csv`) | ✗ (data) |
| `data_test/` | 5 real-world test videos — never used for training or tuning | ✗ |
| `cache/` | local scratch (MediaPipe of data_test, self-test videos, logs) | ✗ |

`modules/`: `mediapipe_pose` (the extractor + hand fallback) · `schema` (Standard Schema, handedness) · `lmenc` (v8 landmark
encoder) · `encoder` (training data, losses) · `parts` / `unisign` (v7 encoder, kept for comparisons) · `sequence` (tagger, synthesis, sequence stage) · `segment` (kinematics, windows) · `spanhead`
(one-pass spotting) · `runtime` (ONNX / PyTorch back-ends, bank) · `decode` (decoder, calibration, filters) · `pipeline` (end to end)
· `face` (blendshape cues) · `language` (evidence-guarded sentence) · `grammar` (TSL word order) · `tts` · `lexicon` · `identity`
(signer faces, data prep only) · `utils` · `env`.

## Data (S3 is the source of truth)

`s3://dsi490-lake-signdata/` (ap-southeast-1), SSO profile in `.env` (`AWS_PROFILE`; `aws sso login --profile …` when it expires).

```bash
python main.py data pull data_prep          # prepared data: Standard-Schema poses (MediaPipe + RTMW) + manifest + shards
python main.py data pull raw_data           # raw videos / landmarks (only to rebuild data_prep)
```
`agent_data/` holds the YouTube single-word lessons harvested by v7 (videos, annotations, QC) and the v8 audit — pushed by the
project owner (see result_v8 §11).

## Training (v8: everything on one laptop GPU; the Vertex AI path stays for larger runs)

```bash
python main.py audit-youtube measure && python main.py audit-youtube decide    # YouTube quality filter (qc/quality_v8.csv)
python main.py prep shards                                            # shards + label merges + daily vocabulary
python main.py train-encoder --run enc-v8a                            # ×3 members (b: --ema 0.998, c: --context_p 0.5 --jitter 0.3)
python main.py train-encoder --ensemble enc-v8a,enc-v8b,enc-v8c --run enc-v8
python main.py train-seq --encoder_run artifacts/runs/enc-v8          # tagger + span head + decoder tables
python main.py eval build --run enc-v8 --seq seq-v8 && python main.py eval spot --seq seq-v8 && python main.py eval tune_natural --seq seq-v8
python main.py eval calibrate --seq seq-v8 && python main.py export-onnx && python main.py train-letters --bank_only
python main.py test --tag v8 && python main.py eval external --tag v8 --fallback && python scripts/make_lab.py --execute
python main.py cloud cleanup                                          # empty the bucket, delete every job, check nothing bills
```

## Principles kept in every version

* **No guessing:** every detected sign is reported as `word` (accepted) / `word?` (uncertain) / `[?] ≈ nearest`; hand transitions
  are labelled `transition`; the sentence layer may only use a segment's candidates and a deterministic evidence guard withholds
  unsupported sentences.
* **Honest evaluation:** held-out *signers* for isolated signs, held-out *sentence templates* for continuous signing, tuning only
  on tune data (out-of-fold tagger outputs); `data_test/` and the self-made sentence videos are only run and scored.
* **One extractor everywhere (v7, v8):** MediaPipe Holistic (+ HandLandmarker fallback) for the training data, the server and the browser.

## Licensing

Raw data came from public endpoints for research. th-sl.com (NADT) carries an explicit AI-training rights reservation; its use
here was approved by the project owner for research only. YouTube clips (v4 and v7 harvest, v8-audited) remain the publishers' — research
use only. Redistribution or release of models trained on these sources is a separate conversation with each rights holder.
