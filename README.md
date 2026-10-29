# ThaiSLM — Thai Sign Language → Thai text → speech

A pose-based Thai Sign Language (TSL) translator for real video and a live camera: it finds where signing happens, spots each
sign, names it from a daily-conversation vocabulary (or the full ~10k-concept dictionary), says how sure it is, lets a language
layer write a Thai sentence from the evidence (never beyond it) — and speaks it.
Results: `result_reporting/result_v7.md` · file map: `result_reporting/file_structure.pdf` · lab: `lab.ipynb`.


## v9 — recognition · calibration · face emotion · speech (prompt_13)

Report: `result_reporting/result_v9.md` (+ pdf) · lab: `lab3.ipynb` · models: `models/model-v9/` (forward, Docker default) ·
`models/face-model/` · `models/speech-model-v1/`. All settings chosen on tune data (`python main.py decision tune --start v8.1`); $0 cloud.
* **Diagnosis first** (`python main.py diagnose ledger`): TSL51 errors are recognition, not segmentation (perfect naming on the current
  segments → WER 0.065; perfect segments with the current reader → 0.26).
* **Second recogniser**: the Thai-only landmark encoder (`onnx/lmenc.onnx`, `lm_bank.npz`) fused with Uni-Sign + hand cue; grammar
  re-rank off; isotonic calibration, τ_accept 0.92 → 0.88 (accepted precision ≥ 0.90 on every tune set); lone letters 0.95 / 0.7 after
  "ชื่อ"; enrolled users (personal bank) run with the v8.1 settings (they win there).
* Held-out WER v8.1 → v9: TSL51 1× / 1.5× / 2× 0.309 / 0.284 / 0.292 → **0.296 / 0.263 / 0.274** · unseen-signer words 0.257 → **0.243**
  · fluent 0.411 → **0.378** · iPhone sentences 0.126 → **0.097** · team-drive per performance 0.550 → **0.508**; accepted words ×1.5–2.
  Enrolled signer (own word clips in the bank): TSL51 **0.119 / 0.109 / 0.137**.
* **Face emotion** (`modules/face_model.py`): 7 emotions from MediaPipe blendshapes, RAVDESS + neutral signing, explained in Thai,
  fused with the signed words and signing energy → `result.face.affect`, API `face.affect`.
* **Speech v1** (`modules/speech.py`, speech env `D:/SLM-labs/envs/speech`): F5-TTS-THAI cloned from emotion references (ElevenLabs
  teacher) — CER 0.06 % vs 8.2 % (old voice), needs a GPU; Qwen3-ASR-0.6B (LoRA tried, not default: hurts real speech).
* **Streaming** (`modules/stream.py`): partial → confirmed words while signing, compute counted.

## v8.1 — recognition (prompt_12): hand-shape cue, automatic fingerspelling, จีน / ประเทศ

Report: `result_reporting/result_v8-1.md` (+ pdf) · lab: `lab2.ipynb` → "2. After Tuning". Models unchanged; CPU only ($0); every setting chosen
on tune data with `python main.py decision hands | tune --write | report | figures`; v8 settings kept in `artifacts/baselines/v8/` (rollback).
* **Hand-shape cue** (`modules/handshape.py`, `models/hand_bank.npz`): finger angles, palm direction, hand place and motion of each segment
  vs every bank clip (metric learned on training signers), fused with the encoder's reading of the whole segment — ฉัน/ชอบ, ชื่อ/เพื่อน now right.
* **Fingerspelling always on** (English A–Z, `spotting.json` `"alphabet": "en"`; 0 inserted letters on 783 tune utterances); **จีน, ประเทศ**
  added; 61 sign **variants** (school / politeness styles) read as their word; split test for fluent signing; grammar re-rank guarded.
* Held-out TSL51 test WER 1× / 1.5× / 2×: 0.394 / 0.364 / 0.376 → **0.309 / 0.284 / 0.292** (Recall@1 1× 0.61 → 0.70); unseen-signer words
  0.295 → **0.253**; fluent sentences 0.448 → **0.397**; iPhone sentences 0.283 → **0.126**. Trade-offs: ECE 0.04 → 0.06, fewer accepted words at 1.5×/2×.

## v8 — decision layer (prompt_11): split signs, calibrated confidence, Accept / Uncertain / Reject

Report: `result_reporting/result_v8.md` (+ pdf) · lab: `lab2.ipynb` → "Phase 2 — Optimized". Encoder / tagger / bank unchanged; tuned on tune
data only with `python main.py decision cache | tune --write | report | figures` (CPU, $0).
* **Split-sign merge** (`decode.merge_split`): a small model gives P(same sign) for neighbouring segments (tagger begin/rest at the cut,
  hand-speed dip, lengths, the union read as one piece) — "แมว → กิน แมว" becomes one sign. Held-out word clips: count error 0.171 → 0.083.
* **Calibrated confidence**: temperature-scaled top-1/top-2 probabilities + P(correct) with 13 features (incl. each word's reliability
  in the bank, `models/class_prior.npz`); ECE on test 0.14 → 0.04, correct words get higher P.
* **Three states** (`decode.decide`): Accept P ≥ 0.90 · Uncertain "อาจหมายถึง …" · Reject P < 0.25 "ไม่สามารถรู้จำได้", with a reason per segment.
* Held-out TSL51 test WER 1× / 1.5× / 2×: 0.432 / 0.374 / 0.386 → **0.394 / 0.364 / 0.376**; accepted precision 0.82 → 0.97 (1×);
  iPhone sentences 0.311 → 0.283; fluent synthetic sentences 0.435 → 0.448 (the trade-off).

## v7.2 — after the first live tests

* **Fingerspelling** A–Z (the one-handed manual alphabet Thai deaf schools teach) and Thai ก–ฮ + 7 vowels: `models/fingerspell.npz`
  + `models/letters_bank.npz`. Held-out people: English 83.8 % top-1 / 95.7 % top-3, Thai 92.7 % / 98.0 %. Automatic in sentences
  (settings chosen so word sentences get no letters: `models/spotting.json` "spell") or the web's "🔤 สะกดอย่างเดียว" button.
  Consecutive letters become one name ("JACK"); speech reads them by Thai letter names.
* **Personal sign memory** ("ท่าของฉัน"): confirming a word / letter on a result card remembers that user's way of signing it
  (browser only; one averaged prototype per word). Unseen signer, TSL51 test sentences: recall 0.65 → 0.80, accepted errors 14 → 12.
* **Web**: speech for uncertain readings (setting), Web-Audio playback that live mode can use, GPU → GPU+CPU → CPU MediaPipe fallback,
  automatic exposure / white balance / sharpening, a face box with live emotion (face not covered), real progress bar, the utterance
  sent during the pause (results ready when the pause ends), no LLM wait when every sign is confident.
* **Data**: +40 YouTube word clips; 32 clips of a channel whose signer looks like a data_test person were removed from both banks.
* Non-regression (TSL51 test sentences, same harness): recall 0.642 → 0.647, accepted errors 15 → 15, WER 0.446 → 0.450 with letters
  on (3 uncertain lone letters in 126 sentences) / 0.444 with letters off. data_test unchanged (R@1 0.41, R@5 0.73).
* New steps: `python main.py letters …`, `train-letters`, `study-clips` (letter settings: since v8.1 chosen by `decision tune`), `eval bank_add`, `harvest discover --words …`.

## Reverse part — text / speech → Thai Sign Language avatar

everse_text2sign/\ (own README inside): Thai / English text or speech → TSL structure → motion from the recorded-sign library → 3D avatar.
\docker compose up --build\ also starts it at **http://localhost:8081** (everse-web\ · everse-api\ · everse-speech\); the forward demo stays at :8080.
Forward (\modules/\, \webapp/\, \models/\) and reverse share nothing at run time; branch \pitty\ is the forward base.

## Start here

```bash
docker compose up --build                            # web demo → http://localhost:8080 (camera, live mode, speech)
```
```bash
conda activate hugging                               # Python 3.10, see requirements.txt
git lfs install && git lfs pull                      # model weights are in Git LFS
python main.py infer --video path/to/video.mp4       # translate a video with models/model-v9/ (ONNX Runtime; --backend torch for the GPU)
python main.py test                                  # data_test/ end-to-end from the raw videos + scores
jupyter lab lab.ipynb                                # data, latent space, tagger, segmentation, results, self-made sentence tests
```
`python main.py` with no arguments lists every command.

## Repository map

| path | what | in git |
|---|---|---|
| `main.py` | the only CLI entry point (infer · test · data · prep · extract · harvest · cloud · train-seq · eval · export-onnx) | ✓ |
| `docker-compose.yml`, `webapp/` | web demo: `web` (Next.js) · `sign-api` (FastAPI + ONNX Runtime) · `tts` (MMS-TTS Thai) · `gateway` (nginx) — see `webapp/README.md` | ✓ |
| `models/` | **the forward models**, one complete model folder each: `model-v9/` (main, default `MODEL`, Docker) · `Jack-model/` (v8.1) · `Boom-model/` · `Pitty-model/`; + `face-model/` (face emotion) · `speech-model-v1/` (voice + ASR) | ✓ (LFS) |
| `modules/` | library (see below) | ✓ |
| `scripts/` | pipeline steps: `data_prep.py` · `extract_mp.py` · `harvest_words.py` · `eval.py` · `export_onnx.py` · `infer.py` · `make_lab.py` · `s3_sync.py` · `md_to_pdf.py` · `pretrained.py` · `collect/` (v4 raw-data harvester) | ✓ |
| `GCP/script/` | Vertex AI: `vertex.py` (upload · load-balanced launch · wait · **cleanup**) · `job_entry.py` (training stages) | ✓ |
| `GCP/result/` | one json per GPU job: machine, region, minutes, estimated cost | ✓ |
| `artifacts/` | `pretrained/` (Uni-Sign base, rebuilt by `python main.py pretrained`) · `eval/` (train-only bank) · `runs/<run>/` (metrics, histories) · `reports/` (evaluation json) | ✓ (no per-clip data) |
| `vocab/` | `daily_conversation.json` (target vocabulary) · `vocab.csv` · `daily_conversation_depth.csv` | ✓ |
| `result_reporting/` | reports (`result_vN`, `architecture_vN`, `latent_space_vN`, `file_structure`, md + pdf), figures, `inference/` outputs | ✓ (no video frames) |
| `cloud_s3/` | local mirror of the S3 data lake: `raw_data/` · `data_prep/` · `agent_data/` (harvested by v7) | ✗ (data) |
| `data_test/` | 5 real-world test videos — never used for training or tuning | ✗ |
| `cache/` | local scratch (MediaPipe of data_test, self-test videos, logs) | ✗ |

`modules/`: `mediapipe_pose` (the extractor) · `schema` (Standard Schema, handedness) · `parts` (encoder input) · `encoder` / `unisign`
(sign encoder + training data) · `sequence` (tagger, synthesis, sequence stage) · `segment` (kinematics, windows) · `spanhead`
(one-pass spotting) · `runtime` (ONNX / PyTorch back-ends, bank) · `decode` (decoder, calibration, filters) · `pipeline` (end to end)
· `face` (blendshape cues) · `language` (evidence-guarded sentence) · `grammar` (TSL word order) · `tts` · `lexicon` · `identity`
(signer faces, data prep only) · `utils` · `env`.

## Data (S3 is the source of truth)

`s3://dsi490-lake-signdata/` (ap-southeast-1), SSO profile in `.env` (`AWS_PROFILE`; `aws sso login --profile …` when it expires).

```bash
python main.py data pull data_prep          # prepared data: Standard-Schema poses (MediaPipe + RTMW) + manifest + shards
python main.py data pull raw_data           # raw videos / landmarks (only to rebuild data_prep)
```
v7 adds `agent_data/` (YouTube single-word lessons harvested for daily-conversation words: videos, annotations, QC) — pushed by
the project owner (see result_v7 §"S3 commands").

## Training (one GPU job, everything else local)

```bash
python main.py prep shards                                            # cloud_s3/data_prep/shards (MediaPipe + RTMW views + harvested)
python main.py cloud upload-code --tag v7 && python main.py cloud upload-data && python main.py cloud upload-models
python main.py cloud launch --tag v7 --stage train_encoder --hours 1.2 --args "--steps 3600 --eval_every 400"
python main.py train-seq --encoder_run artifacts/runs/<encoder_run>   # tagger + span head + decoder tables (local GPU)
python main.py eval build --run <encoder_run> --seq <sequence_run> && python main.py eval grammar && python main.py eval spot --seq <sequence_run>
python main.py eval isolated --run <encoder_run> --v6 && python main.py test && python main.py eval selftest --make
python main.py export-onnx && python scripts/make_lab.py --execute
python main.py cloud cleanup                                          # empty the bucket, delete every job, check nothing bills
```

## Principles kept in every version

* **No guessing:** every detected sign is reported as `word` (accepted) / `word?` (uncertain) / `[?] ≈ nearest`; hand transitions
  are labelled `transition`; the sentence layer may only use a segment's candidates and a deterministic evidence guard withholds
  unsupported sentences.
* **Honest evaluation:** held-out *signers* for isolated signs, held-out *sentence templates* for continuous signing, tuning only
  on tune data (out-of-fold tagger outputs); `data_test/` and the self-made sentence videos are only run and scored.
* **One extractor everywhere (v7):** MediaPipe Holistic for the training data, the server and the browser.

## Licensing

Raw data came from public endpoints for research. th-sl.com (NADT) carries an explicit AI-training rights reservation; its use
here was approved by the project owner for research only. YouTube clips (v4 and v7 harvest) remain the publishers' — research
use only. Redistribution or release of models trained on these sources is a separate conversation with each rights holder.
