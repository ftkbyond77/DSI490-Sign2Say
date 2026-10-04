# ThaiSLM — Thai Sign Language → Thai text → speech

A pose-based Thai Sign Language (TSL) translator for real video and a live camera: it finds where signing happens, spots each
sign, names it from a daily-conversation vocabulary (or the full ~10k-concept dictionary), says how sure it is, lets a language
layer write a Thai sentence from the evidence (never beyond it) — and speaks it.
Results: `result_reporting/result_v7.md` · file map: `result_reporting/file_structure.pdf` · lab: `lab.ipynb`.


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
* New steps: `python main.py letters …`, `train-letters`, `tune-spelling`, `study-clips`, `eval bank_add`, `harvest discover --words …`.

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
| `main.py` | the only CLI entry point (infer · test · data · prep · extract · harvest · cloud · train-seq · eval · export-onnx) | ✓ |
| `docker-compose.yml`, `webapp/` | web demo: `web` (Next.js) · `sign-api` (FastAPI + ONNX Runtime) · `tts` (MMS-TTS Thai) · `gateway` (nginx) — see `webapp/README.md` | ✓ |
| `models/` | **the deployable model** — everything inference loads (+ `onnx/`, `mediapipe/`). See `models/README.md` | ✓ (LFS) |
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
