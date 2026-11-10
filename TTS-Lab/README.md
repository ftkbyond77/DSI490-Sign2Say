# TTS-Lab — Speech-model-v4-TTS

Development, experiments and report for the ThaiSLM voice. Model (used by the app): `models/Speech-model-v4-TTS/` (README there). Report: `TTS-result-v2.pdf` (v1: `TTS-result-v1.pdf`).
v4 = male first-word fix (reference choice + silence tail + prompt-format training), youthful voices, 7-emotion control, 2.3–2.6x lower latency. Honest status (details in the report):
emotion control by the model alone is weak (a PSOLA layer makes it audible), generated voices are estimated at ≈ 26–27 years (target 18–23), nothing is human-rated.

| file | role |
|---|---|
| `prep.py` | `select` (metadata filters) · `fetch` (download shard by shard) · `label` (age + emotion, GPU; `label 30000` = exactly 30,000 clips) · `build` (manifest, splits, voice presentation, emotion labels) |
| `train.py` | fine-tune: multi-grade, prompt-format samples, onset-weighted loss, gap samples, emotion tags + PSOLA transforms (`--gate` = 2-step check, `--preflight` = CPU check) |
| `tts.py` | inference API (copied into the model folder) |
| `evaluate.py` | `gen` · `score` · `affect` · `latency` · `pack` (listening pack) · `compare` (tables + paired bootstrap → `data/results.json`) |
| `vertex_tts.py` | `upload` / `launch --minutes N [--no_eval --regions R]` / `cleanup` (reuses `GCP/script`) |
| `make_figures.py` | `figures/` |
| `TTS-lab.ipynb` | executed demo: neutral, 7 sentences, 5x5 words, gaps, 7 emotions + sequences + intensity sweep, onset regression, latency v3 vs v4 |

Rebuild order: `prep.py select/fetch` → `prep.py label 30000` (env with funasr) → `prep.py build` → `vertex_tts.py upload` → `launch` → `evaluate.py gen/score/affect/compare` → `make_figures.py` → `vertex_tts.py cleanup`.
Cost lesson: the job timeout counts from provisioning start; copy the checkpoint before anything slow (fixed in `vertex_tts.py`).
`data/` keeps what reproduces the numbers (manifest, labels, selection, results, ablation scores, run logs, the trained checkpoint, stack measurements); raw shards and audio clips were deleted (`prep.py fetch` rebuilds them, ≈ 26 GB download, ≈ 7 GB of FLAC).
