# models/model-v10 — the deployable ThaiSLM forward model (v10)

v10 **recognises exactly like v9**: the networks (Uni-Sign encoder, tagger, span head, Thai landmark encoder), the banks and the
decision settings are byte-identical to `models/model-v9`. The v10 candidate for recognition — extractor-averaged exemplars
(MediaPipe + RTMW-X view of each bank clip) + the decision layer re-tuned for them — was better on tune data but failed the release
gate on the held-out test sets (coverage / calibration / TSL51 2× / fluent sentences worse), so it does not ship; it is
kept as the opt-in **v10x** in `artifacts/v10_experimental/` (below; `result_reporting/result_v10.md` §3).

What v10 adds around the recogniser: streaming settings for real-time use (`spotting.json` → `"stream"`: partial → confirmed
words, `python main.py decision stream --write`), speech v2 (`models/speech-model-v2`: a human-like hesitation instead of a guessed
word), the optional RTMW tier-A extractor (`modules/wholebody.py`, not the default — extractor study §2), the automatic release
gate (`python main.py decision gate`) and `real-time-lab.ipynb`. Face: `models/face-model` (unchanged in v10). Rollback:
`models/model-v9` (`THAISLM_MODELS_DIR=./models/model-v9 docker compose up`).

## Files (since v7)

Everything inference loads, and nothing else. Training-only material lives in `artifacts/` (`pretrained/` Uni-Sign base weights,
`eval/` train-only bank, `runs/` job outputs, `reports/` evaluation json).

```python
from modules.pipeline import SignPipeline
pipe = SignPipeline(model_dir="models/model-v10", backend="onnx", device="cpu", vocab="conversation")      # ONNX Runtime + NumPy only (what the web API runs)
r = pipe.run("video.mp4")                                                     # MediaPipe → … → r["sentence"], r["segments"]
r = pipe.run_landmarks(pipe.landmarks_from_browser(frames, (H, W)))           # landmarks already extracted in the browser
pipe = SignPipeline(backend="torch", device="cuda")                           # development: models/*.pt on the GPU
```
or `python main.py infer --video video.mp4 [--vocab full] [--tts]`.

| file | what it is | used by |
|---|---|---|
| `mediapipe/holistic_landmarker.task` | **the extractor** — MediaPipe Holistic (tasks 1.0.1): pose 33, hands 21+21, face 478 + 52 blendshapes. The same file runs in Python (training data, server) and in the browser (web demo) | `modules/mediapipe_pose.py`, `webapp/web` |
| `encoder.pt` | sign encoder: Uni-Sign ST-GCN + mT5 encoder (top 8 layers fine-tuned) + embedding head — full state dict, fp16 | `modules.encoder.load_encoder` |
| `mt5_config.json` | architecture config of the encoder's mT5 part (so `encoder.pt` loads without the 361 MB base checkpoint) | encoder |
| `tagger.pt` | frame tagger (BiGRU over encoder frame states + 13 kinematic features): non-sign / sign-begin / sign-inside | pipeline |
| `span_head.npz` | **one-pass spotting head** (distilled from the encoder): window embeddings from the utterance's frame states, ~free | `modules/spanhead.py` |
| `bank.npz` | vocabulary bank: one embedding per labelled clip (`Z` fp16, concept `c`, clip / dataset / signer) — append-only | spotting |
| `concepts.json` | concept names (synonyms merged: กิน = ทาน = รับประทาน) | spotting, output |
| `vocab_conversation.json` | the daily-conversation vocabulary (conversation mode) + signers per word | pipeline (`vocab="conversation"`) |
| `null_proto.npy` | embedding of "not a sign" (rest, hand transitions) — the competitor in spotting | decoder |
| `freq_prior.npy` | Thai National Corpus word-frequency prior (re-ranks words inside a segment, never the segmentation) | decoder |
| `spotting.json` | decoder settings per vocabulary (window grid, λ, word prior, tempo normalisation, exact re-score, transition / repeat filters) chosen on held-out tune data | pipeline |
| `segment_calibration.json` | P(correct) model per vocabulary + accepted / uncertain thresholds (precision per tier inside) | status |
| `grammar.json`, `roles.json` | TSL word-order patterns (TSL51) and word roles — sentence layer | `modules/language.py` |
| `face_template.npz` | mean face shape for the Standard Schema's face points | `modules/schema.py` |
| `config.json` | provenance: runs, sizes, version | — |
| `onnx/encoder.onnx` · `onnx/tagger.onnx` (+ `encoder.int8.onnx` when parity allows) | production graphs; `parity.json` = ONNX Runtime vs PyTorch | `modules/runtime.py` |

## Inference in one picture

```
landmarks (MediaPipe, browser or server) → Standard Schema 25 fps → right-hand-dominant canonical form
  → encoder ONE pass → frame states H ─┬→ tagger: P(sign), P(begin)          → tempo normalisation (fast signing)
                                       └→ span head: window embeddings      → bank scores → semi-Markov decoder (segments)
  → exact re-score of the chosen segments (encoder on those windows only) → calibrated status (accepted / uncertain / unknown /
    transition) → repeat filter → TSL word-order re-rank → face cues (question / negation / emotion)
  → evidence-guarded Thai sentence (OpenAI or rules) → speech (MMS-TTS Thai)
```

## Updating the vocabulary without retraining

The bank is append-only: a new word, or a new signer of a known word = extract the clip with `modules/mediapipe_pose.extract`,
embed it with the encoder (`TorchBackend.embed` / `OnnxBackend.embed` on `parts.prep(...)`), append `(Z, c)` to `bank.npz` (and the
name to `concepts.json` / `vocab_conversation.json` if new). A new fine-tune is only needed to make the space itself better.

## Added in v9

| file | what it is | used by |
|---|---|---|
| `onnx/lmenc.onnx` | Thai-only landmark encoder ensemble (3 × Conv+Transformer, 768-d), the second opinion on each sign segment | `modules/lmenc.OnnxLM`, `SignPipeline.hand_rescore` |
| `lm_bank.npz` | its embedding of every bank clip (`Z` fp16, `c`, `ok`) — built by `python main.py decision lmbank` | pipeline |
| `hand_bank.npz`, `class_prior.npz` | hand-shape cue bank (v8.1) · per-word reliability (refitted for v9) | pipeline / calibration |

## Changed in v10

| file | what changed |
|---|---|
| `spotting.json` | `"stream"` added — streaming step / look-ahead / stability chosen on tune replays (`python main.py decision stream --write`); every other key = v9 |
| `config.json` | version v10 |
| everything else | byte-identical to `models/model-v9` |

## Opt-in: v10x (not shipped)

`artifacts/v10_experimental/` = `bank.npz` (13,282 / 19,656 entries = normalised mean of the MediaPipe and RTMW-X embedding of the
same clip, `view_averaged` flag), `eval/bank_eval.npz` (the same for the train-only bank), and the decision settings re-tuned for
that bank (`spotting.json`: landmark-encoder weight 0.4, hand weight 1.2, merge 0.4, split 0.6; `segment_calibration.json`:
τ accept 0.88; `class_prior.npz`). Evaluate: `python main.py decision report --versions v9,v10x`. Use in an app: copy
`models/model-v10` to a new folder, overwrite those four files, point `THAISLM_MODELS_DIR` at it. Reproduce: `python main.py
decision views` then `python main.py decision tune --start v9 --significant --write` (both default to `artifacts/v10_experimental`).
