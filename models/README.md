# models/ — the deployable ThaiSLM model (v7)

Everything inference loads, and nothing else. Training-only material lives in `artifacts/` (`pretrained/` Uni-Sign base weights,
`eval/` train-only bank, `runs/` job outputs, `reports/` evaluation json).

```python
from modules.pipeline import SignPipeline
pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation")      # ONNX Runtime + NumPy only (what the web API runs)
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
