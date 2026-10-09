# models/ — the deployable ThaiSLM model (v8)

Everything inference loads, and nothing else. Training-only material lives in `artifacts/` (`pretrained/` Uni-Sign base weights for the v7 comparisons,
`eval/` train-only bank, `runs/` job outputs, `reports/` evaluation json).

```python
from modules.pipeline import SignPipeline
pipe = SignPipeline(backend="onnx", device="cpu", vocab="daily")             # ONNX Runtime + NumPy only (what the web API runs)
r = pipe.run("video.mp4")                                                     # MediaPipe → … → r["sentence"], r["segments"]
r = pipe.run_landmarks(pipe.landmarks_from_browser(frames, (H, W)))           # landmarks already extracted in the browser
pipe = SignPipeline(backend="torch", device="cuda")                           # development: models/*.pt on the GPU
```
or `python main.py infer --video video.mp4 [--vocab conversation] [--tts]`.

| file | what it is | used by |
|---|---|---|
| `mediapipe/hand_landmarker.task` | MediaPipe HandLandmarker — fills hands Holistic misses near a raised pose wrist (browser + server); `result["hand_tracking"]` reports it | `modules/mediapipe_pose.py`, `webapp/web` |
| `mediapipe/holistic_landmarker.task` | **the extractor** — MediaPipe Holistic (tasks 1.0.1): pose 33, hands 21+21, face 478 + 52 blendshapes. The same file runs in Python (training data, server) and in the browser (web demo) | `modules/mediapipe_pose.py`, `webapp/web` |
| `encoder.pt` | sign encoder: ensemble of three Thai-only landmark encoders (122 features/frame → Conv1D + Transformer → 256-d each, concatenated 768-d), 10.2 M parameters, trained from scratch (no ASL/CSL pre-training) | `modules.lmenc.load_model` |
| `tagger.pt` | frame tagger (BiGRU over the encoder's 768-d frame states + 13 kinematic features): non-sign / sign-begin / sign-inside | pipeline |
| `span_head.npz` | **one-pass spotting head** (distilled from the encoder): window embeddings from the utterance's frame states, ~free | `modules/spanhead.py` |
| `bank.npz` | vocabulary bank: one embedding per labelled clip (`Z` fp16, concept `c`, clip / dataset / signer) — append-only | spotting |
| `concepts.json` | concept names (synonyms merged: กิน = ทาน = รับประทาน) | spotting, output |
| `vocab_daily.json` | **default** daily vocabulary: 1,366 words = conversation list ∪ words signed by ≥ 2 people with TNC frequency ≥ 50 | pipeline (`vocab="daily"`) |
| `vocab_conversation.json` | the 327-word daily-conversation list + signers per word | pipeline (`vocab="conversation"`) |
| `null_proto.npy` | embedding of "not a sign" (rest, hand transitions) — the competitor in spotting | decoder |
| `freq_prior.npy` | Thai National Corpus word-frequency prior (re-ranks words inside a segment, never the segmentation) | decoder |
| `spotting.json` | decoder settings per vocabulary (window grid, λ, word prior, tempo normalisation, exact re-score, transition / repeat filters) chosen on held-out tune data (v8: natural-phrase tune half) | pipeline |
| `segment_calibration.json` | P(correct) model per vocabulary + accepted / uncertain thresholds (v8 rule: precision ≥ 0.85 on every tune speed) | status |
| `fingerspell.npz`, `letters_bank.npz` | fingerspelling classifier + letter bank (A–Z, ก–ฮ) | `modules/fingerspell.py` |
| `grammar.json`, `roles.json`, `roles_language.json` | TSL word-order patterns (TSL51) and word roles — sentence layer | `modules/language.py` |
| `face_template.npz` | mean face shape for the Standard Schema's face points | `modules/schema.py` |
| `config.json` | provenance: encoder kind (`lmenc`), runs, sizes, version | `modules/runtime.py` |
| `onnx/encoder.onnx` (43 MB, dynamic length) · `onnx/tagger.onnx` | production graphs; `parity.json` = ONNX Runtime vs PyTorch | `modules/runtime.py` |

## Inference in one picture

```
landmarks (MediaPipe + hand fallback, browser or server) → Standard Schema 25 fps → right-hand-dominant canonical form
  → encoder ONE pass → frame states H ─┬→ tagger: P(sign), P(begin)          → tempo normalisation (fast signing)
                                       └→ span head: window embeddings      → bank scores → semi-Markov decoder (segments)
  → exact re-score of the chosen segments (encoder on those windows only) → calibrated status (accepted / uncertain / unknown /
    transition) → repeat filter → TSL word-order re-rank → face cues (question / negation / emotion)
  → evidence-guarded Thai sentence (OpenAI or rules) → speech (VachanaTTS)
```

## Updating the vocabulary without retraining

The bank is append-only: a new word, or a new signer of a known word = extract the clip with `modules/mediapipe_pose.extract`,
embed it with the encoder (`TorchBackend.embed` / `OnnxBackend.embed` on `backend.prep(...)`), append `(Z, c)` to `bank.npz` (and the
name to `concepts.json` / `vocab_daily.json` if new; `python main.py eval bank_add`). A new fine-tune is only needed to make the space itself better.
