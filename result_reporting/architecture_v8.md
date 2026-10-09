# ThaiSLM — Architecture v8 (camera → landmarks → signs → Thai sentence → speech)

**Scope:** how v8 is built end to end, for reuse in the project report. Changes from v7 are marked **[v8]**.
Results: `result_v8.md`. Representation space: `latent_space_v8.md`. Lab: `lab.ipynb`.

---

## 1. System at a glance

```mermaid
flowchart LR
  subgraph BROWSER["Browser (Next.js)"]
    CAM[camera / upload] --> MPB[MediaPipe Holistic<br/>tasks-vision 1.0.1]
    MPB --> FB["hand fallback [v8]<br/>HandLandmarker on missed raised hands"]
    FB --> EP[live endpointing<br/>hands up → down]
  end
  subgraph API["sign-api (FastAPI + ONNX Runtime)"]
    SCH[Standard Schema<br/>25 fps · canonical hands] --> FEAT["landmark features [v8]<br/>body frame · hand shape · position · motion"]
    FEAT --> ENC["Thai-only encoder ensemble [v8]<br/>3 × conv-transformer, one pass → frame states H"]
    ENC --> TAG[BiGRU tagger<br/>P(begin / inside)]
    ENC --> SPAN[span head → window embeddings]
    SPAN --> BANK[vocabulary bank<br/>conversation 327 · daily 1,366]
    TAG --> DEC[semi-Markov decoder<br/>+ transition filter]
    BANK --> DEC --> CAL[calibration<br/>accepted ≥ 0.8 / uncertain / unknown]
    CAL --> LANG[evidence-guarded sentence<br/>gpt-4.1-mini or rules]
    HT["hand-tracking report [v8]"]
  end
  TTS["tts service [v8]<br/>VachanaTTS Thai (ONNX)"]
  EP -- landmarks --> SCH
  LANG -- confirmed sentence only --> TTS
```

## 2. Data **[v8]**

| step | rule | effect |
|---|---|---|
| YouTube audit (`scripts/audit_youtube.py`) | measured cuts, people, zoom, signer size, signing-hand framing, hand visibility, label agreement with other signers; eye check on contact sheets (clip vs dictionary reference) | 525 / 561 word clips kept; fingerspelled words, ceremony graphics, split screen, wrongly labelled signers removed |
| label normalisation (`data_prep.py v8_concept`) | "word (source dictionary)" and TSL51 politeness forms → the word | 9,988 → 9,605 concepts; one class per word |
| vocabularies (`data_prep.py daily_vocab`) | conversation = curated list ∪ TSL51; daily = conversation ∪ common Thai words (TNC ≥ 50) signed by ≥ 2 people | 327 / 1,366 words; the 9.6k "full" mode is no longer offered in the app |

Everything else is unchanged from v7: MediaPipe Holistic for every video (the same `.task` file as the browser), the Standard
Schema at 25 fps, left-dominant signers mirrored, held-out **signers** for evaluation, TSL51 sentence templates split tune / test.

## 3. Hands **[v8]**

Holistic finds a hand only through the pose wrist; a blurred, fast hand loses it (measured: blur, not fps, is the cause). A
MediaPipe **HandLandmarker** (palm detector on the whole frame) runs only on frames where a raised, visible wrist has no hand, and
each hand it finds goes to the nearer wrist (within ~1 shoulder width). Same rule in the browser (`webapp/web/lib/holistic.ts
fillHands`) and on the server (`modules/mediapipe_pose.py fill_hands`). Every result carries a hand-tracking report (share of
raised-hand frames with 21 points per side, fallback count, longest gap, fps).

## 4. Sign encoder **[v8]** (`modules/lmenc.py`, `scripts/train_encoder.py`)

**Input features** (NumPy, identical in training, ONNX runtime and the API): body frame = shoulder centre and the clip's shoulder
width; per hand: 20 joints relative to the wrist ÷ hand size (handshape), wrist and centre in the body frame, centre relative to
the nose, hand size (depth), seen flag; right − left hand; velocities of wrists / hand centres and handshape change; nose, shoulders,
elbows, wrists. 122 numbers per frame.

**Member model:** Linear → 4 × [depthwise Conv1D block (k 7, GLU) + Transformer block (4 heads, explicit attention)] → frame states
H [T, 256] → masked mean ‖ attention pool → MLP → 256-d L2-normalised embedding. 3.4 M parameters.

**Training** (local RTX 2050, ~50 min per member, resumable): CosFace (s 30, m 0.25) over 7.9k concepts + 0.5 · SupCon; batches of
64 concepts × 2 instances with a *different signer* as the positive, hard negatives re-mined from the proxies every 1,000 steps;
"not a sign" class from rest, real transitions and straddling windows; augmentation = signer retargeting, mirror, speed 0.7–1.6×,
affine, finger jitter, hand drop-outs, webcam frame-rate; RTMW views of the same videos as extra positives (p 0.25).
Selection: mean MRR on the held-out-signer **val** sets.

**Release = ensemble of three members** (frame states concatenated → 768; embedding = concatenation / √3):

| member | difference |
|---|---|
| v8a | the recipe above |
| v8b | + EMA of weights (0.998) |
| v8c | + **co-articulation context** (p 0.5: 3–10 frames of other signs glued before / after, retargeted, 1–4 frame transition) and ±30 % boundary jitter |

The ensemble was chosen on the TSL51 **tune** sentences (word windows of an unseen signer in real sentences): it is the most robust
to window boundaries. 10.2 M parameters in total (v7: 91.9 M), one ONNX fp32 graph of 43 MB for the whole ensemble,
parity cos 1.000.

## 5. Continuous signing (unchanged design, retrained for v8)

Encoder once over the utterance → H → (a) BiGRU tagger on [H ‖ 13 kinematic features] → P(sign begin / inside); (b) span head
distilled from the encoder's own window embeddings → embeddings of all candidate windows from H → bank scores → semi-Markov decoder
→ exact re-score of the chosen segments → calibrated P(correct) → transition / repeat filters → TSL word-order re-rank.
**[v8]** decoder grid allows 8-frame minimum segments (fast signing), tuned separately for conversation and daily vocabularies on
the tune templates with out-of-fold taggers.

## 6. Language and speech

Language layer unchanged (accepted words only, evidence guard, rules composer offline). **[v8]** speech = VachanaTTS (Thai VITS
voices, ONNX Runtime; `.env` `TTS_PROVIDER=vachana`, `TTS_VOICE_VACHANA=th_f_1|th_f_2|th_m_1|th_m_2`), chosen with a Thai speech
recogniser as judge (CER 0.022 vs 0.107 for the v7 voice), 15× faster than real time on CPU, no PyTorch in the TTS image.

## 7. Deployment

| piece | technology |
|---|---|
| model files | `models/`: `encoder.pt` (ensemble), `onnx/encoder.onnx` + `tagger.onnx`, `span_head.npz`, `bank.npz`, `vocab_{conversation,daily}.json`, settings, `mediapipe/{holistic,hand}_landmarker.task` |
| runtime | `modules/runtime.py`: ONNX Runtime + NumPy (or PyTorch for development); encoder kind from `models/config.json` |
| service | `docker compose up --build`: gateway (nginx :8080) · web (Next.js) · sign-api (FastAPI, ONNX Runtime) · tts (VachanaTTS) |
| browser | MediaPipe Holistic + HandLandmarker fallback on the device; GPU light-correction filter; repeated frames skipped |
| diagnosis | `python main.py diagnose <video or saved request> --intended "…"`; saved live requests carry the hand-tracking report |

## 8. Compute and cost

All v8 training and evaluation ran on the laptop (RTX 2050 4 GB, Ryzen 5 7535HS): encoder members ~50 min each, sequence stage
~1 h 45 min, decoder tuning ~10 min. No cloud GPU job was needed this round.
