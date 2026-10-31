# speech-model-v1 — the signer's sentence in a natural, emotional voice (+ Thai ASR for the hearing partner)

Code: `modules/speech.py` · build / evaluate: `scripts/speech.py` (`python main.py speech …`, speech env `D:/SLM-labs/envs/speech` =
the `hugging` env + transformers 4.57, qwen-asr, f5-tts, f5-tts-th). Report: `result_reporting/result_v9.md` §6.

**Voice (sign → speech)** — `plan_speech(result, affect)` decides what to say (the evidence-guarded sentence / accepted words, never
rejected segments) and how (emotion + intensity from `models/face-model` fusion, speed from signing tempo and arousal);
`ExpressiveTTS.speak(plan)` synthesises it with **F5-TTS-THAI** (VIZINTZOR/F5-TTS-THAI `model_1000000.pt`, CC-BY-4.0, loaded from
the Hugging Face cache) cloned from an **emotion reference** in `refs/` — 7 emotions × 2 voices, Thai lines spoken with ElevenLabs
eleven_v3 audio tags (the "teacher", made once, `refs.json`). The emotion is carried by the reference's prosody (zero-shot,
no fine-tuning). Needs a GPU for interactive use (RTX 2050: real-time factor ≈ 1.5 at 32 steps; CPU is far too slow).

**Ear (speech → text)** — `ThaiASR`: **Qwen3-ASR-0.6B** (Apache-2.0), base model by default. `asr_lora/` is an opt-in LoRA
(r 16, decoder attention + MLP, 1 epoch, lr 3e-5) trained on 1,228 short everyday utterances (ElevenLabs teacher voices + F5
clones; no test text): it lowers CER on unseen synthetic phrases 4.2 → 1.3 % but raises it on real read speech (FLEURS 6.7 → 8.4 %),
so it is **not** the default.

| file | what |
|---|---|
| `speech_config.json` | base checkpoints, F5 architecture, min intensity for an emotional reference |
| `refs.json`, `refs/*.wav` | emotion reference prompts (voice × emotion → wav + its text) |
| `asr_lora/` | optional Qwen3-ASR adapter (see above) |
Data: `cloud_s3/speech-data/` (ElevenLabs teacher audio, ASR train / test jsonl) — not in git.
