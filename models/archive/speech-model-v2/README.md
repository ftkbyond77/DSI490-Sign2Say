# speech-model-v2 — the signer's sentence in a natural, emotional voice that hesitates like a person (ThaiSLM v10)

Code: `modules/speech.py` · build / evaluate: `scripts/speech.py` (`python main.py speech …`, speech env `D:/SLM-labs/envs/speech`) ·
service: `webapp/tts/app.py` (`POST /v1/speak_plan`) · report: `result_reporting/result_v10.md` §6.

**What changed from v1 (speech-model-v1)**

1. **Natural disfluency for unread signs** — `plan_speech(result, affect, disfluency="filler")` turns the forward result into units
   in signing order: `say` (accepted words; tentative ones only when the user asks) and `hesitate` (a filler + a pause) where a sign
   could not be read (`[?]` / unknown, or an uncertain word that is not spoken). v1 skipped those signs ("ฉัน ___ พ่อ" → "ฉัน พ่อ").
   Rules: never a guessed word (the `[?]` segment's nearest word is never said; fillers are not content words) · neighbouring unread
   signs = ONE hesitation · an unread sign at the end = silence · a sentence that starts unread opens with "เอ่อ" · the pause follows
   the unread signs' duration (0.25 s + 0.6 × duration, 0.35–1.4 s) · fillers rotate so the voice does not repeat "อืม อืม".
   Modes: `filler` (default) · `pause` (silence only) · `skip` (v1 behaviour).
2. **Fillers** (`fillers/`, `fillers.json`): อืม · เอ่อ · เอ่อ… · แบบว่า… — two takes per voice, spliced between separately synthesised
   phrases with 15 ms fades, loudness matched to the voice. Made with the voice's own F5 clone (ElevenLabs teacher quota was used up).
3. **Two voices behind one interface** — `load_voice(model_dir)`: F5-TTS-THAI (GPU; cloned from `refs/`, emotion carried by the
   reference) or the CPU voice (MMS-TTS Thai + the same plan and fillers) on machines without a GPU (F5 on CPU: real-time factor ≫ 10).
   Docker: `tts` (CPU, default) and `tts-gpu` (compose profile `gpu`); sign-api tries the GPU voice first.
4. **ASR choice** — see `speech_config.json` `asr` and the report (Qwen3-ASR base vs Thonburian Whisper medium on real speech only).

| file | what |
|---|---|
| `speech_config.json` | F5 checkpoint / architecture, min intensity for an emotional reference, disfluency defaults, CPU voice, ASR choice |
| `refs.json`, `refs/*.wav` | emotion reference prompts (unchanged from v1: 7 emotions × 2 voices, ElevenLabs eleven_v3 teacher) |
| `fillers.json`, `fillers/*.wav` | hesitation fillers per voice |

Data: `cloud_s3/speech-data/` (not in git). Licences: F5-TTS-THAI CC-BY-4.0, MMS-TTS CC-BY-NC-4.0 (research), Qwen3-ASR Apache-2.0.
