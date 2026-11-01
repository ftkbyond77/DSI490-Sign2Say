# Speech-model-v4-TTS

ThonburianTTS (F5-TTS, 337M) fine-tuned twice (v1 → v3 → v4): youthful male / female voices, a gap marker `…`, **7 emotions** (FER2013 classes) and
~2.3–2.6x lower latency than v3 (16 ODE steps). Rollback: `models/Speech-model-v3-TTS`. Full evidence and limits: `TTS-Lab/TTS-result-v2.pdf`.

```python
import sys; sys.path.insert(0, "models/Speech-model-v4-TTS")
from tts import Speech
s = Speech("models/Speech-model-v4-TTS")                                # GPU recommended (env 490-TTS)
wav, sr = s.say("ฉันชอบแมว", voice="male")                               # no expression = neutral (== v3 path)
wav, sr = s.say("ฉันชอบแมว", voice="female", emotion="happy", intensity=0.5)   # intensity 0..1 (0 = neutral); default 0.5 = subtle
wav, sr = s.say([("ดีใจจัง", "happy"), ("แล้วไงต่อ", "surprise")])         # pieces -> one waveform
wav, sr = s.say("ฉัน … แม่")                                             # "…" / "___" / "[?]" = natural pause or soft hum
s.save("out.wav", wav, sr)
```

* **Voices** (`voices.json`): `male`, `female` (defaults), `male_2`, `female_2`, plus `*_long` (longer references of the same speakers). References are 3–4 s clips of
  held-out dataset speakers (anonymised ids; perceived age ≈ 24 by an estimator, not a true age). A voice = reference prompt; `ref_tail_s` (0.3 s) adds silence after it.
* **Emotions**: `neutral, happy, sad, angry, fear, disgust, surprise` (alias `curious` → surprise at 0.4x). Two parts: a learned tag symbol in front of the text (7 unused vocab symbols, intensity =
  tag-embedding mix) and a PSOLA prosody layer (`emotion_dsp` in `config.json`; needs `praat-parselmouth`) with the same templates the tags were trained on. **Measured:** the model's own tag moves
  pitch only slightly (independent SER at chance); the prosody layer reproduces the templated F0 / level / tempo shifts; its naturalness is not rated (listening pack: `TTS-Lab/data/run/gen_v4/listening_pack.html`).
  `s.say(..., prosody=False)` = model tag only.
* **Speed**: `default_nfe` 16 (EPSS). 12 steps lost quality in the gate; 32 steps = `nfe=32`.
* **Files**: `model.safetensors` (fp16), `vocab.txt` (unchanged), `voices.json`, `refs/`, `config.json`, `tts.py`.
* **Service**: `webapp/tts` (`POST /v1/speak {text | pieces, voice, emotion?, intensity?, speed?, gap_s?}`); `SPEECH_MODEL_DIR` selects this folder or v3.
* **Licence**: weights derive from ThonburianTTS (CC-BY-NC-SA) → non-commercial / share-alike; reference voices come from real dataset speakers (consent unknown). Research use only.
* **Weights are not in git**: `model.safetensors` is 674 MB (GitHub's limit is 100 MB per file, and `.safetensors` is not under LFS). Everything else needed to use or retrain the model is here;
  get the weights from the project owner (same file as `TTS-Lab/data/run/model.safetensors`) or retrain with `TTS-Lab/vertex_tts.py` (~$0.6 for a 20-minute run; see `TTS-Lab/README.md`). `models/Speech-model-v3-TTS/model.safetensors` is excluded the same way.
