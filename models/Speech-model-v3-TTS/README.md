# Speech-model-v3-TTS — ThonburianTTS (F5-TTS) fine-tuned: natural Thai voice, male / female, gap-aware ("…")

Used by the app: `webapp/tts/app.py` (`POST /tts/v1/speak {text, voice: male|female}` behind the gateway; compose service `tts`, GPU, this folder mounted read-only).
Development, training code, evaluation and the full report: `TTS-Lab/` (`TTS-result-v1.pdf`).

```python
import sys; sys.path.insert(0, "models/Speech-model-v3-TTS")
from tts import Speech
s = Speech("models/Speech-model-v3-TTS")
wav, sr = s.say("ฉันอยากไปประเทศ …", voice="feminine_1")        # "…" / "___" / "[?]" = a sign not read → pause or soft hum
```

| file | what |
|---|---|
| `model.safetensors` | fine-tuned weights (fp16, 337M) — 2,774 updates on 11.8 h of S+ Thai speech + synthetic gap samples |
| `vocab.txt` | character vocabulary (unchanged from ThonburianTTS) |
| `voices.json`, `refs/` | 4 reference voices: `masculine_1/2`, `feminine_1/2` (the app's "male" = `masculine_1`, "female" = `feminine_1`) |
| `config.json` | architecture, default gap length (0.8 s), minimum generated duration, training summary |
| `tts.py` | inference (text normalisation, gap-aware duration, voice prompt) |

Licence: derived from ThonburianTTS weights (CC-BY-NC-SA files in its repo) → non-commercial; dataset Chalermdej/yodas2_sidon_th_tts is cc-by-3.0 (attribution).
