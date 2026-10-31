"""ThaiSLM tts — FastAPI micro-service: the signer's sentence (+ facial emotion) → speech (WAV). Speech v2 (models/speech-model-v2).

  POST /v1/speak       {text, emotion?, intensity?}                                    → audio/wav (v9 API, unchanged)
  POST /v1/speak_plan  {segments, affect?, voice?, speak_uncertain?, disfluency?, …}  → {wav_b64, sr, plan}
                       the forward result itself: modules/speech.plan_speech decides what is said (accepted words in signing order,
                       a natural hesitation — filler + pause — where a sign could not be read, never a guessed word) and how
                       (emotion, intensity, speed)
  GET  /v1/health
Voice (SPEECH_DEVICE, default auto): F5-TTS-THAI cloned from the emotion references on a GPU (the `gpu` compose profile image, or the
speech env on a workstation), MMS-TTS Thai (VITS) on CPU — the same plan and the same fillers either way (modules/speech.load_voice).
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path as _P

_ps = _P(__file__).resolve().parents
_root = _ps[2] if len(_ps) > 2 else _ps[-1]
if (_root / "modules").is_dir() and str(_root) not in sys.path:      # running from the repository (no Docker)
    sys.path.insert(0, str(_root))

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from modules.tts import synthesize_wav_bytes, wav_bytes

SPEECH_MODEL = _P(os.environ.get("SPEECH_MODEL", _root / "models" / "speech-model-v2"))
SPEECH_DEVICE = os.environ.get("SPEECH_DEVICE", "auto")
_VOICE = {}


def voice():
    if "v" not in _VOICE:
        from modules.speech import load_voice
        _VOICE["v"] = load_voice(SPEECH_MODEL, device=None if SPEECH_DEVICE == "auto" else SPEECH_DEVICE)
    return _VOICE["v"]


@lru_cache(maxsize=512)
def _cached(text: str, emotion: str, intensity: float) -> bytes:   # greetings and short daily sentences repeat a lot
    return synthesize_wav_bytes(text, emotion, intensity)

app = FastAPI(title="ThaiSLM tts", version="10.0")


class SpeakRequest(BaseModel):
    text: str
    emotion: str = "neutral"
    intensity: float = 0.0


class PlanRequest(BaseModel):
    segments: list = []
    affect: dict = {}
    sentence: str = ""
    provider: str = "rules"
    timings_ms: dict = {}
    voice: str = "female"
    speak_uncertain: bool = False
    p_min: float = 0.5
    disfluency: str = "filler"          # filler | pause | skip


@app.on_event("startup")
def warm():
    synthesize_wav_bytes("สวัสดี")
    if SPEECH_MODEL.exists():
        voice()


@app.get("/v1/health")
def health():
    v = _VOICE.get("v")
    return dict(ok=True, voice="facebook/mms-tts-tha", plan_voice=type(v).__name__ if v else None, speech_model=str(SPEECH_MODEL.name))


@app.post("/v1/speak")
def speak(req: SpeakRequest):
    text = req.text.replace("[?]", "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    if len(text) > 400:
        raise HTTPException(413, "text too long")
    return Response(_cached(text, req.emotion, round(float(req.intensity), 1)), media_type="audio/wav")


@app.post("/v1/speak_plan")
def speak_plan(req: PlanRequest):
    import time
    from modules.speech import plan_speech
    if req.disfluency not in ("filler", "pause", "skip"):
        raise HTTPException(400, "disfluency must be filler | pause | skip")
    result = dict(segments=req.segments, sentence=req.sentence, fusion=dict(provider=req.provider), timings_ms=req.timings_ms)
    plan = plan_speech(result, req.affect, voice=req.voice, speak_uncertain=req.speak_uncertain, p_min=req.p_min, disfluency=req.disfluency)
    t = time.perf_counter()
    y, sr = voice().speak(plan)
    ms = (time.perf_counter() - t) * 1000
    return dict(wav_b64=base64.b64encode(wav_bytes(y, sr)).decode() if len(y) else "", sr=sr, synth_ms=round(ms, 1),
                plan=dict(text=plan.text, units=plan.units, emotion=plan.emotion, intensity=plan.intensity, speed=plan.speed, reason=plan.reason))
