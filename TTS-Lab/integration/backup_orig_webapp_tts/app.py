"""ThaiSLM tts — FastAPI micro-service: Thai text (+ the signer's facial emotion) → speech (WAV).

  POST /v1/speak  {text, emotion?, intensity?}  → audio/wav
  GET  /v1/health
Voice: facebook/mms-tts-tha (VITS, offline). Emotion → speaking rate / pitch (modules/tts.py SpeechStyle).
"""
from __future__ import annotations

import sys
from pathlib import Path as _P

_ps = _P(__file__).resolve().parents
_root = _ps[2] if len(_ps) > 2 else _ps[-1]
if (_root / "modules").is_dir() and str(_root) not in sys.path:      # running from the repository (no Docker)
    sys.path.insert(0, str(_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from functools import lru_cache

from modules.tts import synthesize_wav_bytes


@lru_cache(maxsize=512)
def _cached(text: str, emotion: str, intensity: float) -> bytes:   # greetings and short daily sentences repeat a lot
    return synthesize_wav_bytes(text, emotion, intensity)

app = FastAPI(title="ThaiSLM tts", version="7.0")


class SpeakRequest(BaseModel):
    text: str
    emotion: str = "neutral"
    intensity: float = 0.0


@app.on_event("startup")
def warm():
    synthesize_wav_bytes("สวัสดี")


@app.get("/v1/health")
def health():
    return dict(ok=True, voice="facebook/mms-tts-tha")


@app.post("/v1/speak")
def speak(req: SpeakRequest):
    text = req.text.replace("[?]", "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    if len(text) > 400:
        raise HTTPException(413, "text too long")
    return Response(_cached(text, req.emotion, round(float(req.intensity), 1)), media_type="audio/wav")
