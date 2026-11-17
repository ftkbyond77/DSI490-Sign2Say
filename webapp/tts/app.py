"""ThaiSLM tts — FastAPI micro-service: text → speech (WAV). Speech v4 (models/Speech-model-v4-TTS: ThonburianTTS fine-tuned, TTS-Lab/).

  POST /v1/speak   {text | pieces, voice?, emotion?, intensity?, speed?, gap_s?}  → audio/wav
                   voice      male | female (or a profile id such as male_2); default male
                   emotion    neutral (default) | happy | sad | angry | fear | disgust | surprise  (the 7 FER2013 classes; no emotion = neutral)
                              | auto = read the sentence (text_emotion.py lexicon; no cue → fallback_emotion, e.g. the face emotion, else neutral)
                   intensity  0..1 (default 0.5; 0 = neutral; subtle at the default)
                   pieces     [{text, emotion?, intensity?}, …] → one waveform (e.g. a sentence with a happy part and a surprised part)
                   text       "…" / "___" / "[?]" = a sign the system could not read → a natural pause or soft hum, never a guessed word
  GET  /v1/voices · GET /v1/health
Engine: the v4 model on a CUDA GPU (SPEECH_MODEL = the mounted model folder; rollback to v3 = mount models/Speech-model-v3-TTS, its voices are
mapped and the emotion is ignored); MMS-TTS Thai (VITS, CPU) when there is no GPU, or for the one request where the GPU fails (out of memory) —
so the speak button always answers.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path as _P

_ps = _P(__file__).resolve().parents
_root = _ps[2] if len(_ps) > 2 else _ps[-1]
if (_root / "modules").is_dir() and str(_root) not in sys.path:      # running from the repository (no Docker)
    sys.path.insert(0, str(_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from modules.tts import synthesize_wav_bytes, wav_bytes

log = logging.getLogger("tts")
SPEECH_MODEL = _P(os.environ.get("SPEECH_MODEL", _root / "models" / "Speech-model-v4-TTS"))
SPEECH_DEVICE = os.environ.get("SPEECH_DEVICE", "auto")          # auto | cpu (cpu = MMS voice only) | cpu-v4 (v4 on the CPU, for a laptop)
MIX_DIR = _P(os.environ.get("SPEECH_MIX_DIR", _root / "TTS-Lab" / "emo_ref"))   # speech_mix.py + policy.json + refs_final.csv (+ refs_final/*.flac)
SPEECH_NFE = int(os.environ.get("SPEECH_NFE", "0"))                    # 0 = the model default (16 steps); fewer steps = faster, a little rougher (try 8 on a CPU)
USE_MIX = os.environ.get("SPEECH_MIX", "1") != "0"                  # 1 (default): per-emotion method of policy.json (emotional reference / PSOLA); 0: plain v4
EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
V3_VOICES = {"male": "masculine_1", "female": "feminine_1"}      # a v3 model folder has no "male"/"female" profile
_E = {}
_LOCK = threading.Lock()                                          # one synthesis at a time on the GPU


def engine():
    """The speech model, loaded once and warmed (reference prompts, kernels); None when there is no GPU (or no model files) → the CPU voice answers."""
    if "e" not in _E:
        _E["e"] = None
        try:
            import torch
            dev = "cuda" if (SPEECH_DEVICE != "cpu" and torch.cuda.is_available()) else ("cpu" if SPEECH_DEVICE == "cpu-v4" else None)   # cpu-v4: the v4 voice on the CPU (slow, ~20 s per sentence; for trying it on a laptop)
            if dev and (SPEECH_MODEL / "model.safetensors").exists():
                sys.path.insert(0, str(SPEECH_MODEL))
                from tts import Speech                              # noqa: E402  (tts.py ships inside the model folder)
                e = Speech(SPEECH_MODEL, device=dev, warm=False)                # warm=False: the constructor's own warm-up needs refs/male.wav (not in the repo); we warm below
                if SPEECH_NFE:
                    e.nfe = SPEECH_NFE
                    log.info("speech steps (nfe) = %s", SPEECH_NFE)
                try:
                    for v in ("male", "female"):                    # warm-up with the real settings: caches both default references
                        e.say("สวัสดีครับ วันนี้อากาศดี", voice=_profile_of(e, v))
                    if getattr(e, "tags", None):
                        e.say("ดีใจจัง", voice=_profile_of(e, "male"), emotion="happy", intensity=0.5)
                    plain_ok = True
                except FileNotFoundError:                           # refs/*.wav are not in the repo (only on the GPU box): plain v4 cannot speak, the mix (THAI-SER refs) can
                    log.warning("v4 default references (models/Speech-model-v4-TTS/refs) missing — plain v4 unavailable, trying the emotion mix only")
                    plain_ok = False
                mix = _load_mix(e)
                if plain_ok or mix is not None:
                    _E["e"], _E["mix"] = e, mix
                else:
                    _E["e"] = None
        except Exception:                                           # noqa: BLE001  (missing f5-tts, OOM at load …)
            log.exception("speech model not available — CPU voice")
            _E["e"] = None
    return _E["e"]


def _load_mix(e):
    """MixSpeech on top of the loaded v4 model (weights unchanged); None when its files are missing or it fails to start (plain v4 then answers)."""
    if not (USE_MIX and getattr(e, "tags", None) and (MIX_DIR / "speech_mix.py").exists() and (MIX_DIR / "refs_final.csv").exists()):
        return None
    import csv
    with open(MIX_DIR / "refs_final.csv", encoding="utf-8") as f:
        missing = [r["path"] for r in csv.DictReader(f) if not (MIX_DIR / r["path"]).exists()]
    if missing:                                                     # the THAI-SER reference clips are local-only (not in git): get them from TTS-Lab/emo_ref
        log.warning("speech mix off: %d reference clips missing under %s (e.g. %s) — plain v4", len(missing), MIX_DIR, missing[0])
        return None
    try:
        sys.path.insert(0, str(MIX_DIR))
        from speech_mix import MixSpeech                            # noqa: E402
        m = MixSpeech(SPEECH_MODEL, refs=MIX_DIR / "refs_final.csv", policy=MIX_DIR / "policy.json", device=e.dev, speech=e)
        m.say("สวัสดีครับ วันนี้อากาศดี", voice="male"); m.say("ดีใจจัง", voice="female", emotion="happy", intensity=0.5)
        return m
    except Exception:                                               # noqa: BLE001
        log.exception("speech mix not available — plain v4")
        return None


def _profile_of(e, voice: str) -> str:
    p = voice if voice in e.voices else V3_VOICES.get(voice, voice)
    if p not in e.voices:
        raise HTTPException(400, f"unknown voice {voice!r}; use male | female | {list(e.voices)}")
    return p


def _profile(voice: str) -> str:
    e = engine()
    return _profile_of(e, voice) if e is not None else voice


@lru_cache(maxsize=512)
def _cached(pieces: tuple, profile: str, speed: float, gap_s: float) -> tuple[bytes, str]:
    """Greetings and short daily sentences repeat a lot. pieces = ((text, emotion, intensity), …). Returns (wav, engine name)."""
    e = engine()
    if e is not None:
        try:
            m = _E.get("mix")
            if m is not None and profile in m.pol["voices"]:       # per-emotion method (TTS-Lab/emo_ref/policy.json); neutral pieces use the same speaker
                import numpy as np
                with _LOCK:
                    parts = []
                    for t, emo, inten in pieces:
                        _t0 = time.time()
                        y, sr = m.say(t, voice=profile, emotion=None if emo == "neutral" else emo, intensity=inten, speed=speed, gap_s=gap_s)
                        log.info("speak %.1fs voice=%s emotion=%s method=%s tries=%s problem=%s len=%d", time.time() - _t0, profile, emo,
                                 m.last.get("method"), m.last.get("tries"), m.last.get("problem"), len(t))
                        parts += [y, np.zeros(int(0.12 * sr), np.float32)]
                return wav_bytes(np.concatenate(parts[:-1]), sr), "thonburian-v4-mix"
            with _LOCK:
                if getattr(e, "tags", None) and (len(pieces) > 1 or pieces[0][1] != "neutral"):
                    y, sr = e.say(list(pieces), voice=profile, speed=speed, gap_s=gap_s)
                else:                                               # neutral text (or a v3 folder: no emotion tags) → one string
                    y, sr = e.say(" ".join(t for t, _, _ in pieces), voice=profile, speed=speed, gap_s=gap_s)
            return wav_bytes(y, sr), "thonburian-v4"
        except RuntimeError:                                        # CUDA out of memory / device error → this request on CPU
            log.exception("GPU synthesis failed — CPU voice for this request")
            import torch
            torch.cuda.empty_cache()
    txt = " ".join(re.sub(r"\[\?\]|_+|…", " ", t).strip() for t, _, _ in pieces)
    emo, inten = next(((emo, i) for _, emo, i in pieces if emo != "neutral"), ("neutral", 0.0))
    return synthesize_wav_bytes(txt.strip(), emo, inten), "mms-cpu"


app = FastAPI(title="ThaiSLM tts", version="12.0")


class Piece(BaseModel):
    text: str
    emotion: str | None = None
    intensity: float | None = Field(None, ge=0.0, le=1.5)


class SpeakRequest(BaseModel):
    text: str | None = None
    pieces: list[Piece] | None = None
    voice: str = "male"
    emotion: str | None = None        # none = neutral; "auto" = read the sentence (lexicon), fallback_emotion if no cue
    fallback_emotion: str | None = None
    intensity: float = Field(0.5, ge=0.0, le=1.5)
    speed: float = Field(1.0, ge=0.6, le=1.5)
    gap_s: float = Field(0.8, ge=0.2, le=2.0)


@app.on_event("startup")
def warm():
    engine()
    try:
        synthesize_wav_bytes("สวัสดี")
    except Exception:                                               # noqa: BLE001  (MMS not downloaded: the v4 voice may still work)
        log.exception("MMS-TTS fallback voice not available")


@app.get("/v1/health")
def health():
    e = _E.get("e")
    return dict(ok=True, engine="thonburian-v4" if e is not None else "mms-cpu", device=getattr(e, "dev", "cpu"), model=SPEECH_MODEL.name,
                voices=["male", "female"], emotions=EMOTIONS if getattr(e, "tags", None) else ["neutral"])


@app.get("/v1/voices")
def voices():
    e = engine()
    return dict(voices=["male", "female"], profiles={k: v["voice_profile"] for k, v in e.voices.items() if "__" not in k} if e is not None else {}, emotions=EMOTIONS)


def _piece(text: str, emotion: str | None, intensity: float | None, default_i: float, fallback: str | None = None) -> tuple[str, str, float]:
    emo = (emotion or "neutral").lower()
    if emo == "auto":                                                # read the meaning of the sentence (text_emotion.py); no cue → the face emotion, else neutral
        from text_emotion import detect                              # noqa: E402
        emo = detect(text)[0] or (fallback or "neutral").lower()
        if emo == "auto":
            emo = "neutral"
    if emo not in EMOTIONS and emo != "curious":
        raise HTTPException(400, f"unknown emotion {emotion!r}; use one of {EMOTIONS}")
    return text.strip(), emo, round(float(default_i if intensity is None else intensity), 2)


@app.post("/v1/speak")
def speak(req: SpeakRequest):
    raw = [_piece(p.text, p.emotion or req.emotion, p.intensity, req.intensity, req.fallback_emotion) for p in req.pieces] if req.pieces else \
          [_piece(req.text or "", req.emotion, None, req.intensity, req.fallback_emotion)]
    pieces = tuple(p for p in raw if p[0].replace("[?]", "").replace("…", "").replace("_", "").strip())
    if not pieces:
        raise HTTPException(400, "nothing to say")                    # empty, or only gaps
    if sum(len(p[0]) for p in pieces) > 400:
        raise HTTPException(413, "text too long")
    wav, used = _cached(pieces, _profile(req.voice), round(req.speed, 2), round(req.gap_s, 2))
    return Response(wav, media_type="audio/wav", headers={"X-Engine": used, "X-Emotion": ",".join(p[1] for p in pieces)})
