"""Speech output behind one interface. Providers: local MMS-TTS Thai (VITS, offline — the default and the web demo's TTS
service), edge-tts, OpenAI TTS. SpeechStyle maps the signer's facial emotion (modules/face.py) to speaking rate / pitch."""
from __future__ import annotations

import io
import os
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import numpy as np

STYLE_MAP = {
    "neutral": dict(rate=1.00, pitch=0), "happy": dict(rate=1.10, pitch=2), "sad": dict(rate=0.90, pitch=-2),
    "angry": dict(rate=1.15, pitch=1), "surprised": dict(rate=1.05, pitch=3),
}


@dataclass
class SpeechStyle:
    emotion: str = "neutral"
    intensity: float = 0.0
    rate: float = 1.0
    pitch_semitones: float = 0.0
    voice: str = "th-TH-PremwadeeNeural"
    persona: str | None = None

    @classmethod
    def from_affect(cls, emotion="neutral", intensity=0.0, **kw):
        m = STYLE_MAP.get(emotion, STYLE_MAP["neutral"])
        k = 0.5 + 0.5 * float(np.clip(intensity, 0, 1))
        return cls(emotion=emotion, intensity=intensity, rate=1 + (m["rate"] - 1) * k,
                   pitch_semitones=m["pitch"] * k, **kw)


class TTSProvider(Protocol):
    name: str
    def synthesize(self, text: str, style: SpeechStyle) -> tuple[np.ndarray, int]: ...


def write_wav(path, audio: np.ndarray, sr: int):
    a = np.clip(audio, -1, 1)
    with wave.open(path if hasattr(path, "write") else str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((a * 32767).astype(np.int16).tobytes())


class LocalMMSTTS:
    """facebook/mms-tts-tha (VITS). Offline; rate via speaking_rate, pitch via resampling."""
    name = "local_mms"

    def __init__(self, model="facebook/mms-tts-tha", device="cpu"):
        from transformers import AutoTokenizer, VitsModel
        try:
            self.tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
            self.model = VitsModel.from_pretrained(model, local_files_only=True).to(device).eval()
        except OSError:                       # not cached yet (fresh container) → download once
            self.tok = AutoTokenizer.from_pretrained(model)
            self.model = VitsModel.from_pretrained(model).to(device).eval()
        self.device, self.sr = device, self.model.config.sampling_rate

    @lru_cache(maxsize=256)
    def _synth(self, text, rate):
        import torch
        self.model.speaking_rate = rate
        ids = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model(**ids).waveform[0].cpu().numpy()

    def synthesize(self, text: str, style: SpeechStyle | None = None):
        style = style or SpeechStyle()
        if not text.strip():
            return np.zeros(1, np.float32), self.sr
        wav = self._synth(text, float(style.rate))
        if style.pitch_semitones:
            f = 2 ** (style.pitch_semitones / 12)
            idx = np.arange(0, len(wav), f)
            wav = np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)
        return wav, self.sr


class EdgeTTSProvider:
    """Microsoft Edge neural voices (network). Returns decoded audio if soundfile/pydub is available."""
    name = "edge"

    def synthesize(self, text: str, style: SpeechStyle | None = None):
        import asyncio
        import edge_tts  # optional dependency
        style = style or SpeechStyle()
        rate = f"{int(round((style.rate - 1) * 100)):+d}%"
        pitch = f"{int(round(style.pitch_semitones * 6)):+d}Hz"

        async def run():
            buf = io.BytesIO()
            async for ch in edge_tts.Communicate(text, style.voice, rate=rate, pitch=pitch).stream():
                if ch["type"] == "audio":
                    buf.write(ch["data"])
            return buf.getvalue()
        mp3 = asyncio.run(run())
        import soundfile as sf
        a, sr = sf.read(io.BytesIO(mp3))
        return a.astype(np.float32), sr


class OpenAITTSProvider:
    name = "openai"

    def synthesize(self, text: str, style: SpeechStyle | None = None):
        from openai import OpenAI
        style = style or SpeechStyle()
        instr = f"พูดภาษาไทยด้วยอารมณ์ {style.emotion} ความเข้ม {style.intensity:.1f}"
        if style.persona:
            instr += f" บุคลิก/สำเนียง: {style.persona}"
        r = OpenAI().audio.speech.create(model=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"), voice="alloy",
                                         input=text, instructions=instr, response_format="wav")
        with wave.open(io.BytesIO(r.read())) as w:
            sr = w.getframerate()
            a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32767
        return a, sr


def get_tts(provider: str | None = None):
    provider = provider or os.environ.get("TTS_PROVIDER", "local")
    if provider == "edge":
        return EdgeTTSProvider()
    if provider == "openai":
        return OpenAITTSProvider()
    return LocalMMSTTS()


_TTS = {}


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    write_wav(buf, audio, sr)
    return buf.getvalue()


LATIN_TH = dict(A="เอ", B="บี", C="ซี", D="ดี", E="อี", F="เอฟ", G="จี", H="เอช", I="ไอ", J="เจ", K="เค", L="แอล", M="เอ็ม", N="เอ็น", O="โอ",
                P="พี", Q="คิว", R="อาร์", S="เอส", T="ที", U="ยู", V="วี", W="ดับเบิลยู", X="เอ็กซ์", Y="วาย", Z="แซด")


def speakable(text: str) -> str:
    """The Thai voice cannot read Latin script: a spelled name is read letter by letter with Thai letter names
    ("JACK" → "เจ เอ ซี เค", "KFC" → "เค เอฟ ซี") — the user spelled letters, so no pronunciation is guessed."""
    import re
    return re.sub(r"[A-Za-z]+", lambda m: " " + " ".join(LATIN_TH[c.upper()] for c in m.group(0)) + " ", text).strip()


def synthesize_wav_bytes(text: str, emotion: str = "neutral", intensity: float = 0.0, provider: str | None = None) -> bytes:
    """Thai text → WAV bytes in the emotion's speaking style (model loaded once per process)."""
    provider = provider or os.environ.get("TTS_PROVIDER", "local")
    if provider not in _TTS:
        _TTS[provider] = get_tts(provider)
    emo = {"surprise": "surprised"}.get(emotion, emotion)
    wav, sr = _TTS[provider].synthesize(speakable(text.replace("[?]", "").strip()), SpeechStyle.from_affect(emo, intensity))
    return wav_bytes(wav, sr)
