"""L6 TTS behind one interface. Providers: local MMS-TTS Thai (VITS, offline), edge-tts, OpenAI TTS.
SpeechStyle carries emotion → rate/pitch mapping and a persona hook for later accent/style work."""
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
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((a * 32767).astype(np.int16).tobytes())


class LocalMMSTTS:
    """facebook/mms-tts-tha (VITS). Offline; rate via speaking_rate, pitch via resampling."""
    name = "local_mms"

    def __init__(self, model="facebook/mms-tts-tha", device="cpu"):
        from transformers import AutoTokenizer, VitsModel
        self.tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.model = VitsModel.from_pretrained(model, local_files_only=True).to(device).eval()
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
