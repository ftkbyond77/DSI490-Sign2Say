"""Speech v1 (models/speech-model-v1/): the signer's sentence spoken in a natural voice that carries their emotion — and the
reverse ear (Thai speech recognition) for the hearing partner.

Voice (sign → speech). Not a fixed "audio display" voice: the forward result decides HOW the sentence is said.
    plan = plan_speech(result, affect)        # text to say, emotion, intensity, speed (signing tempo + arousal), pauses
    wav, sr = ExpressiveTTS(model_dir).speak(plan)
  * Synthesiser: F5-TTS-THAI (VIZINTZOR/F5-TTS-THAI, CC-BY-4.0; flow-matching, zero-shot voice + style cloning from a 3–8 s
    reference). The emotion is carried by the REFERENCE: models/speech-model-v1/refs/<voice>_<emotion>.wav — short Thai lines spoken
    in that emotion (made once with ElevenLabs v3 audio tags as the "teacher"), so the cloned prosody (pitch range, energy,
    tempo, voice quality) follows the signer's affect. Low intensity → the neutral reference with a small speed / energy shift.
  * Only what the forward layer decided to say is spoken (accepted words / the evidence-guarded sentence); nothing is invented.
Ear (speech → text): Qwen3-ASR-0.6B (Apache-2.0) with an optional LoRA adapter (models/speech-model-v1/asr_lora) fine-tuned on
short daily-conversation utterances; also the automatic intelligibility judge of the voice (character error rate).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
# speaking-rate and loudness shifts per emotion at full intensity (applied on top of the reference's own style)
STYLE = {"neutral": (1.00, 1.00), "happy": (1.06, 1.10), "sad": (0.88, 0.85), "angry": (1.08, 1.20), "fear": (1.05, 0.95),
         "disgust": (0.95, 1.00), "surprise": (1.08, 1.10)}
HF_TTS, HF_ASR = "VIZINTZOR/F5-TTS-THAI", "Qwen/Qwen3-ASR-0.6B"


@dataclass
class SpeechPlan:
    text: str
    emotion: str = "neutral"
    intensity: float = 0.0
    speed: float = 1.0
    gain: float = 1.0
    voice: str = "female"
    reason: list = field(default_factory=list)


def plan_speech(result, affect=None, voice="female", speak_uncertain=False, p_min=0.5):
    """Forward result (+ fused affect, modules/face_model.fuse_affect) → what to say and how.
    Text = the LLM sentence of the language layer when it wrote one (evidence-guarded), else the words in the order they were
    signed: accepted words, plus uncertain words with P(correct) ≥ p_min when speak_uncertain (the web's "speak tentative"
    setting) — never rejected segments, never a low-P guess. (The rule composer's reordering is not spoken: it can scramble a
    correct reading.) Speed follows the signing tempo (fast signing → slightly faster speech) and arousal."""
    affect = affect or {}
    fu = result.get("fusion") or {}
    text = (result.get("sentence") or "").strip() if fu.get("provider") not in ("rules", "none", None) else ""
    reason = []
    if not text:
        words = [s["nearest"] for s in result.get("segments", []) if s.get("status") == "accepted"
                 or (speak_uncertain and s.get("status") == "uncertain" and s.get("p_correct", 0.0) >= p_min)]
        text = " ".join(words)
        reason.append("คำตามลำดับที่ทำท่า (ยืนยันแล้ว" + (f" + อาจหมายถึงที่ P ≥ {p_min:.1f})" if speak_uncertain else ")"))
    emo = affect.get("emotion", "neutral") if affect.get("emotion") in EMOTIONS else "neutral"
    inten = float(np.clip(affect.get("intensity", 0.0), 0, 1))
    rate, gain = STYLE[emo]
    k = 0.4 + 0.6 * inten
    speed = 1 + (rate - 1) * k
    tempo = float(result.get("timings_ms", {}).get("tempo_factor", 1.0) or 1.0)
    if tempo > 1.2:
        speed *= 1.0 + min(0.1, 0.05 * (tempo - 1.0)); reason.append("ทำท่าเร็ว → พูดเร็วขึ้นเล็กน้อย")
    speed *= 1.0 + 0.05 * float(np.clip(affect.get("arousal", 0.0), -1, 1))
    if emo != "neutral":
        reason.append(f"น้ำเสียง{emo} ระดับ {inten:.0%} ตามสีหน้า/คำที่ทำ")
    return SpeechPlan(text=text, emotion=emo, intensity=inten, speed=float(np.clip(speed, 0.8, 1.2)), gain=1 + (gain - 1) * k, voice=voice,
                      reason=reason)


def thai_text_norm(text):
    """Text for the synthesiser: Arabic digits spelled out in Thai, letters of a fingerspelled name read by name, tidy spaces."""
    digits = "ศูนย์ หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า".split()
    text = re.sub(r"\d", lambda m: digits[int(m.group())], text)
    letters = {c: n for c, n in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "เอ บี ซี ดี อี เอฟ จี เอช ไอ เจ เค แอล เอ็ม เอ็น โอ พี คิว อาร์ เอส ที ยู วี ดับเบิลยู เอ็กซ์ วาย แซด".split())}
    text = re.sub(r"\b([A-Z])\b", lambda m: letters[m.group(1)], text)
    return re.sub(r"\s+", " ", text).strip()


class ExpressiveTTS:
    """F5-TTS-THAI + the emotion reference bank of models/speech-model-v1/ (refs.json: voice → emotion → {wav, text})."""

    def __init__(self, model_dir, device=None, nfe_step=32):
        """The Thai model's own inference package (f5-tts-th: its DiT variant with pe_attn_head, Thai number / ๆ normalisation,
        syllable-aware chunking) — the generic f5-tts loader reads this checkpoint with the wrong attention layout."""
        import torch
        from f5_tts_th.utils_infer import load_model, load_vocoder
        from huggingface_hub import hf_hub_download
        self.md = Path(model_dir)
        cfg = json.loads((self.md / "speech_config.json").read_text(encoding="utf-8"))
        self.cfg = cfg
        self.refs = json.loads((self.md / "refs.json").read_text(encoding="utf-8"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = hf_hub_download(HF_TTS, cfg["tts_ckpt"]); vocab = hf_hub_download(HF_TTS, "vocab.txt")
        self.model = load_model(cfg["tts_arch"], ckpt, mel_spec_type="vocos", vocab_file=vocab, device=self.device)
        self.vocoder = load_vocoder("vocos", device=self.device)
        self.nfe = nfe_step

    def reference(self, voice, emotion, intensity):
        v = self.refs.get(voice) or next(iter(self.refs.values()))
        e = emotion if (emotion in v and intensity >= self.cfg.get("min_intensity", 0.25)) else "neutral"
        return self.md / v[e]["wav"], v[e]["text"], e

    def speak(self, plan: SpeechPlan, seed=7):
        import torch
        from f5_tts_th.utils_infer import infer_process, preprocess_ref_audio_text
        if not plan.text:
            return np.zeros(0, np.float32), 24000
        ref, ref_text, used = self.reference(plan.voice, plan.emotion, plan.intensity)
        ra, rt = preprocess_ref_audio_text(str(ref), ref_text)
        torch.manual_seed(seed)
        wav, sr, _ = infer_process(ra, rt, thai_text_norm(plan.text), self.model, self.vocoder, mel_spec_type="vocos", nfe_step=self.nfe,
                                   speed=plan.speed * self.cfg.get("base_speed", 1.0), cfg_strength=2.0, set_max_chars=100, use_ipa=False,
                                   device=self.device)
        wav = np.asarray(wav, np.float32) * plan.gain
        peak = float(np.abs(wav).max()) if len(wav) else 1.0
        if peak > 0.98:
            wav *= 0.98 / peak
        plan.reason.append(f"เสียงต้นแบบ: {plan.voice}/{used}")
        return wav, sr


class ThaiASR:
    """Qwen3-ASR-0.6B (+ the LoRA adapter models/speech-model-v1/asr_lora when use_lora — opt-in: it helps short synthetic phrases
    but costs real read speech (FLEURS CER 6.7 → 8.4 %), so the base model is the default)."""

    def __init__(self, model_dir=None, device="cuda", use_lora=False):
        import torch
        from qwen_asr import Qwen3ASRModel
        self.m = Qwen3ASRModel.from_pretrained(HF_ASR, dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map=device,
                                               max_inference_batch_size=8, max_new_tokens=128)
        lora = Path(model_dir) / "asr_lora" if model_dir else None
        self.lora = bool(use_lora and lora and lora.exists())
        if self.lora:
            from peft import PeftModel
            self.m.model = PeftModel.from_pretrained(self.m.model, str(lora)).merge_and_unload()

    def __call__(self, audio, sr=16000):
        items = audio if isinstance(audio, list) else [audio]
        items = [a if isinstance(a, str) else (np.asarray(a, np.float32), sr) for a in items]
        out = [r.text for r in self.m.transcribe(items, language="Thai")]
        return out if isinstance(audio, list) else out[0]


def cer(ref, hyp):
    """Character error rate without spaces / punctuation (Thai has no word spaces)."""
    norm = lambda s: re.sub(r"[\s\.,!?\"'“”‘’…\-]", "", s or "")       # noqa: E731
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)
