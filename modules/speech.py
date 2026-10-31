"""Speech (v1 → v2: models/speech-model-v2/): the signer's sentence spoken in a natural voice that carries their emotion — and the
reverse ear (Thai speech recognition) for the hearing partner.

Voice (sign → speech). Not a fixed "audio display" voice: the forward result decides WHAT is said and HOW.
    plan = plan_speech(result, affect)        # units to say (words / hesitations), emotion, intensity, speed (signing tempo + arousal)
    wav, sr = load_voice(model_dir).speak(plan)   # F5-TTS-THAI on a GPU, the CPU voice (MMS-TTS) otherwise — same plan, same fillers
  * Synthesiser: F5-TTS-THAI (VIZINTZOR/F5-TTS-THAI, CC-BY-4.0; flow-matching, zero-shot voice + style cloning from a 3–8 s
    reference). The emotion is carried by the REFERENCE: <model>/refs/<voice>_<emotion>.wav — short Thai lines spoken in that
    emotion (made once with ElevenLabs v3 audio tags as the "teacher"), so the cloned prosody (pitch, energy, tempo, voice quality)
    follows the signer's affect. Low intensity → the neutral reference with a small speed / energy shift.
  * Only what the forward layer decided to say is spoken (accepted words / the evidence-guarded sentence); nothing is invented.
  * v2 natural disfluency: a sign the system could not read ([?] / unknown, or an uncertain word the user does not want spoken)
    is no longer skipped ("ฉัน ___ พ่อ" used to come out as "ฉัน พ่อ"). The speaker hesitates the way people do while searching for
    a word — a Thai filler ("อืม", "เอ่อ", "เอ่อ…", "แบบว่า…") and / or a pause as long as the unread sign took (capped) — so the
    listener hears that something was signed there, without a guessed word: a filler is never a content word and the [?] segment's
    nearest word is never said. Neighbouring unread signs make ONE hesitation; one at the end of the sentence is silence; a
    transition between signs is not a hesitation. Fillers are pre-recorded per voice (<model>/fillers/, fillers.json) and spliced
    between separately synthesised phrases (short cross-fades), so they keep a hesitant prosody instead of a read-aloud one.
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


FILLERS = {"um": "อืม", "uh": "เอ่อ", "uh_long": "เอ่อ…", "like": "แบบว่า…"}     # hesitation sounds — none of them is a content word


@dataclass
class SpeechPlan:
    text: str
    emotion: str = "neutral"
    intensity: float = 0.0
    speed: float = 1.0
    gain: float = 1.0
    voice: str = "female"
    reason: list = field(default_factory=list)
    units: list = field(default_factory=list)     # v2: [{"kind": "say", "text"} | {"kind": "hesitate", "filler", "pause_s"}] in signing order


def _hesitation(dur_s, n_signs, at_start, prev, rng, mode):
    """One hesitation for `n_signs` unread signs lasting `dur_s` → {kind, filler, pause_s}. The pause follows the signing time
    (0.25 s + 0.6 × the signs' duration, 0.35–1.4 s); a short gap stays a pause, a longer one gets a filler that is not the one just
    used; a sentence that starts unread opens with "เอ่อ"."""
    pause = float(np.clip(0.25 + 0.6 * dur_s, 0.35, 1.4))
    if mode == "pause":
        return dict(kind="hesitate", filler=None, pause_s=pause)
    if at_start:
        f = "uh"
    elif pause < 0.5:
        f = None
    elif pause >= 1.1 and n_signs >= 2:
        f = "like" if prev != "like" else "uh_long"
    elif pause >= 0.9:
        f = "uh_long" if prev != "uh_long" else "um"
    else:
        opts = [x for x in ("um", "uh") if x != prev]
        f = opts[rng.randint(0, len(opts))]
    return dict(kind="hesitate", filler=f, pause_s=pause * (0.5 if f else 1.0))


def plan_speech(result, affect=None, voice="female", speak_uncertain=False, p_min=0.5, disfluency="filler"):
    """Forward result (+ fused affect, modules/face_model.fuse_affect) → what to say and how.
    Text = the LLM sentence of the language layer when it wrote one (evidence-guarded), else the signs in the order they were signed:
    accepted words, plus uncertain words with P(correct) ≥ p_min when speak_uncertain (the web's "speak tentative" setting) — never
    rejected segments, never a low-P guess. (The rule composer's reordering is not spoken: it can scramble a correct reading.)
    disfluency: "filler" (v2 default — unread signs become a human hesitation, see the module docstring) · "pause" (silence only) ·
    "skip" (v1: unread signs are left out). Speed follows the signing tempo (fast signing → slightly faster speech) and arousal."""
    affect = affect or {}
    fu = result.get("fusion") or {}
    text = (result.get("sentence") or "").strip() if fu.get("provider") not in ("rules", "none", None) else ""
    reason, units = [], []
    if not text:
        segs = sorted([s for s in result.get("segments", []) if s.get("status") != "transition"], key=lambda s: s.get("f0", s.get("t0", 0)))

        def said(s):
            return s.get("status") == "accepted" or (speak_uncertain and s.get("status") == "uncertain" and s.get("p_correct", 0.0) >= p_min)
        words = [s["nearest"] for s in segs if said(s)]
        text = " ".join(words)
        reason.append("คำตามลำดับที่ทำท่า (ยืนยันแล้ว" + (f" + อาจหมายถึงที่ P ≥ {p_min:.1f})" if speak_uncertain else ")"))
        if disfluency in ("filler", "pause") and words:
            rng = np.random.RandomState(sum(ord(c) for c in text) % (2 ** 31))      # reproducible variation per sentence
            k, prev, n_gap = 0, None, 0
            while k < len(segs):
                j = k
                if said(segs[k]):
                    while j < len(segs) and said(segs[j]):
                        j += 1
                    units.append(dict(kind="say", text=" ".join(s["nearest"] for s in segs[k:j])))
                    k = j
                    continue
                while j < len(segs) and not said(segs[j]):
                    j += 1
                if j < len(segs):                        # an unread sign at the very end is silence (nothing follows to wait for)
                    dur = float(sum(max(0.0, (s.get("t1") or 0) - (s.get("t0") or 0)) for s in segs[k:j]))
                    u = _hesitation(dur, j - k, not units, prev, rng, disfluency)
                    units.append(u); prev = u["filler"] or prev; n_gap += j - k
                k = j
            if n_gap:
                reason.append(f"ท่าที่อ่านไม่ได้ {n_gap} ท่า → เว้นจังหวะ/คำลังเล (ไม่เดาคำ)")
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
                      reason=reason, units=units if any(u["kind"] == "hesitate" for u in units) else [])


def _fade(y, sr, ms=15):
    n = min(len(y) // 2, int(sr * ms / 1000))
    if n > 1:
        y = y.copy(); r = np.linspace(0, 1, n, dtype=np.float32)
        y[:n] *= r; y[-n:] *= r[::-1]
    return y


def inline_text(plan):
    """The plan as ONE text for the synthesiser (inline rendering): fillers written into the sentence where the unread signs were,
    a pause as an ellipsis — the voice keeps the sentence's prosody instead of reading isolated words."""
    parts = []
    for u in plan.units:
        if u["kind"] == "say":
            parts.append(u["text"])
        else:
            parts.append((FILLERS[u["filler"]].rstrip("…") + "…") if u.get("filler") else "…")
    return " ".join(parts)


def render(plan, say, fillers, sr, inline=False):
    """Plan units → one waveform. Spliced (default): each "say" phrase synthesised by say(text) → (wav, sr), hesitations = a filler clip
    of this voice (resampled to sr, at the plan's loudness) followed by silence, every piece faded in / out over 15 ms.
    inline=True: one synthesis of `inline_text(plan)`. Without units: say(text)."""
    if plan.units and inline:
        y, r = say(inline_text(plan))
        return np.asarray(y, np.float32), r
    if not plan.units:
        if not plan.text:
            return np.zeros(0, np.float32), sr
        y, r = say(plan.text)
        return np.asarray(y, np.float32), r
    out = []
    for u in plan.units:
        if u["kind"] == "say":
            y, r = say(u["text"])
            y = np.asarray(y, np.float32)
            if r != sr:
                y = np.interp(np.arange(0, len(y), r / sr), np.arange(len(y)), y).astype(np.float32)
            out.append(_fade(y, sr))
        else:
            f = fillers.get(plan.voice, u["filler"]) if u.get("filler") else None
            if f is not None:
                out.append(_fade(f * plan.gain, sr))
            out.append(np.zeros(int(sr * u["pause_s"]), np.float32))
    y = np.concatenate(out) if out else np.zeros(0, np.float32)
    peak = float(np.abs(y).max()) if len(y) else 1.0
    return (y * (0.98 / peak) if peak > 0.98 else y), sr


class Fillers:
    """<model>/fillers.json {voice: {filler: [wav, …]}} → filler clips at the voice's sample rate (one of the takes, rotating)."""

    def __init__(self, model_dir, sr):
        self.md, self.sr, self.n = Path(model_dir), sr, {}
        f = self.md / "fillers.json"
        self.table = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        self.cache = {}

    def get(self, voice, name):
        takes = (self.table.get(voice) or next(iter(self.table.values()), {})).get(name) or []
        if not takes:
            return None
        k = self.n.get((voice, name), 0); self.n[(voice, name)] = k + 1
        path = self.md / takes[k % len(takes)]
        if path not in self.cache:
            import soundfile as sf
            y, r = sf.read(str(path), dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            if r != self.sr:
                y = np.interp(np.arange(0, len(y), r / self.sr), np.arange(len(y)), y).astype(np.float32)
            self.cache[path] = y.astype(np.float32)
        return self.cache[path]


def thai_text_norm(text):
    """Text for the synthesiser: Arabic digits spelled out in Thai, letters of a fingerspelled name read by name, tidy spaces."""
    digits = "ศูนย์ หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า".split()
    text = re.sub(r"\d", lambda m: digits[int(m.group())], text)
    letters = {c: n for c, n in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "เอ บี ซี ดี อี เอฟ จี เอช ไอ เจ เค แอล เอ็ม เอ็น โอ พี คิว อาร์ เอส ที ยู วี ดับเบิลยู เอ็กซ์ วาย แซด".split())}
    text = re.sub(r"\b([A-Z])\b", lambda m: letters[m.group(1)], text)
    return re.sub(r"\s+", " ", text).strip()


def torch_dtype(device):
    """Weights / activations dtype for a device: bf16 on CUDA that supports it (fp16 on older NVIDIA), fp32 on Apple MPS and CPU."""
    import torch
    if str(device).startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


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
        from .utils import device as resolve_device
        self.device = resolve_device(device)            # cuda → mps → cpu (F5 is only interactive on a GPU)
        ckpt = hf_hub_download(HF_TTS, cfg["tts_ckpt"]); vocab = hf_hub_download(HF_TTS, "vocab.txt")
        self.model = load_model(cfg["tts_arch"], ckpt, mel_spec_type="vocos", vocab_file=vocab, device=self.device)
        self.vocoder = load_vocoder("vocos", device=self.device)
        self.nfe = nfe_step

    def reference(self, voice, emotion, intensity):
        v = self.refs.get(voice) or next(iter(self.refs.values()))
        e = emotion if (emotion in v and intensity >= self.cfg.get("min_intensity", 0.25)) else "neutral"
        return self.md / v[e]["wav"], v[e]["text"], e

    def say(self, text, plan, seed=7):
        import torch
        from f5_tts_th.utils_infer import infer_process, preprocess_ref_audio_text
        ref, ref_text, used = self.reference(plan.voice, plan.emotion, plan.intensity)
        ra, rt = preprocess_ref_audio_text(str(ref), ref_text)
        torch.manual_seed(seed)
        wav, sr, _ = infer_process(ra, rt, thai_text_norm(text), self.model, self.vocoder, mel_spec_type="vocos", nfe_step=self.nfe,
                                   speed=plan.speed * self.cfg.get("base_speed", 1.0), cfg_strength=2.0, set_max_chars=100, use_ipa=False,
                                   device=self.device)
        self._used = used
        return np.asarray(wav, np.float32) * plan.gain, sr

    def speak(self, plan: SpeechPlan, seed=7, inline=None):
        if not plan.text:
            return np.zeros(0, np.float32), 24000
        if not hasattr(self, "fillers"):
            self.fillers = Fillers(self.md, 24000)
        inline = self.cfg.get("disfluency", {}).get("render") == "inline" if inline is None else inline
        wav, sr = render(plan, lambda t: self.say(t, plan, seed), self.fillers, 24000, inline=inline)
        plan.reason.append(f"เสียงต้นแบบ: {plan.voice}/{getattr(self, '_used', 'neutral')}")
        return wav, sr


class CPUVoice:
    """The voice on a machine without a GPU (F5 needs one: CPU real-time factor ≫ 10): MMS-TTS Thai (modules/tts.py, VITS, offline)
    with the emotion's speaking rate / pitch, and the SAME plan and fillers as the GPU voice."""

    def __init__(self, model_dir, device="cpu", why="ไม่มี GPU สำหรับ F5"):
        from .tts import LocalMMSTTS
        self.md, self.why = Path(model_dir), why                  # why this voice and not F5 (shown in the plan's reasons)
        f = self.md / "speech_config.json"
        self.cfg = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        self.tts = LocalMMSTTS(device="cpu")
        self.fillers = Fillers(self.md, self.tts.sr)

    def speak(self, plan: SpeechPlan, seed=7, inline=None):
        from .tts import SpeechStyle, speakable
        if not plan.text:
            return np.zeros(0, np.float32), self.tts.sr
        style = SpeechStyle.from_affect({"surprise": "surprised"}.get(plan.emotion, plan.emotion), plan.intensity)
        style.rate *= plan.speed
        inline = self.cfg.get("disfluency", {}).get("cpu_render") == "inline" if inline is None else inline
        wav, sr = render(plan, lambda t: self.tts.synthesize(speakable(t.replace("…", " ")), style), self.fillers, self.tts.sr, inline=inline)
        plan.reason.append(f"เสียง CPU (MMS-TTS) — {self.why}")
        return wav * plan.gain, sr


def load_voice(model_dir, device=None, nfe_step=32):
    """The best voice this machine can run interactively: F5-TTS-THAI on CUDA / Apple MPS, else the CPU voice."""
    from .utils import device as resolve_device
    d = resolve_device(device)
    if d == "cpu":
        return CPUVoice(model_dir, why="เลือก CPU ไว้" if device == "cpu" else "ไม่มี GPU สำหรับ F5")
    try:
        return ExpressiveTTS(model_dir, device=d, nfe_step=nfe_step)
    except ImportError:                               # f5-tts-th not installed in this environment
        return CPUVoice(model_dir, why="env นี้ไม่มี F5 (ใช้ speech env / speech server)")


class ThaiASR:
    """Qwen3-ASR-0.6B (+ the LoRA adapter models/speech-model-v1/asr_lora when use_lora — opt-in: it helps short synthetic phrases
    but costs real read speech (FLEURS CER 6.7 → 8.4 %), so the base model is the default)."""

    def __init__(self, model_dir=None, device=None, use_lora=False):
        from qwen_asr import Qwen3ASRModel
        from .utils import device as resolve_device
        device = resolve_device(device)                 # cuda → mps → cpu
        self.m = Qwen3ASRModel.from_pretrained(HF_ASR, dtype=torch_dtype(device), device_map=device,
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
