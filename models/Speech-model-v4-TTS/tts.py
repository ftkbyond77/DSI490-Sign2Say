"""Inference for a ThonburianTTS (F5-TTS) model dir: voice profiles, gap marker, 7 FER2013 emotions with intensity, low latency.

    from tts import Speech
    s = Speech("models/Speech-model-v4-TTS")                       # model.safetensors, vocab.txt, voices.json, refs/*.wav, config.json
    wav, sr = s.say("ฉันชอบแมว", voice="male")                       # no expression -> neutral
    wav, sr = s.say("ฉันชอบแมว", emotion="happy", intensity=0.5)     # intensity 0..1 (0 == neutral), up to 1.5 allowed
    wav, sr = s.say([("ดีใจจัง", "happy"), ("แล้วไงต่อ", "surprise")]) # pieces -> one waveform
    wav, sr = s.say("ฉัน … แม่")                                     # "…" / "___" / "[?]" -> natural pause or soft hum
    s.save("out.wav", wav, sr)

Voice = reference-audio prompt (voices.json). Emotion = one tag symbol in front of the text (config.json "emotion_tags"; an unused vocab symbol, so
the checkpoint stays loadable); intensity interpolates the tag's embedding between the neutral tag and the emotion tag.
"""
import json, re, time
from pathlib import Path

import numpy as np, soundfile as sf, torch

ARCH = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)   # same as the baseline loader
GAP = "…"
EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
ALIASES = {"curious": ("surprise", 0.4), "fearful": ("fear", 1.0), "disgusted": ("disgust", 1.0), "surprised": ("surprise", 1.0), "joy": ("happy", 1.0)}
# emotion -> (shift of mean F0 in semitones, F0-range scale, speaking-rate factor, gain dB) at strength 1.0: the same descriptive templates (after Banse & Scherer 1996)
# the model's tags were trained with; the strength s scales every deviation linearly
TEMPLATE = {"happy": (1.5, 1.30, 1.06, 2.0), "sad": (-1.5, 0.60, 0.85, -3.0), "angry": (1.0, 1.35, 1.08, 4.0),
            "fear": (3.0, 1.10, 1.12, -1.0), "disgust": (-1.0, 0.90, 0.90, -1.0), "surprise": (3.5, 1.50, 1.00, 2.0)}
_TH = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def normalize(text: str) -> str:
    """Same text policy as training: Thai digits -> Arabic, every 'unreadable' marker -> one '…', tidy spaces."""
    t = (text or "").translate(_TH)
    t = re.sub(r"\[\?\]|_{2,}|\.{2,}|…+|\*{2,}", GAP, t)
    t = re.sub(r"(?:\s*…\s*){2,}", f" {GAP} ", t)                  # neighbouring unread signs = ONE gap
    t = re.sub(r"\s*…\s*", f" {GAP} ", t)
    return re.sub(r"\s+", " ", t).strip()


def emotionize(y, emo, s, sr=24000):
    """Praat PSOLA: move the clip's own F0 mean/range, rate and level toward `emo` by strength s (0..1). Returns y unchanged when Praat is missing or the clip has too few pitch points."""
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return y
    st, rg, rate, db = TEMPLATE[emo]
    snd = parselmouth.Sound(np.asarray(y, np.float64), sr)
    man = call(snd, "To Manipulation", 0.01, 70, 450)
    pt = call(man, "Extract pitch tier"); n = call(pt, "Get number of points")
    if n < 3:
        return y
    t = np.array([call(pt, "Get time from index", i) for i in range(1, n + 1)]); f = np.array([call(pt, "Get value at index", i) for i in range(1, n + 1)])
    med = np.median(f); f2 = med * 2 ** (s * st / 12) * (f / med) ** (1 + s * (rg - 1))
    new = call("Create PitchTier", "p", 0, snd.xmax)
    for a_, b_ in zip(t, f2):
        call(new, "Add point", float(a_), float(b_))
    call([new, man], "Replace pitch tier")
    dur = call("Create DurationTier", "d", 0, snd.xmax); r = 1 / (1 + s * (rate - 1))
    call(dur, "Add point", 0.0, r); call(dur, "Add point", snd.xmax, r)
    call([dur, man], "Replace duration tier")
    out = call(man, "Get resynthesis (overlap-add)").values[0].astype(np.float32) * 10 ** (s * db / 20)
    return np.clip(out, -0.99, 0.99)


class Speech:
    def __init__(self, model_dir, device=None, warm=True):
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT
        self.dir = Path(model_dir)
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.loads((self.dir / "config.json").read_text(encoding="utf-8")) if (self.dir / "config.json").exists() else {}
        self.cfg = cfg
        self.gap_s = cfg.get("gap_seconds", 0.8)
        self.min_gen_s = cfg.get("min_gen_seconds", 0.9)
        self.nfe, self.cfg_strength, self.sway = cfg.get("default_nfe", 32), cfg.get("default_cfg", 2.0), cfg.get("default_sway", 0.0)
        self.lead_s = cfg.get("lead_seconds", 0.0)                   # extra duration allowance in front of the generated text
        self.ref_tail_s = cfg.get("ref_tail_s", 0.3)                # silence after the reference (F5 itself leaves 50 ms; a reference cut mid-sound loses the first word)
        self.joint_vocoder = cfg.get("joint_vocoder", False)         # decode ref+generated mel together, then cut (avoids the vocoder edge at the cut)
        self.tags = cfg.get("emotion_tags", {})                      # {"happy": "∞", ...}; empty -> checkpoint without emotion control
        self.dsp = cfg.get("emotion_dsp", False)                     # prosody layer after the model: the model's own tag moves pitch only slightly (measured), PSOLA makes the emotion audible
        self.dsp_strength = cfg.get("emotion_dsp_strength", 1.0)
        self.emo_rate = cfg.get("emotion_rate", {})                  # speaking-rate factor at intensity 1 (the same templates the model was trained with): sad slower, happy faster
        self.voices = json.loads((self.dir / "voices.json").read_text(encoding="utf-8"))
        self.vocoder = load_vocoder("vocos", device=self.dev)
        self.model = load_model(DiT, cfg.get("arch", ARCH), str(self.dir / "model.safetensors"), mel_spec_type="vocos", vocab_file=str(self.dir / "vocab.txt"),
                                ode_method="euler", use_ema=True, device=self.dev)
        self.vmap = self.model.vocab_char_map
        self._emb = self.model.transformer.text_embed.text_embed.weight
        self._refs, self.timing = {}, {}
        if warm:
            self.say("ทดสอบ", nfe=4)

    # ------------------------------------------------------------------ voices
    def _voice(self, voice):
        voice = voice or next(iter(self.voices))
        if voice not in self.voices:
            raise KeyError(f"unknown voice {voice!r}; available: {list(self.voices)}")
        if voice not in self._refs:
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
            import torchaudio
            v = self.voices[voice]
            path, text = preprocess_ref_audio_text(str(self.dir / v["ref"]), v["ref_text"], show_info=lambda *_: None)
            a, sr = torchaudio.load(path)
            a = a.mean(0, keepdim=True)
            rms = a.pow(2).mean().sqrt()
            if rms < 0.1:
                a = a * 0.1 / rms
            if sr != 24000:
                a = torchaudio.functional.resample(a, sr, 24000)
            if len(text[-1].encode("utf-8")) == 1:
                text += " "
            dur = a.shape[-1] / 24000
            rate = dur / max(len(text.encode("utf-8")), 1)                       # speaking rate from the speech itself (before padding)
            extra = int(max(self.ref_tail_s - 0.05, 0) * 24000)
            if extra:
                a = torch.cat([a, torch.zeros(1, extra)], 1)
            self._refs[voice] = dict(audio=a.to(self.dev), text=text, dur=dur, rms=float(rms), rate=rate)
        return self._refs[voice]

    # ------------------------------------------------------------------ emotion
    def _tag(self, emotion, intensity):
        """-> (tag symbol or '', alpha, rate multiplier, emotion name or None). Aliases map to a base emotion at a lower intensity."""
        if emotion in ALIASES:
            emotion, k = ALIASES[emotion]; intensity = intensity * k
        if emotion is None or emotion == "neutral" or not self.tags:
            return "", None, 1.0, None
        if emotion not in self.tags:
            raise KeyError(f"unknown emotion {emotion!r}; use one of {EMOTIONS}")
        return self.tags[emotion], intensity, 1.0 + min(intensity, 1.0) * (self.emo_rate.get(emotion, 1.0) - 1.0), emotion

    def _set_intensity(self, tag, alpha):
        i = self.vmap[tag] + 1                                       # F5 shifts token ids by one (0 = padding)
        j = self.vmap[self.tags["neutral"]] + 1 if "neutral" in self.tags else None
        self._saved = (i, self._emb.data[i].clone())
        if alpha != 1.0 and j is not None:
            self._emb.data[i] = self._emb.data[j] + alpha * (self._saved[1] - self._emb.data[j])

    def _restore(self):
        if getattr(self, "_saved", None):
            self._emb.data[self._saved[0]] = self._saved[1]; self._saved = None

    # ------------------------------------------------------------------ generation
    def _gen(self, v, text, tag, alpha, speed, gs, nfe, cfg, sway, seed, lead):
        from f5_tts.model.utils import convert_char_to_pinyin
        t0 = time.perf_counter()
        n_gap = text.count(GAP)
        gen_s = max(len(text.replace(GAP, "").encode("utf-8")) * v["rate"] / speed + n_gap * gs, self.min_gen_s) + lead
        full = convert_char_to_pinyin([v["text"] + tag + text])
        ref_len = v["audio"].shape[-1] // 256
        dur = ref_len + int(gen_s * 24000 / 256)
        if seed is not None:
            torch.manual_seed(seed)
        t1 = time.perf_counter()
        if tag:
            self._set_intensity(tag, alpha)
        try:
            with torch.inference_mode():
                mel, _ = self.model.sample(cond=v["audio"], text=full, duration=dur, steps=nfe, cfg_strength=cfg, sway_sampling_coef=sway)
        finally:
            if tag:
                self._restore()
        if self.dev == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        with torch.inference_mode():
            mel = mel.to(torch.float32).permute(0, 2, 1)
            wav = self.vocoder.decode(mel if self.joint_vocoder else mel[:, :, ref_len:])
            if self.joint_vocoder:
                wav = wav[:, ref_len * 256:]
            wav = wav * (v["rms"] / 0.1 if v["rms"] < 0.1 else 1.0)
            wav = wav.squeeze().float().cpu().numpy()
        t3 = time.perf_counter()
        for k, x in (("prep", t1 - t0), ("ode", t2 - t1), ("vocoder", t3 - t2)):
            self.timing[k] = self.timing.get(k, 0.0) + x
        return wav

    def say(self, text, voice=None, emotion=None, intensity=0.5, speed=1.0, gap_s=None, nfe=None, cfg=None, sway=None, seed=None, lead=None, prosody=None):
        pieces = [(text, emotion, intensity)] if isinstance(text, str) else [(p[0], p[1] if len(p) > 1 else emotion, p[2] if len(p) > 2 else intensity) for p in text]
        v = self._voice(voice)
        nfe, cfg = nfe or self.nfe, self.cfg_strength if cfg is None else cfg
        sway = self.sway if sway is None else sway
        lead = self.lead_s if lead is None else lead
        gs = self.gap_s if gap_s is None else gap_s
        self.timing = {}
        out, sr = [], 24000
        for t, emo, inten in pieces:
            t = normalize(t)
            if not t:
                raise ValueError("empty text")
            if len(t) > 400:
                raise ValueError("text too long (max 400 characters)")
            tag, alpha, rm, ename = self._tag(emo, inten)
            use_dsp = ename in TEMPLATE and (self.dsp if prosody is None else prosody)
            if use_dsp:
                rm = 1.0                                                  # the PSOLA layer applies the tempo itself
            for seg in self._segments(t, v["rate"], speed * rm, gs):
                w = self._gen(v, seg, tag, alpha, speed * rm, gs, nfe, cfg, sway, seed, lead)
                if use_dsp:
                    w = emotionize(w, ename, min(alpha, 1.0) * self.dsp_strength, sr)
                out.append(w); out.append(np.zeros(int(0.12 * sr), np.float32))
        return np.concatenate(out[:-1]), sr

    @staticmethod
    def _segments(text, rate, speed, gs, max_s=13.0):
        parts, cur = re.split(r"(?<=[\.\?!])\s+|\n", text), ""
        segs = []
        for p in parts:
            cand = (cur + " " + p).strip()
            est = len(cand.replace(GAP, "").encode("utf-8")) * rate / speed + cand.count(GAP) * gs
            if cur and est > max_s:
                segs.append(cur); cur = p
            else:
                cur = cand
        return segs + [cur] if cur else segs

    @staticmethod
    def save(path, wav, sr):
        sf.write(path, wav, sr, subtype="PCM_16")
