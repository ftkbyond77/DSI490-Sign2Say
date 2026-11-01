"""Inference for a ThonburianTTS (F5-TTS) model dir, with voice profiles and "gap" support.

    from tts import Speech
    s = Speech("models/Speech-model-v3-TTS")               # model.safetensors, vocab.txt, voices.json, refs/*.wav
    wav, sr = s.say("ฉัน … แม่", voice="feminine_1")        # "…" / "___" / "[?]" -> natural pause or soft hum
    s.save("out.wav", wav, sr)

Voice = reference-audio prompt (voices.json: ref wav + its transcript + presentation 'masculine'/'feminine'), as in F5-TTS.
"""
import json, re
from pathlib import Path

import numpy as np, soundfile as sf, torch

ARCH = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)   # same as the baseline loader
GAP = "…"
_TH = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def normalize(text: str) -> str:
    """Same text policy as training: Thai digits -> Arabic, every 'unreadable' marker -> one '…', tidy spaces."""
    t = (text or "").translate(_TH)
    t = re.sub(r"\[\?\]|_{2,}|\.{2,}|…+|\*{2,}", GAP, t)
    t = re.sub(r"(?:\s*…\s*){2,}", f" {GAP} ", t)                  # neighbouring unread signs = ONE gap
    t = re.sub(r"\s*…\s*", f" {GAP} ", t)
    return re.sub(r"\s+", " ", t).strip()


class Speech:
    def __init__(self, model_dir, device=None, seed=0):
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT
        self.dir, self.seed = Path(model_dir), seed
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.loads((self.dir / "config.json").read_text(encoding="utf-8")) if (self.dir / "config.json").exists() else {}
        self.gap_s = cfg.get("gap_seconds", 0.8)
        self.min_gen_s = cfg.get("min_gen_seconds", 0.9)
        self.voices = json.loads((self.dir / "voices.json").read_text(encoding="utf-8"))
        self.vocoder = load_vocoder("vocos", device=self.dev)
        self.model = load_model(DiT, ARCH, str(self.dir / "model.safetensors"), mel_spec_type="vocos", vocab_file=str(self.dir / "vocab.txt"),
                                ode_method="euler", use_ema=True, device=self.dev)

    def say(self, text, voice=None, speed=1.0, gap_s=None, nfe=32, cfg=2.0):
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
        text = normalize(text)
        if not text:
            raise ValueError("empty text")
        if len(text) > 400:
            raise ValueError("text too long (max 400 characters)")
        v = self.voices.get(voice or next(iter(self.voices)))
        if v is None:
            raise KeyError(f"unknown voice {voice!r}; available: {list(self.voices)}")
        ref, ref_text = preprocess_ref_audio_text(str(self.dir / v["ref"]), v["ref_text"], show_info=lambda *_: None)
        info = sf.info(ref); ref_dur = info.frames / info.samplerate
        rate = ref_dur / max(len(ref_text.encode("utf-8")), 1)                  # seconds per utf-8 byte of the reference speech
        gs = self.gap_s if gap_s is None else gap_s
        out, sr = [], 24000
        for seg in self._segments(text, rate, speed, gs):
            n_gap = seg.count(GAP)
            gen = max(len(seg.replace(GAP, "").encode("utf-8")) * rate / speed + n_gap * gs, self.min_gen_s)   # one word needs >= ~0.9 s
            wav, sr, _ = infer_process(ref, ref_text, seg, self.model, self.vocoder, mel_spec_type="vocos", target_rms=0.1, nfe_step=nfe,
                                       cfg_strength=cfg, sway_sampling_coef=0.0, speed=speed, fix_duration=ref_dur + gen, device=self.dev,
                                       show_info=lambda *_: None, progress=None)
            out.append(np.asarray(wav, np.float32)); out.append(np.zeros(int(0.12 * sr), np.float32))
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
