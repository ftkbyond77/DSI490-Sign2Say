"""Speech with the per-emotion method of policy.json on top of Speech-model-v4-TTS (weights unchanged). NOT run yet - written 2026-11-14.

    from speech_mix import MixSpeech
    s = MixSpeech("../../models/Speech-model-v4-TTS")            # device: cuda if available, else cpu (do not use mps: audible artefacts)
    wav, sr = s.say("ฉันได้รับข้อความแล้ว", voice="female", emotion="angry", intensity=0.6)
    s.save("x.wav", wav, sr)
N = neutral reference, no emotion; A = neutral reference + tag + PSOLA (with headroom, no hard clip); B = the speaker's emotional reference;
C = B + a mild PSOLA layer. Unknown emotion or missing reference -> N.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PEAK = 10 ** (-1 / 20)


class MixSpeech:
    def __init__(self, model_dir, refs=HERE / "refs.csv", policy=HERE / "policy.json", device=None, speech=None):
        sys.path.insert(0, str(Path(model_dir).resolve()))
        import tts as _tts
        self._emo = _tts.emotionize
        _tts.emotionize = self._headroom                                  # v4 hard-clips after its PSOLA gain
        self.pol = json.loads(Path(policy).read_text(encoding="utf-8"))
        self.s = speech if speech is not None else _tts.Speech(model_dir, device=device, warm=False)   # speech= reuses a loaded model (the tts service)
        self.have = set()
        for r in csv.DictReader(open(refs, encoding="utf-8")):
            p = Path(r["path"]); p = p if p.is_absolute() else HERE / p
            self.s.voices[f"{r['speaker']}__{r['emotion']}"] = {"ref": str(p), "ref_text": r["ref_text"]}
            self.have.add((r["speaker"], r["emotion"]))

    def _headroom(self, y, emo, s, sr=24000):
        return self._emo(np.asarray(y) * 0.25, emo, s, sr) * 4.0

    def _method(self, spk, emo):
        m = (self.pol.get("overrides", {}).get(spk, {})).get(emo) or self.pol["methods"][emo]
        if m in ("B", "C") and (spk, emo) not in self.have:
            m = "A" if emo in getattr(self.s, "tags", {}) else "N"
        return "N" if emo == "neutral" else m

    def _make(self, m, text, spk, emo, intensity, kw):
        if m == "N":
            return self.s.say(text, voice=f"{spk}__neutral", **kw)
        if m == "A":
            return self.s.say(text, voice=f"{spk}__neutral", emotion=emo, intensity=intensity, **kw)
        wav, sr = self.s.say(text, voice=f"{spk}__{emo}", **kw)
        if m == "C":
            wav = self._headroom(wav, emo, self.pol.get("psola_c", 0.6) * min(max(intensity / 0.5, 0.5), 1.5), sr)
        return wav, sr

    def problem(self, wav, sr, text, spk):
        """Cheap checks for failures that can be measured (NOT a detector of every 'crackle' a listener hears)."""
        g = self.pol.get("guard", {})
        if not len(wav) or not np.all(np.isfinite(wav)): return "non-finite / empty"
        if float(np.mean(np.abs(wav) >= 0.985)) > g.get("clip_frac", 0.0005): return "clipped samples"
        if float(np.sqrt(np.mean(wav ** 2))) < 0.003: return "near-silent"
        rate = self.s._voice(f"{spk}__neutral")["rate"]                          # seconds per text byte of this voice
        lo, hi = g.get("dur_ratio", [0.5, 2.2]); r = (len(wav) / sr) / max(len(text.encode("utf-8")) * rate, 0.3)
        if not lo <= r <= hi: return f"duration {r:.1f}x the expected"
        return None

    def say(self, text, voice="female", emotion=None, intensity=0.5, seed=0, **kw):
        """-> (wav, sr). self.last = dict(method, tries, problem) says what was used and whether the guard stepped in."""
        if str(self.s.dev) == "mps":
            raise RuntimeError("mps gives audible artefacts with this model (measured 2026-11-12): use cpu or cuda")
        spk = self.pol["voices"].get(voice, voice)
        emo = emotion if emotion in self.pol["methods"] else "neutral"
        m = self._method(spk, emo); ladder = [m] + [x for x in ("A", "N") if x != m and (x == "N" or emo in getattr(self.s, "tags", {}))]
        tries, why = 0, None
        for mm in ladder:
            for k in range(1 + (self.pol.get("guard", {}).get("max_retries", 2) if mm == ladder[0] else 0)):
                wav, sr = self._make(mm, text, spk, emo, intensity, dict(kw, seed=seed + k)); tries += 1
                peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
                wav = wav * (PEAK / peak) if peak > PEAK else wav
                why = self.problem(wav, sr, text, spk)
                if not why:
                    self.last = dict(method=mm, tries=tries, problem=None); return wav, sr
        self.last = dict(method=ladder[-1], tries=tries, problem=why)
        return wav, sr

    def save(self, path, wav, sr):
        self.s.save(path, wav, sr)
