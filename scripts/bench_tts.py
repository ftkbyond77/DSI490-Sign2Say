"""v8 speech-output choice: Thai TTS voices compared on what can be measured without a listening panel.

  python main.py bench-tts [--asr biodatlab/whisper-th-medium-combined] [--out cache/tts_bench]

Per voice, on daily-conversation sentences of the kind the sentence layer produces:
  intelligibility  a Thai speech recogniser (open-source Whisper fine-tuned on Thai) transcribes the synthetic speech; character
                   error rate against the input text (lower = clearer pronunciation, fewer dropped / garbled syllables)
  speed            real-time factor on this CPU (synthesis time ÷ audio length; < 1 = faster than real time), first-call excluded
  prosody          pitch range (st. dev. of voiced F0 in semitones): the v7 voice sounded flat / robotic to the user
WAV files are written to --out so a person can listen. Nothing here touches the sign model.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from modules.utils import get_logger, write_json

log = get_logger("bench_tts")
SENTENCES = ["สวัสดีครับ", "ฉันชื่อแจ็ค", "วันนี้ฉันจะมานำเสนองาน", "สวัสดีทุกคน วันนี้ผมจะนำเสนองาน", "วันนี้ฉันรู้สึกอยากกินไก่ทอด",
             "สวัสดีอาจารย์ วันนี้สบายดีไหม", "พ่อดื่มน้ำ", "ฉันรักเพื่อน", "ฉันปลอบเพื่อนที่ร้องไห้", "ไปกินข้าวด้วยกันไหม",
             "ขอบคุณมากครับ", "ขอโทษนะคะ", "คุณชื่ออะไร", "บ้านของคุณอยู่ที่ไหน", "พรุ่งนี้เราจะไปโรงเรียน", "แม่ทำอาหารอร่อยมาก",
             "ฉันไม่สบาย ต้องไปหาหมอ", "น้องชอบแมว", "เมื่อวานฝนตกหนัก", "ยินดีที่ได้รู้จัก"]
VOICES = [("local", "facebook/mms-tts-tha"), ("local", "VIZINTZOR/MMS-TTS-THAI-FEMALEV2"), ("vachana", "th_f_1"), ("vachana", "th_m_1"),
          ("vachana", "th_f_2"), ("vachana", "th_m_2")]


def make(provider, voice):
    from modules.tts import LocalMMSTTS, VachanaTTS
    return LocalMMSTTS(model=voice) if provider == "local" else VachanaTTS(voice=voice)


def cer(ref, hyp):
    import re
    r = re.sub(r"[\s.,!?ๆ]", "", ref); h = re.sub(r"[\s.,!?ๆ]", "", hyp)
    d = np.arange(len(h) + 1)
    for i, c in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, x in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (c != x))
    return d[len(h)] / max(len(r), 1)


def pitch_range(wav, sr):
    """St. dev. of F0 (semitones) over voiced frames — torchaudio's pitch detector (no extra dependency)."""
    import torch
    import torchaudio.functional as AF
    f0 = AF.detect_pitch_frequency(torch.from_numpy(wav.astype(np.float32))[None], sr, frame_time=0.02, freq_low=70, freq_high=450)[0].numpy()
    e = np.array([np.sqrt(np.mean(wav[i:i + int(0.02 * sr)] ** 2)) for i in range(0, len(wav), int(0.02 * sr))])[:len(f0)]
    f0 = f0[:len(e)][e > 0.3 * e.max()]                      # voiced (loud) frames only
    f0 = f0[(f0 > 70) & (f0 < 450)]
    return float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 10 else float("nan")


def resample16(wav, sr):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(sr), 16000)
    return resample_poly(wav.astype(np.float32), 16000 // g, int(sr) // g).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="biodatlab/whisper-th-medium-combined")
    ap.add_argument("--out", default=str(ROOT / "cache" / "tts_bench"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from modules.tts import write_wav
    audio = {}
    rows = []
    for prov, voice in VOICES:
        try:
            tts = make(prov, voice)
        except Exception as e:  # noqa: BLE001
            log.warning("%s/%s unavailable: %s", prov, voice, e); continue
        tts.synthesize("สวัสดี")                                      # first call (model warm-up) not timed
        t_syn = t_aud = 0.0
        for k, s in enumerate(SENTENCES):
            t = time.perf_counter(); wav, sr = tts.synthesize(s + " ")   # (space: bypass the provider's cache of the warm-up)
            t_syn += time.perf_counter() - t; t_aud += len(wav) / sr
            name = f"{prov}_{voice.split('/')[-1]}"
            write_wav(out / f"{name}_{k:02d}.wav", wav, sr)
            audio[(name, k)] = (wav, sr)
        rows.append(dict(voice=name, rtf=round(t_syn / t_aud, 3), sec_per_sentence=round(t_syn / len(SENTENCES), 3)))
        log.info("%s rtf %.3f", name, t_syn / t_aud)
        del tts
    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=a.asr, device=dev, torch_dtype=torch.float16 if dev == 0 else torch.float32)
    for r in rows:
        errs, prs, hyps = [], [], []
        for k, s in enumerate(SENTENCES):
            wav, sr = audio[(r["voice"], k)]
            w16 = resample16(wav, sr)
            hyp = asr({"raw": w16, "sampling_rate": 16000}, generate_kwargs=dict(language="th", task="transcribe"))["text"]
            errs.append(cer(s, hyp)); hyps.append(hyp); prs.append(pitch_range(wav, sr))
        r.update(cer=round(float(np.mean(errs)), 4), cer_max=round(float(np.max(errs)), 3), pitch_sd_semitones=round(float(np.nanmean(prs)), 2),
                 examples=[dict(text=s, heard=h) for s, h in list(zip(SENTENCES, hyps))[:6]])
        log.info("%s: CER %.3f · pitch sd %.2f st · RTF %.3f", r["voice"], r["cer"], r["pitch_sd_semitones"], r["rtf"])
    write_json(ROOT / "artifacts" / "reports" / "tts_bench.json", dict(asr=a.asr, sentences=SENTENCES, voices=rows))


if __name__ == "__main__":
    main()
