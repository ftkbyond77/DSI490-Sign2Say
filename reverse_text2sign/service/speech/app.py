"""reverse-speech — the heavy language models, kept out of the lean motion API (one container, int8 CTranslate2):

  POST /asr          audio (webm / ogg / wav / mp3, ≤ 60 s) + lang (th | en) → {text, lang, ms, reliable, flags, char_probs, …}
  POST /asr/stream   incremental recognition while the person is still speaking (v2): the browser posts the audio recorded so
                     far every ~1 s with a session id; pauses (voice activity) close a chunk that is recognised once and kept,
                     the open tail is recognised quickly for display → {committed, tentative}; with final=1 → the final text
  POST /mt           {text} English → Thai: facebook/nllb-200-distilled-600M (beam 2)
  GET  /health       device, models, load time

Models (result_v2.md §7): Thai = biodatlab Whisper fine-tuned on Thai — `medium` on a GPU (real-speech CER 3.8 % vs 6.1 % for
`small`, 0.7 s per short utterance on an RTX 2050), `small` on a CPU (medium is ~3.5× slower there); English = faster-whisper
base.en. SPEECH_DEVICE=auto|cpu|cuda, SPEECH_TH_MODEL=auto|small|medium override the choice.

Reliability (prompt v2 §E): Whisper fine-tuned on Thai video subtitles invents a greeting ("สวัสดีครับ", "ขอบคุณที่ติดตาม")
on very short or unclear audio. A transcript is flagged unreliable — the UI then asks to repeat or type instead of signing it —
when it is physically impossible speech (> 25 characters per second of speech), repeats itself inside a short utterance, is a
known subtitle phrase on < 1.2 s of speech, or is empty. The flag never changes the text; the decision stays with the person.
"""
from __future__ import annotations

import io
import os
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

MODELS = Path(os.environ.get("SPEECH_MODELS", "/models"))
THREADS = int(os.environ.get("SPEECH_THREADS", "4"))
DEVICE = os.environ.get("SPEECH_DEVICE", "auto")
TH_MODEL = os.environ.get("SPEECH_TH_MODEL", "auto")
FULL_FINAL_S = float(os.environ.get("SPEECH_FULL_FINAL_S", "12"))   # final pass re-decodes the whole audio up to this length
SR = 16000
app = FastAPI(title="reverse-speech", version="2.0")
_lock = threading.Lock()
M: dict = {}
SESSIONS: dict = {}
SUBTITLE_PHRASES = ("สวัสดีครับ", "สวัสดีค่ะ", "ขอบคุณที่ติดตาม", "ขอบคุณที่รับชม", "กดติดตาม", "ซับไตเติ้ล", "อืม", "อิอิ")


def _device():
    if DEVICE != "auto":
        return DEVICE
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


@app.on_event("startup")
def _load():
    import ctranslate2
    import sentencepiece as spm
    from faster_whisper import WhisperModel
    from faster_whisper.vad import get_vad_model
    t0 = time.time()
    get_vad_model()
    dev = _device()
    th = TH_MODEL if TH_MODEL != "auto" else ("medium" if dev == "cuda" else "small")
    if not (MODELS / f"whisper-th-{th}").exists():
        th = "small"
    ct = "int8_float16" if dev == "cuda" else "int8"
    M["asr_th"] = WhisperModel(str(MODELS / f"whisper-th-{th}"), device=dev, compute_type=ct, cpu_threads=THREADS)
    M["asr_en"] = WhisperModel(str(MODELS / "whisper-base-en"), device=dev, compute_type=ct, cpu_threads=THREADS)
    M["mt"] = ctranslate2.Translator(str(MODELS / "nllb-600m"), device="cpu", compute_type="int8", intra_threads=THREADS)
    M["sp"] = spm.SentencePieceProcessor(model_file=str(MODELS / "nllb-600m" / "sentencepiece.bpe.model"))
    M.update(device=dev, th_model=f"whisper-th-{th}", compute_type=ct, load_s=round(time.time() - t0, 2))


@app.get("/health")
def health():
    return dict(ok="mt" in M, load_s=M.get("load_s"), device=M.get("device"), th_model=M.get("th_model"),
                sessions=len(SESSIONS))


def _decode(data):
    from faster_whisper.audio import decode_audio
    try:
        return decode_audio(io.BytesIO(data), sampling_rate=SR)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(415, f"cannot decode audio: {type(e).__name__}") from e


def _transcribe(wav, lang, beam=3, vad=True, words=False, retry=True):
    """→ (text, char_probs, segments). Thai/English model; very short replies that the VAD filters out entirely are retried
    without it (v1 fix: ใช่ / ได้ / ฝนตก). Latency guard (v2): a hallucinating decode ("สวัสดีครับ สวัสดีครับ …" through every
    temperature fallback) took 30–45 s on a CPU and blocked the queue — tokens are capped by the audio length (48 + 30 per
    second: twice what fast Thai speech needs, measured; a tighter cap truncated sentences). words=True adds per-character
    probabilities (not used by default: the probability gate of the corrector showed no benefit, result_v2.md §7.4)."""
    model = M["asr_en"] if lang == "en" else M["asr_th"]
    cap = int(48 + 30 * len(wav) / SR)
    kw = dict(language="en" if lang == "en" else "th", beam_size=beam, condition_on_previous_text=False, without_timestamps=True,
              word_timestamps=words, max_new_tokens=min(cap, 440))
    with _lock:                                   # one decode at a time per container (scale with --scale)
        segs = list(model.transcribe(wav, vad_filter=vad, **kw)[0])
        if vad and retry and not "".join(s.text for s in segs).strip():
            segs = list(model.transcribe(wav, vad_filter=False, **kw)[0])
    text, probs = "", []
    for s in segs:
        if text:
            text += " "; probs.append(1.0)
        if words and s.words:
            t = "".join(w.word for w in s.words)
            p = [round(float(w.probability), 3) for w in s.words for _ in w.word]
            lead, trail = len(t) - len(t.lstrip()), len(t) - len(t.rstrip())
            probs += p[lead:len(p) - trail]
            text += t.strip()
        else:
            text += s.text.strip()
            probs += [1.0] * len(s.text.strip())
    return text, (probs if len(probs) == len(text) else None), segs


def _speech_seconds(wav):
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(wav, VadOptions(min_silence_duration_ms=300, speech_pad_ms=100))
    return sum(t["end"] - t["start"] for t in ts) / SR if ts else len(wav) / SR


def reliability(text, wav, segs):
    """flags of an impossible / invented transcript (module docstring). → (reliable, [reasons], speech seconds)."""
    t = re.sub(r"\s+", "", text)
    sec = max(_speech_seconds(wav), 0.1)
    reasons = []
    if not t:
        reasons.append("no speech recognised")
    if len(t) / sec > 25:
        reasons.append(f"{len(t) / sec:.0f} characters per second of speech — faster than anyone speaks")
    words = text.split()
    if len(words) >= 2 and len(set(words)) < len(words) and sec < 3:
        reasons.append("the same phrase repeated within a short utterance")
    if sec < 1.2 and any(p in t for p in SUBTITLE_PHRASES):
        reasons.append("a stock subtitle phrase on very short audio (a known recogniser hallucination)")
    lp = [s.avg_logprob for s in segs]
    if lp and min(lp) < -1.2:
        reasons.append(f"low recogniser confidence (log-prob {min(lp):.2f})")
    return not reasons, reasons, round(sec, 2)


@app.post("/asr")
async def asr(audio: UploadFile = File(...), lang: str = Form("th")):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio")
    t0 = time.time()
    wav = _decode(data)
    if len(wav) > SR * 62:
        raise HTTPException(413, "audio longer than 60 s")
    text, probs, segs = _transcribe(wav, lang)
    ok, why, sec = reliability(text, wav, segs)
    return dict(text=text, lang=lang, ms=round((time.time() - t0) * 1000), seconds=round(len(wav) / SR, 2), speech_seconds=sec,
                reliable=ok, flags=why, char_probs=probs, model=M["th_model"] if lang != "en" else "whisper-base.en",
                device=M["device"])


@app.post("/asr/stream")
async def asr_stream(audio: UploadFile = File(...), lang: str = Form("th"), session: str = Form(""), final: int = Form(0)):
    """incremental recognition (module docstring). A chunk is closed by ≥ 0.5 s of silence after it and recognised once; the
    open tail is decoded greedily for display. final=1: the whole audio is re-decoded when it is short (best accuracy — chunking
    at pauses was measured slightly WORSE on real speech, result_v2.md §7.2), else committed chunks + the tail."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    data = await audio.read()
    t0 = time.time()
    sid = session or uuid.uuid4().hex
    st = SESSIONS.setdefault(sid, dict(chunks=[], t=time.time()))
    st["t"] = time.time()
    for k in [k for k, v in SESSIONS.items() if time.time() - v["t"] > 120]:   # forget abandoned sessions
        SESSIONS.pop(k, None)
    if not data:
        return dict(session=sid, committed="", tentative="", final=bool(final))
    wav = _decode(data)
    if len(wav) > SR * 62:
        raise HTTPException(413, "audio longer than 60 s")
    dur = len(wav) / SR
    if final and dur <= FULL_FINAL_S:
        text, probs, segs = _transcribe(wav, lang)
        ok, why, sec = reliability(text, wav, segs)
        SESSIONS.pop(sid, None)
        return dict(session=sid, final=True, text=text, char_probs=probs, reliable=ok, flags=why, speech_seconds=sec,
                    seconds=round(dur, 2), ms=round((time.time() - t0) * 1000), mode="full", model=M["th_model"], device=M["device"])
    ts = get_speech_timestamps(wav, VadOptions(min_silence_duration_ms=500, speech_pad_ms=150))
    done_until = st["chunks"][-1][1] if st["chunks"] else 0
    for t in ts:                                          # close chunks followed by ≥ 0.5 s of silence (or everything at the end)
        a, b = t["start"], t["end"]
        if b <= done_until:
            continue
        closed = final or (len(wav) - b) / SR >= 0.5
        if not closed:
            break
        piece = wav[max(done_until, a - 3200):min(len(wav), b + 3200)]
        text, _, segs = _transcribe(piece, lang, vad=False, words=False)
        if not reliability(text, piece, segs)[0]:             # an invented greeting on a short / unclear chunk: not shown
            text = ""
        st["chunks"].append((a, b, text))
        done_until = b
    committed = " ".join(c[2] for c in st["chunks"] if c[2]).strip()
    tentative = ""
    tail = wav[done_until:]
    if not final and len(tail) > SR * 0.4:
        piece = tail[-SR * 12:]
        # interim text only where the voice detector hears speech (no retry without it — that is where Whisper invents
        # "สวัสดีครับ"), and only if it passes the same reliability check as the final text
        tentative, _, segs = _transcribe(piece, lang, beam=1, vad=True, words=False, retry=False)
        if not reliability(tentative, piece, segs)[0]:
            tentative = ""
    out = dict(session=sid, committed=committed, tentative=tentative, final=bool(final), seconds=round(dur, 2),
               ms=round((time.time() - t0) * 1000))
    if final:
        ok, why, sec = reliability(committed, wav, [])
        out.update(text=committed, reliable=ok, flags=why, speech_seconds=sec, mode="incremental", model=M["th_model"],
                   device=M["device"])
        SESSIONS.pop(sid, None)
    return out


class MTReq(BaseModel):
    text: str = Field(..., max_length=1000)


@app.post("/mt")
def mt(req: MTReq):
    t0 = time.time()
    sp = M["sp"]
    src = ["eng_Latn"] + sp.encode(req.text.strip(), out_type=str) + ["</s>"]
    with _lock:
        r = M["mt"].translate_batch([src], target_prefix=[["tha_Thai"]], beam_size=2, max_decoding_length=200,
                                    return_scores=True)
    hyp = r[0].hypotheses[0]
    return dict(text=sp.decode(hyp[1:] if hyp and hyp[0] == "tha_Thai" else hyp), score=round(float(r[0].scores[0]), 3),
                model="nllb-200-distilled-600M", ms=round((time.time() - t0) * 1000))
