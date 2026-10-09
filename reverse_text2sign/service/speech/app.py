"""reverse-speech — the heavy language models, kept out of the lean motion API (one container, CPU, int8 CTranslate2):

  POST /asr   audio (webm / ogg / wav / mp3, ≤ 60 s) + lang (th | en) → {text, lang, ms, avg_logprob, no_speech}
              Thai: biodatlab/whisper-th-small-combined (Whisper small fine-tuned on Thai, converted to CT2 int8)
              English: Systran/faster-whisper-base.en
  POST /mt    {text} English → Thai: facebook/nllb-200-distilled-600M (CT2 int8, beam 2)
  GET  /health
Models are converted from the official checkpoints at image build time (see Dockerfile) and loaded once at start.
"""
from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

MODELS = Path(os.environ.get("SPEECH_MODELS", "/models"))
THREADS = int(os.environ.get("SPEECH_THREADS", "4"))
app = FastAPI(title="reverse-speech", version="1.0")
_lock = threading.Lock()
M: dict = {}


@app.on_event("startup")
def _load():
    import ctranslate2
    import sentencepiece as spm
    from faster_whisper import WhisperModel
    t0 = time.time()
    M["asr_th"] = WhisperModel(str(MODELS / "whisper-th-small"), device="cpu", compute_type="int8", cpu_threads=THREADS)
    M["asr_en"] = WhisperModel(str(MODELS / "whisper-base-en"), device="cpu", compute_type="int8", cpu_threads=THREADS)
    M["mt"] = ctranslate2.Translator(str(MODELS / "nllb-600m"), device="cpu", compute_type="int8", intra_threads=THREADS)
    M["sp"] = spm.SentencePieceProcessor(model_file=str(MODELS / "nllb-600m" / "sentencepiece.bpe.model"))
    M["load_s"] = round(time.time() - t0, 2)


@app.get("/health")
def health():
    return dict(ok="mt" in M, load_s=M.get("load_s"))


@app.post("/asr")
async def asr(audio: UploadFile = File(...), lang: str = Form("th")):
    from faster_whisper.audio import decode_audio
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio")
    t0 = time.time()
    try:
        wav = decode_audio(io.BytesIO(data), sampling_rate=16000)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(415, f"cannot decode audio: {type(e).__name__}") from e
    if len(wav) > 16000 * 62:
        raise HTTPException(413, "audio longer than 60 s")
    model = M["asr_en"] if lang == "en" else M["asr_th"]
    kw = dict(language="en" if lang == "en" else "th", beam_size=3, condition_on_previous_text=False, without_timestamps=True)
    with _lock:                                   # one decode at a time per container (CPU bound); scale with --scale
        segs = list(model.transcribe(wav, vad_filter=True, **kw)[0])
        vad = True
        if not "".join(x.text for x in segs).strip():
            # very short replies ("ใช่", "ได้", "ฝนตก") can fall entirely under the VAD's minimum speech length → retry without it
            segs = list(model.transcribe(wav, vad_filter=False, **kw)[0])
            vad = False
    text = " ".join(s.text.strip() for s in segs).strip()
    lp = [s.avg_logprob for s in segs]
    return dict(text=text, lang=lang, ms=round((time.time() - t0) * 1000), seconds=round(len(wav) / 16000, 2), vad=vad,
                avg_logprob=round(sum(lp) / len(lp), 3) if lp else None,
                no_speech=round(max((s.no_speech_prob for s in segs), default=1.0), 3))


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
