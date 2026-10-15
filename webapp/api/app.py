"""ThaiSLM sign-api — FastAPI micro-service: landmarks (or a video) → Thai words / sentence (+ speech via the tts service).

  POST /v1/translate         JSON {frames:[{t, pose, lh, rh, face, bs}], width, height, vocab?, llm?, speak?, face_question?, face_baseline?}
                             (browser MediaPipe)
  POST /v1/translate/video   multipart file=<video>  (?vocab=conversation&llm=1&speak=0&face_question=0)          (server MediaPipe)
  …/stream                   the same two endpoints as NDJSON: {"stage", "p"} progress lines, then {"result": …}
  GET  /v1/health · /v1/info · /v1/vocab

Stateless (model files are read-only, mounted at /app/models) → scale horizontally: `docker compose up --scale sign-api=3`.
Inference = ONNX Runtime + NumPy (no PyTorch). The language layer uses OpenAI when OPENAI_API_KEY is set, otherwise rules.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P

_ps = _P(__file__).resolve().parents
_root = _ps[2] if len(_ps) > 2 else _ps[-1]
if (_root / "modules").is_dir() and str(_root) not in sys.path:      # running from the repository (no Docker)
    sys.path.insert(0, str(_root))

import base64
import json
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from modules.pipeline import SignPipeline
from modules.utils import read_json

MODEL_DIR = Path(os.environ.get("THAISLM_MODELS", "/app/models"))
TTS_URL = os.environ.get("TTS_URL", "http://tts:8001")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", 25 * 60))

app = FastAPI(title="ThaiSLM sign-api", version="7.0")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","), allow_methods=["*"], allow_headers=["*"])
_pipes: dict[tuple, SignPipeline] = {}
_lock = threading.Lock()


def pipe(vocab="conversation", llm=True) -> SignPipeline:
    key = (vocab, bool(llm))
    with _lock:
        if key not in _pipes:
            _pipes[key] = SignPipeline(MODEL_DIR, backend="onnx", device=os.environ.get("ORT_DEVICE", "cpu"), vocab=vocab, use_llm=llm)
        return _pipes[key]


@app.on_event("startup")
def warm():
    p = pipe()
    import numpy as np
    from modules.parts import prep
    rng = np.random.RandomState(0)
    p.be.embed([prep(rng.uniform(0.3, 0.7, (30, 133, 2)).astype(np.float32), np.ones((30, 133), np.float32))])   # first-call latency


class Frame(BaseModel):
    t: float = Field(..., description="timestamp in ms")
    pose: list[list[float]] = []
    lh: list[list[float]] = []
    rh: list[list[float]] = []
    face: list[list[float]] = []
    bs: list[float] = []


class MySign(BaseModel):
    word: str
    z: str = Field(..., description="base64 float16 embedding [768] as returned in a segment's `emb`")


def _b64_to_vec(z: str):
    import numpy as np
    try:
        v = np.frombuffer(base64.b64decode(z), np.float16).astype(np.float32)
        return v if v.shape == (768,) else None
    except Exception:  # noqa: BLE001
        return None


def _vec_to_b64(v) -> str:
    import numpy as np
    return base64.b64encode(np.asarray(v, np.float16).tobytes()).decode()


class TranslateRequest(BaseModel):
    frames: list[Frame]
    width: int
    height: int
    vocab: str = "conversation"
    llm: bool = True
    speak: bool = False
    speak_tentative: bool = False               # also speak a withheld (uncertain) reading — the user's choice in the web UI
    face_question: bool = False                 # experimental: furrowed brows at the end → yes/no question (ไหม)
    face_baseline: list[float] | None = None    # the signer's resting-face blendshapes (52), measured by the client
    user_bank: list[MySign] = []                # this user's confirmed signs (personal sign memory, kept by the client)
    alphabet: str = "en"                        # fingerspelled letters: en (A–Z) | th (ก–ฮ, vowels) | both | off
    spell_only: bool = False                    # this utterance is fingerspelling only (the "สะกดอย่างเดียว" button)


def _speak(result, tentative=False):
    """Speech for the confirmed sentence; with tentative=True also for a withheld reading (the client marks it as uncertain).
    Unknown-word marks are never read out."""
    text = result.get("sentence") or ""
    if not text and tentative:
        text = (result.get("fusion") or {}).get("tentative_sentence_th") or ""
    text = text.replace("[?]", " ").strip()
    if not text:
        return None
    face = result.get("face") or {}
    try:
        r = httpx.post(f"{TTS_URL}/v1/speak", json=dict(text=text, emotion=face.get("emotion", "neutral"), intensity=face.get("intensity", 0.0)), timeout=30)
        r.raise_for_status()
        return base64.b64encode(r.content).decode()
    except Exception:  # noqa: BLE001  (speech is optional; text is the result)
        return None


def _public(result):
    keep = ("title", "duration_s", "n_frames", "signer_visible_frac", "signing_frac", "mirrored_to_right_hand", "sentence", "vocab", "model", "timings_ms")
    out = {k: result.get(k) for k in keep}
    out["segments"] = [dict(t0=s["t0"], t1=s["t1"], status=s["status"], word=s["nearest"], p=round(s.get("p_correct", 0), 3), kind=s.get("kind", "sign"),
                            candidates=[c["word"] for c in s["candidates"][:5]], **({"emb": _vec_to_b64(s["emb"])} if "emb" in s else {}))
                       for s in result.get("segments", [])]
    f = result.get("fusion") or {}
    out["language"] = {k: f.get(k) for k in ("sentence_th", "tentative_sentence_th", "meaning_th", "evidence_gloss", "evidence_guard", "provider", "chosen")}
    face = result.get("face") or {}
    out["face"] = {k: face.get(k) for k in ("question_yesno", "question_furrow", "question_wh", "negation", "affirmation", "emotion", "emotion_th", "intensity",
                                            "face_frames")}
    out["face"]["brow_furrow_rel"] = (face.get("cues") or {}).get("brow_furrow_rel")
    return out


@app.get("/v1/health")
def health():
    return dict(ok=True, models=str(MODEL_DIR), loaded=[f"{k[0]}{'+llm' if k[1] else ''}" for k in _pipes])


@app.get("/v1/info")
def info():
    cfg = read_json(MODEL_DIR / "config.json")
    p = pipe()
    return dict(version=cfg.get("version"), built=cfg.get("built"), concepts_full=len(p.concepts), bank_clips=int(len(p.bank.c)),
                conversation_concepts=int(len(p.bank.concepts_present)), llm=bool(os.environ.get("OPENAI_API_KEY")), extractor="MediaPipe Holistic 1.0.1")


@app.get("/v1/vocab")
def vocab():
    return read_json(MODEL_DIR / "vocab_conversation.json")


STAGE_P = dict(schema=0.05, tagger=0.15, spotting=0.45, face=0.7, language=0.8)


def _check(req: TranslateRequest):
    if not req.frames:
        raise HTTPException(400, "no frames")
    if len(req.frames) > MAX_FRAMES:
        raise HTTPException(413, f"too many frames (max {MAX_FRAMES})")
    if req.vocab not in ("conversation", "full"):
        raise HTTPException(400, "vocab must be conversation | full")


SAVE_DIR = os.environ.get("THAISLM_SAVE_REQUESTS", "")      # diagnosis: keep each utterance's landmarks + result (landmarks only, no video)


def _save_request(req: TranslateRequest, out: dict):
    if not SAVE_DIR:
        return
    try:
        import gzip
        d = Path(SAVE_DIR); d.mkdir(parents=True, exist_ok=True)
        rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), width=req.width, height=req.height, vocab=req.vocab, alphabet=req.alphabet,
                   n_user_signs=len(req.user_bank), frames=[f.model_dump() for f in req.frames],
                   result={k: out.get(k) for k in ("sentence", "language", "face", "timings_ms", "duration_s")})
        # copies without the embedding — the response itself must keep "emb" (the client saves it when the user teaches a word)
        rec["result"]["segments"] = [{k: v for k, v in sgm.items() if k != "emb"} for sgm in out.get("segments") or []]
        with gzip.open(d / f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False)
    except Exception:  # noqa: BLE001  (diagnosis must never break a request)
        pass


def _frames_job(req: TranslateRequest, progress=None):
    t = time.perf_counter()
    say = progress or (lambda stage, p: None)
    p = pipe(req.vocab, req.llm)
    raw = p.landmarks_from_browser([f.model_dump() for f in req.frames], (req.height, req.width))
    ub = [(m.word, v) for m in req.user_bank[:600] if (v := _b64_to_vec(m.z)) is not None]
    r = p.run_landmarks(raw, title="camera", progress=lambda st, f: say(st, STAGE_P.get(st, f)), face_question=req.face_question,
                        face_baseline=req.face_baseline, user_bank=ub, alphabet=req.alphabet if req.alphabet in ("en", "th", "both", "off") else "off",
                        spell_only=req.spell_only)
    out = _public(r)
    if req.speak and (r.get("sentence") or (req.speak_tentative and (r.get("fusion") or {}).get("tentative_sentence_th"))):
        say("tts", 0.9)
        out["audio_wav_b64"] = _speak(r, req.speak_tentative)
        out["spoken"] = "sentence" if r.get("sentence") else "tentative"
    out["timings_ms"]["total_ms"] = round((time.perf_counter() - t) * 1000, 1)
    _save_request(req, out)
    return out


def _video_job(path, name, vocab, llm, speak, face_question, progress=None, speak_tentative=False, alphabet="en", spell_only=False):
    t = time.perf_counter()
    say = progress or (lambda stage, p: None)
    p = pipe(vocab, llm)
    r = p.run(path, face_question=face_question, alphabet=alphabet if alphabet in ("en", "th", "both", "off") else "off", spell_only=spell_only,
              progress=lambda st, f: say(st, 0.6 * f if st == "mediapipe" else 0.6 + 0.4 * STAGE_P.get(st, f)))
    if r.get("error"):
        raise HTTPException(422, r["error"])
    r["title"] = name
    out = _public(r)
    if speak and (r.get("sentence") or (speak_tentative and (r.get("fusion") or {}).get("tentative_sentence_th"))):
        say("tts", 0.93)
        out["audio_wav_b64"] = _speak(r, speak_tentative)
        out["spoken"] = "sentence" if r.get("sentence") else "tentative"
    out["timings_ms"]["total_ms"] = round((time.perf_counter() - t) * 1000, 1)
    return out


def _stream(job):
    """Run job(progress) in a worker thread; stream progress lines and the result as NDJSON (one JSON object per line)."""
    q: queue.Queue = queue.Queue()

    def work():
        try:
            q.put(dict(result=job(lambda stage, p: q.put(dict(stage=stage, p=round(float(p), 3))))))
        except HTTPException as e:
            q.put(dict(error=str(e.detail), status=e.status_code))
        except Exception as e:  # noqa: BLE001
            q.put(dict(error=f"{type(e).__name__}: {e}", status=500))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def gen():
        yield json.dumps(dict(stage="received", p=0.02)) + "\n"
        while (m := q.get()) is not None:
            yield json.dumps(m, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


def _save_upload(file: UploadFile):
    suffix = Path(file.filename or "v.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        return tmp.name


@app.post("/v1/translate")
def translate(req: TranslateRequest):
    _check(req)
    return _frames_job(req)


@app.post("/v1/translate/stream")
def translate_stream(req: TranslateRequest):
    _check(req)
    return _stream(lambda progress: _frames_job(req, progress))


@app.post("/v1/translate/video")
def translate_video(file: UploadFile = File(...), vocab: str = Query("conversation"), llm: bool = Query(True), speak: bool = Query(False),
                    face_question: bool = Query(False), speak_tentative: bool = Query(False), alphabet: str = Query("en"),
                    spell_only: bool = Query(False)):
    path = _save_upload(file)
    try:
        return _video_job(path, file.filename, vocab, llm, speak, face_question, speak_tentative=speak_tentative, alphabet=alphabet,
                          spell_only=spell_only)
    finally:
        os.unlink(path)


@app.post("/v1/translate/video/stream")
def translate_video_stream(file: UploadFile = File(...), vocab: str = Query("conversation"), llm: bool = Query(True), speak: bool = Query(False),
                           face_question: bool = Query(False), speak_tentative: bool = Query(False), alphabet: str = Query("en"),
                           spell_only: bool = Query(False)):
    path = _save_upload(file)

    def job(progress):
        try:
            return _video_job(path, file.filename, vocab, llm, speak, face_question, progress, speak_tentative, alphabet, spell_only)
        finally:
            os.unlink(path)
    return _stream(job)
