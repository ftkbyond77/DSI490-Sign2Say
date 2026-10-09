"""reverse-api — text / speech → Thai Sign Language avatar motion (FastAPI).

  GET  /api/health                 status, library size, load time
  POST /api/translate              {text, lang: auto|th|en, speed, nmm} → {ir, motion}
  POST /api/render                 {ir, speed, nmm} → {ir, motion}     (re-render an IR edited by the user / an expert)
  GET  /api/sign/{concept}         one dictionary sign (preview)
  GET  /api/lookup?q=              signable words starting with / containing q (gloss editor autocomplete)
  POST /api/asr                    audio (webm / wav / ogg) + lang → {text, lang, ms}   (proxied to the speech service)
  POST /api/export/vrma            {ir | text, speed} → .vrma (VRM Animation: three-vrm / UniVRM in Unity)
  GET  /api/examples               example sentences
  GET  /api/rest?avatar=           the idle pose
  GET  /api/avatars                the characters of the picker (every *.vrm in vrm/; optimised copies + thumbnails at /avatars/)
Requests that produce motion take `avatar` (a catalogue id): the motion is fitted to that character's body (t2s/motion/fit.py).
Stateless; the motion library is memory-mapped (many workers share it). Responses are gzip-compressed.
"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(os.environ.get("T2S_HOME", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from t2s import avatar as avatars  # noqa: E402
from t2s.pipeline import Engine  # noqa: E402

SPEECH_URL = os.environ.get("T2S_SPEECH_URL", "http://speech:8002")
MAX_CHARS = int(os.environ.get("T2S_MAX_CHARS", "400"))
EXAMPLES = [
    "สวัสดีครับ ผมชื่อสมชาย", "คุณชื่ออะไร", "ฉันชอบกินไข่", "ห้องน้ำอยู่ที่ไหน", "พรุ่งนี้คุณจะไปโรงเรียนไหม",
    "ฉันไม่เข้าใจ ช่วยพูดช้าๆ ได้ไหม", "วันนี้อากาศร้อนมาก", "ขอบคุณมากครับ", "ฉันหิวข้าว", "แม่ของฉันเป็นครู",
    "Hello, how are you?", "Where is the hospital?",
]

app = FastAPI(title="ThaiSLM reverse — text/speech → Thai sign", version="1.0")
app.add_middleware(GZipMiddleware, minimum_size=800)
ENGINE: Engine | None = None
CACHE: OrderedDict = OrderedDict()          # (text, lang, speed, nmm) → response; small LRU for repeated phrases
CACHE_MAX = 256


@app.on_event("startup")
def _load():
    global ENGINE
    import threading
    threading.Thread(target=avatars.catalogue, name="avatar-scan", daemon=True).start()   # thumbnails / fits / optimised copies
    ENGINE = Engine()
    ENGINE.run("สวัสดี ฉันชื่ออะไร")              # warm-up: pythainlp dictionaries / tagger, page-in the first clips


class TranslateReq(BaseModel):
    text: str = Field(..., max_length=2000)
    lang: str = "auto"
    speed: float = Field(1.0, ge=0.4, le=2.0)
    nmm: bool = True
    source: str = "text"
    phrasebook: bool = True
    avatar: str | None = None


class RenderReq(BaseModel):
    ir: dict
    speed: float = Field(1.0, ge=0.4, le=2.0)
    nmm: bool = True
    avatar: str | None = None


def _speech_ok():
    try:
        return httpx.get(f"{SPEECH_URL}/health", timeout=0.8).json().get("ok", False)
    except Exception:  # noqa: BLE001
        return False


@app.get("/api/health")
def health():
    e = ENGINE
    sp = _speech_ok()
    return dict(ok=e is not None, signs=len(e.lib.clips) if e else 0, english=sp, speech=sp,
                load_s=round(e.load_s, 2) if e else None)


@app.get("/api/rest")
def rest(avatar: str | None = None):
    """the idle pose (one frame) — the browser breathes / blinks on top of it while nothing plays."""
    import numpy as np
    from t2s.motion.codec import encode
    from t2s.motion.compose import rest_pose
    from t2s.motion.retarget import MORPHS
    q = rest_pose()[None]
    fit = avatars.fit_for(avatar)
    return encode(fit(q) if fit else q, np.zeros((1, len(MORPHS)), np.float32))


@app.get("/api/avatars")
def avatar_list():
    return dict(avatars=[e for e, _ in avatars.catalogue()])


@app.get("/api/examples")
def examples():
    return dict(examples=EXAMPLES)


@app.post("/api/translate")
def translate(req: TranslateReq):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"text longer than {MAX_CHARS} characters — please split it")
    fit = avatars.fit_for(req.avatar)
    key = (text, req.lang, round(req.speed, 2), req.nmm, req.source, req.phrasebook, req.avatar if fit else None)
    if key in CACHE:
        CACHE.move_to_end(key)
        return CACHE[key]
    t0 = time.time()
    try:
        ir, motion = ENGINE.run(text, lang=req.lang, speed=req.speed, source=req.source, nmm=req.nmm, phrasebook=req.phrasebook,
                                fit=fit)
    except RuntimeError as e:
        raise HTTPException(422, str(e)) from e
    ir.confidence["t_total_ms"] = round((time.time() - t0) * 1000, 1)
    out = dict(ir=ir.to_dict(), motion=motion)
    CACHE[key] = out
    if len(CACHE) > CACHE_MAX:
        CACHE.popitem(last=False)
    return out


@app.post("/api/render")
def render(req: RenderReq):
    t0 = time.time()
    ir, motion = ENGINE.render_ir_dict(req.ir, speed=req.speed, nmm=req.nmm, fit=avatars.fit_for(req.avatar))
    ir.confidence["t_total_ms"] = round((time.time() - t0) * 1000, 1)
    return dict(ir=ir.to_dict(), motion=motion)


@app.get("/api/sign/{concept}")
def sign(concept: str, speed: float = 1.0, avatar: str | None = None):
    if concept not in ENGINE.lib:
        raise HTTPException(404, "no recorded sign for this word")
    ir, motion = ENGINE.render_ir_dict(dict(units=[dict(gloss=concept, kind="sign", concept=concept, source_words=[concept],
                                                        match="lemma", confidence={"translation": 1.0})], thai=concept),
                                       speed=speed, nmm=False, fit=avatars.fit_for(avatar))
    return dict(ir=ir.to_dict(), motion=motion)


@app.get("/api/lookup")
def lookup(q: str, limit: int = 12):
    q = q.strip().lower()
    if not q:
        return dict(items=[])
    ent = ENGINE.mapper.entries
    starts = [k for k in ent if k.startswith(q)]
    contains = [k for k in ent if q in k and not k.startswith(q)] if len(starts) < limit else []
    items = []
    for k in sorted(starts, key=len)[:limit] + sorted(contains, key=len)[: max(0, limit - len(starts))]:
        items.append(dict(text=k, concept=ent[k][0]["concept"], how=ent[k][0]["how"]))
    return dict(items=items[:limit])


@app.post("/api/asr")
async def asr(audio: UploadFile = File(...), lang: str = Form("th")):
    data = await audio.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(413, "audio longer than ~60 s")
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{SPEECH_URL}/asr", files={"audio": (audio.filename or "speech.webm", data, audio.content_type)},
                             data={"lang": lang})
    except httpx.HTTPError as e:
        raise HTTPException(503, f"speech recognition unavailable: {type(e).__name__}") from e
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text[:300])
    return r.json()


class ExportReq(BaseModel):
    ir: dict | None = None
    text: str | None = None
    speed: float = 1.0
    avatar: str | None = None


@app.post("/api/export/vrma")
def export_vrma(req: ExportReq):
    from t2s.motion.vrma import to_vrma
    fit = avatars.fit_for(req.avatar)
    if req.ir:
        ir, motion = ENGINE.render_ir_dict(req.ir, speed=req.speed, fit=fit)
    elif req.text:
        ir, motion = ENGINE.run(req.text, speed=req.speed, fit=fit)
    else:
        raise HTTPException(400, "ir or text required")
    data = to_vrma(motion)
    return Response(data, media_type="model/gltf-binary", headers={"Content-Disposition": 'attachment; filename="signing.vrma"'})


# character files (development; in Docker nginx serves /avatars/ from the shared cache volume and /avatars/src/ from vrm/)
app.mount("/avatars/src", StaticFiles(directory=avatars.SRC_DIR, check_dir=False), name="avatar-src")
app.mount("/avatars", StaticFiles(directory=avatars.CACHE_DIR, check_dir=False), name="avatars")
