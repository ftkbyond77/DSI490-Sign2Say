# ThaiSLM web demo — sign → Thai text → speech, in the browser

```bash
docker compose up --build        # from the repository root; then open http://localhost:8080
```
(optional) put `OPENAI_API_KEY=...` in the root `.env` to let an LLM write the Thai sentence (otherwise a rules composer does it).

## What you can do

| mode | how | what happens |
|---|---|---|
| **บันทึกเอง** (record) | เปิดกล้อง → เริ่ม → sign a sentence → หยุด → ประมวลผล | landmarks of the recorded frames go to the API; the result (words, sentence) is shown and spoken |
| **โหมดสด** (live) | เปิดกล้อง → choose โหมดสด → raise your hands and sign; put them down to finish | hands up = an utterance starts, hands down for 1.5 s (setting: 0.9 / 1.5 / 2.5 s — longer if you sign word by word) = it ends and is translated + spoken automatically, again and again. A hand raise shorter than 0.6 s is skipped quietly |
| **อัปโหลดวิดีโอ** | pick a video file | the video is processed by MediaPipe **in your browser** (same path as the camera) |
| **ให้เซิร์ฟเวอร์แยกจุด** | pick a video file | the server extracts the landmarks (same MediaPipe model in Python) |

Options (remembered in this browser):

| option | default | what it does |
|---|---|---|
| เสียงพูด | ออกเสียงทุกผล | `ไม่ออกเสียง` · `เฉพาะที่มั่นใจ` (only a sentence the evidence supports) · `ทุกผล` (also an uncertain reading — the card marks it "(ไม่แน่ใจ)"). The 🔊 button on any card speaks it on request. Speech is played through Web Audio unlocked by your first click, so live-mode results (which arrive without a click) are heard |
| สะกดนิ้ว | อังกฤษ A–Z | fingerspelled letters: English A–Z (the one-handed manual alphabet Thai deaf schools teach), Thai ก–ฮ + 7 vowels, both, or off. Consecutive letters become one name ("J A C K" → **JACK**); speech reads Latin letters by their Thai names ("เจ เอ ซี เค") |
| คลังคำ | บทสนทนาประจำวัน | the daily-conversation vocabulary, or the full ~10k-concept dictionary (more words, less certainty) |
| LLM | on | writes the Thai sentence when some signs are uncertain; when every sign is confident the rules composer is used (no 1–3 s wait) |
| ปรับภาพอัตโนมัติ | on | brightness / contrast / white balance (and display sharpening) adapted every 0.3 s — dark rooms and washed-out webcams look alike to MediaPipe |
| ทดลอง: ขมวดคิ้ว = ไหม | off | experimental yes/no-question cue measured against your own resting face (not reliable across signers: 0/8 on an unseen signer) |

**ท่าของฉัน (personal sign memory).** Click a word on a result card → choose the right word → that segment is remembered for you
(in this browser only; up to 3 examples per word, averaged into one prototype on the server). On unseen TSL51 test sentences of
a new signer, remembering 2 examples per word raised recall 0.65 → 0.80 and confident-correct words 141 → 178 while confident errors
went 14 → 12. The shared model never changes.

On the camera view: body and hands (boxed, labelled มือซ้าย / มือขวา) and one orange box around the face with the live emotion — the
face itself is not covered; the "ตรวจพบ" bar still checks that face, brows / expression and both hands are extracted. The HUD shows
the MediaPipe mode and fps; a progress bar shows the real server stages while a sentence is processed. In live mode the utterance is
sent after 0.5 s of hands-down, so the result is usually ready when the pause ends.

**Speed / robustness, chosen automatically per machine:** MediaPipe runs fully on the GPU when the browser's WebGL can run the
face-expression model; on GPUs where that model fails ("No support of const") it runs body / hands / face mesh on the GPU every frame
and the expressions on the CPU every 5th frame (≈ 1.5–3× faster than CPU-only); otherwise CPU only. A frame that fails never stops
the camera loop. Light correction (the same rule as the server's for uploads) raised hand detection on test videos from 0.61 → 0.69
(dark), 0.70 → 0.74 (over-bright), 0.69 → 0.73 (low contrast) and left good footage unchanged.

Every word is shown with its evidence: green = confident, amber `word?` = uncertain, red `[?] ≈ word` = a sign the model does not
know (nearest learned word).


**v7.4 (camera & cards):** the camera opens in its full-sensor mode (native aspect, often 4:3 — the widest and tallest view it has, zoom at minimum) and the stage takes that shape; hands are drawn as their 21 tracked points (the model's input), no boxes; the face box shows the live expression from a trained model (face-api FaceExpressionNet on the MediaPipe face crop, ~7 ms, ≤ 7×/s; blendshape rules only if it cannot load), e.g. “😊 ยิ้ม → มีความสุข”. Result cards have a stable identity, so a word taught on one card stays on that card when new results arrive (before, the ✓ moved to the newest card).

**v7.3 (after real webcam tests):** a word is shown, spoken and used in the sentence only when its calibrated P(correct) ≥ 0.8
(`models/spotting.json` "accept_min"); others appear as **[?]** (no guessed word) — click one, pick or type the word you signed,
and it is remembered for you. Live mode ignores movements shorter than 1.2 s / with < 0.6 s of hands up (scratching, adjusting) and
uses the body's wrist points too, so a hand that drops out for a moment no longer ends the sentence; hand drop-outs ≤ 0.5 s are
interpolated (`hand_gap_s`). Saved settings were reset (an earlier "full vocabulary + both alphabets" setting made it slow and
noisy). Each utterance's landmarks + result are kept in `cache/live_requests/` for diagnosis (landmarks only, no video;
`THAISLM_SAVE_REQUESTS=` disables it); `python scripts/replay_live.py <folder>` replays them through the running service.

## Architecture (micro-services)

```
browser ─ camera → MediaPipe Holistic (WASM, @mediapipe/tasks-vision 1.0.1) → landmarks only (no video leaves the device)
   │  POST /api/v1/translate/stream {frames, width, height, vocab, llm, speak, face_question, face_baseline}
   ▼
gateway (nginx :8080) ──/───► web       Next.js 16 (React 19) UI
                     ├─/api/► sign-api  FastAPI + ONNX Runtime + NumPy (stateless — `docker compose up --scale sign-api=3`)
                     └─/tts/► tts       FastAPI + MMS-TTS Thai (VITS, CPU)
sign-api → tts (internal) when speak=true · sign-api → OpenAI (optional) for the sentence layer
```

| service | folder | image | notes |
|---|---|---|---|
| gateway | `webapp/gateway/nginx.conf` | nginx:1.27-alpine | one origin (camera needs a secure context → localhost is fine; use HTTPS in production); resolves scaled replicas through Docker DNS |
| web | `webapp/web` | node:22-alpine (Next.js standalone) | serves MediaPipe WASM + the `.task` model from its own origin |
| sign-api | `webapp/api` | python:3.11-slim | mounts `./models` read-only (`THAISLM_MODELS_DIR` to use another model folder) |
| tts | `webapp/tts` | python:3.11-slim + CPU torch | the Thai voice is baked into the image |

API reference: `GET /api/v1/health · /api/v1/info · /api/v1/vocab`, `POST /api/v1/translate`, `POST /api/v1/translate/video`
(multipart `file`), and the same two as `…/stream` (NDJSON: `{"stage","p"}` progress lines, then `{"result"}`; the gateway does
not buffer them), `POST /tts/v1/speak {text, emotion, intensity}` → `audio/wav`. Interactive docs: http://localhost:8080/api/docs
is not exposed through the gateway by default; run `docker compose exec sign-api curl localhost:8000/docs`.

## Scaling to a product

* sign-api is stateless and CPU-only → horizontal scaling behind the gateway (or any load balancer / Cloud Run / k8s HPA); a GPU
  node can serve the same image with `onnxruntime-gpu` and `ORT_DEVICE=cuda`.
* Pose extraction runs on the user's device, so server cost per request is small (one encoder pass per utterance + a few windows).
* Models are a mounted volume: a new bank (new words / signers) is a file swap, no rebuild.

## Development without Docker

```bash
cd webapp/web && npm install && npm run dev                    # http://localhost:3000 (proxies /api → :8000, /tts → :8001)
uvicorn app:app --app-dir webapp/api --port 8000               # from the repo root, env THAISLM_MODELS=models
uvicorn app:app --app-dir webapp/tts --port 8001
```
