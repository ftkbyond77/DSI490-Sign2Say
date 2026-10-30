# Live frontend A/B — 2026-10-28

Experimental only, default off; no production containers/models/configs changed.
API uses read-only Jack-model v8.1, localhost:18081. Web localhost:18082.

| Case | URL |
|---|---|
| A: raw + original face cadence | http://127.0.0.1:18082/?liveExp=1 |
| B: raw + reduced expression cadence | http://127.0.0.1:18082/?liveExp=1&faceEvery=3 |
| C: median 3 only | http://127.0.0.1:18082/?liveExp=1&consensus=3 |
| D: both | http://127.0.0.1:18082/?liveExp=1&faceEvery=3&consensus=3 |
| Optional median 5 | http://127.0.0.1:18082/?liveExp=1&faceEvery=3&consensus=5 |

Close camera before switching URLs. Keep one camera tab active. Use identical camera resolution, lighting, vocabulary, auto-light, fingerspelling and speech settings across cases. Experiment starts in Live, conversation, LLM off and speech off. Use an agreed sentence (e.g. สวัสดี ขอบคุณ), normal/fast, three repeats per case; rotate order. Export “บันทึกผล A/B (JSON)” before reload. It includes original/filtered landmarks, observation timestamps, FPS, frame-gap p95, hand visibility fractions, tracking wall time, expression-dispatch time, predictions and camera settings. Frames limited to last 3000 observations; no images/video recorded. These visibility fractions include hands-down periods, so compare equivalent intervals.

## What actually changed

- Existing ExpressionModel was already Promise-based, busy-guarded and throttled to >140 ms. This does NOT establish face as the FPS bottleneck.
- faceEvery=3 calls ExpressionModel every third captured observation, with minimum 300 ms dispatch interval. It remains on main thread and shares GPU resources; it is NOT a Worker implementation. Holistic face mesh/blendshapes remain unchanged, including CPU fallback every eighth frame. Face label held between updates; face loss clears it. Server face analysis unchanged.
- consensus=3/5 is a causal coordinate median for each hand, NOT a direction-consensus gate. It never waits until still, never drops observations, never fills absent hands, and keeps original timestamps/body/face. Warmup, missing hand and >250 ms gap reset/bypass the filter. It can lag/distort real fast movement; not promoted.
- In all liveExp cases, duplicate video.currentTime observations are skipped equally. Therefore compare A–D with liveExp=1, not an uninstrumented baseline. No claim this reduces Tagger load (frame count is unchanged).
- Median lag is approximately (window-1)/2 frame intervals on smooth motion: ~67–154 ms for window3 at 13–30 fps; ~133–308 ms for window5. Not a measured end-to-end latency result.
- Expression dispatch timing excludes asynchronous GPU completion; do not call it total face latency.
- API caches requests separately under Boom/v81_port/outputs/live_experiment_requests. Experiment flags/raw pairs are in downloaded JSON, not currently in API cache. Keep exports to identify cases.

## Verification

TypeScript noEmit passed; algorithm invariant tests passed (no recognition mocks): disabled identity, median spike, timestamp preservation, missing-hand reset, long-gap reset, valid flags. HTTP frontend 200, /api/v1/info confirms v8.1 with conversation 328. Live FPS/accuracy not measured until a person signs. No WER without manually verified reference. Dev build warnings from pre-existing face-api dynamic require remain; verify camera/expression fallback in actual browser.

## Local server commands

From this worktree root, API in one terminal:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR="$PWD/Boom/v81_port/outputs/.mplconfig" \
PYTHONPATH="$PWD/Boom/v81_port/outputs/live_server_deps:$PWD/Boom/v81_port/outputs/api_deps:$PWD" \
THAISLM_MODELS=/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-jack/models/Jack-model \
THAISLM_SAVE_REQUESTS="$PWD/Boom/v81_port/outputs/live_experiment_requests" ORT_THREADS=2 \
/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say/Boom/venv-sign2say/bin/python -B -m uvicorn webapp.api.app:app --host 127.0.0.1 --port 18081
```

Web in another:

```bash
cd webapp/web
SIGN_API=http://127.0.0.1:18081 WATCHPACK_POLLING=true npm run dev -- --webpack --port 18082 --hostname 127.0.0.1
```

Use existing local node_modules/public assets installed for this experiment. Stop only these foreground processes with Ctrl-C. No Docker changes. The dev server adds overhead, so numbers are exploratory; a controlled production build is needed before claiming speedup.

## GPU compatibility fallback

If MediaPipe reports unsupported GPU delegate operations at the warmup probe, use `&delegate=cpu` on every A/B URL. This skips Holistic GPU probes and selects the CPU expression backend. It is a compatibility workaround, not a speed improvement. Keep the same delegate across comparisons. GPU attempt cleanup now closes a partially created graph if the companion fails. The console message alone does not establish that camera startup failed, since the original path already attempts fallback. CPU mode passed TypeScript checks; actual camera FPS remains to be measured on the user's browser.

## Framing guide (2026-10-29)

Opt-in `&guide=1` or checkbox below video. Dashed head/shoulder/body reference and safe-edge outline use source-image coordinates, so they share the video's mirror/aspect transform. Advice is based on confident nose/shoulder positions, apparent shoulder width and detected hand proximity to image edges. Missing hands at rest do not block readiness. No frames, predictions or endpoint decisions are changed. Green means heuristic body framing only, not correct hand tracking/sign recognition. Shoulder widths .18–.46, center tolerance .12 and other limits are unvalidated UX heuristics, not measured training distribution. Rotation can look like small shoulder width; use as advice only. No light/occlusion/identity inference.

Experiment exports include per-observation framing measurements and current toggle. Export control moved outside camera overlay so it cannot be hidden by video or cover the guide. TypeScript and pure function invariant tests pass. Actual camera placement/recognition effect still needs paired human recordings. Compare same model/delegate/faceEvery/resolution, same sentence with/without guide; do not simultaneously enable consensus. Save each run immediately. No inference reruns, test data access, model changes or Docker changes.
