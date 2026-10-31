# Face cadence default 3

Base: boom-v81-port, 2136b86. Saved pre-existing experiments separately on boom-live-experiment (07179ab).

## Behavior
- Previously ExpressionModel.update and its blendshape fallback were called every eligible face-visible camera frame (1). Now called on frames 1, 4, 7... (3). Last expression remains visible between updates.
- Existing internal 140 ms / busy throttle remains. This is a scheduling interval, not a guarantee of inference every third frame.
- `?faceEvery=N` accepts positive safe integers; absent/invalid values default to 3. `?faceEvery=1` restores previous invocation frequency.
- CPU companion blendshapes remain every 8 eligible frames. Holistic, hand/body detection, camera size, models and expression engine are unchanged.
- `?perf=nofx` retains its existing Live-mode bypass of auxiliary expression updates, companion detection and blendshape payload. It does NOT disable Holistic internal face detection, model loading/warmup or server face analysis. No claim that all face computation stops.
- Optional `face_every` records requested cadence on Live payload/cache. `perf_mode=nofx` means that cadence is bypassed; it is not measured inference FPS. Legacy clients/cache omit the field safely.
- No A/B, framing-guide or hand-median feature is included in this branch's change.

## Validation
- TypeScript: `tsc --noEmit` passed.
- Phase 2.2 unittest suite: 23/23 passed.
- No real-camera FPS/accuracy improvement has been measured for this branch. Existing live services were not restarted.

## PR scope
Target boom as requested. Since base v8.1 port PR #7 is still pending, the full PR includes that prerequisite commit; the face-only change is the new commit after 2136b86. Do not merge automatically.
