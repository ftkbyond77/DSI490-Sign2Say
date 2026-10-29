# cam=wide experiment — 2026-10-29

Branch boom-cam-wide, based on boom-v81-port commit 2136b86 (draft PR #7). This is a stacked draft so the camera diff can be reviewed independently. No unrelated uncommitted face/framing/consensus changes included; MediaPipe modules, recognition settings and model assets unchanged.

## Behavior

`?cam=wide` requests exactly once:

```json
{"video":{"facingMode":"user","width":{"ideal":1280},"height":{"ideal":720},"resizeMode":"none"},"audio":false}
```

It returns without the existing resolution-selection loop, applyConstraints, or zoom adjustment. There is no silent retry to another size. `ideal` dimensions and plain resizeMode are requests, not guarantees: always inspect actual settings. Without cam=wide, the previous getUserMedia request, numeric cam height selection, resolution loop and zoom code are unchanged. No new diagnostics field is emitted in that path.

## Diagnostics

Wide mode exports `camera_capture` and capture history plus current translation results from a button BELOW the viewport. Each snapshot includes requested constraints, getSettings(), getCapabilities() or null, getConstraints(), capture time, and separate flags:

- resizeModeConstraintRecognized: browser supported-constraints advertisement.
- resizeModeCapabilityPresent: track capabilities explicitly contain a resizeMode list.
- resizeModeNoneSupported: true/false when list exists, null when absent/unavailable.
- resizeModeNoneApplied: true only if actual settings.resizeMode equals none; false for another value, null if omitted.

Camera translations carry optional camera_capture via stream API; file uploads do not inherit camera metadata. API accepts it as optional dict and caches it without affecting inference. Reader pattern: `record.get("camera_capture")`; writer uses `.get` on request data. Old requests/caches remain valid. Raw device/group identifiers returned by getSettings/getCapabilities remain in LOCAL export/cache; no captured diagnostics committed.

## Real browser observation

In the Codex in-app browser on this machine at localhost:18083/?cam=wide, after opening camera, the rendered diagnostics read:

> กล้อง wide: 1280 × 720 · resizeMode none: รองรับ · ใช้งานจริง

Thus that browser/track reported a capability list containing none and actual resizeMode none. This does not establish identical sensor FOV to another browser/mode or recognition improvement. The earlier 720×720 exports did not log capabilities and cannot establish their support. Other browsers with no resizeMode capability explicitly export null/false-presence instead of guessing.

The unchanged MediaPipe GPU warmup also emitted unsupported-GPU-operation console errors; GPU+CPU initially rendered, then the browser showed a page-load/error overlay. Closed the test tab to release its camera. Camera capability was observed, but this is NOT a passing end-to-end recognition/FPS evaluation. Do not change MediaPipe under this PR.

## Validation

- `tsc --noEmit`: passed.
- Phase 2.2 unittest discovery: 23/23 passed.
- Camera contract test: wide performs one request and no applyConstraints; missing/false/true support distinguished; default request and numeric cam=480 retained.
- Request/cache tests: old request omission and new metadata round-trip passed.
- `git diff --check`: passed.

No Docker operations, model writes, training, TSL51/Eval-Test evaluation, or merge performed. Existing localhost:18082 experiment untouched. localhost:18083 is a separate dev server for inspecting this branch; its current backend was the older local API during UI verification, so cache integration is covered by schema/cache tests, not claimed as a browser-to-cache end-to-end run. Deploy/restart only the isolated API with this branch before collecting a wide A/B dataset; never assume old API preserves unknown optional fields.
