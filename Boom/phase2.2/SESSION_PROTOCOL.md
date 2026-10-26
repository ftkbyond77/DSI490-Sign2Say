# Boom Phase 2.2: live capture protocol

This is a local capture and analysis protocol, not a request to run inference
now. Capture and inference stay on the participant's machine. Do not commit
recordings, screenshots, extracted landmarks, live caches or generated results.

## Participants and sessions

- Recruit at least 3 distinct signers. Each signer completes at least 2
  sessions on different days, with a short break between sessions.
- Use the same 20 prompts below in the displayed order in every session.
  Record one utterance per prompt; note any failed/repeated capture in the
  local capture log.
- Assign a distinct session ID and pseudonymous signer ID to each capture.
- Mark configuration-selection sessions `tune`. Reserve a different session
  and signer for `eval`. Never use Eval-Test or any held-out evaluation
  session/signer to select a configuration.

## Camera, lighting, and signing

- Use the same camera position for all sessions: eye-level, landscape, stable
  full upper body in frame, both hands and face visible, no digital zoom.
- Keep camera and signer still, about an arm's length away.
- Use even, diffuse frontal lighting. Avoid backlighting, flicker, strong
  shadows, and a bright window behind the signer.
- Keep the background uncluttered and clothing distinct from it. Avoid gloves
  and reflective hand accessories.
- Start and stop each prompt with hands at rest. Sign naturally at a consistent
  pace while facing the camera. Leave about one second between prompts.
- Use conversation vocabulary and do not activate spell-only mode or personal
  sign memory. Replay uses the saved alphabet setting but those other request
  options and personal-sign embeddings are not stored in the diagnostic cache.
- Record device, browser and configured camera frame rate in the metadata CSV.
  The manifest separately computes observed frame rate from saved timestamps.

## Prompt script

The gloss list is the exact reference token sequence for scoring. Punctuation
and spoken paraphrase are not scored. All glosses appear in
`models/vocab_conversation.json`.

| # | Prompt to convey | Reference glosses |
|---:|---|---|
| 1 | I want to eat rice. | ฉัน อยาก กิน ข้าว |
| 2 | Today I drink coffee. | วันนี้ ฉัน ดื่ม กาแฟ |
| 3 | Tomorrow we go to school. | พรุ่งนี้ เรา ไป โรงเรียน |
| 4 | What food do you like? | คุณ ชอบ อาหาร อะไร |
| 5 | I am hungry; I want bread. | ฉัน หิว อยาก ขนมปัง |
| 6 | My family is happy. | ครอบครัว ฉัน มีความสุข |
| 7 | I want to go to the market. | ฉัน อยาก ไป ตลาด |
| 8 | Where is the hospital? | โรงพยาบาล อยู่ ที่ไหน |
| 9 | I will meet my friend tomorrow. | พรุ่งนี้ ฉัน เจอ เพื่อน |
| 10 | Please help me. | ช่วย ฉัน ด้วย |
| 11 | I am tired and want to rest. | ฉัน เหนื่อย อยาก พัก |
| 12 | The weather is cold today. | วันนี้ หนาว |
| 13 | I like to read a book. | ฉัน ชอบ อ่าน หนังสือ |
| 14 | Where is the bathroom? | ห้องน้ำ อยู่ ที่ไหน |
| 15 | I am going to eat fruit. | ฉัน กิน ผลไม้ |
| 16 | Thank you; goodbye. | ขอบคุณ ลาก่อน |
| 17 | My younger sister is at home. | น้องสาว อยู่บ้าน |
| 18 | I want to buy water. | ฉัน อยาก ซื้อ น้ำ |
| 19 | What time will we meet? | เรา เจอ เมื่อไร |
| 20 | I feel good today. | วันนี้ ฉัน รู้สึก ดี |

Keep this reference fixed before collection. Use the 1-based script index in
the metadata for each capture.

## Local files and metadata

The sign-api diagnostic writer stores gzip JSON under
`cache/live_requests/*.json.gz`. Confirmed record fields are `time`, `width`,
`height`, `vocab`, `alphabet`, `n_user_signs`, `frames`, and `result`. Each
frame contains `t` in milliseconds and browser landmark fields `pose`, `lh`,
`rh`, `face`, and `bs`. The writer does not save a request ID, signer, session,
device or browser. Tools use the exact cache filename as request ID and require
a separate metadata CSV; they never infer those fields from timestamps.

Copy `capture_metadata.template.csv` to a local file and add one row per cache
filename. Required columns:

`request_id,session_id,signer_id,script_index,split,device,browser,advertised_fps`

Use `tune` or `eval` for `split`, the exact cache filename (including
`.json.gz`) for `request_id`, and pseudonymous signer IDs. An optional
`offline_video` column may contain a checkout-relative path to the exact
consented source video for capture-gap analysis. Keep identifying information
out of this CSV. The CSV, cache, local videos and generated results are local
only.

## Analysis boundaries

Tune only on `tune` sessions. A candidate passes only if every paired
tune-session WER delta versus baseline is at most `0.003`, mean improvement is
greater than `0.002`, and accepted precision is at least `0.90`. Freeze the
selection, then evaluate once on the separate session and signer marked `eval`.
Report every session. Eval data and Eval-Test must never influence selection.

For capture-gap analysis, retain a consented local source video only when the
participant agrees. Extract offline landmarks from that exact video using the
production MediaPipe and pipeline path. Compare paired request IDs. Do not
commit video or offline landmarks. Synthetic unit tests access no camera,
video, cache or model.
