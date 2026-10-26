# Live hand/body dropout diagnosis — blocked on recordings

## Scope and actual state
- Branch: boom-live-diag, starting commit a0b4f76, same as local boom at inspection.
- cache/live_requests does not exist in this checkout. Directory search under DSI490 found no live_requests directory. No request was opened or replayed.
- The previously mentioned eight recordings, 7–12 fps and 30–67% visibility are supplied context, NOT measurements produced by this audit.
- Pipeline, models, config, phase2.2 and phase2.1combined/outputs were not modified. No TSL51 test or Eval-Test accessed.

## What the source code saves (not confirmed from real recordings)
| Item | Source evidence | Limitation |
|---|---|---|
| Timestamp | frames.t, milliseconds | Actual fps unavailable without recordings |
| Body | pose: 33 x [x,y,z,visibility] | Client substitutes 1 when visibility absent; from_browser also replaces some zero visibility with 1 when coordinates nonzero |
| Hands | lh/rh: 21 x [x,y,z], or empty | No per-landmark hand confidence; 21 points present does not establish tracking accuracy |
| Handedness | Left/right slots only | No handedness class score, persistent track ID or explicit handedness events; cannot count true swaps reliably |
| Face | 68 points, blendshapes | No RGB video, exposure or image brightness saved |
| Predictions | result.segments, sentence, timing | Predicted output is not ground truth; cannot label correct/incorrect from it |
| Personal bank | n_user_signs only | Embeddings are not saved; requests with personal signs cannot be exactly replayed from the record alone |
| Tagger | not saved by _save_request | begin/inside requires rerunning tagger; recording-time model hash/config are absent |

Sources: webapp/web/lib/holistic.ts pack; modules/mediapipe_pose.py from_browser; webapp/api/app.py _save_request; scripts/replay_live.py.

## Code facts that affect interpretation
1. models/spotting.json sets hand_gap_s=0.5. fill_hand_gaps uses mean confidence of 21 points >0.3 to mark a hand present. Only internal, bounded whole-hand gaps are filled when run length <= round(gap_s/median interval) (minimum one frame). A gap shorter than 0.5 seconds is NOT automatically filled: leading/trailing gaps, irregular timestamps and partial detection matter.
2. resample uses the minimum confidence of adjacent samples, but ALSO restores a single-sided point from the nearer sample if the nearest interpolation distance is <0.25. It is incorrect to describe this implementation as only minimum-confidence resampling.
3. Standard Schema present = mean(nose,left shoulder,right shoulder)>0.5; failing frames are zeroed before canonical_hands. Raw body rates do not directly equal post-harmonisation rates: template imputation and gap/resample processing happen first.
4. dominant_hand aggregates activity over the entire request: movement in shoulder-width units plus elevation, using visible hands. canonical_hands flips only if dominant side is L and share>=0.62. Fewer than two valid shoulder frames returns R,0.5. One flip boolean applies to the whole request; it does NOT toggle frame-by-frame. Decisions may differ across separate live requests.
5. There is no public pipeline/config switch for disabling canonical_hands in this checkout. The helper's margin argument exists but the pipeline calls the default directly. No no-flip replay was performed; it would require an explicitly isolated experimental wrapper, not an invented config flag.
6. No min_hand_frac consumer exists in current modules/pipeline.py or modules/decode.py. Setting post.min_hand_frac=0/.3 alone is a no-op, not evidence that a hand filter works or is safe. Do not copy a friend's filter into production for this diagnosis.
7. replay_live.py POSTs into the running service, can request LLM, and may cause diagnostic saves depending on server settings. It was NOT executed for this read-only audit.

## Requested quantitative results
| Measure | Result |
|---|---|
| Requests / raw fps / left-right hand rates | N/A — recordings absent |
| Body visibility / present-zeroed fraction | N/A — recordings absent |
| Gap histograms in frames and ms | N/A — recordings absent, no invented plots |
| Gaps >0.5s, short gaps actually filled | N/A — requires raw streams and matching preprocessing |
| Handedness switches | Not established; explicit tracking identities/scores not saved |
| Flips and prediction changes | N/A — no requests, no supported no-flip config |
| Gap overlap with begin/inside | N/A — tagger output must be derived, not saved |
| Rotation associated with dropout | N/A — requires observed endpoints; palm-normal proxy cannot establish what happened during an unobserved interval |
| Incorrect vs correct segment visibility | N/A — ground-truth words/times not established |
| uncertain/reject vs accepted visibility | Computable descriptively once recordings exist; accepted must not be treated as correct |
| Original vs min_hand_frac 0 / .3 | Not run; key unsupported in current pipeline |

## Diagnosis and next evidence
No data-supported ranking of (a) tracking reacquisition, (b) low fps, (c) handedness/canonicalisation or (d) lighting can be made yet. A long gap alone cannot distinguish occlusion from tracker reacquisition. Timestamp-derived sparse fps alone cannot prove rotation caused the dropout. Without RGB/exposure metadata, lighting cannot be measured retrospectively.

Obtain the existing *.json.gz recordings from the actual THAISLM_SAVE_REQUESTS directory or Docker volume, with human reference words and optional sign intervals. Run diagnose.py --input /absolute/path to inventory real fields and raw gaps. This script currently performs only raw data audit, not full replay/rotation inference. Then add isolated read-only replay using the exact recording model/config if available; flag unmatched versions and unavailable personal-bank state. Any rotation statistic must report its eligible bounded-gap denominator and ambiguous/edge-on cases. Hand swaps require extra evidence beyond a coordinate jump. Do not choose parameters from these diagnostic comparisons.

## Git hygiene
Root .gitignore already ignores cache/; git check-ignore confirms cache/live_requests/example.json.gz. No root edit needed. Local .gitignore excludes media, model arrays and plotting caches. Commit only files under Boom/live_diag; no input landmark streams or face images.
