# Phase 2.2 live tuning scaffold

All paths accepted by these tools are relative to the repository checkout. The
default input is `cache/live_requests/`; outputs default to ignored
`cache/phase2.2/`. No script is run automatically by tests.

1. Follow `SESSION_PROTOCOL.md`; copy `capture_metadata.template.csv` to the
   local-only `Boom/phase2.2/capture_metadata.csv` and fill every capture row.
2. Build the manifest:

   ```sh
   python Boom/phase2.2/live_manifest.py
   ```

3. Explicitly replay the local cache through the production pipeline. Results
   append by `(request_id, case)` and can safely resume:

   ```sh
   python Boom/phase2.2/live_replay.py
   ```

4. Aggregate per-utterance, per-session and corpus metrics, paired bootstrap
   intervals, and tune gate checks:

   ```sh
   python Boom/phase2.2/live_metrics.py
   ```

5. To measure the capture/offline difference, add checkout-relative,
   consented source video paths in the optional `offline_video` column. This
   command reads those videos and invokes production landmark extraction, so
   it is only for a deliberate local run:

   ```sh
   python Boom/phase2.2/capture_gap.py
   ```

The replay uses the saved browser timestamps through `SignPipeline`'s browser
landmark conversion and standardisation path. This path performs production
timestamp-aware resampling. A candidate `tau_accept` is assigned directly to
`pipeline.cal["tau_accept"]` after construction, avoiding the production
constructor's `max(calibration_tau, spotting.accept_min)` behavior.
`hand_gap_s` is assigned to `pipeline.spot_all["hand_gap_s"]`, which is the
exact setting read by `SignPipeline.standardise` before production calls
`fill_hand_gaps`. The current `models/spotting.json` baseline is `0.5` seconds;
the plan compares it to `0.3`, `0.8`, and `1.0`. Every replay row records its effective value and left/right
post-fill frame coverage and estimated imputed-frame fraction. An output frame
is classified as imputed when its nearest original timestamped frame lacked
that hand but the standardized output contains it; reported sides are mapped
back to the signer's physical left/right even when canonicalisation mirrored
the frame.

The cache writer does not save request ID, signer/session identity, device,
browser, advertised camera fps or source video path. These values are therefore
required in the separate metadata CSV and are never guessed. Actual fps is
computed from adjacent cache frame timestamps. Cache rows must use the
conversation vocabulary because the fixed script glosses reference that
vocabulary.

The diagnostic writer also does not persist all inference request options or
personal-sign embeddings. For controlled comparisons this replay explicitly
uses LLM off, face analysis off, and no personal sign memory; it retains the
saved alphabet setting, assumes spell-only mode was off, and refuses a capture
with nonzero `n_user_signs`.
Therefore the replay is production-pipeline aligned for landmarks and spotting
configuration, but cannot exactly reconstruct a historical request that used
the omitted options. Its latency includes browser-landmark conversion plus
`run_landmarks`, not browser/network overhead or video extraction.

## Read-only hand/body diagnostics

Run `python Boom/phase2.2/live_hand_body_diag.py` to inspect gzip JSON captures
without invoking the model or printing/writing landmark coordinates. It writes
one metrics-only CSV per request and `summary.md` under
`cache/phase2.2/hand_body_diag/`. It validates the root `frames` list, numeric
strictly increasing `t` timestamps, `pose` (33 points with visibility at index
3), and `lh`/`rh` (21 points with normalized x/y). Empty pose/hand lists mean
that detector channel was absent for the frame. A nonempty array with another
shape fails before metrics are produced.

The browser frame schema has no separate per-hand confidence values. Hand
point scores are therefore treated as binary, using the same nonzero x/y
presence rule used when `from_browser` builds the production schema; the hand
rate is `count(present points) / 21 > 0.3`, matching `fill_hand_gaps`. Body
scores use the pose visibility field and the nose/shoulder clearing rule from
the Phase 2.1 test engine: average nose/left-shoulder/right-shoulder visibility
must be strictly greater than `0.5`; otherwise the frame is counted as cleared.

Hand gaps are contiguous runs where that hand's 21-point mean presence score
does not exceed `0.3`. Gap milliseconds use the interval from the first missing
frame timestamp up to the next frame timestamp; leading/trailing gaps are
extended by the median observed frame interval. This estimates capture-time
duration for irregular frame timestamps. Millisecond histograms use 100 ms
bins; frame histograms use exact run lengths. The long-gap fraction counts
durations strictly greater than `hand_gap_s` (default `0.5` s); the companion
fraction includes durations equal to or shorter than the threshold.

The current cache writer schema does not include a handedness label. If labels
are available, the diagnostic only accepts `handedness` as either a frame-level
string or a top-level list aligned one-for-one with `frames`; accepted labels
are `left`/`right` or `L`/`R`. It counts changes only between adjacent frames
where both labels exist. Other fields, including `mirrored_to_right_hand`, are
not treated as handedness; absent labels are reported as `ไม่มีข้อมูล`, never
inferred.

Synthetic tests:

```sh
python -m unittest discover -s Boom/phase2.2/tests -v
```

The tests use synthetic records and temporary files only. They do not read
real caches or media, load models, or run inference.
