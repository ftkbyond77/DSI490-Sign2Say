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

The script locates the checkout root by searching its parent directories for
`.git` and `modules/`, so the default input resolves to
`<checkout>/cache/live_requests/`, not `Boom/cache/...`. A missing cache folder
or a folder with no `*.json.gz` files is a fatal, explicit error rather than an
empty successful report.

The browser frame schema has no separate per-hand confidence values. Hand
point scores are therefore treated as binary, using the same nonzero x/y
presence rule used when `from_browser` builds the production schema. The
diagnostic reports two distinct rates:

- `*_hand_visible_rate` counts a frame when more than 30% of the 21 points are
  present (at least 7/21). This is a partial-detection diagnostic cue, not the
  production `fill_hand_gaps` hand-presence mask.
- `*_hand_full_rate` requires all 21 points. This matches the current
  `from_mp75` confidence construction, which marks a hand's points present only
  when all 21 source points are present; `fill_hand_gaps` then applies its
  mean-score `> 0.3` test to that mask. Please confirm this interpretation
  against `modules/schema.py` and `modules/mediapipe_pose.py` before relying on
  the diagnostic as a pipeline metric.

Body scores use the visibility field at pose index 3. To mirror
`modules/mediapipe_pose.py::from_browser`, a point with zero visibility but
nonzero x/y is treated as visibility 1. The diagnostic's
`cleared_by_present_rate` is the fraction where mean effective nose/left
shoulder/right shoulder confidence is `<= 0.5`, matching the negation of
`present = sc[:, [nose, left_shoulder, right_shoulder]].mean(1) > 0.5` in
`Boom/phase2.1combined/test_engine.py`. This is a schema-level reproduction of
that present gate; the tool does not claim to observe a later model/pipeline
output mask. The scores are binary as they are after `to_mp75`/`from_mp75`:
pose visibility must be `> 0.5` and x/y nonzero, except the browser adapter's
zero-visibility/nonzero-coordinate fallback, which sets visibility to 1.

Hand gaps use the all-21-points `hand_full_rate` mask, matching the current
production conversion's all-points hand confidence mask before
`fill_hand_gaps` applies mean-score `> 0.3`. The partial `hand_visible_rate`
(at least 7/21) is not used to determine pipeline-like gap lengths. Gap
duration is estimated as missing-frame timestamp span plus one median frame
period; for a leading gap it is capped at one period, avoiding counting the
interval before the first captured frame. A gap is `fillable` only when it has
a full-hand frame on both sides and its estimated duration is at most
`hand_gap_s`; edge gaps are always `not_fillable`, even when shorter than the
threshold. The CSV provides fillable, edge and too-long counts separately.
Millisecond histograms use 100 ms bins; frame histograms use exact run lengths.
The long-gap fraction counts durations strictly greater than `hand_gap_s`
(default `0.5` s); the companion fraction includes durations at or below the
threshold, including unfillable edge gaps.

The current cache writer does not save per-frame handedness. If labels are
present, the diagnostic only accepts `handedness` as either a frame-level
string or a top-level list aligned one-for-one with `frames`; accepted labels
are `left`/`right` or `L`/`R`. It counts changes only between adjacent frames
where both labels exist. Without labels it reports `ไม่มีข้อมูล` and does not
infer handedness.

The sign-api cache writer now stores the existing `result.mirrored_to_right_hand`
flip boolean and a `result.hand_canonicalization` summary containing dominant
hand, dominant-hand activity share, the `0.62` decision margin, decision rule
and reason. The summary is returned internally by the same canonicalization
calculation that determines the existing flip; it does not change predictions
or the API response. Cache records also store top-level `hand_gap_s`, `n_frames`
and timestamp-derived `fps_actual`. No additional landmark coordinates or face
data are added. The diagnostic prefers `hand_canonicalization.flipped` and
falls back to legacy explicit boolean result keys when absent. Old cache files
remain readable and report flip metadata as `ไม่มีข้อมูล`; other old fields
remain derived from their recorded frames.

To check a newly recorded request locally, perform one sentence in the web app
with request saving enabled, then run:

```sh
Boom/venv-sign2say/bin/python Boom/phase2.2/live_hand_body_diag.py
```

Open `cache/phase2.2/hand_body_diag/summary.md` and confirm the request's
`Flip data` column says `มีข้อมูล` and its `Flipped` value is `True` or `False`.
The per-request CSV additionally shows the decision reason/share and recorded
`hand_gap_s`, frame count and estimated FPS.

`scripts/replay_live.py` is not a faithful replay of saved request settings:
its `--vocab` and `--alphabet` values come from command-line options (defaults
`conversation` and `en`), while the diagnostic cache stores the original
`vocab` and `alphabet`. Set replay options to match those saved values before
comparing replay outputs with the original result.

Synthetic tests:

```sh
python -m unittest discover -s Boom/phase2.2/tests -v
```

The tests use synthetic records and temporary files only. They do not read
real caches or media, load models, or run inference.
