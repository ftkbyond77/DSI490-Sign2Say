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

Synthetic tests:

```sh
python -m unittest discover -s Boom/phase2.2/tests -v
```

The tests use synthetic records and temporary files only. They do not read
real caches or media, load models, or run inference.
