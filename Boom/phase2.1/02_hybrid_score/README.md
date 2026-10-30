# Hybrid phase 2.1 — one scientific notebook

Open `hybrid_phase2_1.ipynb`. All experiment functions live in `experiment_analysis.py`; production inference calls `modules.pipeline.SignPipeline`, dictionary construction calls `modules.symbolic.build_dictionary`.

## Run

1. Run setup cells. The inference interpreter is always `Boom/venv-sign2say/bin/python`.
2. Inspect `ground_truth.json`: sentence text is the filename explicitly confirmed by the user. Initial gloss tokens use newmm and remain editable. A confirmed Thai sentence does not provide sign boundaries.
3. Run dictionary and coverage cells.
4. Leave `RUN_INFERENCE=True` for a new real six-video experiment. This extracts each video once, compares 0 and 0.30 on shared landmarks and stores raw outputs. Set False only to read a completed run.
5. Continue through metrics, paired bootstrap, coverage diagnostics and all six segment visualizations.

No training or model-weight modification occurs. Other held-out video folders are not opened.

## Measurements

- WER: identical newmm segmentation of filename sentence and concatenated retrieval output.
- sentence_WER: identical segmentation of filename sentence and composed pipeline sentence.
- proposed_gloss_WER: proposed reference sign tokens against per-segment words; review reference units/order before interpreting it as sign recognition WER.
- Temporal Segment F1 / Recall@1/5 / accepted precision require human timed segments in seconds. These remain N/A without them.
- Presence Recall is explicitly separated from temporal recall.
- p_correct remains the original calibrator output, not a verified Hybrid confidence.
- Rule agreement and weighted sum are not calibrated probabilities.
- Timing separates extraction/inference and excludes model initialization. One run is not a stable performance benchmark.

## Outputs

`boom_datatest/scientific/`: manifest.json, results.json, metrics.csv, segments.csv, top5.csv, comparison.csv, REPORT.md and figures.

Inference overwrites the current results after each complete clip; copy the whole scientific directory before preserving multiple runs. An incomplete run is rejected by the analysis loader.

Legacy `hybrid030_tsl51_test_1x.json` is not used: its original run did not establish an enabled symbolic stream. The new experiment does not claim a negative result based on that file.

## Verified during implementation

Notebook schema and Python code syntax validated. Real venv project imports, real video metadata, dictionary generation, TSL51 reference coverage and figure rendering checked. Six-video inference has not been run as part of this edit; the user starts it in the notebook. No scores or metrics have been fabricated.


## Native crash fix (2026-10-19)

MediaPipe 1.0.1 aborted in DrishtiMetalHelper / TensorsToDetectionsCalculator while creating Holistic. This experiment loads MediaPipe 1.1.0 from `runtime_packages/`, installed from the local wheel in `runtime_wheels/`. Anaconda and the shared venv installation are unchanged. `subprocess_environment()` passes this package path to child processes; CLI imports also select the experiment package directory. The original `.task` model is unchanged. Treat the package upgrade as experiment provenance when comparing older results.

The inference notebook cell calls `ex.launch_inference()`, streams combined stdout/stderr, and saves `boom_datatest/scientific/inference.log`. It reports the native error tail rather than hiding it behind CalledProcessError.


## Finger-state tracking

Notebook section 1.3 calls `handshape_tracking.py` for actual video frames. It saves anatomical L/R hand states, per-joint angles, raw hand arrays, event intervals, annotated H.264 video and timeline in `boom_datatest/handshape/<video stem>/`. The first sample (`ฉันแอบชอบคุณ`) was executed on all 388 real frames. This is an unvalidated geometry baseline, not a learned classifier. Missing/ambiguous evidence abstains rather than becoming a fist. Many frames remain unknown in the sample; do not equate successful execution with validated recognition accuracy. No retrieval scores are modified by this experiment.
