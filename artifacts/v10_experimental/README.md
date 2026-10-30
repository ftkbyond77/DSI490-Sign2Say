# v10x — the v10 recognition candidate (opt-in, NOT shipped)

v10x = v9's networks + **extractor-averaged exemplars** (each bank clip that also exists as RTMW-X landmarks is embedded in both views
and its entry is the normalised mean) + the decision layer re-tuned on tune data for that bank. It was better than v9 on every tune
set, but **failed the release gate on the held-out test sets** in both tuning rounds, so `models/model-v10` ships v9's recognition and
v10x stays here as an option (prompt rule: a part that makes some metric worse and cannot be fixed without looking at test does not ship).

| file | what |
|---|---|
| `bank.npz` | production bank: 13,282 / 19,656 entries view-averaged (`view_averaged` flag per entry) — `python main.py decision views` |
| `eval/bank_eval.npz` | the same for the train-only bank used to evaluate the test sets |
| `spotting.json`, `segment_calibration.json`, `class_prior.npz` | re-tuned decision layer (round 2, release-aligned rules): landmark-encoder weight 0.4, hand weight 1.2, merge 0.4, split 0.6, v9's re-ranker, no isotonic, τ accept 0.88 — `python main.py decision tune --start v9 --significant --write` |
| `vocab_conversation.json` | = model-v10 (the harness reads the vocabulary from the version folder) |
| `run1/` | round 1 (original tune rules) settings + its tune log — test results in `artifacts/reports/decision_v10_run1.json`, `gate_v10_run1.json` |

The landmark encoder and its bank are v9's (`models/model-v9/lm_bank.npz`, `onnx/lmenc.onnx`).

* Evaluate: `python main.py decision report --versions v9,v10x` · gate: `python main.py decision gate --new v10x`
  (`artifacts/reports/gate_v10x.json`; round-2 tune log `decision_tune_v10.json`; round 1 = `decision_v10_run1.json` + `gate_v10_run1.json`).
* Use in an app: copy `models/model-v10` to a new folder, overwrite `bank.npz`, `spotting.json`, `segment_calibration.json`,
  `class_prior.npz` with these, and point `THAISLM_MODELS_DIR` at it.
* Results and why it failed: `result_reporting/result_v10.md` §3.
