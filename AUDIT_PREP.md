# Prep audit and reversible bank experiments

Run from the repository with NumPy installed (CPU only):

```bash
python scripts/audit_bank.py --eda ../eda --out artifacts/reports/prep-audit
python -m unittest discover -s tests -p 'test_audit_bank.py' -v
```

Outputs: `report.md`, a searchable `report.html`, `summary.json` (input fingerprints),
`bank_audit.jsonl` (one record per bank row), `word_coverage.json` (all conversation
words) and `review.json` (200 prioritised examples). No source bank is changed.

The audit joins exact metadata UIDs, explicit YouTube IDs, and normalised raw
paths. Ambiguous joins are not used. Existing EDA is sparse whole-video sampling
using separate hand/face detectors; it is not production Holistic span quality.
Missing quality/metadata and unverified signer labels must not be interpreted as
bad clips or verified distinct people. Word counts refer to instances, not quality.

Actual duplicate groups come from EDA SHA-256 results. When local copies exist,
the audit verifies their hashes and links them for review. Different concepts on
identical videos are review candidates, not automatic proof of incorrect labels.

## Filter an experiment after review

Edit a copy of `review.json`: set an item's action to `exclude` and give a reason.
`pending` and `keep` leave rows unchanged. There is no automatic label merge or
re-embedding. Then use a new, empty output directory:

```bash
python scripts/audit_bank.py --build-review artifacts/reports/prep-audit/review.json --out artifacts/experiments/review-01
```

Source fingerprints must still match. The command independently filters
`models/bank.npz` and `artifacts/eval/bank_eval.npz`, preserving concept indices,
embeddings and all row metadata. It never builds an evaluation bank by copying
the deployment bank. Existing evaluation splits are inherited, not independently
certified by this audit. `comparison.json` records lost concepts and removals.

To compare against held-out validation, use `SignPipeline(bank_file=...)` with
the original train-only bank and experimental `bank_eval.npz`, keeping encoder,
vocabulary, calibration and evaluation clips fixed. Report lost-vocabulary
coverage alongside recognition metrics. Do not deploy the experimental bank
before that comparison. Prepared poses/test clips are required for accuracy
comparison; the audit does not invent metrics when those files are missing.
