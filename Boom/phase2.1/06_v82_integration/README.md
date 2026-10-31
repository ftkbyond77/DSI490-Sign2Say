# v8.2 integration — waiting for actual v8.1

Open `v82_integration.ipynb`. `preflight.py` audits real Eval-Test files and training metadata without model inference or face recognition. There are no copied team metrics.

Current checkout: v7.2. `scripts/decision.py`, `modules/handshape.py`, `models/hand_bank.npz` and `artifacts/eval/hand_bank_eval.npz` are missing. Remote main and pitty config files also report v7.2 at the inspected commits. Obtain the exact v8.1 checkout and artifacts first. Do not overwrite production models, manufacture a v8.1 backup from v7.2, or implement approximate v8.1 features from a narrative report.

`trial_plan.json` is an inactive experiment plan, **not a selected v8.2 configuration**. H must itself pass all tune gates; individual successes do not imply their combination passes. Baseline test inference is deferred to the final frozen evaluation batch. Test has been inspected in earlier work, so this is not an untouched holdout. Eval-Test is never used to select settings.

Leak checks use exact file hashes and declared signer metadata. Folder names are not proof of unseen people. Person-level separation remains unknown without verified signer IDs. Six Boom videos may already have been used for development; see the hash audit, not filenames alone.

No production change has been made, so rollback is unnecessary. An actual backup and reversible deployment command will be produced only after authentic v8.1 is available and the tuning decision has been made.
