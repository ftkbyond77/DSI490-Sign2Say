# Decoder tuning: real Boom + TSL51 experiments

Open `decoder_tuning.ipynb`. It calls `decoder_experiment.py` and `tsl51_experiment.py`.

## Boom

Nine executable configurations × six real videos, plus one explicitly blocked CSLS case. Cases are baseline, activation threshold, minimum duration, exact rescoring, two post-processing thresholds, grammar weight, stride, and the proposed combined settings **without CSLS**. Production weights, bank and spotting configuration are not edited. Runtime settings are copied for each case.

Landmark extraction is cached with clip/model/extractor-code/version fingerprints. Compare inference timings only; these are single measurements and concurrent work can affect them. Corpus WER uses identical newmm tokenization for filename reference sentences and concatenated retrieval words. It is not a verified temporal gloss evaluation.

## TSL51

The run processes all 126 `tagger_train`/tune sentences, comparing baseline with combined_without_csls. Uses `artifacts/eval/bank_eval.npz` and the same standardization/canonicalization as `pitty/scripts/sentence_eval.py`. Speed=1, no hold-cut, symbolic disabled, no fingerspelling or LLM. It never opens the 126 test pose files.

WER is computed by the actual `modules.decode.wer` function and cross-checked against edit-operation counts. Presence Recall@1/5 follows pitty's definition. Temporal F1 and accepted-word precision are unavailable without verified timed ground truth; automatic alignment spans are not treated as human labels.

## Outputs and reruns

- `outputs/latest.json`: latest completed Boom run.
- `outputs/latest_tsl51.json`: latest completed TSL51 tune run.
- Each run has raw outputs, effective configs, provenance, metrics and a report.
- `progress.csv` on TSL51 contains partial progress, never advertised as completed results.
- Notebook `RUN` / `RUN_TSL51` are false to display saved results. Set true deliberately to rerun.
- No automatic promotion to production. After selecting on development data, use a separate held-out evaluation.

## Corrections to the proposed plan

`freq_gamma` boosts frequent words in this repository. `alignment_score_cut` is bank-building metadata, not a runtime threshold. `csls_alpha` needs a hub vector; `models/hub.npy` is absent. `p_sign_min` acts jointly with length/nullness rules. `exact_rescore` may improve or harm accuracy and adds encoder work. These points are reflected in the notebook rather than assuming every suggested change will help.
