# v7.2 audit experiment

Entry point: `decoder_tuning.ipynb`, section A–C. All computations call real project functions. No retraining or production configuration edits. Existing modified files in `modules/` are not reset or committed.

- `audit_experiments.py --grammar-audit`: reconstruct source text and perform 60-point leave-role-template-out grid on captured real tune visual candidates. Includes zero-weight controls. This is explicitly **offline CV**, not 60 fresh encoder runs.
- `audit_experiments.py --bank-audit`: original ONNX encoder on automatically aligned tune spans; metadata and cosine comparisons in eval and production banks. Production contains sentence-derived exemplars and is diagnostic only.
- `full_audit_run.py --run --dataset tune [--speed 2]`: full pipeline, seven prespecified cases. Exact input-keyed neural/decoder cache shared across cases of the same clip; postprocess, grammar, repeat collapse and composer run again. Not a latency benchmark.
- `full_audit_run.py --run --dataset boom`: real extracted pose cache from six source videos, separate results and original filename ground truth; Thai newmm text tokens are not human sign boundaries.
- `full_audit_run.py --run --dataset test --final-test`: explicit final-only path, with one-run lock. Do not remove lock to retune. User has previously reported test outcomes, so this is a frozen reproduction on a previously evaluated test, not untouched validation.
- `audit_report.py`: reports, source video/pose paths and human review queue.

## Baselines and controls

`grammar_off`: visual-only grammar control. `grammar_original`: current v7.2 grammar (known text-label leakage). `grammar_clean_selected`: selected tune-CV settings with only tune-trained grammar, held-out role-template during tune. Filter cases each retain original grammar to isolate filter effects. All models and banks hashed; evaluation bank used for TSL51, production bank for Boom as in previous experiment. No cross-dataset metric pooling.

## Interpretation limits

`min_hand_frac` is an opt-in experimental hook in this runner, **not a production v8 port**. It uses confidence of standardised hand landmarks, potentially including interpolation; no claim that it measures raw MediaPipe observations. A genuine mouth-level sign may lose hands through occlusion. Lowering `tau_accept` affects filtering and grammar eligibility as well as visible text.

Aligned word spans are automatic, not verified temporal ground truth. Segment F1, time-aligned accepted correctness and physical mismatch claims require annotations. Ordered Levenshtein accepted correctness is reported separately. A wrong nearest neighbour can arise from segmentation, a variant, detector error or label noise; no bank item is removed automatically.

Grammar source reconstruction exactly matches current `models/grammar.json` from all 252 source recordings / 62 unique texts. This identifies test reference text leakage; the model contains no source IDs, so the historical recording list is reconstructed, not certified from a build log.
