# face-model — signer facial emotion (ThaiSLM v9, shipped unchanged in v10)

> v10: the face v2 candidates (more neutral signing, emotion-word signing, mouthing mask, a domain input) were evaluated with signer-level cross-validation and none was better on every metric, so this model (v1) stays — `artifacts/reports/face_model_v2.json`, `result_reporting/result_v10.md` §5.

What the signer **feels** while signing, from the face — separate from the sign model (`models/model-v9`) and loaded by
`SignPipeline` from `models/face-model` (or `THAISLM_FACE_MODEL`). Code: `modules/face_model.py`, training: `scripts/face_model.py`.

| | |
|---|---|
| input | MediaPipe Holistic face: 52 blendshapes + 68 face points per frame — the same extractor as the sign pipeline (browser and server), so live landmarks work without video |
| features | per blendshape: mean, 90th percentile, rise above the stretch's own 10th percentile (relative to this person's resting face), spread; head yaw / pitch / roll amplitude (209) |
| model | standardise → multinomial logistic regression (7 classes) + temperature — every decision splits into per-feature contributions, which become the Thai explanation |
| classes | neutral · happy · sad · angry · fear · disgust · surprise → valence / arousal (circumplex) + intensity |
| fusion | `fuse_affect`: face (main evidence) + emotion words read with confidence (รัก, เสียใจ, โกรธ …) + signing energy (hand speed → arousal); grammatical face (question brows, negation head shake — `modules/face.py`) is reported apart and never read as emotion |
| training data | RAVDESS video speech (24 actors × 60 clips, acted emotions, CC BY-NC-SA 4.0 — research / non-commercial) + 720 neutral samples of dictionary signing (training-role signers, no emotion words) |
| files | `face_emotion.npz` (weights, NumPy only), `model_card.json` (data, CV metrics, confusion matrix) |

Evaluation (held out, `model_card.json` / `artifacts/reports/face_model_signs.json`): see `result_reporting/result_v9.md` §5.
Rebuild: `python main.py face-data ravdess` (features only are kept in `cloud_s3/face-data/`) → `python main.py face-model train` →
`python main.py face-model signs`.
