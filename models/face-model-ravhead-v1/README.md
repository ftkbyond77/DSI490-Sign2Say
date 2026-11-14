# face-model-ravhead-v1

Face branch v2: facial expression from the face picture (live label on the camera + the voice's emotion).
Used by `modules/face_fer.py` via `THAISLM_FACE_FER` (docker-compose default: this folder).

* members: `face_fer.onnx` (v2-ferplus, 48 px), `enet_b2_8_feat.onnx` (HSEmotion B2, 96 px, logits + 1408-d features)
* `rav_head.npz`: RAVDESS-trained head over a 2 s window (+ neutral dictionary signing), neutral prior 0.5 (`config.json` → `head.prior_shift`)
* RAVDESS 24 actors, leave-actors-out: clips macro-F1 0.723 (blendshape face model 0.591, previous ensemble 0.456)
* team signer clips: false emotion on neutral frames 3.9% (previous ensemble 2.0%)

Details: `model_card.json`, `Face-Lab/face-report-v2.pdf`. Research / education use only (RAVDESS CC BY-NC-SA 4.0, AffectNet).
