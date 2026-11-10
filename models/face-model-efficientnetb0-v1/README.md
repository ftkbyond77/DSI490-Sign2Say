# face-model-efficientnetb0-v1 — facial expression (FER2013, 7 classes)

An **expression category** from a face picture (angry, disgust, fear, happy, sad, surprise, neutral) — not a diagnosis of a feeling.
Independent late-fusion branch: it reads only the face crop, never changes sign landmarks / sign output, and only feeds the speech layer
(`modules/face_fer.py`: crop -> model -> quality gate -> unknown -> EMA + hysteresis -> `speech_affect`).

| | |
|---|---|
| Model | EfficientNet-B0 (ImageNet pretrained, fine-tuned), 4.0 M params, 15 MB ONNX |
| Input | `face` float32 [N,1,48,48] gray 0..255 (resize / normalise are inside the graph, net res 96) |
| Output | `logits` [N,7] (order in `config.json`); the app applies temperature 1.0 and a deployment prior (`prior_shift`: neutral +0.6, signers rest on a neutral face) |
| FER2013 test (7178) | acc 0.681, Macro-F1 0.663, UAR 0.670 (with the prior; flip averaging is not used); ECE 0.032 |
| Clean test (no train duplicates, 6569) | acc 0.659, Macro-F1 0.625 |
| CPU latency (ORT, 2 threads, batch 1) | ~4-5 ms |
| Unknown rule | confidence < 0.48, margin < 0.10 or quality < 0.25 -> `unknown` |

Files: `face_fer.onnx` (served), `face_fer.pt` (EMA weights), `config.json` (classes, crop geometry, thresholds, smoother), `model_card.json` (metrics).
Run: `docker compose up` mounts this folder into sign-api (`FACE_FER_DIR`). Full numbers + comparison with MobileNetV3-Large: `Face-Lab/face-report-v1.pdf`.
