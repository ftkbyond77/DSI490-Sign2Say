# PoC/yolo — hand proposal detector (research, lab4-handExtractor.ipynb)

| | |
|---|---|
| Model | Ultralytics **YOLO11n-pose** (`weights/yolo11n-pose.pt`, official release v8.3.0), fine-tuned for hand boxes → `weights/hand_yolo11n.onnx` |
| Data | Ultralytics **hand-keypoints** (18,776 train / 7,992 val, 21 keypoints) — https://github.com/ultralytics/assets/releases/download/v0.0.0/hand-keypoints.zip → `data/` (git-ignored) |
| Training | CPU, imgsz 320, 25 % of train, 5 epochs (a quick PoC, not a tuned model); only the box is used downstream |
| Not used for training | data_test/, iphone_test/, any ThaiSLM clip |
| License | Ultralytics = AGPL-3.0 — check before shipping |
