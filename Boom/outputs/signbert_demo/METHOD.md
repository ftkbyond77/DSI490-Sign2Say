# SignBERT-inspired without MANO

Reference: Hu et al., SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition, ICCV 2021. https://arxiv.org/abs/2110.05382

This is an independently written adaptation, NOT an official implementation or a faithful reproduction of the paper's benchmark results.

## Retained concepts

Hand-pose tokens, graph features, time/chirality embeddings, Transformer context, masked coordinate reconstruction pretraining, and supervised attention-pooled classification.

## Explicit differences

- **MANO decoder removed.** A two-layer MLP directly regresses 21 x/y joints per hand token. No mesh, hand-shape prior, camera projection, or MANO regularization is used. This removes a central contribution of the original paper.
- MediaPipe-family ONNX hand/pose models from OpenCV Zoo instead of MMPose. OpenCV DNN CPU runs extraction without MediaPipe runtime, OpenGL or Metal. Observed/not-observed masks replace calibrated per-joint confidence; missing joints never become reconstruction targets. Hand slots are associated with detected body wrists when both wrists are available.
- Shoulder-centered/scaled coordinates; no hand crop to 256×256.
- Compact 128-dimensional encoder, 4 heads, 3 layers, a simple two-layer hand graph and mean joint pooling. Visibility is an additional input channel. This graph/pooling is not a replication of the original implementation.
- Source video is processed at approximately 15 fps, followed by 64 uniform frames in the feature cache, 32 deterministic temporal samples for training/inference. No random temporal training sampling or RGB/pose late fusion.
- Half of observed hand tokens are selected for equal-probability joint masking/disturbance, frame masking, or identity. Reconstruction loss averages L1 error on selected observed joints.
- Pretraining uses only the selected local clips; no large external pretraining corpus or pretrained SignBERT checkpoint.
- CPU training with Adam. This is a small engineering demo rather than a comparative research result.

## Evaluation boundary

All 20 selected source recordings are training-only. No independent validation/test set exists. Synthetic tensors used for integration tests are saved separately under work/, never as demo training data or demo weights. Training-set Top-1/Top-5 must not be represented as signer-independent accuracy.
