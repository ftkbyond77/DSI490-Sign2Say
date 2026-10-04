"""Signer identity from faces (data preparation only — never used by the sign model).

Why: held-out evaluation needs held-out PEOPLE. Sources without a signer column (th_sl, YouTube channels — one person may run two
channels, one channel may show several people) get signer ids from face embeddings, and test videos (data_test) are checked
against every training signer so that no test person is also a training person.

face crop  = box around the 68 MediaPipe face points (modules/mediapipe_pose.py raw output) at a few frames of the clip
embedding  = DINOv2-base CLS of the crop (generic visual identity features; centred over the corpus before comparing)
"""
from __future__ import annotations

import numpy as np


def face_crops(video, raw, fracs=(0.3, 0.5, 0.7), size=(112, 140)):
    """RGB face crops at the given fractions of the clip (only frames where MediaPipe found a face)."""
    import cv2
    t, face, hw = raw["t"], raw["face"], raw["hw"]
    ok = np.abs(face).sum((1, 2)) > 0
    if not ok.any():
        return []
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    crops = []
    good = np.flatnonzero(ok)
    for f in fracs:
        i = good[min(len(good) - 1, int(f * len(good)))]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t[i] * fps)))
        ok_, fr = cap.read()
        if not ok_:
            continue
        H, W = fr.shape[:2]
        sx, sy = W / max(int(hw[1]), 1), H / max(int(hw[0]), 1)       # extraction may have used a crop / resize
        p = face[i] * np.array([hw[1] * sx, hw[0] * sy])
        x0, y0 = p.min(0); x1, y1 = p.max(0)
        w = max(x1 - x0, y1 - y0) * 1.25
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a0, b0 = int(max(0, cx - w / 2)), int(max(0, cy - w * 0.6))
        a1, b1 = int(min(W, cx + w / 2)), int(min(H, cy + w * 0.65))
        if a1 - a0 < 16 or b1 - b0 < 16:
            continue
        crops.append(cv2.cvtColor(cv2.resize(fr[b0:b1, a0:a1], size), cv2.COLOR_BGR2RGB))
    cap.release()
    return crops


def embed_faces(crops, bs=64, device="cuda"):
    """[N,768] L2-normalised DINOv2-base CLS embeddings."""
    import torch
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    net = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    if device == "cuda":
        net = net.half()
    E = np.zeros((len(crops), 768), np.float32)
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            x = proc(images=crops[i:i + bs], return_tensors="pt")["pixel_values"].to(device)
            x = x.half() if device == "cuda" else x
            E[i:i + bs] = net(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy()
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
