"""Round 8 data pack for Colab: every RAVDESS 24-actor face crop (cache/ravdess24, 5 fps, 96x96 gray) + the neutral dictionary-signing
crops (cache/sign_neutral.npz, smiling clips left out like rav_head.sign_neutral(max_happy=0.2)), JPEG q95 to keep the upload small.
  python pack_rav_ft.py → cache/rav_ft_pack.npz   (labels in data.CLASSES order; signing crops: actor = 0, label = neutral)"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

H = Path(__file__).resolve().parent; ROOT = H.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(H))
from modules.face_fer import softmax   # noqa: E402
from compare24 import EMO              # noqa: E402
import rav_head as R                   # noqa: E402

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
MAX_HAPPY = 0.2


def main():
    jpg, label, actor, clip, inten, q, t = [], [], [], [], [], [], []
    for f in sorted(R.DATA.glob("actor_*.npz")):
        z = dict(np.load(f)); a = int(z["actor"])                     # dict: each array decompressed once
        for j in range(len(z["crop_q"])):
            ci = int(z["crop_clip"][j])
            jpg.append(cv2.imencode(".jpg", z["crop_g96"][j], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
            label.append(CLASSES.index(EMO[int(z["emotion"][ci])])); actor.append(a); clip.append(a * 100 + ci)
            inten.append(int(z["intensity"][ci])); q.append(float(z["crop_q"][j])); t.append(float(z["crop_t"][j]))
        print(f.name, flush=True)
    s = dict(np.load(H / "cache" / "sign_neutral.npz"))
    bias = np.array([0, 0, 0, 0, 0, 0, 0.75])
    hap = ((softmax(s["v2"] / 0.7 + bias) + softmax(s["b2"][:, [R.HSE.index(c) for c in CLASSES]] + bias)) / 2)[:, CLASSES.index("happy")]
    kept = 0
    for c in np.unique(s["clip"]):
        idx = np.where((s["clip"] == c) & (s["q"] >= 0.25))[0]
        if not len(idx) or hap[idx].mean() >= MAX_HAPPY:
            continue
        for j in idx:
            jpg.append(cv2.imencode(".jpg", s["g96"][j], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
            label.append(CLASSES.index("neutral")); actor.append(0); clip.append(10000 + int(c)); inten.append(0); q.append(float(s["q"][j])); t.append(float(s["t"][j]))
            kept += 1
    off = np.r_[0, np.cumsum([len(b) for b in jpg])]
    np.savez_compressed(H / "cache" / "rav_ft_pack.npz", jpg=np.frombuffer(b"".join(jpg), np.uint8), off=off, label=np.array(label), actor=np.array(actor),
                        clip=np.array(clip), intensity=np.array(inten), q=np.array(q, np.float32), t=np.array(t, np.float32), classes=np.array(CLASSES))
    print(f"{int((np.array(actor) > 0).sum())} RAVDESS crops + {kept} neutral signing crops → cache/rav_ft_pack.npz ({off[-1] / 1e6:.0f} MB jpeg)")


if __name__ == "__main__":
    main()
