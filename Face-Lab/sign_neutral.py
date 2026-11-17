"""In-domain NEUTRAL faces for the RAVDESS-trained picture head (the same idea as the blendshape model's 720 neutral signing samples):
dictionary sign clips in data_test/jack_test (th_sl / TTRS / tsl51 / YouTube signers; citation signs, a calm face) — NOT the team's
Eval-Test clips, which stay the held-out signer check (real_check.py). Folders whose word may carry a feeling (รู้สึก, สบายดี) are left out.
Per clip at 5 fps: 48 / 96 px crops + quality → B2 features / logits + v2 logits (the same as rav_head.py) → cache/sign_neutral.npz
  python sign_neutral.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import FaceFER, crop48, crop_box, face_quality, hand_overlap   # noqa: E402
from rav_head import B2F, V2, _b2_run                                              # noqa: E402

SKIP = ("15_", "21_")
OUT = ROOT / "Face-Lab" / "cache" / "sign_neutral.npz"


def main(every_s=0.2):
    import cv2
    import onnxruntime as ort
    from modules.mediapipe_pose import Holistic
    clips = sorted(p for p in glob.glob(str(ROOT / "data_test" / "jack_test" / "*" / "*.mp4")) if not Path(p).parent.name.startswith(SKIP))
    G, G96, Q, T, C = [], [], [], [], []
    for ci, p in enumerate(clips):
        cap = cv2.VideoCapture(p); vf = cap.get(cv2.CAP_PROP_FPS) or 30; hol = Holistic(); i = 0; nxt = 0.0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            t = i / vf; i += 1
            if t + 1e-6 < nxt:
                continue
            nxt += every_s
            if fr.shape[0] > 720:
                fr = cv2.resize(fr, (fr.shape[1] * 720 // fr.shape[0], 720), interpolation=cv2.INTER_AREA)
            _, lh, rh, face, _ = hol.frame(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), t * 1000.0)
            if np.abs(face).sum() == 0:
                continue
            box = crop_box(face, fr.shape[1], fr.shape[0]); gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); g = crop48(gray, box)
            G.append(g); G96.append(crop48(gray, box, 96)); T.append(t); C.append(ci)
            Q.append(face_quality(g, box[2], hand_overlap(face, [lh[:, :2], rh[:, :2]])))
        cap.release(); hol.close()
    G, G96 = np.stack(G), np.stack(G96)
    so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.log_severity_level = 3
    b2 = ort.InferenceSession(str(B2F), so, providers=["CPUExecutionProvider"])
    feat, zb = _b2_run(b2, G96)
    np.savez_compressed(OUT, clip=np.array(C), t=np.array(T, np.float32), q=np.array(Q, np.float32), feat=feat.astype(np.float16), b2=zb, g96=G96,
                        v2=FaceFER(V2).logits(G).astype(np.float32), names=np.array([str(Path(c).relative_to(ROOT)) for c in clips]))
    print(f"{len(clips)} clips, {len(G)} crops → {OUT}")


if __name__ == "__main__":
    main()
