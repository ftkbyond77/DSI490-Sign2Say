"""Harvest the face geometry of the RAVDESS clips for the brow-furrow cue: per sampled frame the 68 face points (as the web sends them)
and the 52 MediaPipe blendshapes, over the WHOLE clip (the first ~0.6 s of each clip is the actor's resting face = a per-clip baseline).
  python brow_harvest.py   -> cache/ravdess_brows.npz   (clip, actor, label, strong, t, rel = t / duration, face [N,68,2], bs [N,52])"""
from __future__ import annotations

import glob
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import CLASSES   # noqa: E402
from video_eval import EMO, RAV         # noqa: E402

OUT = ROOT / "Face-Lab" / "cache" / "ravdess_brows.npz"


def main(every_s=0.2):
    import cv2
    from modules.mediapipe_pose import Holistic
    rows = []
    for z in sorted(glob.glob(str(RAV / "Video_Speech_Actor_*.zip"))):
        with zipfile.ZipFile(z) as zf, tempfile.TemporaryDirectory() as tmp:
            for name in sorted(n for n in zf.namelist() if os.path.basename(n).startswith("02-")):
                parts = os.path.basename(name)[:-4].split("-")
                if parts[2] not in EMO:
                    continue
                p = zf.extract(name, tmp)
                cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 30; dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
                hol = Holistic(); i = 0; last = -9
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    t = i / fps; i += 1
                    if t - last < every_s:
                        continue
                    if fr.shape[0] > 720:
                        fr = cv2.resize(fr, (fr.shape[1] * 720 // fr.shape[0], 720), interpolation=cv2.INTER_AREA)
                    _, _, _, face, bs = hol.frame(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), t * 1000.0)
                    if np.abs(face).sum() == 0:
                        continue
                    last = t
                    rows.append(dict(clip=os.path.basename(name)[:-4], actor=int(parts[6]), label=CLASSES.index(EMO[parts[2]]), strong=parts[3] == "02",
                                     t=t, rel=t / dur, face=np.asarray(face, np.float32)[:, :2], bs=np.asarray(bs, np.float32)))
                cap.release(); hol.close(); os.remove(p)
        print(os.path.basename(z), len(rows), flush=True)
    np.savez_compressed(OUT, **{k: np.array([r[k] for r in rows]) for k in ("clip", "actor", "label", "strong", "t", "rel")},
                        face=np.stack([r["face"] for r in rows]), bs=np.stack([r["bs"] for r in rows]))


if __name__ == "__main__":
    main()
