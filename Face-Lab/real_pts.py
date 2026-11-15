"""The 68 face points (pixels) of the real_check frames (same clips, same 0.5 s sampling as real_check.harvest) → cache/real_pts.npz,
row-aligned with cache/real_faces.npz (checked on clip + time).   python real_pts.py"""
from __future__ import annotations
import glob, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Face-Lab"))
import real_check   # noqa: E402  (sets MP_HOLISTIC_MODEL)


def main():
    import cv2
    from modules.mediapipe_pose import Holistic
    S = np.load(real_check.CACHE, allow_pickle=True)
    clips = sorted(glob.glob(str(ROOT / "data_test" / "team-drive" / "Eval-Test" / "*" / "*.*")))
    P, C, T = [], [], []
    for ci, p in enumerate(clips):
        cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 30; hol = Holistic(); i = 0; last = -9
        while True:
            ok, fr = cap.read()
            if not ok: break
            t = i / fps; i += 1
            if t - last < 0.5: continue
            if fr.shape[0] > 720: fr = cv2.resize(fr, (fr.shape[1] * 720 // fr.shape[0], 720), interpolation=cv2.INTER_AREA)
            _, _, _, face, _ = hol.frame(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), t * 1000.0)
            if np.abs(face).sum() == 0: continue
            last = t
            P.append(np.asarray(face, np.float64)[:, :2] * [fr.shape[1], fr.shape[0]]); C.append(ci); T.append(t)
        cap.release(); hol.close(); print(ci, flush=True)
    C, T = np.array(C), np.array(T)
    ok = len(C) == len(S["clip"]) and (C == S["clip"]).all() and np.allclose(T, S["t"], atol=1e-3)
    print("aligned with real_faces:", ok, len(C), len(S["clip"]))
    np.savez_compressed(ROOT / "Face-Lab" / "cache" / "real_pts.npz", pts=np.stack(P).astype(np.float32), clip=C, t=T, aligned=ok)


if __name__ == "__main__":
    main()
