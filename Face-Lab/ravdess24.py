"""RAVDESS, all 24 actors, harvested ONCE for both face models so they are compared on the same frames under the same protocol as
scripts/face_model.py (the blendshape model, models/face-model):
  * every clip (modality 02, video only, speech) at 10 fps: 52 blendshapes + 68 face points (the production extractor, MediaPipe Holistic)
    = exactly what scripts/face_data.py stores for the blendshape model;
  * every 2nd of those frames (5 fps): the 48x48 and 96x96 gray face crops + quality, for the picture models (Face-Lab).
Labels from the file name; calm is kept (the blendshape model folds it into neutral). One actor zip (~0.55 GB, Zenodo 1188976,
CC BY-NC-SA 4.0, research use) at a time: download (several HTTP ranges in parallel) → harvest → zip deleted (only the local zips of
actors 1-4 in data_test/ravdess are kept). Output: cache/ravdess24/actor_XX.npz (no video kept).

  python ravdess24.py [--actors 1-24] [--keep-ahead 1]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.face_fer import crop48, crop_box, face_quality, hand_overlap   # noqa: E402

OUT = ROOT / "Face-Lab" / "cache" / "ravdess24"
TMP = ROOT / "cache" / "face_tmp"
LOCAL = ROOT / "data_test" / "ravdess"
URL = "https://zenodo.org/records/1188976/files/Video_Speech_Actor_{:02d}.zip?download=1"


def download(url, path, parts=6, tries=10):
    """Parallel HTTP-range download (Zenodo is slow per connection) into path; resumes each range after a drop."""
    import requests
    if path.exists():
        return path
    n = int(requests.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])
    tmp = Path(str(path) + ".part")
    with open(tmp, "wb") as f:
        f.truncate(n)
    bounds = [(i * n // parts, (i + 1) * n // parts - 1) for i in range(parts)]
    err = []

    def get(a, b):
        pos = a
        for k in range(tries):
            try:
                with requests.get(url, headers={"Range": f"bytes={pos}-{b}"}, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    if r.status_code != 206:
                        raise RuntimeError("server ignored the range")
                    with open(tmp, "r+b") as f:
                        f.seek(pos)
                        for ch in r.iter_content(1 << 20):
                            f.write(ch); pos += len(ch)
                if pos > b:
                    return
            except Exception as e:  # noqa: BLE001
                print(f"  {path.name} range {a}: retry {k + 1} ({type(e).__name__})", flush=True); time.sleep(5)
        err.append((a, b))

    th = [threading.Thread(target=get, args=ab) for ab in bounds]
    [t.start() for t in th]; [t.join() for t in th]
    if err:
        raise RuntimeError(f"download failed {path.name}: {err}")
    tmp.replace(path)
    return path


def harvest(zpath, actor, fps=10.0):
    import cv2
    from modules.mediapipe_pose import Holistic
    L = dict(bs=[], face=[], t=[], clip=[])          # 10 fps landmark track
    C = dict(g=[], g96=[], q=[], t=[], clip=[])      # 5 fps crops
    meta = dict(name=[], emotion=[], intensity=[], statement=[])
    with zipfile.ZipFile(zpath) as zf, tempfile.TemporaryDirectory() as tmp:
        names = sorted(n for n in zf.namelist() if os.path.basename(n).startswith("02-") and n.endswith(".mp4"))
        for n in names:
            p = zf.extract(n, tmp)
            cap = cv2.VideoCapture(p); vf = cap.get(cv2.CAP_PROP_FPS) or 30
            hol = Holistic(); i = 0; k = 0; nxt = 0.0; ci = len(meta["name"])
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                t = i / vf; i += 1
                if t + 1e-6 < nxt:
                    continue
                nxt += 1.0 / fps
                if fr.shape[0] > 720:
                    fr = cv2.resize(fr, (fr.shape[1] * 720 // fr.shape[0], 720), interpolation=cv2.INTER_AREA)
                _, lh, rh, face, bs = hol.frame(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), t * 1000.0)
                L["bs"].append(bs.astype(np.float16)); L["face"].append(face.astype(np.float16)); L["t"].append(t); L["clip"].append(ci)
                if k % 2 == 0 and np.abs(face).sum() > 0:
                    box = crop_box(face, fr.shape[1], fr.shape[0])
                    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); g = crop48(gray, box)
                    C["g"].append(g); C["g96"].append(crop48(gray, box, 96)); C["t"].append(t); C["clip"].append(ci)
                    C["q"].append(face_quality(g, box[2], hand_overlap(face, [lh[:, :2], rh[:, :2]])))
                k += 1
            cap.release(); hol.close(); os.remove(p)
            f = [int(x) for x in Path(n).stem.split("-")]
            meta["name"].append(Path(n).stem); meta["emotion"].append(f[2]); meta["intensity"].append(f[3]); meta["statement"].append(f[4])
    clip = np.array(L["clip"]); off = np.r_[0, np.cumsum(np.bincount(clip, minlength=len(meta["name"])))]
    np.savez_compressed(OUT / f"actor_{actor:02d}.npz", bs=np.stack(L["bs"]), face=np.stack(L["face"]), t=np.array(L["t"], np.float32), off=off,
                        crop_g=np.stack(C["g"]), crop_g96=np.stack(C["g96"]), crop_q=np.array(C["q"], np.float32), crop_t=np.array(C["t"], np.float32),
                        crop_clip=np.array(C["clip"]), actor=np.int32(actor), fps=np.float32(fps), **{k: np.array(v) for k, v in meta.items()})
    return len(meta["name"]), len(clip), len(C["g"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--actors", default="1-24"); ap.add_argument("--keep-ahead", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.actors.split("-"))
    todo = [k for k in range(lo, hi + 1) if not (OUT / f"actor_{k:02d}.npz").exists()]
    src = {k: LOCAL / f"Video_Speech_Actor_{k:02d}.zip" for k in todo if (LOCAL / f"Video_Speech_Actor_{k:02d}.zip").exists()}
    remote = [k for k in todo if k not in src]
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)                                       # one download at a time (disk), the next one while harvesting
    fut = {}

    def ahead(i):
        for k in remote[i:i + 1 + a.keep_ahead]:
            if k not in fut:
                fut[k] = pool.submit(download, URL.format(k), TMP / f"actor_{k:02d}.zip")
    ahead(0)
    for k in todo:
        t0 = time.time()
        if k in src:
            z = src[k]
        else:
            ahead(remote.index(k)); z = fut[k].result()
        n = harvest(z, k)
        if k not in src:
            z.unlink()
        print(f"actor {k:02d}: {n[0]} clips, {n[1]} frames @10fps, {n[2]} crops, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
