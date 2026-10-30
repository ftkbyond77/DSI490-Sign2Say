"""Face-emotion training data for models/face-model/: RAVDESS video (Livingstone & Russo 2018, Zenodo 1188976, CC BY-NC-SA 4.0 —
research use) → the production extractor (MediaPipe Holistic, modules/mediapipe_pose.py) → 52 blendshapes + 68 face points per frame.

    python main.py face-data ravdess [--actors 1-24] [--fps 10]   # → cloud_s3/face-data/ravdess/actor_XX.npz (features only)

One actor zip (~0.5 GB) at a time: download → the 60 video-only speech clips (modality 02) → landmarks → zip deleted. Labels come
from the file name: modality-channel-EMOTION-INTENSITY-statement-repetition-ACTOR (emotion 1 neutral · 2 calm · 3 happy · 4 sad ·
5 angry · 6 fearful · 7 disgust · 8 surprised; intensity 1 normal · 2 strong). No video is kept.
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "cloud_s3" / "face-data" / "ravdess"
TMP = ROOT / "cache" / "face_tmp"
URL = "https://zenodo.org/records/1188976/files/Video_Speech_Actor_{:02d}.zip?download=1"
RAVDESS_EMO = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fear", 7: "disgust", 8: "surprise"}


def download(url, path, tries=8):
    """Resumable (HTTP Range) download into path.part → path. Zenodo serves ~0.3 MB/s per connection, so several run in parallel."""
    import requests
    part = Path(str(path) + ".part")
    for k in range(tries):
        try:
            have = part.stat().st_size if part.exists() else 0
            with requests.get(url, stream=True, timeout=60, headers={"Range": f"bytes={have}-"} if have else {}) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                with open(part, "ab" if have and r.status_code == 206 else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            break
        except Exception as e:                       # noqa: BLE001 — retry flaky downloads
            print(f"  {path.name}: download interrupted ({e}), retry {k + 1}", flush=True)
            time.sleep(10)
    else:
        raise RuntimeError(f"cannot download {url}")
    part.replace(path)


def cmd_ravdess(a):
    from concurrent.futures import ThreadPoolExecutor
    from modules.mediapipe_pose import extract
    OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.actors.split("-"))
    todo = [k for k in range(lo, hi + 1) if not (OUT / f"actor_{k:02d}.npz").exists()]
    pool = ThreadPoolExecutor(a.parallel)
    fut = {k: pool.submit(lambda k=k: (TMP / f"actor_{k:02d}.zip").exists() or download(URL.format(k), TMP / f"actor_{k:02d}.zip")) for k in todo}
    for actor in todo:
        out = OUT / f"actor_{actor:02d}.npz"
        z = TMP / f"actor_{actor:02d}.zip"
        fut[actor].result()
        t = time.time()
        rec = dict(bs=[], face=[], t=[], name=[], emotion=[], intensity=[], statement=[], actor=[])
        with zipfile.ZipFile(z) as zf:
            names = sorted(n for n in zf.namelist() if Path(n).name.startswith("02-") and n.endswith(".mp4"))
            for n in names:
                p = TMP / Path(n).name
                p.write_bytes(zf.read(n))
                raw = extract(p, fps=a.fps, normalize=False)
                p.unlink()
                if raw is None:
                    continue
                f = [int(x) for x in Path(n).stem.split("-")]
                rec["bs"].append(raw["bs"].astype(np.float16)); rec["face"].append(raw["face"].astype(np.float16)); rec["t"].append(raw["t"])
                rec["name"].append(Path(n).stem); rec["emotion"].append(RAVDESS_EMO[f[2]]); rec["intensity"].append(f[3])
                rec["statement"].append(f[4]); rec["actor"].append(f[6])
        off = np.r_[0, np.cumsum([len(b) for b in rec["bs"]])]
        np.savez_compressed(out, bs=np.concatenate(rec["bs"]), face=np.concatenate(rec["face"]), t=np.concatenate(rec["t"]), off=off,
                            name=np.array(rec["name"]), emotion=np.array(rec["emotion"]), intensity=np.array(rec["intensity"]),
                            statement=np.array(rec["statement"]), actor=np.array(rec["actor"]), fps=np.float32(a.fps))
        z.unlink()
        print(f"actor {actor:02d}: {len(rec['name'])} clips, {off[-1]} frames, {time.time() - t:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["ravdess"])
    ap.add_argument("--actors", default="1-24")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    dict(ravdess=cmd_ravdess)[a.step](a)


if __name__ == "__main__":
    main()
