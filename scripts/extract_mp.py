"""Re-extract every video source with the production extractor (MediaPipe Holistic, modules/mediapipe_pose.py) — local CPU, free.

  python scripts/extract_mp.py [--datasets a,b] [--workers 5] [--limit N]

Reads data_prep/params/video_items.csv (clip id → raw video; `python main.py prep pose_items`) + any extra video lists in
cloud_s3/agent_data/*/videos.csv, writes
  data_prep/pose_mp_raw/<dataset>/<clip>.npz   raw MediaPipe arrays (pose 33, hands, 68 face points, 52 blendshapes, timestamps)
  data_prep/pose_mp/<dataset>/<clip>.npz       Standard Schema (extractor 'mp') — what training, banks and evaluation read
Resumable: finished clips are skipped. YouTube sources are cropped to the signer box found by the v6 RTMW signer lock (an inset
interpreter becomes a full-size signer); dictionary videos are used full-frame.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from modules.utils import CACHE, DATA, PREP, RAW, get_logger, safe_id

log = get_logger("extract_mp")
OUT_RAW, OUT = PREP / "pose_mp_raw", PREP / "pose_mp"
ORDER = ["tsl51_user_sentence", "tsl51_user_sign", "ttrs", "th_sl", "agent", "youtube_sentence", "youtube_continuous", "youtube_bigsign"]


def signer_box(rtmw_raw, hw_video, pad=0.6):
    """Static crop (pixels) around the RTMW-locked signer: upper body + hands over the whole clip, padded by `pad` shoulder widths."""
    z = np.load(rtmw_raw)
    kp, sc = z["kp"].astype(np.float32), z["sc"].astype(np.float32)
    H, W = hw_video
    pts = np.concatenate([kp[:, list(range(0, 11)) + list(range(91, 133))]], 1)
    ok = np.concatenate([sc[:, list(range(0, 11)) + list(range(91, 133))]], 1) > 0.5
    if ok.sum() < 50:
        return None
    xy = pts[ok] * np.array([W, H])
    x0, y0 = np.percentile(xy, 1, axis=0); x1, y1 = np.percentile(xy, 99, axis=0)
    shok = (sc[:, 5] > 0.5) & (sc[:, 6] > 0.5)
    sw = float(np.median(np.linalg.norm((kp[shok, 5] - kp[shok, 6]) * np.array([W, H]), axis=1))) if shok.any() else (x1 - x0) / 2
    x0, x1 = x0 - pad * sw, x1 + pad * sw
    y0, y1 = y0 - 0.9 * sw, y1 + pad * sw
    box = [max(0, x0), max(0, y0), min(W, x1), min(H, y1)]
    if (box[2] - box[0]) * (box[3] - box[1]) > 0.8 * W * H:          # signer already fills the frame
        return None
    return box


def _one(item):
    from modules.mediapipe_pose import extract, handedness_check, save_raw, to_standard
    from modules.schema import save
    cid, dataset, video, out_name, rtmw_raw = item
    out_raw, out = OUT_RAW / dataset / f"{out_name}.npz", OUT / dataset / f"{out_name}.npz"
    if out.exists() and out_raw.exists():
        return cid, "skip", 0, None
    if not Path(video).exists():
        return cid, "missing", 0, None
    t0 = time.time()
    try:
        crop = None
        if rtmw_raw and Path(rtmw_raw).exists():
            import cv2
            cap = cv2.VideoCapture(str(video)); hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))); cap.release()
            crop = signer_box(rtmw_raw, hw)
        raw = extract(video, fps=25.0, crop=crop)
        if raw is None or len(raw["t"]) < 2:
            return cid, "empty", 0, None
        save_raw(out_raw, raw)
        save(out, to_standard(raw, src=dataset))
        return cid, "ok", len(raw["t"]), dict(sec=round(time.time() - t0, 1), hand_check=handedness_check(raw),
                                               hands=float(((np.abs(raw["lh"]).sum((1, 2)) > 0) | (np.abs(raw["rh"]).sum((1, 2)) > 0)).mean()),
                                               body=float((raw["pose"][:, [11, 12], 3].min(1) > 0.5).mean()), crop=crop)
    except Exception as e:  # noqa: BLE001
        return cid, f"error: {type(e).__name__}: {str(e)[:160]}", 0, None


def items(datasets=None):
    it = pd.read_csv(PREP / "params" / "video_items.csv")              # python main.py prep pose_items
    rows = []
    for r in it.itertuples():
        yt = r.dataset.startswith("youtube")
        rtmw_raw = PREP / "pose_raw" / r.out_rel if yt else None
        rows.append((r.clip_id, r.dataset, str(RAW / r.rel_path), Path(r.out_rel).stem, str(rtmw_raw) if rtmw_raw else None))
    for f in sorted((DATA / "agent_data").glob("*/videos.csv")):         # harvested by this project (scripts/harvest_words.py)
        a = pd.read_csv(f)
        rows += [(r.clip_id, "agent", str(f.parent / r.video), safe_id(r.clip_id), None) for r in a.itertuples()]
    if datasets:
        rows = [r for r in rows if r[1] in datasets]
    rank = {d: i for i, d in enumerate(ORDER)}
    return sorted(rows, key=lambda r: rank.get(r[1], 99))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--reverse", action="store_true", help="work from the end of the list (a second process can share the load)")
    a = p.parse_args()
    todo = items(set(a.datasets.split(",")) if a.datasets else None)
    todo = [r for r in todo if not ((OUT / r[1] / f"{r[3]}.npz").exists() and (OUT_RAW / r[1] / f"{r[3]}.npz").exists())]
    if a.reverse:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    log.info("MediaPipe extraction: %d clips to do (%s)", len(todo), pd.Series([r[1] for r in todo]).value_counts().to_dict() if todo else {})
    from multiprocessing import Pool
    t0, frames, done, bad, qc = time.time(), 0, 0, [], []
    with Pool(a.workers, maxtasksperchild=200) as ex:
        for cid, st, n, info in ex.imap_unordered(_one, todo, chunksize=1):
            done += 1; frames += n
            if st.startswith(("error", "missing", "empty")):
                bad.append((cid, st))
            elif info:
                qc.append(dict(clip_id=cid, frames=n, **{k: v for k, v in info.items() if k != "crop"}))
            if done % 200 == 0 or done == len(todo):
                el = time.time() - t0
                log.info("%d/%d clips · %d frames · %.1f fps · %.0f min elapsed · ETA %.0f min · problems %d", done, len(todo), frames,
                         frames / max(el, 1), el / 60, el / done * (len(todo) - done) / 60, len(bad))
    if qc:
        q = pd.DataFrame(qc)
        f = PREP / "qc" / "mp_extraction.csv"
        q.to_csv(f, mode="a", header=not f.exists(), index=False)
        log.info("handedness check (left-hand root nearer the left wrist): median %.3f · hands seen %.2f · body seen %.2f",
                 q.hand_check.median(), q.hands.mean(), q.body.mean())
    if bad:
        log.warning("problems: %s", bad[:20])


if __name__ == "__main__":
    main()
