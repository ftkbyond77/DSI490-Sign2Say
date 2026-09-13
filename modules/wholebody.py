"""v4 L1: whole-body keypoints (RTMW-X, 133 kpts: body 17 · feet 6 · face 68 · left hand 21 · right hand 21).

Signer box comes from the v1 signer-locked body track (or the Articulators when no track exists), expanded to
the upper body so the hands get more pixels than a full-person detector box would give. RTMW runs batched on the
GPU; CPU workers decode + warp crops. Output per clip (Uni-Sign convention):
  artifacts/pose_v4/<clip>.npz   kp [T,133,2] in [0,1] (x/W, y/H) · sc [T,133] · fps · hw · t0
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

import numpy as np

from .utils import ART, get_logger, safe_id

log = get_logger("wholebody")
POSE_DIR = ART / "pose_v4"
RTMW = Path(os.path.expanduser("~/.cache/rtmlib/hub/checkpoints")) / "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx"
INPUT_WH = (192, 256)
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def pose_path(clip_id: str) -> Path:
    return POSE_DIR / f"{safe_id(clip_id)}.npz"


def upper_body_box(body_kp: np.ndarray, W: int, H: int, min_conf=0.3):
    """body_kp [17,3] (px) → xyxy covering head, shoulders and the reachable signing space of both hands."""
    ok = body_kp[:, 2] >= min_conf
    if ok[5] and ok[6]:
        sw = float(np.linalg.norm(body_kp[5, :2] - body_kp[6, :2]))
    elif ok[:11].sum() >= 3:
        pts = body_kp[:11][ok[:11], :2]
        sw = float(max(np.ptp(pts[:, 0]), 1.0)) * 0.6
    else:
        return None
    pts = body_kp[:11][ok[:11], :2]
    cx = float(np.mean(body_kp[[5, 6], 0])) if ok[5] and ok[6] else float(pts[:, 0].mean())
    top = float(pts[:, 1].min()) - 0.9 * sw
    bottom = max(float(pts[:, 1].max()), float(np.mean(body_kp[[5, 6], 1])) + 1.8 * sw) + 0.2 * sw
    half = 1.35 * sw
    xs = [cx - half, cx + half] + [float(body_kp[k, 0]) + s * 0.45 * sw for k in (9, 10) if ok[k] for s in (-1, 1)]
    box = np.array([min(xs), top, max(xs), bottom], np.float32)
    return np.clip(box, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])


def prep_crop(frame: np.ndarray, box: np.ndarray):
    """→ uint8 warped crop [256,192,3] (BGR, as rtmlib), center, scale. Normalisation happens batched on the GPU side."""
    from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
    center, scale = bbox_xyxy2cs(np.asarray(box, np.float32), padding=1.25)
    img, scale = top_down_affine(INPUT_WH, scale, center, frame)
    return np.clip(img, 0, 255).astype(np.uint8), center.astype(np.float32), np.asarray(scale, np.float32)


def normalize_crops(crops_u8: np.ndarray) -> np.ndarray:
    return ((crops_u8.astype(np.float32) - MEAN) / STD).transpose(0, 3, 1, 2).copy()


def decode(simcc_x, simcc_y, centers, scales):
    from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum
    locs, vals = get_simcc_maximum(simcc_x, simcc_y)
    kp = locs / 2.0
    kp = kp / np.array(INPUT_WH, np.float32) * scales[:, None, :]
    kp = kp + centers[:, None, :] - scales[:, None, :] / 2
    return kp, vals


class RTMWBatch:
    def __init__(self, device="cuda"):
        import onnxruntime as ort
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(RTMW), providers=prov)
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, crops_u8: np.ndarray, centers, scales, bs=48):
        kps, scs = [], []
        for i in range(0, len(crops_u8), bs):
            sx, sy = self.sess.run(None, {self.name: normalize_crops(crops_u8[i:i + bs])})
            k, s = decode(sx, sy, centers[i:i + bs], scales[i:i + bs])
            kps.append(k); scs.append(s)
        return np.concatenate(kps), np.concatenate(scs)


# ----------------------------------------------------------------------------- per-video (inference)
def extract_video(path, fps=12.5, device="cuda", start_s=0.0, max_s=None, art=None, rtmw=None):
    """Signer lock with v1 Articulators, then RTMW on the upper-body box. Returns dict like the npz."""
    from .articulators import Articulators
    from .data import decode_frames
    art = art or Articulators(device=device, threads=4)
    rtmw = rtmw or RTMWBatch(device)
    art.reset()
    frames = np.concatenate(list(decode_frames(path, fps=fps, start_s=start_s, max_s=max_s)))
    H, W = frames.shape[1:3]
    crops, cs, ss, last = [], [], [], None
    for f in frames:
        kp, _, _ = art.step(f)
        box = upper_body_box(kp, W, H)
        box = box if box is not None else (last if last is not None else np.array([0, 0, W - 1, H - 1], np.float32))
        last = box
        c, ce, sc = prep_crop(f, box)
        crops.append(c); cs.append(ce); ss.append(sc)
    kp, sc = rtmw(np.stack(crops), np.stack(cs), np.stack(ss))
    return dict(kp=(kp / np.array([W, H], np.float32)).astype(np.float32), sc=sc.astype(np.float32), fps=np.float32(fps),
                hw=np.array([H, W]), t0=np.float32(start_s), frames=frames)


# ----------------------------------------------------------------------------- corpus job (uses v1 tracks)
def _worker(task_q, out_q):
    from .data import decode_frames
    while True:
        item = task_q.get()
        if item is None:
            break
        cid, path, track, start_s, n_frames = item
        try:
            tr = np.load(track)
            bk = tr["kpts"]
            crops, cs, ss, last = [], [], [], None
            i = 0
            H = W = None
            for ch in decode_frames(path, fps=12.5, start_s=start_s, max_s=n_frames / 12.5 + 0.2):
                H, W = ch.shape[1:3]
                for f in ch:
                    if i >= len(bk):
                        break
                    box = upper_body_box(bk[i], W, H)
                    box = box if box is not None else (last if last is not None else np.array([0, 0, W - 1, H - 1], np.float32))
                    last = box
                    c, ce, sc = prep_crop(f, box)
                    crops.append(c); cs.append(ce); ss.append(sc)
                    i += 1
                if len(crops) >= 384:
                    out_q.put(("chunk", cid, np.stack(crops), np.stack(cs), np.stack(ss))); crops, cs, ss = [], [], []
            if crops:
                out_q.put(("chunk", cid, np.stack(crops), np.stack(cs), np.stack(ss)))
            if i == 0:
                out_q.put(("error", cid, "no frames")); continue
            out_q.put(("ok", cid, (H, W), start_s))
        except Exception:
            out_q.put(("error", cid, traceback.format_exc()[-600:]))


def run_corpus(items, workers=8):
    """items: (clip_id, media_path, track_npz, start_s, n_frames)."""
    todo = [it for it in items if not pose_path(it[0]).exists()]
    log.info("wholebody job: %d/%d clips", len(todo), len(items))
    if not todo:
        return
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    rtmw = RTMWBatch("cuda")
    ctx = mp.get_context("spawn")
    task_q, out_q = ctx.Queue(), ctx.Queue(maxsize=workers * 2)
    procs = [ctx.Process(target=_worker, args=(task_q, out_q), daemon=True) for _ in range(workers)]
    for p in procs:
        p.start()
    for it in todo:
        task_q.put(it)
    for _ in procs:
        task_q.put(None)
    done = err = frames = 0
    acc: dict = {}
    t0 = time.time()
    while done + err < len(todo):
        try:
            msg = out_q.get(timeout=900)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                break
            continue
        if msg[0] == "chunk":
            _, cid, crops, cs, ss = msg
            k, s = rtmw(crops, cs, ss)
            acc.setdefault(cid, []).append((k, s))
        elif msg[0] == "ok":
            _, cid, (H, W), start_s = msg
            parts = acc.pop(cid, [])
            kp = np.concatenate([p[0] for p in parts]); sc = np.concatenate([p[1] for p in parts])
            np.savez_compressed(pose_path(cid), kp=(kp / np.array([W, H], np.float32)).astype(np.float16),
                                sc=sc.astype(np.float16), fps=np.float32(12.5), hw=np.array([H, W]), t0=np.float32(start_s))
            done += 1; frames += len(kp)
            if done % 200 == 0:
                el = time.time() - t0
                log.info("%d/%d clips · %d frames · %.0f fr/s · eta %.1f min", done, len(todo), frames, frames / el,
                         el / done * (len(todo) - done) / 60)
        else:
            err += 1
            acc.pop(msg[1], None)
            log.error("%s: %s", msg[1], msg[2])
    log.info("wholebody finished: %d ok, %d errors, %.1f min", done, err, (time.time() - t0) / 60)
