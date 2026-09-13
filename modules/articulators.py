"""L1 Articulator extraction: person detection (YOLOX-tiny, every N frames) + RTMPose-s body
→ signer lock (wrist confidence + size + layout zone, with hysteresis) → crop geometry for
L-hand / R-hand / face at native resolution → 112 px crops + validity, pose features, active mask."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from .utils import load_cfg

CKPT = Path(os.path.expanduser("~/.cache/rtmlib/hub/checkpoints"))
NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LELB, RELB, LWRI, RWRI = range(11)
STREAMS = ("hand_l", "hand_r", "face")   # anatomical left/right of the signer
GRAY = 114


def _session_opts(sess_obj, threads: int, device: str):
    """rtmlib does not expose thread counts; rebuild the ORT session so N workers do not oversubscribe."""
    if device != "cpu":
        return
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess_obj.session = ort.InferenceSession(sess_obj.onnx_model, sess_options=so, providers=["CPUExecutionProvider"])


class Articulators:
    def __init__(self, device: str = "cpu", threads: int = 2, crop_size: int | None = None, cfg: dict | None = None):
        import contextlib, io
        from rtmlib import RTMPose, YOLOX
        self.cfg = cfg or load_cfg("data")
        a = self.cfg["articulators"]
        self.crop_size = crop_size or self.cfg["crop_size"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.det = YOLOX(str(CKPT / a["det"]["model"]), model_input_size=tuple(a["det"]["input"]),
                             score_thr=a["det"]["score_thr"], backend="onnxruntime", device=device)
            self.pose = RTMPose(str(CKPT / a["pose"]["model"]), model_input_size=tuple(a["pose"]["input"]),
                                backend="onnxruntime", device=device)
        _session_opts(self.det, threads, device)
        _session_opts(self.pose, threads, device)
        self.det_every = a["det"]["every_n_frames"]
        self.max_persons = a["det"]["max_persons"]
        self.ema = a["box_ema"]
        self.min_conf = a["min_kpt_conf"]
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, frame_wh: tuple[int, int] | None = None, signer_zone=None):
        self.i = 0
        self.lock_box = None
        self.lock_score = 0.0
        self.switch_count = 0
        self.prev = None
        self.wh = frame_wh
        self.zone = signer_zone
        self.last_geom = np.zeros((3, 3), np.float32)

    # ------------------------------------------------------------------ helpers
    def _score(self, box, kp, sc, W, H, max_area):
        wl = self.cfg["signer_lock"]["weights"]
        wrist = float(np.mean(sc[[LWRI, RWRI]]))
        area = (box[2] - box[0]) * (box[3] - box[1]) / max(max_area, 1)
        zone = 1.0
        if self.zone is not None:
            cx, cy = (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H
            z = self.zone
            zone = float(z[0] <= cx <= z[2] and z[1] <= cy <= z[3])
        return wl["wrist_conf"] * wrist + wl["size"] * area + wl["zone"] * zone

    @staticmethod
    def _iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / max(u, 1e-6)

    def _box_from_kpts(self, kp, sc, W, H):
        ok = sc[:11] >= self.min_conf
        if ok.sum() < 4:
            return None
        pts = kp[:11][ok]
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        sw = self._shoulder_width(kp, sc) or (x1 - x0) * 0.5
        box = np.array([x0 - 0.8 * sw, y0 - 0.8 * sw, x1 + 0.8 * sw, max(y1, kp[LSHO, 1]) + 1.0 * sw], np.float32)
        return np.clip(box, [0, 0, 0, 0], [W, H, W, H])

    def _shoulder_width(self, kp, sc):
        if sc[LSHO] >= self.min_conf and sc[RSHO] >= self.min_conf:
            return float(np.linalg.norm(kp[LSHO] - kp[RSHO]))
        if sc[LEAR] >= self.min_conf and sc[REAR] >= self.min_conf:
            return 2.6 * float(np.linalg.norm(kp[LEAR] - kp[REAR]))
        return None

    def geometry(self, kp, sc):
        """[3,3] (cx, cy, side) for hand_l, hand_r, face and validity [3]."""
        c = self.cfg["articulators"]["crops"]
        geom = self.last_geom.copy()
        valid = np.zeros(3, np.uint8)
        sw = self._shoulder_width(kp, sc)
        if sw is None or sw < 4:
            return geom, valid
        for j, (wri, elb) in enumerate(((LWRI, LELB), (RWRI, RELB))):
            if sc[wri] < self.min_conf:
                continue
            w = kp[wri]
            if sc[elb] >= self.min_conf:
                fore = kp[wri] - kp[elb]
                ctr = w + c["hand"]["center_shift"] * fore
                side = c["hand"]["side_ratio"] * np.linalg.norm(fore) * c["hand"]["pad"]
            else:
                ctr, side = w, 0.0
            side = float(np.clip(side, c["hand"]["min_side_shoulder"] * sw, 1.2 * sw))
            geom[j] = (ctr[0], ctr[1], side)
            valid[j] = 1
        if sc[NOSE] >= self.min_conf:
            eyes = [kp[k] for k in (LEYE, REYE) if sc[k] >= self.min_conf]
            ctr = np.mean([kp[NOSE]] + eyes, axis=0)
            if sc[LEAR] >= self.min_conf and sc[REAR] >= self.min_conf:
                side = c["face"]["side_ratio_ears"] * np.linalg.norm(kp[LEAR] - kp[REAR]) * c["face"]["pad"]
            else:
                side = 0.0
            side = float(np.clip(side, c["face"]["min_side_shoulder"] * sw, 1.1 * sw))
            geom[2] = (ctr[0], ctr[1], side)
            valid[2] = 1
        self.last_geom = geom
        return geom, valid

    def crops(self, frame, geom, size=None):
        size = size or self.crop_size
        out = np.full((3, size, size, 3), GRAY, np.uint8)
        for j in range(3):
            cx, cy, side = geom[j]
            if side <= 1:
                continue
            s = size / side
            M = np.array([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]], np.float32)
            out[j] = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(GRAY, GRAY, GRAY))
        return out

    # ------------------------------------------------------------------ main step
    def step(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        kp = np.zeros((17, 2), np.float32)
        sc = np.zeros(17, np.float32)
        found = False
        if self.i % self.det_every == 0 or self.lock_box is None:
            boxes = self.det(frame)
            boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
            if len(boxes):
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes = boxes[np.argsort(-areas)[: self.max_persons]]
                kps, scs = self.pose(frame, bboxes=boxes.tolist())
                scores = [self._score(b, k, s, W, H, areas.max()) for b, k, s in zip(boxes, kps, scs)]
                best = int(np.argmax(scores))
                cur = None
                if self.lock_box is not None:
                    ious = [self._iou(b, self.lock_box) for b in boxes]
                    if max(ious) > 0.3:
                        cur = int(np.argmax(ious))
                pick = best
                if cur is not None and cur != best:
                    if scores[best] > self.cfg["signer_lock"]["switch_ratio"] * scores[cur]:
                        self.switch_count += 1
                    else:
                        self.switch_count = 0
                    pick = best if self.switch_count >= self.cfg["signer_lock"]["switch_patience"] else cur
                    if pick == best:
                        self.switch_count = 0
                kp, sc = kps[pick], scs[pick]
                self.lock_box = boxes[pick]
                found = True
        elif self.lock_box is not None:
            kps, scs = self.pose(frame, bboxes=[self.lock_box.tolist()])
            kp, sc = kps[0], scs[0]
            found = True
        if found:
            kb = self._box_from_kpts(kp, sc, W, H)
            if kb is not None:
                self.lock_box = self.ema * self.lock_box + (1 - self.ema) * kb
            elif self.i % self.det_every != 0:
                self.lock_box = None
        self.i += 1
        geom, valid = self.geometry(kp, sc) if found else (self.last_geom.copy(), np.zeros(3, np.uint8))
        return np.concatenate([kp, sc[:, None]], 1).astype(np.float32), geom, valid


# ---------------------------------------------------------------------- per-clip post-processing
def pose_features(kpts: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
    """[T,17,3] → [T,44]: upper-body (11 kpts) xy normalised by mid-shoulder & shoulder width + velocity."""
    T = len(kpts)
    xy, c = kpts[:, :11, :2], kpts[:, :11, 2]
    ok_s = (kpts[:, LSHO, 2] >= min_conf) & (kpts[:, RSHO, 2] >= min_conf)
    mid = (kpts[:, LSHO, :2] + kpts[:, RSHO, :2]) / 2
    sw = np.linalg.norm(kpts[:, LSHO, :2] - kpts[:, RSHO, :2], axis=1)
    if ok_s.any():
        med_sw = np.median(sw[ok_s])
        mid[~ok_s] = np.median(mid[ok_s], 0)
        sw[~ok_s] = med_sw
    sw = np.maximum(sw, 1.0)
    f = (xy - mid[:, None]) / sw[:, None, None]
    f[c < min_conf] = 0
    f = f.reshape(T, 22)
    v = np.diff(f, axis=0, prepend=f[:1])
    return np.clip(np.concatenate([f, v], 1), -5, 5).astype(np.float32)


_MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def mirror_kpts(kpts: np.ndarray, width: int) -> np.ndarray:
    """Horizontal mirror of COCO-17 keypoints with left/right identities swapped (matches flip_swap)."""
    k = kpts[:, _MIRROR].copy()
    k[..., 0] = np.where(k[..., 2] > 0, width - 1 - k[..., 0], 0)
    return k


def active_mask(kpts: np.ndarray, mode: str = "clip", raise_thr: float = 0.15, move_thr: float = 0.12,
                dilate: int = 1, min_conf: float = 0.3) -> np.ndarray:
    """Rest-pose trimming: a frame is active if a wrist is raised above its rest height or moving."""
    T = len(kpts)
    if T == 0:
        return np.zeros(0, bool)
    sw = np.linalg.norm(kpts[:, LSHO, :2] - kpts[:, RSHO, :2], axis=1)
    sw = np.where(sw > 1, sw, np.median(sw[sw > 1]) if (sw > 1).any() else 100.0)
    act = np.zeros(T, bool)
    for w in (LWRI, RWRI):
        y = kpts[:, w, 1] / sw
        x = kpts[:, w, 0] / sw
        ok = kpts[:, w, 2] >= min_conf
        if ok.sum() < 2:
            continue
        if mode == "clip":
            edge = np.r_[np.arange(min(3, T)), np.arange(max(T - 3, 0), T)]
            edge = edge[ok[edge]]
            rest = np.median(y[edge]) if len(edge) else np.percentile(y[ok], 90)
        else:  # streaming / continuous: rest = lowest typical wrist position (largest y)
            rest = np.percentile(y[ok], 90)
        raised = (rest - y) > raise_thr
        sp = np.r_[0, np.hypot(np.diff(x), np.diff(y))]
        if T >= 3:  # 3-tap median suppresses single-frame keypoint jitter
            sp = np.median(np.stack([np.r_[sp[:1], sp[:-1]], sp, np.r_[sp[1:], sp[-1:]]]), 0)
        act |= ok & (raised | (sp > move_thr))
    if dilate:
        k = np.ones(2 * dilate + 1)
        act = np.convolve(act.astype(float), k, "same") > 0
    return act


def chroma_masks(crops: np.ndarray, lo, hi) -> np.ndarray:
    """[T,3,S,S,3] BGR → [T,3,S,S] uint8 (255 = chroma background)."""
    T, K, S = crops.shape[:3]
    hsv = cv2.cvtColor(crops.reshape(T * K, S, S, 3).reshape(-1, S, 3), cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return m.reshape(T, K, S, S)


def mask_regions(frames: np.ndarray, regions, fill) -> np.ndarray:
    if not regions:
        return frames
    frames = frames.copy()
    H, W = frames.shape[1:3]
    for x0, y0, x1, y1 in regions:
        frames[:, int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = fill
    return frames


def parse_regions(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    return json.loads(s) if isinstance(s, str) else s
