"""Tier A pose extractor (v10): RTMW whole-body — 133 COCO-WholeBody points with a score per point — on a TRACKED signer box,
ONNX Runtime on the GPU. The alternative to MediaPipe Holistic (modules/mediapipe_pose.py) for a machine / server with a GPU.

Restored from the v4–v6 corpus extractor (git 86098cb^:modules/wholebody.py: YOLOX signer lock + RTMW-X on the upper-body box) and
rebuilt for a live camera:
  * the person detector (YOLOX-tiny, 416²) runs only every `redetect` frames or when the track is lost; in between the box comes
    from the previous frame's own body keypoints (`upper_body_box`) — temporal tracking instead of a detector pass per frame;
  * one RTMW pass per frame on the 256×192 upper-body crop (head, shoulders and the reachable signing space get the pixels);
    `fp16=True` runs a half-precision copy of the model (built once next to the FP32 file, inputs / outputs stay FP32);
  * scores are decoded like the corpus extraction (min of the two SimCC axis maxima), so tier-A landmarks are the same kind of
    numbers as the RTMW training clips of cloud_s3/data_prep/pose/.
Output = the MediaPipe raw dict (t, hw, pose 33×4, lh / rh 21×3, face 68×2, bs 52 = zeros: RTMW has no blendshapes) PLUS
kp133 [T,133,2] (x/W, y/H) and sc133 [T,133]. Every consumer of MediaPipe output (hands-up rule, framing, face cues) keeps working,
and `mediapipe_pose.to_standard` harmonises kp133 / sc133 exactly like the RTMW corpus (schema.from_rtmw: scores kept, face
template, hand gaps ≤ 0.2 s filled). A hand whose mean score is below `hand_thr` is written as missing in lh / rh (MediaPipe's
convention); the scores themselves are kept in sc133.

    wb = WholeBody(model="x", device="auto", fp16=True)
    fr = wb.frame(rgb, t_ms)                 # one camera frame → dict(pose, lh, rh, face, bs, kp133, sc133)
    raw = extract("video.mp4")               # a whole file at 25 fps
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

CKPT = Path(os.environ.get("RTMW_DIR", Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"))
MODELS = {   # name → (file, input W×H, download zip)
    "x": ("rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx", (192, 256),
          "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip"),
    "x384": ("rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx", (288, 384),        # v10 Pose Gate candidate: more pixels per hand
             "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"),
    "l": ("rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx", (192, 256),
          "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip"),
}
DETECTOR = ("yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx", (416, 416),
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip")
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
# COCO-WholeBody body 17 → MediaPipe pose 33 (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
COCO2MP = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}
FACE68 = slice(23, 91); LH133 = slice(91, 112); RH133 = slice(112, 133)


def ensure(name_url):
    """Checkpoint file (the rtmlib cache by default, RTMW_DIR to override); downloaded + unzipped on first use."""
    fname, _, url = name_url
    p = CKPT / fname
    if p.exists():
        return p
    import io
    import urllib.request
    import zipfile
    CKPT.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(url, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        member = next(m for m in z.namelist() if m.endswith(".onnx"))
        p.write_bytes(z.read(member))
    return p


def fp16_copy(path):
    """<model>.fp16.onnx next to the FP32 file (float16 weights / compute, float32 inputs / outputs); built once."""
    out = Path(path).with_suffix(".fp16.onnx")
    if not out.exists():
        import onnx
        from onnxconverter_common import float16
        m = float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True)
        onnx.save(m, str(out))
    return out


def upper_body_box(body, W, H, min_conf=0.3):
    """body [17,3] (px, score) → xyxy covering head, shoulders and the reachable signing space of both hands (v4 rule)."""
    ok = body[:, 2] >= min_conf
    if ok[5] and ok[6]:
        sw = float(np.linalg.norm(body[5, :2] - body[6, :2]))
    elif ok[:11].sum() >= 3:
        sw = float(max(np.ptp(body[:11][ok[:11], 0]), 1.0)) * 0.6
    else:
        return None
    pts = body[:11][ok[:11], :2]
    cx = float(np.mean(body[[5, 6], 0])) if ok[5] and ok[6] else float(pts[:, 0].mean())
    top = float(pts[:, 1].min()) - 0.9 * sw
    bottom = max(float(pts[:, 1].max()), float(np.mean(body[[5, 6], 1])) + 1.8 * sw) + 0.2 * sw
    half = 1.35 * sw
    xs = [cx - half, cx + half] + [float(body[k, 0]) + s * 0.45 * sw for k in (9, 10) if ok[k] for s in (-1, 1)]
    box = np.array([min(xs), top, max(xs), bottom], np.float32)
    return np.clip(box, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])


def to_mp_raw_frame(kp, sc, hand_thr=0.3):
    """One frame of 133 points (x/W, y/H) + scores → the MediaPipe raw frame (pose 33×4 with visibility = score, hands 21×3 or zeros,
    face 68×2, no blendshapes)."""
    pose = np.zeros((33, 4), np.float32)
    for c, m in COCO2MP.items():
        pose[m] = (kp[c, 0], kp[c, 1], 0.0, sc[c])
    for m, j in ((9, 23 + 54), (10, 23 + 48)):        # MediaPipe mouth left / right ← iBUG 54 (signer's left) / 48
        pose[m] = (kp[j, 0], kp[j, 1], 0.0, sc[j])
    lh = np.zeros((21, 3), np.float32); rh = np.zeros((21, 3), np.float32)
    if sc[LH133].mean() >= hand_thr:
        lh[:, :2] = kp[LH133]
    if sc[RH133].mean() >= hand_thr:
        rh[:, :2] = kp[RH133]
    face = kp[FACE68].astype(np.float32) if sc[FACE68].mean() >= hand_thr else np.zeros((68, 2), np.float32)
    return pose, lh, rh, face, np.zeros(52, np.float32)


class WholeBody:
    """YOLOX-tiny signer lock + box tracking + RTMW (x | l) on the upper-body crop. ONNX Runtime providers from
    modules.runtime.ort_providers (CUDA → DirectML → CPU)."""

    def __init__(self, model="x", device="auto", fp16=True, redetect=10, hand_thr=0.3):
        import onnxruntime as ort
        from .runtime import ort_providers, ort_session
        self.prov = ort_providers(device)
        so = ort.SessionOptions(); so.log_severity_level = 3
        gpu = self.prov[0] != "CPUExecutionProvider"
        path = ensure(MODELS[model])
        self.fp16 = bool(fp16 and gpu)
        self.pose = ort_session(fp16_copy(path) if self.fp16 else path, so, self.prov)
        self.det = ort_session(ensure(DETECTOR), so, self.prov)
        self.prov = self.pose.get_providers()               # what actually loaded (a failed GPU provider → CPU)
        self.in_wh = MODELS[model][1]
        self.redetect, self.hand_thr = int(redetect), hand_thr
        self.reset()
        self.timing = dict(det_ms=[], pose_ms=[])

    def reset(self):
        self.box, self.n, self.lost = None, 0, True

    # ------------------------------------------------------------------ detector
    def detect(self, bgr):
        """Largest confident person box (xyxy px) — the signer; None if nobody is in the picture. The YOLOX export is end-to-end
        (NMS inside): dets [1, N, 5] (x0, y0, x1, y1, score) in input pixels + labels [1, N] (0 = person)."""
        import cv2
        H, W = bgr.shape[:2]
        S = DETECTOR[1][0]
        r = min(S / H, S / W)
        img = np.full((S, S, 3), 114, np.uint8)
        img[:int(H * r), :int(W * r)] = cv2.resize(bgr, (int(W * r), int(H * r)), interpolation=cv2.INTER_LINEAR)
        dets, labels = self.det.run(None, {self.det.get_inputs()[0].name: img.transpose(2, 0, 1)[None].astype(np.float32)})
        b = dets[0][(labels[0] == 0) & (dets[0][:, 4] > 0.5)]
        if not len(b):
            return None
        boxes = b[:, :4] / r
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return np.clip(boxes[int(np.argmax(area))], [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1]).astype(np.float32)

    # ------------------------------------------------------------------ pose on a box
    def _crop(self, bgr, box):
        """Top-down crop exactly as the corpus extraction (rtmlib bbox_xyxy2cs(padding 1.25) + top_down_affine, rotation 0): the box
        padded, widened to the model's aspect, centre → crop centre, uniform scale."""
        import cv2
        x0, y0, x1, y1 = np.asarray(box, np.float32)
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        w, h = (x1 - x0) * 1.25, (y1 - y0) * 1.25
        W_, H_ = self.in_wh
        scale = np.array([w, w * H_ / W_] if w > h * W_ / H_ else [h * W_ / H_, h], np.float32)
        s = W_ / scale[0]
        M = np.array([[s, 0, W_ / 2 - s * center[0]], [0, s, H_ / 2 - s * center[1]]], np.float32)
        img = cv2.warpAffine(bgr, M, (int(W_), int(H_)), flags=cv2.INTER_LINEAR)
        x = ((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x), center, scale

    def _pose(self, bgr, box):
        x, center, scale = self._crop(bgr, box)
        sx, sy = self.pose.run(None, {self.pose.get_inputs()[0].name: x})
        mx, my = sx.max(2)[0], sy.max(2)[0]
        loc = np.stack([sx.argmax(2)[0], sy.argmax(2)[0]], 1).astype(np.float32) / 2.0        # SimCC split ratio 2
        kp = loc / np.array(self.in_wh, np.float32) * scale + center - scale / 2
        return kp, np.minimum(mx, my).astype(np.float32)                                      # corpus scores: min of the two axes

    def frame(self, rgb, t_ms=None, bgr=None):
        """One frame (RGB like MediaPipe; or bgr=) → dict(pose, lh, rh, face, bs, kp133 [133,2] x/W y/H, sc133 [133])."""
        if bgr is None:
            bgr = np.ascontiguousarray(rgb[..., ::-1])
        H, W = bgr.shape[:2]
        if self.box is None or self.lost or self.n % self.redetect == 0:
            t0 = time.perf_counter()
            b = self.detect(bgr)
            self.timing["det_ms"].append((time.perf_counter() - t0) * 1000)
            if b is not None and (self.box is None or self.lost):
                self.box = _expand_person_box(b, W, H)     # first frame / lost track; otherwise keep the keypoint-tracked box
        self.n += 1
        if self.box is None:
            kp = np.zeros((133, 2), np.float32); sc = np.zeros(133, np.float32)
        else:
            t0 = time.perf_counter()
            kp, sc = self._pose(bgr, self.box)
            self.timing["pose_ms"].append((time.perf_counter() - t0) * 1000)
            nb = upper_body_box(np.c_[kp[:17], sc[:17]], W, H)
            self.lost = nb is None or float(sc[[5, 6]].min()) < 0.3
            if nb is not None:
                self.box = nb
            kp = kp / np.array([W, H], np.float32)
        pose, lh, rh, face, bs = to_mp_raw_frame(kp, sc, self.hand_thr)
        return dict(pose=pose, lh=lh, rh=rh, face=face, bs=bs, kp133=kp.astype(np.float32), sc133=np.clip(sc, 0, None).astype(np.float32))


def _expand_person_box(b, W, H):
    """Detector person box → a box around its upper half with arm room (the first frame; afterwards the keypoints track it)."""
    x0, y0, x1, y1 = b
    h, w = y1 - y0, x1 - x0
    cx = (x0 + x1) / 2
    half = max(w, 0.55 * h) * 0.75
    box = np.array([cx - half, y0 - 0.05 * h, cx + half, y0 + 0.75 * h], np.float32)
    return np.clip(box, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])


def stack_frames(frames, ts, hw):
    """Per-frame dicts of WholeBody.frame + timestamps (s) → the raw dict (MediaPipe keys + kp133 / sc133)."""
    st = lambda k: np.stack([f[k] for f in frames]).astype(np.float32)          # noqa: E731
    t = np.asarray(ts, np.float64)
    return dict(t=t - t[0], hw=np.array(hw, np.int32), pose=st("pose"), lh=st("lh"), rh=st("rh"), face=st("face"), bs=st("bs"),
                kp133=st("kp133"), sc133=st("sc133"), src_fps=np.float32(len(t) / max(t[-1] - t[0], 1e-3)), extractor="rtmw")


def extract(video, fps=25.0, model="x", device="auto", fp16=True, redetect=10, wb=None, max_s=None):
    """Video file → raw dict (frames sampled at ≤ fps on their own timestamps, like mediapipe_pose.extract)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(src_fps) or src_fps <= 1 or src_fps > 240:
        src_fps = 25.0
    wb = wb or WholeBody(model, device, fp16, redetect)
    wb.reset()
    frames, ts, i, next_t, hw = [], [], 0, 0.0, None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = i / src_fps; i += 1
            if max_s is not None and t > max_s:
                break
            if t + 1e-6 < next_t:
                continue
            next_t = t + 1.0 / fps - 0.25 / src_fps
            hw = bgr.shape[:2]
            frames.append(wb.frame(None, t * 1000.0, bgr=bgr)); ts.append(t)
    finally:
        cap.release()
    return stack_frames(frames, ts, hw) if frames else None


def save_raw(path, raw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: (v.astype(np.float16) if isinstance(v, np.ndarray) and v.dtype == np.float32 and k != "t" else v)
                                 for k, v in raw.items() if k != "extractor"}, extractor="rtmw")
