"""Helpers for Face-lab.ipynb: the face branch on a video clip and on the webcam, the way the web app uses it
(MediaPipe face points -> 48x48 gray crop -> models/face-model-*-v1 -> EMA + hysteresis).  Nothing here touches a sign model."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MP_HOLISTIC_MODEL", str(ROOT / "models" / "model-v10" / "mediapipe" / "holistic_landmarker.task"))
from modules.face_fer import CLASSES, ExpressionSmoother, FaceBranch, crop48, crop_box, face_quality, hand_overlap   # noqa: E402

COL = {"angry": (60, 60, 230), "disgust": (60, 160, 60), "fear": (200, 80, 160), "happy": (60, 200, 240), "sad": (230, 120, 60),
       "surprise": (240, 200, 60), "neutral": (200, 200, 200), "unknown": (120, 120, 120)}


def find_model():
    return sorted((ROOT / "models").glob("face-model-*"))[-1]


class Tracker:
    """One frame at a time: Holistic -> face crop -> raw result -> smoothed label (the notebook's stand-in for the browser client)."""

    def __init__(self, model_dir=None, every_s=0.18):
        from modules.mediapipe_pose import Holistic
        self.fb = FaceBranch(model_dir or find_model(), threads=2)
        self.hol = Holistic()
        self.sm = ExpressionSmoother(self.fb.model.classes, **self.fb.sm_cfg)
        self.crop_cfg = {k: self.fb.model.cfg["crop"][k] for k in ("scale", "shift_up")}
        self.every, self.last, self.shown, self.raw = every_s, -1e9, None, None
        self.log = []

    def close(self):
        self.hol.close()

    def step(self, bgr, t_s):
        """-> (shown dict | None, raw dict | None, box). Runs the classifier at most every `every_s` seconds."""
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pose, lh, rh, face, bs = self.hol.frame(rgb, t_s * 1000.0)
        if np.abs(face).sum() == 0:
            self.raw = dict(expression="unknown", valid=False, confidence=0.0, quality=0.0, probs=None)
            return self.sm.update(dict(self.raw, t=t_s, probs={c: 0 for c in CLASSES})), self.raw, None
        h, w = bgr.shape[:2]
        box = crop_box(face, w, h, **self.crop_cfg)
        if t_s - self.last >= self.every:
            self.last = t_s
            g = crop48(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), box)
            q = face_quality(g, box[2], hand_overlap(face, [lh[:, :2], rh[:, :2]]))
            self.raw = self.fb.model.frame_results(g[None], [q], [t_s])[0]
            self.shown = self.sm.update(self.raw)
            self.log.append((t_s, self.raw["top"], self.raw["top_confidence"], self.raw["expression"], self.shown["expression"], q))
        return self.shown, self.raw, box


def draw(bgr, shown, raw, box):
    import cv2
    if box is None:
        cv2.putText(bgr, "no face", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * max(bgr.shape[0] / 480.0, 1.0), (0, 0, 255), 2)
        return bgr
    x, y, s = (int(v) for v in box)
    e = shown["expression"] if shown else "unknown"
    k = max(bgr.shape[0] / 480.0, 1.0)                                  # text / line size follows the picture size
    cv2.rectangle(bgr, (x, y), (x + s, y + s), COL[e], max(2, int(2 * k)))
    txt = f"{e} {shown['confidence']:.0%}" if shown and shown["valid"] else "unknown"
    if raw is not None:
        txt += f"  | raw {raw['top']} {raw['top_confidence']:.0%} q{raw['quality']:.2f}"
    cv2.putText(bgr, txt, (max(x, 4), max(y - int(8 * k), int(18 * k))), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, COL[e], max(2, int(2 * k)))
    return bgr


def run_video(path, out_path=None, max_s=None, step=1, max_h=720, sheet_every_s=0.5):
    """Whole clip -> (log, annotated thumbnails). The log rows are (t, raw top, raw conf, raw label, smoothed label, quality);
    frames are downscaled to max_h first (phone clips are 1080p / 4K); out_path = optional annotated mp4."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tr, i, thumbs, last_sheet, vw = Tracker(), 0, [], -1e9, None
    try:
        while True:
            ok, fr = cap.read()
            if not ok or (max_s and i / fps > max_s):
                break
            if i % step == 0:
                if fr.shape[0] > max_h:
                    fr = cv2.resize(fr, (int(fr.shape[1] * max_h / fr.shape[0]), max_h), interpolation=cv2.INTER_AREA)
                shown, raw, box = tr.step(fr, i / fps)
                out = draw(fr, shown, raw, box)
                if out_path:
                    if vw is None:
                        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (out.shape[1], out.shape[0]))
                    vw.write(out)
                if i / fps - last_sheet >= sheet_every_s:
                    last_sheet = i / fps
                    thumbs.append((i / fps, cv2.cvtColor(cv2.resize(out, (out.shape[1] * 360 // out.shape[0], 360)), cv2.COLOR_BGR2RGB)))
            i += 1
    finally:
        cap.release(); tr.close()
        if vw is not None:
            vw.release()
    return tr.log, thumbs


def stability(log):
    """raw vs smoothed: label changes per second, unknown share, mean segment length (s)."""
    if len(log) < 3:
        return {}
    t = np.array([l[0] for l in log]); raw = [l[3] for l in log]; sm = [l[4] for l in log]
    dur = max(t[-1] - t[0], 1e-6)
    ch = lambda s: sum(a != b for a, b in zip(s[:-1], s[1:]))   # noqa: E731
    return dict(seconds=round(dur, 1), raw_changes_per_s=round(ch(raw) / dur, 2), smoothed_changes_per_s=round(ch(sm) / dur, 2),
                raw_unknown=round(float(np.mean([r == "unknown" for r in raw])), 3), smoothed_unknown=round(float(np.mean([s == "unknown" for s in sm])), 3),
                mean_segment_s=round(dur / max(ch(sm) + 1, 1), 2))


def webcam(seconds=30, cam=0, width=640):
    """Live view inside the notebook (stop early with the notebook's interrupt button): the same pipeline as the web app, frames shown
    as JPEGs (OpenCV here has no window support). Returns the log; prints the achieved FPS and the classifier latency."""
    import cv2
    from IPython.display import Image, display
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if not cap.isOpened():
        raise RuntimeError("camera not available")
    tr, t0, n = Tracker(), time.perf_counter(), 0
    handle = display(display_id=True)
    try:
        while time.perf_counter() - t0 < seconds:
            ok, fr = cap.read()
            if not ok:
                break
            t = time.perf_counter() - t0
            shown, raw, box = tr.step(fr, t)
            n += 1
            out = draw(fr.copy(), shown, raw, box)
            cv2.putText(out, f"{n / max(t, 1e-3):.0f} fps", (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            handle.update(Image(data=cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release(); tr.close()
    print(f"{n} frames in {time.perf_counter() - t0:.1f}s = {n / max(time.perf_counter() - t0, 1e-3):.1f} fps")
    return tr.log
