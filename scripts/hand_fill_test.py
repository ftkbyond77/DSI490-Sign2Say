"""Hand-fallback check on real videos with two-hand contact / hand-on-body signs (ชื่อ, สวัสดี, ขอบคุณ, ขอโทษ, ฉัน).

Runs Holistic and a HandLandmarker on every frame once, then compares the fill rules offline:
  holistic  — Holistic alone
  v8        — modules.mediapipe_pose.fill_hands as released in v8 (wrist-distance identity)
  new       — fill_hands after the v8.1 fix (whole-hand identity, contact-safe assignment)
Metric = hand_report "hit" (raised visible wrist with 21 points) per side, and duplicates (a filled hand that is the other side's
hand). Videos are subsampled to ~15 fps like the browser.

  python scripts/hand_fill_test.py [--per_word 6] [--out artifacts/reports/hand_fill_test.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules import mediapipe_pose as mpp  # noqa: E402

WORDS = ["ชื่อ", "สวัสดี", "ขอบคุณ", "ขอโทษ", "ฉัน"]


def videos(per_word):
    out = []
    us = ROOT / "cloud_s3" / "raw_data" / "TSL51" / "videos" / "user_sign"
    for w in WORDS:
        fs = sorted(f for f in us.glob("*.mp4") if f.name.split("_")[0] == w)[:per_word]
        out += [("tsl51_user", w, f) for f in fs]
    import pandas as pd
    c = pd.read_parquet(ROOT / "cloud_s3" / "data_prep" / "manifest" / "clips.parquet", columns=["dataset", "concept", "source_file"])
    c = c[(c.dataset == "th_sl") & c.concept.isin(WORDS)]
    out += [("th_sl", r.concept, ROOT / "cloud_s3" / r.source_file) for r in c.itertuples() if (ROOT / "cloud_s3" / r.source_file).exists()]
    return out


def run_video(path, step_fps=15.0):
    from mediapipe.tasks.python import BaseOptions, vision
    import mediapipe as mp
    hol = mpp.Holistic(hand_fallback=False)
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(mpp.HAND_MODEL_PATH)), running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / step_fps)))
    rows, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t_ms = i * 1000.0 / fps
            pose, lh, rh, _, _ = hol.frame(rgb, t_ms)
            f = dict(pose=pose, lh=lh, rh=rh)
            h = hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), int(t_ms))
            det = [np.array([[p.x, p.y, p.z] for p in lm], np.float32) for lm in (h.hand_landmarks or [])]
            rows.append((t_ms / 1000.0, f, det))
        i += 1
    cap.release(); hol.close(); hands.close()
    return rows


def evaluate(rows, rule):
    T = len(rows)
    raw = dict(t=np.array([r[0] for r in rows]), pose=np.zeros((T, 33, 4), np.float32), lh=np.zeros((T, 21, 3), np.float32),
               rh=np.zeros((T, 21, 3), np.float32))
    fb, dup = [], 0
    for i, (_, f, det) in enumerate(rows):
        pose, lh, rh = f["pose"], f["lh"].copy(), f["rh"].copy()
        raw["pose"][i] = pose
        filled = (False, False)
        if rule != "holistic" and np.abs(pose).sum() > 0 and det:
            fn = mpp.fill_hands_v8 if rule == "v8" else mpp.fill_hands
            lh2, rh2, filled = fn(pose, lh, rh, det)
            sw = float(np.hypot(*(pose[11, :2] - pose[12, :2]))) or 0.2
            for k, (a, b) in enumerate(((lh2, rh2), (rh2, lh2))):
                if filled[k] and np.abs(b).sum() > 0 and np.linalg.norm(a[:, :2] - b[:, :2], axis=1).mean() < 0.1 * sw:
                    dup += 1
            lh, rh = lh2, rh2
        raw["lh"][i], raw["rh"][i] = lh, rh
        fb.append(filled)
    rep = mpp.hand_report(raw, fb)
    rep["duplicates"] = dup
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_word", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "reports" / "hand_fill_test.json"))
    a = ap.parse_args()
    res = {}
    for src, w, f in videos(a.per_word):
        rows = run_video(f)
        res[f"{src}:{f.name}"] = dict(word=w, source=src, **{r: evaluate(rows, r) for r in ("holistic", "v8", "new")})
        e = res[f"{src}:{f.name}"]
        print(f"{src:10s} {w:8s} " + "  ".join(f"{r}: L {e[r]['left']['hit']} R {e[r]['right']['hit']} dup {e[r]['duplicates']}" for r in ("holistic", "v8", "new")), flush=True)
    summ = {}
    for r in ("holistic", "v8", "new"):
        for side in ("left", "right"):
            n = sum(v[r][side]["n"] for v in res.values())
            got = sum((v[r][side]["hit"] or 0) * v[r][side]["n"] for v in res.values())
            summ[f"{r}_{side}"] = round(got / max(n, 1), 3)
        summ[f"{r}_duplicates"] = sum(v[r]["duplicates"] for v in res.values())
        summ[f"{r}_max_gap_mean"] = round(float(np.mean([max(v[r]["left"]["max_gap"], v[r]["right"]["max_gap"]) for v in res.values()])), 2)
    by_word = {w: {r: round(float(np.mean([(v[r]["left"]["hit"] or 1) * 0.5 + (v[r]["right"]["hit"] or 1) * 0.5 for v in res.values() if v["word"] == w])), 3)
                   for r in ("holistic", "v8", "new")} for w in WORDS}
    out = dict(summary=summ, by_word=by_word, videos=res)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(summary=summ, by_word=by_word), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
