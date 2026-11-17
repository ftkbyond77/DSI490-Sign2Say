"""Labelled-video check of the face branch on RAVDESS (Livingstone & Russo 2018, CC BY-NC-SA 4.0): actors say two fixed sentences with an
acted emotion. Unlike real_check.py (unlabelled signer clips), every clip here has a label, so this measures both over-reacting AND missed
expressions. It is NOT sign language and the emotions are acted (usually stronger than a signer's face).

  python video_eval.py [--bundles ../models/face-model-efficientnetb0-v1 candidates/face-model-efficientnetb0-v2-ferplus]
  -> cache/ravdess_faces.npz (face crops, built once)  +  results/video_eval_ravdess.json

Data: ../data_test/ravdess/Video_Speech_Actor_XX.zip (Zenodo 1188976). Only the video-only files (modality 02) are used.
File name = modality-channel-emotion-intensity-statement-repetition-actor; emotion 01 neutral, 02 calm (left out: no FER class),
03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised; intensity 01 normal, 02 strong.
Frames: one every 0.25 s between 25 % and 90 % of the clip (the expression is built up after the onset), same crop / quality as the app.
Clip score = mean probability of its usable frames (quality >= 0.25). Run with an environment that has mediapipe (conda env thaislm).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MP_HOLISTIC_MODEL", str(ROOT / "models" / "model-v10" / "mediapipe" / "holistic_landmarker.task"))
from modules.face_fer import CLASSES, FaceFER, crop48, crop_box, face_quality, hand_overlap   # noqa: E402

RAV = ROOT / "data_test" / "ravdess"
CACHE = ROOT / "Face-Lab" / "cache" / "ravdess_faces.npz"
EMO = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"}   # 02 calm: no FER class


def harvest(every_s=0.25, lo=0.25, hi=0.90, scale=1.22, shift_up=0.16):
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
                cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 30; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); dur = n / fps
                hol = Holistic(); i = 0; last = -9
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    t = i / fps; i += 1
                    if t < lo * dur or t > hi * dur or t - last < every_s:
                        continue
                    if fr.shape[0] > 720:
                        fr = cv2.resize(fr, (fr.shape[1] * 720 // fr.shape[0], 720), interpolation=cv2.INTER_AREA)
                    _, lh, rh, face, _ = hol.frame(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), t * 1000.0)
                    if np.abs(face).sum() == 0:
                        continue
                    last = t
                    box = crop_box(face, fr.shape[1], fr.shape[0], scale, shift_up)
                    g = crop48(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), box)
                    x, y, sd = [int(v) for v in box]                              # colour crop of the same box (for colour models, e.g. HSEmotion)
                    sub = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)[max(y, 0):y + sd, max(x, 0):x + sd]
                    col = cv2.resize(sub, (160, 160), interpolation=cv2.INTER_AREA) if min(sub.shape[:2]) > 8 else np.zeros((160, 160, 3), np.uint8)
                    rows.append(dict(col=col, clip=os.path.basename(name)[:-4], actor=int(parts[6]), label=CLASSES.index(EMO[parts[2]]), strong=parts[3] == "02",
                                     t=t, g=g, q=face_quality(g, box[2], hand_overlap(face, [lh[:, :2], rh[:, :2]]))))
                cap.release(); hol.close(); os.remove(p)
        print(os.path.basename(z), len(rows), flush=True)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **{k: np.array([r[k] for r in rows]) for k in ("clip", "actor", "label", "strong", "t", "q")}, g=np.stack([r["g"] for r in rows]), col=np.stack([r["col"] for r in rows]))


def scores(y, p, k=len(CLASSES)):
    """accuracy, macro-F1 over the classes present in y, per-class recall / F1, and the two numbers that matter for the app:
    expression recall (non-neutral truth answered as SOME non-neutral class) and neutral kept (neutral truth answered neutral)."""
    nu = CLASSES.index("neutral")
    pc, f1s = {}, []
    for c in range(k):
        tp, fp, fn = int(((p == c) & (y == c)).sum()), int(((p == c) & (y != c)).sum()), int(((p != c) & (y == c)).sum())
        if (y == c).sum() == 0:
            continue
        r = tp / (tp + fn); pr = tp / (tp + fp) if tp + fp else 0.0; f = 2 * pr * r / (pr + r) if pr + r else 0.0
        pc[CLASSES[c]] = dict(recall=round(r, 3), precision=round(pr, 3), f1=round(f, 3), n=int((y == c).sum())); f1s.append(f)
    return dict(n=int(len(y)), accuracy=round(float((p == y).mean()), 4), macro_f1=round(float(np.mean(f1s)), 4),
                expression_recall=round(float((p[y != nu] != nu).mean()), 4) if (y != nu).any() else None,
                neutral_kept=round(float((p[y == nu] == nu).mean()), 4) if (y == nu).any() else None,
                predicted_neutral_share=round(float((p == nu).mean()), 4), per_class=pc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", default=[str(ROOT / "models" / "face-model-efficientnetb0-v1"), str(ROOT / "Face-Lab" / "candidates" / "face-model-efficientnetb0-v2-ferplus")])
    ap.add_argument("--priors", nargs="+", type=float, default=[0.0, 0.4, 0.6])
    ap.add_argument("--actors", nargs="+", type=int, default=None, help="only these actors (train.py uses 1-2 as dev, so 3-4 is the clean test)")
    ap.add_argument("--out", default="video_eval_ravdess.json")
    a = ap.parse_args()
    if not CACHE.exists():
        harvest()
    R = np.load(CACHE)
    ok = (R["q"] >= 0.25) & (np.isin(R["actor"], a.actors) if a.actors else True)
    clip, lab, strong, g = R["clip"][ok], R["label"][ok], R["strong"][ok], R["g"][ok]
    clips = sorted(set(clip))
    out = dict(source="RAVDESS video-only speech, actors " + ",".join(str(x) for x in sorted(set(R["actor"][ok]))), n_frames=int(ok.sum()), n_frames_low_quality=int((~ok).sum()),
               n_clips=len(clips), clips_per_class={CLASSES[c]: int(sum(1 for cl in clips if lab[clip == cl][0] == c)) for c in range(7)}, models={})
    for b in a.bundles:
        m = FaceFER(Path(b)); nm = Path(b).name.replace("face-model-", "")
        z = np.concatenate([m.logits(g[i:i + 256]) for i in range(0, len(g), 256)]); out["models"][nm] = {}   # batches: one big batch at res 112 runs out of memory
        for pr in a.priors:
            bias = np.array([pr if c == "neutral" else 0.0 for c in m.classes])
            P = np.exp((z + m.T * bias) / m.T - ((z + m.T * bias) / m.T).max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            cp = np.stack([P[clip == cl].mean(0) for cl in clips]); cy = np.array([lab[clip == cl][0] for cl in clips]); cs = np.array([strong[clip == cl][0] for cl in clips])
            out["models"][nm][f"prior_{pr}"] = dict(frames=scores(lab, P.argmax(1)), clips=scores(cy, cp.argmax(1)),
                                                    clips_normal_intensity=scores(cy[~cs], cp[~cs].argmax(1)), clips_strong_intensity=scores(cy[cs], cp[cs].argmax(1)))
            c = out["models"][nm][f"prior_{pr}"]["clips"]
            print(f"{nm:28s} prior {pr:.1f}  clip acc {c['accuracy']:.3f}  macroF1 {c['macro_f1']:.3f}  expression recall {c['expression_recall']:.3f}  neutral kept {c['neutral_kept']}", flush=True)
    (ROOT / "Face-Lab" / "results" / a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
