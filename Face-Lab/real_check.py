"""Real-signer check of the face branch on the 18 test clips (Boom / Jack / Pitty, data_test/team-drive/Eval-Test):
face crops are sampled every 0.5 s (cached in cache/real_faces.npz), then every model bundle is run through the same unknown rule + smoother as the app.
There are NO expression labels for these clips. The judged set below is a SMALL, hand-made one (looked at the crops): 3 clear smiles, the frames that are
ambiguous are left out, every other frame with a usable face (quality >= 0.25) counts as neutral. It tells "stable / over-reacting", not accuracy.

  python real_check.py [--bundles efficientnetb0 resnet18]     -> results/real_check.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "Face-Lab" / "cache" / "real_faces.npz"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Face-Lab"))
os.environ.setdefault("MP_HOLISTIC_MODEL", str(ROOT / "models" / "model-v10" / "mediapipe" / "holistic_landmarker.task"))
from modules.face_fer import CLASSES, ExpressionSmoother, FaceFER   # noqa: E402

HAPPY = {(12, 1.5), (15, 0.0), (15, 1.5)}                                                  # clip index (sorted file list), seconds
AMBIGUOUS = lambda c, t: c == 0 or (c == 3 and t >= 9) or (c == 5 and t >= 7) or (c == 9 and 3.0 <= t <= 3.5) or (c == 4 and 4.4 <= t <= 6.2)   # noqa: E731
EXPRESSIVE = [(12, 1.5, "smile"), (15, 0.0, "smile"), (15, 1.5, "smile"), (3, 10.5, "question brows"), (5, 7.5, "question brows"), (5, 9.0, "question brows"),
              (0, 0.0, "sad line"), (0, 1.5, "sad line"), (9, 3.0, "head back")]


def harvest():
    import demo
    from modules.face_fer import crop48, crop_box, face_quality, hand_overlap
    from modules.mediapipe_pose import Holistic
    import cv2
    fb = demo.FaceBranch(demo.find_model(), threads=2)
    cc = {k: fb.model.cfg["crop"][k] for k in ("scale", "shift_up")}
    rows = []
    clips = sorted(glob.glob(str(ROOT / "data_test" / "team-drive" / "Eval-Test" / "*" / "*.*")))
    for ci, p in enumerate(clips):
        cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 30; hol = Holistic(); i = 0; last = -9
        while True:
            ok, fr = cap.read()
            if not ok: break
            t = i / fps; i += 1
            if t - last < 0.5: continue
            if fr.shape[0] > 720: fr = cv2.resize(fr, (fr.shape[1] * 720 // fr.shape[0], 720), interpolation=cv2.INTER_AREA)
            pose, lh, rh, face, bs = hol.frame(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), t * 1000.0)
            if np.abs(face).sum() == 0: continue
            last = t
            box = crop_box(face, fr.shape[1], fr.shape[0], **cc)
            g = crop48(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), box)
            q = face_quality(g, box[2], hand_overlap(face, [lh[:, :2], rh[:, :2]]))
            x, y, s = [int(v) for v in box]
            col = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)[max(y, 0):y + s, max(x, 0):x + s], (96, 96)) if s > 8 else np.zeros((96, 96, 3), np.uint8)
            pr = fb.model.frame_results(g[None], [q], [t])[0]
            rows.append(dict(clip=ci, name=os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p), t=t, g=g, col=col, q=q, top=pr["top"], conf=pr["top_confidence"], exp=pr["expression"], probs=np.array([pr["probs"][c] for c in demo.CLASSES]), bs=bs))
        cap.release(); hol.close(); print(ci, os.path.basename(p), sum(r["clip"] == ci for r in rows), flush=True)
    np.savez_compressed(CACHE, clip=[r["clip"] for r in rows], name=[r["name"] for r in rows], t=[r["t"] for r in rows], g=np.stack([r["g"] for r in rows]),
                        col=np.stack([r["col"] for r in rows]), q=[r["q"] for r in rows], top=[r["top"] for r in rows], conf=[r["conf"] for r in rows], exp=[r["exp"] for r in rows], probs=np.stack([r["probs"] for r in rows]), bs=np.stack([r["bs"] for r in rows]))
    print("saved", len(rows))


def judged(clip, t, q):
    lab = []
    for i in range(len(q)):
        if q[i] < 0.25:
            lab.append(-1)                                                                  # hand over the face / tiny / dark: the app answers unknown
        elif any(h[0] == clip[i] and abs(h[1] - t[i]) < 0.26 for h in HAPPY):
            lab.append(CLASSES.index("happy"))
        else:
            lab.append(-2 if AMBIGUOUS(clip[i], t[i]) else CLASSES.index("neutral"))
    return np.array(lab)


def run(bundle, g, q, t, clip, prior=None):
    tmp = Path(tempfile.mkdtemp()); shutil.copy(bundle / "face_fer.onnx", tmp)
    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    if prior is not None:
        cfg["prior_shift"] = prior
    (tmp / "config.json").write_text(json.dumps(cfg))
    m = FaceFER(tmp); res = m.frame_results(g, q, t); shown = np.empty(len(g), object)
    for c in sorted(set(clip)):
        sm = ExpressionSmoother(m.classes, **cfg["smoother"])
        for i in np.where(clip == c)[0]:
            shown[i] = sm.update(res[i])["expression"]
    shutil.rmtree(tmp, ignore_errors=True)
    return m, np.array([r["expression"] for r in res]), shown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", default=[p.name.split("-")[2] for p in sorted((ROOT / "models").glob("face-model-*-v1"))])
    a = ap.parse_args()
    if not CACHE.exists():
        CACHE.parent.mkdir(exist_ok=True); harvest()
    R = np.load(CACHE, allow_pickle=True)
    clip, t, g, q = R["clip"], R["t"], R["g"], R["q"]
    lab = judged(clip, t, q); n, h = lab == 6, lab == 3
    out = dict(n_frames=int(len(g)), judged=dict(neutral=int(n.sum()), happy=int(h.sum()), ambiguous=int((lab == -2).sum()), face_covered=int((lab == -1).sum())), models={})
    for nm in a.bundles:
        B = ROOT / "models" / f"face-model-{nm}-v1"
        if not B.exists():
            continue
        rows = {}
        for label, prior in (("no prior", {}), ("shipped", None)):
            m, raw, shown = run(B, g, q, t, clip, prior)
            bias = {c: float(b) for c, b in zip(m.classes, m.bias) if b}
            row = dict(prior=bias, raw_answered_on_neutral=round(float((raw[n] != "unknown").mean()), 3), shown_neutral=round(float((shown[n] == "neutral").mean()), 3),
                       shown_wrong_expression=round(float(((shown[n] != "neutral") & (shown[n] != "unknown")).mean()), 3), shown_unknown=round(float((shown[n] == "unknown").mean()), 3),
                       smiles_shown_happy=f"{int((shown[h] == 'happy').sum())}/{int(h.sum())}",
                       per_signer={w: dict(shown_neutral=round(float((shown[n & np.isin(clip, list(cs))] == "neutral").mean()), 3),
                                          wrong=round(float(((shown[n & np.isin(clip, list(cs))] != "neutral") & (shown[n & np.isin(clip, list(cs))] != "unknown")).mean()), 3))
                                   for w, cs in {"Boom": range(0, 6), "Jack": range(6, 11), "Pitty": range(11, 18)}.items()})
            rows[label] = row
            if prior is not None:
                continue
        m0 = FaceFER(B); m0.bias[:] = 0; p = m0.probs(g)                                      # the model itself, no prior: expressive frames
        ex = []
        for c_, tt, why in EXPRESSIVE:
            i = int(np.argmin(np.where(clip == c_, abs(t - tt), 9e9))); top = np.argsort(-p[i])[:2]
            ex.append(dict(clip=int(c_), t=round(float(t[i]), 1), what=why, top=[(CLASSES[k], round(float(p[i, k]), 2)) for k in top]))
        out["models"][nm] = dict(stability=rows, expressive_frames=ex)
        print(nm, json.dumps(rows["shipped"], ensure_ascii=False)[:400])
    (ROOT / "Face-Lab" / "results" / "real_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
